"""The always-available visual tier: composed environments and staged figures.

This runs anywhere, with no model, no download and no GPU. It is not photoreal
and does not pretend to be; it aims to look *designed* -- a graded environment
with a horizon and a light source, and clean geometric figures staged by shot
type -- so a film rendered here reads as a deliberate motion-graphics style
rather than as a broken render.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .base import IDENTITY, STILL, CharacterRef, ShotRequest, ShotVisual

# Shot type -> (figure height as a fraction of frame height, vertical anchor).
_STAGING = {
    "establishing": (0.30, 0.86),
    "wide": (0.62, 0.92),
    "medium": (1.05, 1.30),
    "close": (1.90, 2.05),
    "insert": (0.0, 0.0),
}


def _hsv_rgb(h: float, s: float, v: float) -> np.ndarray:
    swatch = Image.new("HSV", (1, 1), (int(h * 255) % 256, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))))
    return np.array(swatch.convert("RGB").getpixel((0, 0)), dtype=np.float32)


def character_palette(name: str, seed: int = 0) -> Tuple[int, int, int]:
    """A stable, distinct colour per character name."""
    from ..model import stable_seed

    hue = (stable_seed(name.upper(), seed) % 1000) / 1000.0
    rgb = _hsv_rgb(hue, 165, 235)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def render_environment(
    prompt: str,
    seed: int,
    size: Tuple[int, int],
    palette_seed: Optional[int] = None,
    style: str = "2d",
) -> Image.Image:
    """A graded environment plate: sky, low sun, layered ridges, grain, vignette.

    ``palette_seed`` is shared by every shot in one production and each shot only
    shifts hue slightly from it. Without that, consecutive shots pick unrelated
    hues and the cuts read as glitches rather than as a look.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    palette_rng = np.random.default_rng(seed if palette_seed is None else palette_seed)

    hue = float(palette_rng.uniform(0, 1)) + float(rng.uniform(-0.055, 0.055))
    warm = hue + 0.5 + float(palette_rng.uniform(-0.05, 0.05))
    # 2D wants flatter, punchier colour; realistic wants it desaturated.
    sat = {"2d": (170, 225), "3d": (140, 195), "realistic": (95, 150)}.get(style, (150, 205))

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    v_norm = yy / max(1, height - 1)

    c_high = _hsv_rgb(hue, palette_rng.uniform(*sat), 118)
    c_low = _hsv_rgb(hue - 0.04, palette_rng.uniform(sat[0] * 0.6, sat[1] * 0.75), 228)
    horizon = float(rng.uniform(0.54, 0.72))
    sky = np.clip(v_norm / horizon, 0, 1)[..., None] ** 1.4
    canvas = c_high[None, None, :] * (1 - sky) + c_low[None, None, :] * sky

    sun_x = float(rng.uniform(0.2, 0.8)) * width
    sun_y = horizon * height - float(rng.uniform(0.02, 0.12)) * height
    sun_r = float(rng.uniform(0.10, 0.20)) * max(width, height)
    glow = np.exp(-(((xx - sun_x) ** 2 + (yy - sun_y) ** 2) / (2 * sun_r**2)))
    canvas += glow[..., None] * _hsv_rgb(warm, 90, 255)[None, None, :] * 0.55

    for layer in range(int(rng.integers(2, 4))):
        depth = (layer + 1) / 4.0
        crest = horizon + depth * float(rng.uniform(0.05, 0.13))
        wobble = np.sin(
            xx / width * np.pi * float(rng.uniform(1.0, 2.6)) + rng.uniform(0, 6.28)
        ) * float(rng.uniform(0.012, 0.035))
        edge = (crest + wobble) * height
        ramp = np.clip((yy - edge) / max(1.0, height * 0.006), -60.0, 60.0)
        mask = 1.0 / (1.0 + np.exp(-ramp))
        shade = _hsv_rgb(hue + 0.02 * layer, 170, 92 - 22 * layer)
        canvas = canvas * (1 - mask[..., None] * 0.92) + shade[None, None, :] * (mask[..., None] * 0.92)

    dx = (xx / width - 0.5) * 2
    dy = (yy / height - 0.5) * 2
    canvas *= (1.0 - 0.40 * np.clip(dx**2 + dy**2, 0, 1))[..., None]
    canvas += rng.normal(0, 3.2, canvas.shape).astype(np.float32)

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return image.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 500.0))


def _draw_figure(
    layer: Image.Image,
    centre_x: float,
    baseline_y: float,
    figure_h: float,
    colour: Tuple[int, int, int],
    speaking: bool,
) -> None:
    """A clean stylised figure: head, shoulders, torso.

    Deliberately geometric rather than pseudo-anatomical -- a simple silhouette
    reads as a design choice, where a crude attempt at a person reads as a bug.
    """
    draw = ImageDraw.Draw(layer, "RGBA")
    # Roughly human proportions: a head about one seventh of standing height.
    head_r = figure_h * 0.075
    head_cy = baseline_y - figure_h + head_r
    neck = head_cy + head_r * 1.35
    shoulder_w = figure_h * 0.30
    torso_w = shoulder_w * 0.82

    # Speaking figures are fully saturated and rim-lit; the rest sit back.
    fill = (*colour, 255) if speaking else (*[int(c * 0.72) for c in colour], 195)
    rim = (255, 255, 255, 165 if speaking else 45)

    # Soft contact shadow grounds the figure against the plate.
    draw.ellipse(
        [centre_x - shoulder_w * 0.62, baseline_y - figure_h * 0.02,
         centre_x + shoulder_w * 0.62, baseline_y + figure_h * 0.035],
        fill=(0, 0, 0, 75),
    )
    if baseline_y - neck > 2:
        # Shoulders as a wide rounded cap, torso tapering below it.
        draw.rounded_rectangle(
            [centre_x - shoulder_w / 2, neck, centre_x + shoulder_w / 2, baseline_y],
            radius=max(2.0, shoulder_w * 0.42),
            fill=fill,
        )
        draw.rectangle(
            [centre_x - torso_w / 2, neck + shoulder_w * 0.38, centre_x + torso_w / 2, baseline_y],
            fill=fill,
        )
    draw.ellipse(
        [centre_x - head_r, head_cy - head_r, centre_x + head_r, head_cy + head_r],
        fill=fill,
        outline=rim,
        width=max(1, int(head_r * 0.14)),
    )


def stage_characters(
    plate: Image.Image,
    characters: Sequence[CharacterRef],
    shot_type: str,
    seed: int = 0,
) -> Image.Image:
    """Composite figures onto an environment plate according to shot type."""
    visible = [c for c in characters if c.name]
    scale, anchor = _STAGING.get(shot_type, _STAGING["medium"])
    if not visible or scale <= 0:
        return plate

    width, height = plate.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    figure_h = height * scale
    baseline = height * anchor

    # Spread evenly, with a gentle deterministic jitter so a two-hander is not
    # perfectly symmetrical.
    count = len(visible)
    for index, character in enumerate(visible):
        fraction = (index + 1) / (count + 1)
        jitter = ((seed + index * 37) % 11 - 5) / 100.0
        centre_x = width * (fraction + jitter * 0.4)
        colour = character.palette or character_palette(character.name, seed)
        _draw_figure(layer, centre_x, baseline, figure_h, colour, character.speaking)

    blurred = layer.filter(ImageFilter.GaussianBlur(radius=max(0.6, height / 900.0)))
    return Image.alpha_composite(plate.convert("RGBA"), blurred).convert("RGB")


class ProceduralBackend:
    """Always available. Environments plus staged figures, no model required."""

    name = "procedural"
    priority = 10

    def __init__(self, config=None) -> None:
        self.config = config

    def available(self) -> bool:
        return True

    def capabilities(self) -> Set[str]:
        # IDENTITY is honest here: a character's palette and staging are stable
        # across shots, which is consistency -- just not photographic likeness.
        return {STILL, IDENTITY}

    def describe(self) -> str:
        return "procedural environments and staged figures; no model, runs anywhere"

    @staticmethod
    def plate_size(request: ShotRequest, oversample: float = 1.28) -> Tuple[int, int]:
        """Work at the frame's aspect ratio, not a square.

        Diffusion models are trained on square-ish latents and need one; this
        backend has no such constraint, and rendering square would mean the
        camera-move crop slices the staged figures off at the top and bottom.
        """
        frame_w, frame_h = request.frame_size
        scale = max(1.0, (request.image_size * oversample) / max(1, min(frame_w, frame_h)))
        return (max(64, int(frame_w * scale)), max(64, int(frame_h * scale)))

    def render(self, request: ShotRequest) -> ShotVisual:
        size = self.plate_size(request)
        plate = render_environment(
            request.prompt,
            seed=request.seed,
            size=size,
            palette_seed=request.palette_seed,
            style=request.style,
        )
        staged = stage_characters(plate, request.characters, request.shot_type, seed=request.seed)
        used = {STILL}
        if request.characters:
            used.add(IDENTITY)
        return ShotVisual(backend=self.name, image=staged, used=used)
