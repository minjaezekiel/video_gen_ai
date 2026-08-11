"""Presets, style tiers, formats and the one config object a render depends on.

Everything that changes how a production looks or how long it takes to render is
declared here as data, so the pipeline never branches on hardware or format in
the middle of a render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
# Quality presets
#
# A preset fixes the *vertical* resolution and the diffusion cost; the aspect
# ratio then decides the width. Keeping them separate means "9:16 at standard
# quality" is expressible without a combinatorial preset table.
# --------------------------------------------------------------------------- #

PRESETS: Dict[str, Dict[str, object]] = {
    "draft": {
        "height": 480,
        "fps": 24,
        "crf": 23,
        "model": "stabilityai/sd-turbo",
        "steps": 1,
        "guidance": 0.0,
        "image_size": 384,
        "negative_prompt": "",
    },
    "standard": {
        "height": 720,
        "fps": 24,
        "crf": 20,
        "model": "stabilityai/sd-turbo",
        "steps": 4,
        "guidance": 0.0,
        "image_size": 512,
        "negative_prompt": "",
    },
    "quality": {
        "height": 1080,
        "fps": 30,
        "crf": 18,
        "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "steps": 25,
        "guidance": 7.5,
        "image_size": 768,
        "negative_prompt": (
            "blurry, low quality, distorted, deformed, watermark, text, "
            "signature, jpeg artifacts, extra limbs"
        ),
    },
}

DEFAULT_PRESET = "standard"

#: width : height multipliers. Widths are rounded to even numbers for H.264.
ASPECTS: Dict[str, float] = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "21:9": 21 / 9,
}

DEFAULT_ASPECT = "16:9"


# --------------------------------------------------------------------------- #
# Style tiers
#
# "2D", "3D" and "realistic" are not three engines -- they are conditioning on
# one pipeline. Each tier supplies a prompt suffix, a negative prompt, and the
# palette the procedural backend uses when no model is available.
# --------------------------------------------------------------------------- #

STYLE_TIERS: Dict[str, Dict[str, object]] = {
    "2d": {
        "label": "2D animation",
        "prompt_suffix": (
            "2D animation still, cel shaded, anime style, clean bold line art, "
            "flat vibrant colours, expressive character design"
        ),
        "negative": "photorealistic, 3d render, photograph, live action",
        "saturation": 1.15,
        "line_art": True,
    },
    "3d": {
        "label": "3D animation",
        "prompt_suffix": (
            "3D animated film still, stylised character render, subsurface "
            "scattering, soft global illumination, volumetric light, depth of field"
        ),
        "negative": "flat, 2d, line art, sketch, photograph",
        "saturation": 1.05,
        "line_art": False,
    },
    "realistic": {
        "label": "Live action",
        "prompt_suffix": (
            "cinematic film still, 35mm anamorphic, shallow depth of field, "
            "natural motivated lighting, photorealistic, colour graded"
        ),
        "negative": "cartoon, anime, 3d render, illustration, painting",
        "saturation": 0.95,
        "line_art": False,
    },
}

DEFAULT_STYLE = "2d"


# --------------------------------------------------------------------------- #
# Production formats
#
# Format drives pacing and structure, not looks: how long, how many shots, how
# dialogue-heavy, and whether the piece needs a call to action or a recap.
# --------------------------------------------------------------------------- #

FORMATS: Dict[str, Dict[str, object]] = {
    "ad": {
        "label": "Advertisement",
        "target_seconds": 30,
        "shot_seconds": (2.0, 4.0),
        "wpm": 170,
        "dialogue_ratio": 0.5,
        "beats": ["hook", "problem", "product", "proof", "call to action"],
        "needs_cta": True,
    },
    "short": {
        "label": "Short film",
        "target_seconds": 90,
        "shot_seconds": (3.0, 7.0),
        "wpm": 140,
        "dialogue_ratio": 0.7,
        "beats": ["setup", "inciting incident", "escalation", "climax", "resolution"],
        "needs_cta": False,
    },
    "educational": {
        "label": "Educational",
        "target_seconds": 120,
        "shot_seconds": (4.0, 8.0),
        "wpm": 145,
        "dialogue_ratio": 0.3,
        "beats": ["question", "context", "explanation", "example", "recap"],
        "needs_cta": False,
    },
    "explainer": {
        "label": "Explainer",
        "target_seconds": 60,
        "shot_seconds": (3.0, 6.0),
        "wpm": 155,
        "dialogue_ratio": 0.35,
        "beats": ["problem", "how it works", "benefit", "next step"],
        "needs_cta": True,
    },
}

DEFAULT_FORMAT = "short"


# --------------------------------------------------------------------------- #
# Shots and pacing
# --------------------------------------------------------------------------- #

SHOT_TYPES: Dict[str, Dict[str, object]] = {
    "establishing": {"framing": "extreme wide shot, full environment", "subject_scale": 0.22},
    "wide": {"framing": "wide shot, full body", "subject_scale": 0.55},
    "medium": {"framing": "medium shot, waist up", "subject_scale": 0.80},
    "close": {"framing": "close-up, head and shoulders", "subject_scale": 1.20},
    "insert": {"framing": "insert shot, detail", "subject_scale": 0.0},
}

CAMERA_MOVES = ("static", "push in", "pull out", "pan left", "pan right", "tilt up")

MIN_SHOT_SECONDS = 1.6
MAX_SHOT_SECONDS = 12.0
SHOT_TAIL_PAD = 0.45
CROSSFADE_SECONDS = 0.35


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class GenerationConfig:
    """Everything that influences a render, resolved into concrete numbers."""

    preset: str = DEFAULT_PRESET
    aspect: str = DEFAULT_ASPECT
    style: str = DEFAULT_STYLE
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[int] = None
    seed: int = 1234
    music: bool = True
    music_volume: float = 0.16
    duck_db: float = -11.0  # how far the bed drops under dialogue
    captions: bool = True
    speaker_labels: bool = True
    use_llm: bool = False
    backend: Optional[str] = None  # force one visual backend, else auto-detect
    output_dir: Path = Path("output")
    scripts_dir: Path = Path("video_scripts")
    characters_dir: Path = Path("characters")
    cache_dir: Path = Path(".cache")

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}")
        if self.aspect not in ASPECTS:
            raise ValueError(f"unknown aspect {self.aspect!r}; choose from {sorted(ASPECTS)}")
        if self.style not in STYLE_TIERS:
            raise ValueError(f"unknown style {self.style!r}; choose from {sorted(STYLE_TIERS)}")

        preset = PRESETS[self.preset]
        if self.resolution is None:
            self.resolution = self.resolution_for(self.preset, self.aspect)
        if self.fps is None:
            self.fps = int(preset["fps"])  # type: ignore[arg-type]
        width, height = self.resolution
        if width % 2 or height % 2:
            raise ValueError("resolution must be even in both dimensions for H.264")

        self.output_dir = Path(self.output_dir)
        self.scripts_dir = Path(self.scripts_dir)
        self.characters_dir = Path(self.characters_dir)
        self.cache_dir = Path(self.cache_dir)

    @staticmethod
    def resolution_for(preset: str, aspect: str) -> Tuple[int, int]:
        """Even-dimensioned frame size for a preset/aspect pair.

        The preset's number is the **short side**, so "standard" means 720p
        whether the video is landscape (1280x720) or portrait (720x1280).
        Treating it as the height instead would make a vertical video 406x720 --
        a quarter of the pixels, and not a resolution anything expects.
        """
        short = int(PRESETS[preset]["height"])  # type: ignore[arg-type]
        ratio = ASPECTS[aspect]
        if ratio >= 1.0:
            width, height = int(round(short * ratio)), short
        else:
            width, height = short, int(round(short / ratio))
        return (width + (width % 2), height + (height % 2))

    @property
    def preset_data(self) -> Dict[str, object]:
        return PRESETS[self.preset]

    @property
    def style_data(self) -> Dict[str, object]:
        return STYLE_TIERS[self.style]

    @property
    def is_portrait(self) -> bool:
        return self.resolution[1] > self.resolution[0]
