"""Production in, MP4 out.

Two invariants drive the design:

* **Measured audio owns the clock.** A shot lasts as long as its dialogue
  actually takes to speak. Computing picture first and stretching sound to fit
  is what makes audio drift away from picture over a long timeline.
* **The whole film is one clip.** Scenes are composited by hand rather than by
  ``concatenate_videoclips(method="compose")``, which runs moviepy's mask
  blitter on every frame instead of only the frames inside a crossfade, and
  measured 4x slower.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy import AudioArrayClip, AudioFileClip, CompositeAudioClip, VideoClip
import moviepy.audio.fx as afx

from .casting import cast_production, discover_voices
from .config import (
    CROSSFADE_SECONDS,
    MAX_AUTO_JOBS,
    MIN_SHOT_SECONDS,
    SHOT_TAIL_PAD,
    GenerationConfig,
)
from .model import Line, Production, Shot, character_key, stable_seed
from . import fastpath, subtitles
from .music import DEFAULT_MOOD, SAMPLE_RATE, duck, to_stereo
from .score import ScoreRequest, default_registry as default_music_registry
from .speech import SpeechSynthesizer, read_wav
from .visuals import CharacterRef, ShotRequest, ShotVisual, character_palette, default_registry

LOGGER = logging.getLogger("video_gen.render")

ProgressCallback = Callable[[float, str], None]

# Camera move -> (start scale, end scale, start pan, end pan). Scale 1.0 is the
# tightest crop the oversample allows; larger values sit further back.
_ZOOM = 1.18
_MOVES: Dict[str, Tuple[float, float, Tuple[float, float], Tuple[float, float]]] = {
    "push in": (_ZOOM, 1.0, (0.5, 0.5), (0.5, 0.5)),
    "pull out": (1.0, _ZOOM, (0.5, 0.5), (0.5, 0.5)),
    "pan left": (1.04, 1.04, (0.85, 0.5), (0.15, 0.5)),
    "pan right": (1.04, 1.04, (0.15, 0.5), (0.85, 0.5)),
    "tilt up": (1.04, 1.04, (0.5, 0.85), (0.5, 0.15)),
    # Never perfectly locked: a completely static frame reads as a freeze.
    "static": (1.02, 1.0, (0.5, 0.52), (0.5, 0.48)),
}


@dataclass
class Caption:
    """A speaker-labelled subtitle, live for one line's span within a shot."""

    start: float
    end: float
    #: (box_h, box_w, 4) uint8. The layer is drawn as RGBA, so keeping it in
    #: 8-bit is lossless and lets both composite paths share one buffer.
    rgba: np.ndarray
    box: Tuple[int, int, int, int]

    @property
    def rgba_bytes(self) -> bytes:
        return np.ascontiguousarray(self.rgba, dtype=np.uint8).tobytes()


@dataclass
class SpokenLine:
    """One line's measured place within its shot, for subtitle sidecars."""

    start: float
    end: float
    character: str
    text: str
    narration: bool


@dataclass
class RenderedShot:
    """A shot reduced to a duration and a function from time to pixels."""

    duration: float
    #: Time within the shot -> raw RGB bytes at the output resolution.
    frame_at: Callable[[float], bytes]
    audio: List[Tuple[float, Path]] = field(default_factory=list)  # (offset, wav)
    speech_windows: List[Tuple[float, float]] = field(default_factory=list)
    #: Per-line spans, in shot-local seconds. Same measured clock as the audio.
    spoken: List[SpokenLine] = field(default_factory=list)


def content_hash(production, config: GenerationConfig) -> str:
    """Identity of a render: everything that would change a single pixel or sample.

    This is what makes `--resume` honest. Hashing the screenplay file's bytes
    would be wrong in both directions -- a comment edit would force a re-render,
    and a changed `--seed` or `--preset` would not. So the hash covers the shot
    list, the cast's voices, and exactly those config fields the output depends
    on. `jobs` is absent on purpose: how many threads rendered a film cannot
    change the film.
    """
    parts: List[str] = [
        production.id,
        production.title,
        production.format,
        production.style,
        production.aspect,
        config.preset,
        # The config's style, not the production's: this is the one the
        # renderer passes to the backend, and it can be overridden per run.
        config.style,
        config.aspect,
        f"{config.resolution[0]}x{config.resolution[1]}@{config.fps}",
        f"seed={config.seed}",
        f"music={int(bool(config.music))}",
        f"captions={int(bool(config.captions))}",
        f"labels={int(bool(config.speaker_labels))}",
        f"lipsync={int(bool(getattr(config, 'lip_sync', False)))}",
        f"backend={config.backend or 'auto'}",
        f"music_backend={config.music_backend or 'auto'}",
        f"volume={config.music_volume:.3f}",
        f"duck={config.duck_db:.2f}",
    ]
    for member in production.cast:
        voice = member.voice
        parts.append(
            f"cast:{member.key}:{voice.voice}:{voice.rate:.4f}:{voice.pitch:.4f}"
            f":{voice.hosted_voice}:{member.reference_image or ''}"
        )
    for scene in production.scenes:
        parts.append(f"scene:{scene.index}:{scene.setting}:{scene.time_of_day}:{scene.mood}")
        for shot in scene.shots:
            # Normalised names, not raw ones: `to_markdown` writes a speaker as
            # `**MAYA:**`, so hashing `line.character` made the hash change the
            # first time a screenplay was saved and loaded back -- which meant
            # --resume never once matched.
            subjects = ",".join(sorted(character_key(name) for name in shot.subjects))
            parts.append(
                f"shot:{shot.index}:{shot.shot_type}:{shot.camera}:{shot.visual}"
                f":{shot.action}:{subjects}"
            )
            for line in shot.lines:
                parts.append(f"line:{line.key}:{line.delivery}:{line.text}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def is_current(production, config: GenerationConfig, output: Path) -> bool:
    """Whether `output` is already this exact render.

    True only when the file exists and the last recorded render of this
    production wrote *that* path from identical content. A render of the same
    production to a different filename does not count.
    """
    output = Path(output)
    if not output.is_file() or output.stat().st_size == 0:
        return False
    wanted = content_hash(production, config)
    for record in reversed(production.renders or []):
        if record.get("content_hash") != wanted:
            continue
        try:
            if Path(str(record.get("output", ""))).resolve() == output.resolve():
                return True
        except OSError:  # noqa: PERF203
            continue
    return False


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def cover_transform(
    src_size: Tuple[int, int], size: Tuple[int, int]
) -> Tuple[float, float, float]:
    """The ``(scale, left, top)`` that ``cover_resize`` applies.

    Returned separately so a point in the source plate -- a mouth, say -- can be
    carried into the resized image without guessing at the same arithmetic in
    two places.
    """
    target_w, target_h = size
    src_w, src_h = src_size
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = max(target_w, int(math.ceil(src_w * scale)))
    resized_h = max(target_h, int(math.ceil(src_h * scale)))
    return scale, (resized_w - target_w) // 2, (resized_h - target_h) // 2


def cover_resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale to fully cover ``size``, centre-cropping the overflow."""
    target_w, target_h = size
    src_w, src_h = image.size
    scale, left, top = cover_transform((src_w, src_h), size)
    resized = image.convert("RGB").resize(
        (
            max(target_w, int(math.ceil(src_w * scale))),
            max(target_h, int(math.ceil(src_h * scale))),
        ),
        Image.LANCZOS,
    )
    return resized.crop((left, top, left + target_w, top + target_h))


#: How many mouth openness steps are pre-drawn. Eight is past the point where a
#: viewer can tell, and it means eight sprites per shot instead of one per frame.
MOUTH_STEPS = 8


def mouth_sprites(width: int, height: int, colour: Tuple[int, int, int]) -> List[bytes]:
    """RGBA sprites for openness 0..1, pre-drawn once per shot.

    Drawing with Pillow inside the per-frame path would cost more than the
    resize it sits next to, so the mouth is quantised and the sprites reused.
    """
    width = max(2, int(width))
    height = max(2, int(height))
    shade = tuple(int(c * 0.30) for c in colour)
    sprites: List[bytes] = []
    for step in range(MOUTH_STEPS):
        openness = step / (MOUTH_STEPS - 1)
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        # A closed mouth is a line; an open one is an ellipse that grows
        # downwards, which is how a jaw actually moves.
        lip = max(1.0, height * 0.16)
        opening = lip + (height - lip) * openness
        top = (height - opening) / 2
        inset = width * 0.5 * (1.0 - 0.22 * openness)
        box = [width / 2 - inset, top, width / 2 + inset, top + opening]
        draw.ellipse(box, fill=(*shade, 225))
        if openness > 0.22:
            # A hint of a lower lip catching the light stops it reading as a hole.
            draw.arc(box, start=20, end=160, fill=(255, 255, 255, 70), width=max(1, int(lip)))
        sprites.append(layer.tobytes())
    return sprites


class FilmRenderer:
    """Renders a `Production` to an MP4 and records the result back into it."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        progress: Optional[ProgressCallback] = None,
        registry=None,
        synthesizer: Optional[SpeechSynthesizer] = None,
        music_registry=None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.progress: ProgressCallback = progress or (lambda pct, msg: None)
        self.registry = registry or default_registry(self.config)
        self.speech = synthesizer or SpeechSynthesizer(config=self.config)
        self.backend = self.registry.select(self.config.backend)
        self.music = music_registry or default_music_registry(
            self.config, getattr(self.config, "providers", None)
        )
        # Which backend actually scored the film, filled in by build_audio, so
        # the render log records what was used rather than what was asked for.
        self.score_backend = ""
        self.score_detail = ""
        #: Sidecars written by the last render, recorded into the screenplay.
        self.subtitle_files: List[Path] = []

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir = self.config.cache_dir / "images"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        # Scratch for intermediate audio only; nothing the user keeps lands here.
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_gen_"))
        self._cleaned = False

    # -- lifecycle --------------------------------------------------------- #

    def cleanup(self) -> None:
        """Idempotent teardown; only ever removes the scratch directory."""
        if self._cleaned:
            return
        self._cleaned = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self) -> "FilmRenderer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    # -- captions ---------------------------------------------------------- #

    def build_caption(self, line: Line, colour: Tuple[int, int, int]) -> Optional[Caption]:
        """Pre-render one subtitle, cropped to its own bounding box.

        Returning a box rather than a full-frame layer keeps per-frame
        compositing to a slice of the image instead of the whole thing.
        """
        text = " ".join(str(line.text).split())
        if not text:
            return None
        width, height = self.config.resolution
        font_size = max(15, int(height * 0.043))
        font = load_font(font_size)
        label_font = load_font(max(12, int(font_size * 0.78)))

        speaker = ""
        if self.config.speaker_labels and not line.is_narration:
            speaker = line.character.upper()

        max_width = int(width * 0.84)
        lines: List[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if not current or font.getbbox(trial)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip(" ,.") + "…"

        line_h = int(font_size * 1.30)
        pad = int(font_size * 0.60)
        label_h = int(label_font.getbbox("Ag")[3] * 1.5) if speaker else 0
        box_h = min(height, line_h * len(lines) + pad * 2 + label_h)
        text_w = max(font.getbbox(l)[2] for l in lines)
        if speaker:
            text_w = max(text_w, label_font.getbbox(speaker)[2])
        box_w = min(width, text_w + pad * 2)
        x0 = max(0, (width - box_w) // 2)
        y0 = max(0, height - box_h - int(height * 0.055))

        layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            [0, 0, box_w - 1, box_h - 1], radius=int(font_size * 0.34), fill=(8, 10, 14, 178)
        )
        cursor = pad
        if speaker:
            # The speaker's own colour is what makes a two-hander readable.
            draw.text(
                ((box_w - label_font.getbbox(speaker)[2]) // 2, cursor - int(pad * 0.25)),
                speaker,
                font=label_font,
                fill=(*colour, 255),
            )
            cursor += label_h
        for index, text_line in enumerate(lines):
            tx = (box_w - font.getbbox(text_line)[2]) // 2
            ty = cursor + index * line_h
            draw.text((tx + 2, ty + 2), text_line, font=font, fill=(0, 0, 0, 190))
            draw.text((tx, ty), text_line, font=font, fill=(246, 247, 250, 255))

        return Caption(
            0.0, 0.0, np.asarray(layer, dtype=np.uint8), (x0, y0, box_w, box_h)
        )

    # -- visuals ----------------------------------------------------------- #

    def _character_refs(self, production: Production, shot: Shot) -> List[CharacterRef]:
        speaking = {line.character.upper() for line in shot.lines if not line.is_narration}
        refs: List[CharacterRef] = []
        for name in shot.subjects:
            member = production.character(name)
            palette = (
                tuple(member.palette)
                if member and member.palette
                else character_palette(name, self.config.seed)
            )
            refs.append(
                CharacterRef(
                    name=member.name if member else name,
                    prompt_fragment=member.prompt_fragment() if member else name,
                    reference_image=Path(member.reference_image)
                    if member and member.reference_image
                    else None,
                    identity_ref=member.identity_ref if member else None,
                    lora=member.lora if member else None,
                    palette=palette,  # type: ignore[arg-type]
                    speaking=name.upper() in speaking,
                )
            )
        return refs

    def shot_visual(self, production: Production, shot: Shot, duration: float) -> ShotVisual:
        """Ask the backend for this shot, with a persistent disk cache."""
        preset = self.config.preset_data
        seed = (self.config.seed + stable_seed(shot.visual or shot.action, shot.index)) % (2**31 - 1)
        request = ShotRequest(
            prompt=shot.visual or shot.action or production.logline,
            negative_prompt=str(preset["negative_prompt"]),
            frame_size=self.config.resolution,
            image_size=int(preset["image_size"]),  # type: ignore[arg-type]
            steps=int(preset["steps"]),  # type: ignore[arg-type]
            guidance=float(preset["guidance"]),  # type: ignore[arg-type]
            seed=seed,
            palette_seed=self.config.seed,
            style=self.config.style,
            shot_type=shot.shot_type,
            camera=shot.camera,
            duration=duration,
            fps=self.config.fps,
            characters=self._character_refs(production, shot),
            want_motion=True,
        )

        cache_key = stable_seed(
            f"{self.backend.name}|{request.prompt}|{self.config.preset}|{self.config.style}"
            f"|{seed}|{shot.shot_type}|{request.image_size}"
        )
        cached = self.image_cache_dir / f"{cache_key:08x}.png"
        if cached.exists():
            try:
                with Image.open(cached) as handle:
                    return ShotVisual(backend=self.backend.name, image=handle.convert("RGB"))
            except Exception:  # noqa: BLE001 - corrupt cache entry
                cached.unlink(missing_ok=True)

        try:
            visual = self.backend.render(request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("backend %s failed on shot %s (%s); using procedural",
                           self.backend.name, shot.index, exc)
            from .visuals.procedural import ProceduralBackend

            visual = ProceduralBackend(self.config).render(request)

        # Only stills are cached; a frame sequence is large and rarely reused.
        if visual.image is not None and not visual.is_motion:
            try:
                # compress_level=1 rather than Pillow's default 6: writing five
                # 1280x720 PNGs was 6.4s of a 43.8s render, almost all of it
                # zlib, for a cache file that is read back once and thrown away.
                # Written beside and renamed so two threads cannot leave a
                # half-written entry behind.
                scratch = cached.with_suffix(f".{threading.get_ident():x}.part")
                visual.image.save(scratch, format="PNG", compress_level=1)
                scratch.replace(cached)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("could not cache shot image: %s", exc)
        return visual

    def _build_openness(
        self, audio: Sequence[Tuple[float, Path]], duration: float
    ) -> Optional[Callable[[float], float]]:
        """Shot-local time -> mouth openness, from the dialogue actually spoken.

        Built per shot from the synthesised wavs, so a mouth moves with the
        words rather than on a loop. Silence between lines closes it, which is
        what makes a shot/reverse-shot pair read correctly.
        """
        if not getattr(self.config, "lip_sync", False) or not audio:
            return None

        fps = float(self.config.fps or 24)
        frames = max(1, int(math.ceil(duration * fps)) + 1)
        levels = [0.0] * frames
        for offset, wav in audio:
            try:
                samples, rate = read_wav(Path(wav))
            except Exception as exc:  # noqa: BLE001 - a missing wav is just silence
                LOGGER.debug("no lip sync for %s: %s", wav, exc)
                continue
            if samples.size == 0 or rate <= 0:
                continue
            span = max(1, int(math.ceil(samples.size / rate * fps)))
            line = fastpath.mouth_openness(samples, rate, fps, span)
            start = int(round(offset * fps))
            for i, value in enumerate(line):
                index = start + i
                if 0 <= index < frames:
                    levels[index] = max(levels[index], float(value))

        def openness(t: float) -> float:
            index = int(t * fps)
            if index < 0 or index >= frames:
                return 0.0
            return levels[index]

        return openness

    def _mouth_plan(
        self, visual: ShotVisual, big: Tuple[int, int]
    ) -> List[Tuple[float, float, int, int, List[bytes]]]:
        """Speaking mouths carried from plate pixels into the zoomed base image.

        Returns `(x, y, w, h, sprites)` per speaking mouth. Empty when lip sync
        is off, when the backend reported no mouths, or when nobody is speaking
        -- all of which simply mean the shot renders as it always did.
        """
        if not getattr(self.config, "lip_sync", False) or visual.image is None:
            return []
        speaking = [m for m in visual.mouths if m.speaking]
        if not speaking:
            return []

        scale, left, top = cover_transform(visual.image.size, big)
        big_w, big_h = big
        plan: List[Tuple[float, float, int, int, List[bytes]]] = []
        for mouth in speaking:
            x = mouth.x * scale - left
            y = mouth.y * scale - top
            width = max(2, int(round(mouth.width * scale)))
            height = max(2, int(round(mouth.height * scale)))
            if not (0 <= x < big_w and 0 <= y < big_h):
                continue  # centre-cropped out of the plate
            plan.append(
                (x, y, width, height, mouth_sprites(width, height, mouth.colour or (210, 170, 160)))
            )
        return plan

    def _base_renderer(
        self,
        visual: ShotVisual,
        duration: float,
        camera: str,
        seed: int,
        openness: Optional[Callable[[float], float]] = None,
    ) -> Callable[[float], bytes]:
        """Frames from a motion clip, or a camera move over a still.

        Returns raw RGB bytes rather than an array so the per-frame chain --
        resize, then caption composite -- never converts between the two.
        """
        out_w, out_h = self.config.resolution

        if visual.is_motion and visual.frames:
            frames = [cover_resize(f, (out_w, out_h)) for f in visual.frames]
            rate = visual.frame_rate or self.config.fps
            buffers = [f.convert("RGB").tobytes() for f in frames]

            def from_frames(t: float) -> bytes:
                # Hold the last frame if the clip is shorter than the dialogue.
                index = min(len(buffers) - 1, max(0, int(t * rate)))
                return buffers[index]

            return from_frames

        big_w, big_h = int(out_w * _ZOOM), int(out_h * _ZOOM)
        base = cover_resize(visual.image, (big_w, big_h))  # type: ignore[arg-type]
        s0, s1, p0, p1 = _MOVES.get(camera, _MOVES["static"])
        # Deterministic jitter so two shots sharing a move are not identical.
        rng = np.random.default_rng(seed)
        jitter = lambda p: (  # noqa: E731
            float(np.clip(p[0] + rng.uniform(-0.06, 0.06), 0, 1)),
            float(np.clip(p[1] + rng.uniform(-0.06, 0.06), 0, 1)),
        )
        p0, p1 = jitter(p0), jitter(p1)
        duration = max(0.1, duration)
        # Converted once per shot, not once per frame: this is a few megabytes
        # and the crop is expressed as an offset into it.
        source = base.convert("RGB").tobytes()
        mouths = self._mouth_plan(visual, (big_w, big_h)) if openness else []

        def ken_burns(t: float) -> bytes:
            u = min(1.0, max(0.0, t / duration))
            eased = u * u * (3.0 - 2.0 * u)  # smoothstep: no jerk at the cuts
            scale = s0 + (s1 - s0) * eased
            crop_w = min(big_w, max(2, int(round(out_w * scale))))
            crop_h = min(big_h, max(2, int(round(out_h * scale))))
            px = p0[0] + (p1[0] - p0[0]) * eased
            py = p0[1] + (p1[1] - p0[1]) * eased
            x = int(round(px * (big_w - crop_w)))
            y = int(round(py * (big_h - crop_h)))
            frame = fastpath.resize_crop(
                source, big_w, big_h, x, y, crop_w, crop_h, out_w, out_h
            )
            if not mouths:
                return frame

            # The crop also scales, so a mouth's size on screen changes as the
            # camera pushes in. Recomputed per frame from the same crop the
            # picture used, which is why this lives inside the closure.
            level = openness(t)
            if level <= 0.02:
                return frame
            step = min(MOUTH_STEPS - 1, max(0, int(round(level * (MOUTH_STEPS - 1)))))
            fx, fy = out_w / crop_w, out_h / crop_h
            for mx, my, mw, mh, sprites in mouths:
                sw = max(2, int(round(mw * fx)))
                sh = max(2, int(round(mh * fy)))
                cx = int(round((mx - x) * fx)) - sw // 2
                cy = int(round((my - y) * fy)) - sh // 2
                if cx < 0 or cy < 0 or cx + sw > out_w or cy + sh > out_h:
                    continue  # the mouth has panned out of frame
                sprite = sprites[step]
                if (sw, sh) != (mw, mh):
                    sprite = (
                        Image.frombuffer("RGBA", (mw, mh), sprite, "raw", "RGBA", 0, 1)
                        .resize((sw, sh), Image.BILINEAR)
                        .tobytes()
                    )
                frame = fastpath.blend_box(frame, out_w, out_h, cx, cy, sw, sh, sprite)
            return frame

        return ken_burns

    # -- shots ------------------------------------------------------------- #

    def render_shot(self, production: Production, shot: Shot, index: int) -> RenderedShot:
        """Synthesise the dialogue, fetch the visual, and bind them to a clock."""
        audio: List[Tuple[float, Path]] = []
        captions: List[Caption] = []
        windows: List[Tuple[float, float]] = []
        spoken: List[SpokenLine] = []
        cursor = 0.0

        for line_number, line in enumerate(shot.lines):
            member = production.character(line.character)
            voice = member.voice if member else None
            wav = self.temp_dir / f"shot{index:03d}_line{line_number:02d}.wav"
            if voice is None:
                from .model import VoiceProfile

                voice = VoiceProfile()
            _, said = self.speech.speak(line.text, voice, wav)

            audio.append((cursor, wav))
            windows.append((cursor, cursor + said))
            spoken.append(
                SpokenLine(cursor, cursor + said, line.character, line.text, line.is_narration)
            )
            colour = (
                tuple(member.palette)
                if member and member.palette
                else character_palette(line.character, self.config.seed)
            )
            caption = self.build_caption(line, colour)  # type: ignore[arg-type]
            if caption is not None and self.config.captions:
                caption.start = cursor
                # Hold the subtitle slightly past the audio so it is readable.
                caption.end = cursor + said + 0.35
                captions.append(caption)
            cursor += said + 0.12  # a beat between lines

        duration = max(MIN_SHOT_SECONDS, cursor + SHOT_TAIL_PAD if shot.lines else shot.duration)
        shot.duration = round(duration, 2)

        visual = self.shot_visual(production, shot, duration)
        base_at = self._base_renderer(
            visual,
            duration,
            shot.camera,
            seed=self.config.seed + index,
            openness=self._build_openness(audio, duration),
        )

        out_w, out_h = self.config.resolution
        overlays = {id(c): c.rgba_bytes for c in captions}

        def frame_at(t: float) -> bytes:
            frame = base_at(t)
            active = next((c for c in captions if c.start <= t < c.end), None)
            if active is None:
                return frame
            x0, y0, box_w, box_h = active.box
            return fastpath.blend_box(
                frame, out_w, out_h, x0, y0, box_w, box_h, overlays[id(active)]
            )

        return RenderedShot(
            duration=duration,
            frame_at=frame_at,
            audio=audio,
            speech_windows=windows,
            spoken=spoken,
        )

    # -- timeline ---------------------------------------------------------- #

    @staticmethod
    def crossfade_for(durations: Sequence[float]) -> float:
        """Crossfade length, capped so it can never swallow a short shot."""
        if len(durations) < 2:
            return 0.0
        fade = min(CROSSFADE_SECONDS, min(durations) / 3.0)
        return fade if fade > 0.05 else 0.0

    @staticmethod
    def shot_starts(durations: Sequence[float], fade: float) -> List[float]:
        """Start time of each shot on the assembled (overlapping) timeline."""
        starts: List[float] = []
        cursor = 0.0
        for duration in durations:
            starts.append(cursor)
            cursor += duration - fade
        return starts

    def build_timeline(self, shots: Sequence[RenderedShot], fade: float) -> VideoClip:
        """Assemble every shot into one clip, dissolving between them.

        Only frames inside an overlap render two shots, so crossfade cost is
        proportional to the overlap rather than to the whole runtime.
        """
        durations = [s.duration for s in shots]
        starts = self.shot_starts(durations, fade)
        total = starts[-1] + durations[-1]
        last = len(shots) - 1

        out_w, out_h = self.config.resolution

        def as_array(buffer: bytes) -> np.ndarray:
            return np.frombuffer(buffer, dtype=np.uint8).reshape(out_h, out_w, 3)

        def frame_function(t: float) -> np.ndarray:
            index = 0
            for i, start in enumerate(starts):
                if t >= start:
                    index = i
                else:
                    break
            index = min(index, last)
            frame = as_array(shots[index].frame_at(t - starts[index]))
            if fade > 0 and index > 0 and t < starts[index - 1] + durations[index - 1]:
                alpha = min(1.0, max(0.0, (t - starts[index]) / fade))
                previous = as_array(shots[index - 1].frame_at(t - starts[index - 1]))
                frame = (
                    previous.astype(np.float32) * (1.0 - alpha)
                    + frame.astype(np.float32) * alpha
                ).astype(np.uint8)
            return frame

        return VideoClip(frame_function=frame_function, duration=total)

    # -- audio ------------------------------------------------------------- #

    def build_audio(
        self,
        production: Production,
        shots: Sequence[RenderedShot],
        starts: Sequence[float],
        total: float,
    ) -> Tuple[CompositeAudioClip, List[AudioFileClip]]:
        """Dialogue pinned to its own shot, plus a scored and ducked bed."""
        clips: List[AudioFileClip] = []
        tracks = []
        windows: List[Tuple[float, float]] = []

        for shot, start in zip(shots, starts):
            for offset, wav in shot.audio:
                clip = AudioFileClip(str(wav))
                clips.append(clip)
                at = start + offset
                if at + clip.duration > total:
                    clip = clip.subclipped(0, max(0.05, total - at))
                tracks.append(
                    clip.with_effects(
                        [afx.AudioFadeIn(0.03), afx.AudioFadeOut(0.08)]
                    ).with_start(at)
                )
            for window_start, window_end in shot.speech_windows:
                windows.append((start + window_start, start + window_end))

        if self.config.music:
            segments: List[Tuple[float, float, str]] = []
            cursor = 0
            for scene in production.scenes:
                shot_count = len(scene.shots)
                if not shot_count or cursor >= len(starts):
                    continue
                begin = starts[cursor]
                last_index = min(len(shots), cursor + shot_count) - 1
                end = starts[last_index] + shots[last_index].duration
                segments.append((begin, min(end, total), scene.mood or DEFAULT_MOOD))
                cursor += shot_count
            request = ScoreRequest(
                segments=segments or [(0.0, total, DEFAULT_MOOD)],
                total_duration=total,
                seed=self.config.seed,
                sample_rate=SAMPLE_RATE,
                style=self.config.style,
                format=production.format,
                title=production.title,
            )
            # The registry falls back to the procedural bed if a hosted
            # provider fails, so a paid outage costs fidelity, never a render.
            score = self.music.render(request, prefer=self.config.music_backend)
            self.score_backend = score.backend
            self.score_detail = score.detail
            bed = duck(
                score.fitted(total, SAMPLE_RATE), windows, depth_db=self.config.duck_db
            )
            tracks.append(
                AudioArrayClip(to_stereo(bed, self.config.music_volume), fps=SAMPLE_RATE)
            )

        return CompositeAudioClip(tracks).with_duration(total), clips

    # -- subtitles --------------------------------------------------------- #

    def _write_subtitles(
        self, rendered: Sequence[RenderedShot], starts: Sequence[float], output: Path
    ) -> List[Path]:
        """Sidecars from the measured per-line timings, offset onto the timeline."""
        if not getattr(self.config, "subtitles", True):
            return []
        spans: List[Tuple[float, float, str, str, bool]] = []
        for shot, start in zip(rendered, starts):
            for line in shot.spoken:
                spans.append(
                    (
                        start + line.start,
                        start + line.end,
                        line.character,
                        line.text,
                        line.narration,
                    )
                )
        cues = subtitles.build_cues(spans)
        if not cues:
            return []
        try:
            return subtitles.write_sidecars(
                output, cues, speaker_labels=self.config.speaker_labels
            )
        except OSError as exc:
            LOGGER.warning("could not write subtitle sidecars: %s", exc)
            return []

    # -- shot scheduling --------------------------------------------------- #

    def worker_count(self, shots: int) -> int:
        """How many shots to render at once.

        Shots are independent: each synthesises its own dialogue and asks the
        backend for its own picture. Most of a shot is spent blocked on a TTS
        subprocess or inside Pillow, both of which release the GIL, so threads
        genuinely overlap -- TTS alone was 25% of a measured render, serialised.

        Clamped to 1 when the visual backend holds one model on one device,
        because concurrent calls would interleave on the same weights.
        """
        if shots <= 1:
            return 1
        if not getattr(self.backend, "parallel_safe", False):
            return 1
        requested = int(getattr(self.config, "jobs", 0) or 0)
        if requested > 0:
            return min(requested, shots)
        return max(1, min(shots, os.cpu_count() or 1, MAX_AUTO_JOBS))

    def _render_shots(
        self, production: Production, shots: Sequence[Shot]
    ) -> Tuple[List[RenderedShot], List[float]]:
        """Render every shot, in order, possibly several at a time."""
        total = len(shots)
        workers = self.worker_count(total)
        rendered: List[Optional[RenderedShot]] = [None] * total
        per_shot: List[float] = [0.0] * total
        done = 0
        guard = threading.Lock()

        def one(index: int) -> None:
            nonlocal done
            shot_started = time.time()
            rendered[index] = self.render_shot(production, shots[index], index)
            per_shot[index] = round(time.time() - shot_started, 2)
            with guard:
                done += 1
                # Reported on completion rather than on start, so the
                # percentage still only moves forwards when shots overlap.
                self.progress(
                    done / total * 90.0,
                    f"Shot {done}/{total} ({shots[index].shot_type})",
                )

        if workers == 1:
            for index in range(total):
                one(index)
        else:
            LOGGER.debug("rendering %d shots across %d workers", total, workers)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot") as pool:
                # list() so the first exception propagates rather than being
                # swallowed with a half-filled shot list.
                list(pool.map(one, range(total)))

        missing = [i for i, value in enumerate(rendered) if value is None]
        if missing:
            raise RuntimeError(f"shots {missing} did not render")
        return [r for r in rendered if r is not None], per_shot

    # -- render ------------------------------------------------------------ #

    def render(self, production: Production, output: Optional[Path] = None) -> Path:
        """Render a production to MP4 and record the result back into its file."""
        shots = production.shots
        if not shots:
            raise ValueError("production has no shots")
        output = Path(output) if output else self.config.output_dir / f"{production.id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        # Anyone without a voice gets cast now, so a hand-written script renders.
        uncast = [c for c in production.cast if not c.voice.voice]
        if uncast:
            cast_production(production.cast, discover_voices())

        # A film with captions and no audible dialogue looks finished and is
        # completely broken, so check the synthesiser before spending minutes
        # rendering rather than discovering it afterwards.
        spoken_lines = sum(len(shot.lines) for shot in shots)
        if spoken_lines and not self.speech.available:
            LOGGER.error(
                "NO SPEECH SYNTHESISER AVAILABLE — every line will be silent and the "
                "film will have subtitles but no voices. On macOS `say` should exist; "
                "on Linux install espeak-ng, or install requirements-ml.txt for neural TTS."
            )

        started = time.time()
        audio_clips: List[AudioFileClip] = []
        total_shots = len(shots)

        try:
            rendered, per_shot = self._render_shots(production, shots)

            self.progress(92.0, "Assembling timeline")
            durations = [r.duration for r in rendered]
            fade = self.crossfade_for(durations)
            video = self.build_timeline(rendered, fade)
            starts = self.shot_starts(durations, fade)

            audio, audio_clips = self.build_audio(production, rendered, starts, video.duration)
            video = video.with_audio(audio)
            rendered_seconds = round(float(video.duration), 2)

            self.progress(95.0, "Encoding")
            self._encode(video, output)

            # After encoding, so a failed render never leaves orphan sidecars
            # claiming to describe a file that does not exist.
            self.subtitle_files = self._write_subtitles(rendered, starts, output)
        finally:
            for clip in audio_clips:
                try:
                    clip.close()
                except Exception:  # noqa: BLE001
                    pass

        elapsed = round(time.time() - started, 2)
        voiced, silent = self.speech.voiced_lines, self.speech.silent_lines
        if spoken_lines and not voiced:
            LOGGER.error(
                "RENDERED WITH NO AUDIBLE DIALOGUE: all %d line(s) fell back to silence. "
                "The video has subtitles but no voices. Cause: %s",
                silent,
                "; ".join(dict.fromkeys(self.speech.failures)) or "unknown",
            )
        elif silent:
            LOGGER.warning(
                "%d of %d line(s) were silent: %s",
                silent, voiced + silent, "; ".join(dict.fromkeys(self.speech.failures)),
            )

        record = {
            "output": str(output),
            "at": datetime.now().isoformat(timespec="seconds"),
            "preset": self.config.preset,
            "style": self.config.style,
            "resolution": f"{self.config.resolution[0]}x{self.config.resolution[1]}",
            "backend": self.backend.name,
            # The encoded length is shorter than the sum of shot durations by
            # one crossfade per cut, so record what was actually written.
            "duration_s": rendered_seconds,
            "elapsed_s": elapsed,
            "shots": total_shots,
            "voiced_lines": voiced,
            "silent_lines": silent,
            "per_shot_s": per_shot,
            "cast": {c.name: c.voice.voice for c in production.speaking_cast},
        }
        record["content_hash"] = content_hash(production, self.config)
        if self.subtitle_files:
            record["subtitles"] = [str(path) for path in self.subtitle_files]
        if self.speech.cache_hits:
            record["audio_cache_hits"] = self.speech.cache_hits
        if self.score_backend:
            record["score_backend"] = self.score_backend
        if getattr(self.speech, "engine_name", ""):
            record["speech_engine"] = self.speech.engine_name
        production.renders.append(record)
        production.status = "rendered"
        self.progress(100.0, f"Done: {output}")
        return output

    def _encode(self, video: VideoClip, output: Path) -> None:
        video.write_videofile(
            str(output),
            fps=self.config.fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="medium",
            threads=max(1, os.cpu_count() or 1),
            temp_audiofile=str(self.temp_dir / "mux_audio.m4a"),
            logger=None,
            # CRF is the quality target; also passing `bitrate=` makes the two
            # settings fight and x264 silently drops one.
            ffmpeg_params=[
                "-crf", str(self.config.preset_data["crf"]),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )
