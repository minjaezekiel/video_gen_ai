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

import logging
import math
import os
import shutil
import tempfile
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
from .config import CROSSFADE_SECONDS, MIN_SHOT_SECONDS, SHOT_TAIL_PAD, GenerationConfig
from .model import Line, Production, Shot, stable_seed
from .music import DEFAULT_MOOD, SAMPLE_RATE, duck, to_stereo
from .score import ScoreRequest, default_registry as default_music_registry
from .speech import SpeechSynthesizer
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
    rgb: np.ndarray
    alpha: np.ndarray
    box: Tuple[int, int, int, int]


@dataclass
class RenderedShot:
    """A shot reduced to a duration and a function from time to pixels."""

    duration: float
    frame_at: Callable[[float], np.ndarray]
    audio: List[Tuple[float, Path]] = field(default_factory=list)  # (offset, wav)
    speech_windows: List[Tuple[float, float]] = field(default_factory=list)


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


def cover_resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale to fully cover ``size``, centre-cropping the overflow."""
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.convert("RGB").resize(
        (
            max(target_w, int(math.ceil(src_w * scale))),
            max(target_h, int(math.ceil(src_h * scale))),
        ),
        Image.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


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

        arr = np.asarray(layer, dtype=np.float32)
        return Caption(0.0, 0.0, arr[..., :3], arr[..., 3:4] / 255.0, (x0, y0, box_w, box_h))

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
                visual.image.save(cached)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("could not cache shot image: %s", exc)
        return visual

    def _base_renderer(
        self, visual: ShotVisual, duration: float, camera: str, seed: int
    ) -> Callable[[float], np.ndarray]:
        """Frames from a motion clip, or a camera move over a still."""
        out_w, out_h = self.config.resolution

        if visual.is_motion and visual.frames:
            frames = [cover_resize(f, (out_w, out_h)) for f in visual.frames]
            rate = visual.frame_rate or self.config.fps
            arrays = [np.asarray(f, dtype=np.uint8) for f in frames]

            def from_frames(t: float) -> np.ndarray:
                # Hold the last frame if the clip is shorter than the dialogue.
                index = min(len(arrays) - 1, max(0, int(t * rate)))
                return arrays[index]

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

        def ken_burns(t: float) -> np.ndarray:
            u = min(1.0, max(0.0, t / duration))
            eased = u * u * (3.0 - 2.0 * u)  # smoothstep: no jerk at the cuts
            scale = s0 + (s1 - s0) * eased
            crop_w = min(big_w, max(2, int(round(out_w * scale))))
            crop_h = min(big_h, max(2, int(round(out_h * scale))))
            px = p0[0] + (p1[0] - p0[0]) * eased
            py = p0[1] + (p1[1] - p0[1]) * eased
            x = int(round(px * (big_w - crop_w)))
            y = int(round(py * (big_h - crop_h)))
            window = base.resize(
                (out_w, out_h), Image.BILINEAR, box=(x, y, x + crop_w, y + crop_h)
            )
            return np.asarray(window, dtype=np.uint8)

        return ken_burns

    # -- shots ------------------------------------------------------------- #

    def render_shot(self, production: Production, shot: Shot, index: int) -> RenderedShot:
        """Synthesise the dialogue, fetch the visual, and bind them to a clock."""
        audio: List[Tuple[float, Path]] = []
        captions: List[Caption] = []
        windows: List[Tuple[float, float]] = []
        cursor = 0.0

        for line_number, line in enumerate(shot.lines):
            member = production.character(line.character)
            voice = member.voice if member else None
            wav = self.temp_dir / f"shot{index:03d}_line{line_number:02d}.wav"
            if voice is None:
                from .model import VoiceProfile

                voice = VoiceProfile()
            _, spoken = self.speech.speak(line.text, voice, wav)

            audio.append((cursor, wav))
            windows.append((cursor, cursor + spoken))
            colour = (
                tuple(member.palette)
                if member and member.palette
                else character_palette(line.character, self.config.seed)
            )
            caption = self.build_caption(line, colour)  # type: ignore[arg-type]
            if caption is not None and self.config.captions:
                caption.start = cursor
                # Hold the subtitle slightly past the audio so it is readable.
                caption.end = cursor + spoken + 0.35
                captions.append(caption)
            cursor += spoken + 0.12  # a beat between lines

        duration = max(MIN_SHOT_SECONDS, cursor + SHOT_TAIL_PAD if shot.lines else shot.duration)
        shot.duration = round(duration, 2)

        visual = self.shot_visual(production, shot, duration)
        base_at = self._base_renderer(visual, duration, shot.camera, seed=self.config.seed + index)

        def frame_at(t: float) -> np.ndarray:
            frame = base_at(t)
            active = next((c for c in captions if c.start <= t < c.end), None)
            if active is None:
                return frame
            frame = frame.copy()
            x0, y0, box_w, box_h = active.box
            region = frame[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float32)
            frame[y0 : y0 + box_h, x0 : x0 + box_w] = (
                region * (1.0 - active.alpha) + active.rgb * active.alpha
            ).astype(np.uint8)
            return frame

        return RenderedShot(
            duration=duration, frame_at=frame_at, audio=audio, speech_windows=windows
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

        def frame_function(t: float) -> np.ndarray:
            index = 0
            for i, start in enumerate(starts):
                if t >= start:
                    index = i
                else:
                    break
            index = min(index, last)
            frame = shots[index].frame_at(t - starts[index])
            if fade > 0 and index > 0 and t < starts[index - 1] + durations[index - 1]:
                alpha = min(1.0, max(0.0, (t - starts[index]) / fade))
                previous = shots[index - 1].frame_at(t - starts[index - 1])
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
        per_shot: List[float] = []
        rendered: List[RenderedShot] = []
        audio_clips: List[AudioFileClip] = []
        total_shots = len(shots)

        try:
            for index, shot in enumerate(shots):
                shot_started = time.time()
                self.progress(
                    index / total_shots * 90.0,
                    f"Shot {index + 1}/{total_shots} ({shot.shot_type})",
                )
                rendered.append(self.render_shot(production, shot, index))
                per_shot.append(round(time.time() - shot_started, 2))

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
