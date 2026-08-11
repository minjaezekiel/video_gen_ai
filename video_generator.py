#!/usr/bin/env python3
"""
Story -> script -> video generator, optimised for CPU-only and offline use.

The pipeline is deliberately layered so that every stage has a zero-download
fallback.  With the optional model stack installed (``requirements-ml.txt``)
scenes are rendered with Stable Diffusion Turbo and narrated with MMS-TTS;
without it, scenes become procedurally composed cards and narration uses the
host operating system's speech synthesiser.  Both paths produce a complete,
playable MP4 -- the difference is fidelity, never functionality.

Usage
-----
    python video_generator.py                 # interactive wizard
    python video_generator.py --self-test     # offline smoke test, no downloads
    python video_generator.py --story-file s.txt --yes --preset draft

See ``decision.md`` for the engineering rationale and ``roadmap.md`` for what
is planned next.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# The macOS python.org build ships without a root certificate store, which makes
# every HTTPS request (including Hugging Face downloads) fail with
# CERTIFICATE_VERIFY_FAILED.  Pointing OpenSSL at certifi fixes it process-wide,
# and must happen before requests/huggingface_hub are imported.
try:  # pragma: no cover - environment dependent
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:  # pragma: no cover
    pass

from moviepy import AudioArrayClip, AudioFileClip, CompositeAudioClip, VideoClip
import moviepy.audio.fx as afx

LOGGER = logging.getLogger("video_gen")

# --------------------------------------------------------------------------- #
# Optional dependency probing
#
# Nothing below is required.  Each flag gates one enhancement; the pipeline has
# a working path for every combination of them being absent.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - depends on install
    import torch

    HAS_TORCH = True
except Exception:  # noqa: BLE001 - a broken torch install must not be fatal
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:  # pragma: no cover
    import diffusers  # noqa: F401

    HAS_DIFFUSERS = HAS_TORCH
except Exception:  # noqa: BLE001
    HAS_DIFFUSERS = False

try:  # pragma: no cover
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = HAS_TORCH
except Exception:  # noqa: BLE001
    HAS_TRANSFORMERS = False

try:  # pragma: no cover
    import soundfile as sf

    HAS_SOUNDFILE = True
except Exception:  # noqa: BLE001
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False


def capability_report() -> Dict[str, bool]:
    """Machine-readable summary of which optional backends are usable."""
    return {
        "torch": HAS_TORCH,
        "diffusers": HAS_DIFFUSERS,
        "transformers": HAS_TRANSFORMERS,
        "soundfile": HAS_SOUNDFILE,
        "system_tts": _system_tts_command() is not None,
    }


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

#: Rendering presets.  ``steps``/``guidance`` are tuned per model family: the
#: distilled turbo checkpoints are trained for 1-4 steps with classifier-free
#: guidance *disabled*, so passing the usual ``guidance_scale=7.5`` and a
#: negative prompt produces washed-out garbage.  That is why guidance and
#: negative prompts live in the preset rather than in global settings.
PRESETS: Dict[str, Dict[str, object]] = {
    "draft": {
        "model": "stabilityai/sd-turbo",
        "steps": 1,
        "guidance": 0.0,
        "image_size": 384,
        "negative_prompt": "",
        "resolution": (854, 480),
        "fps": 24,
        "crf": 23,
    },
    "standard": {
        "model": "stabilityai/sd-turbo",
        "steps": 4,
        "guidance": 0.0,
        "image_size": 512,
        "negative_prompt": "",
        "resolution": (1280, 720),
        "fps": 24,
        "crf": 20,
    },
    "quality": {
        "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "steps": 25,
        "guidance": 7.5,
        "image_size": 768,
        "negative_prompt": (
            "blurry, low quality, distorted, deformed, watermark, text, "
            "signature, jpeg artifacts, extra limbs"
        ),
        "resolution": (1920, 1080),
        "fps": 30,
        "crf": 18,
    },
}

DEFAULT_PRESET = "standard"

#: Candidate treatments.  Three *different* pacings beat three samples of one
#: model: small models resampled three times give three near-identical outputs,
#: which makes the accept step meaningless.
VARIANTS: Dict[str, Dict[str, object]] = {
    "Cinematic": {
        "scenes": (4, 5),
        "wpm": 130,
        "style": (
            "cinematic wide shot, anamorphic lens, dramatic volumetric lighting, "
            "shallow depth of field, film grain, muted teal and amber palette"
        ),
        "feel": "Wide establishing shots, slower narration, longer holds.",
    },
    "Fast-cut": {
        "scenes": (7, 8),
        "wpm": 165,
        "style": (
            "dynamic close-up, bold saturated colours, high contrast, "
            "strong diagonal composition, crisp detail"
        ),
        "feel": "Punchy and quick; many short scenes, energetic narration.",
    },
    "Documentary": {
        "scenes": (5, 6),
        "wpm": 145,
        "style": (
            "natural available light, documentary photography, 35mm lens, "
            "realistic texture, neutral colour grade"
        ),
        "feel": "Measured and explanatory; evenly weighted scenes.",
    },
}

MIN_SCENE_SECONDS = 2.5
MAX_SCENE_SECONDS = 12.0
SCENE_TAIL_PAD = 0.55  # breathing room after narration ends
CROSSFADE_SECONDS = 0.4


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Scene:
    """A single shot: one generated still, one line of narration."""

    index: int
    visual: str
    narration: str
    duration: float = MIN_SCENE_SECONDS
    overlay: str = ""

    def to_markdown(self) -> str:
        parts = [f"## Scene {self.index} — {self.duration:.1f}s", ""]
        parts.append(f"**Visual:** {self.visual}")
        parts.append("")
        parts.append(f"**Narration:** {self.narration}")
        if self.overlay:
            parts.append("")
            parts.append(f"**Overlay:** {self.overlay}")
        return "\n".join(parts)


@dataclass
class VideoScript:
    """A candidate or accepted script, persisted as ``video_scripts/<id>.md``."""

    id: str
    title: str
    variant: str
    scenes: List[Scene]
    source_story: str = ""
    status: str = "candidate"  # candidate -> accepted -> rendered
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    config: Dict[str, object] = field(default_factory=dict)
    renders: List[Dict[str, object]] = field(default_factory=list)

    @property
    def estimated_duration(self) -> float:
        return sum(s.duration for s in self.scenes)

    def to_markdown(self) -> str:
        front = {
            "id": self.id,
            "title": self.title,
            "variant": self.variant,
            "status": self.status,
            "created": self.created,
            "source_story": self.source_story,
            "config": self.config,
            "renders": self.renders,
        }
        header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=88)
        body = "\n\n".join(s.to_markdown() for s in self.scenes)
        return f"---\n{header}---\n\n# {self.title}\n\n{body}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "VideoScript":
        front, body = _split_frontmatter(text)
        return cls(
            id=str(front.get("id", "unknown")),
            title=str(front.get("title", "Untitled")),
            variant=str(front.get("variant", "Cinematic")),
            scenes=_parse_scenes(body),
            source_story=str(front.get("source_story", "") or ""),
            status=str(front.get("status", "candidate")),
            created=str(front.get("created", "")),
            config=dict(front.get("config") or {}),
            renders=list(front.get("renders") or []),
        )


@dataclass
class GenerationConfig:
    """Everything that influences a render, in one place."""

    preset: str = DEFAULT_PRESET
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[int] = None
    seed: int = 1234
    music: bool = True
    music_volume: float = 0.12
    overlays: bool = True
    use_llm: bool = False
    voice_rate: float = 1.0
    output_dir: Path = Path("output")
    scripts_dir: Path = Path("video_scripts")
    cache_dir: Path = Path(".cache")

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}")
        preset = PRESETS[self.preset]
        if self.resolution is None:
            self.resolution = tuple(preset["resolution"])  # type: ignore[arg-type]
        if self.fps is None:
            self.fps = int(preset["fps"])  # type: ignore[arg-type]
        if self.resolution[0] % 2 or self.resolution[1] % 2:
            raise ValueError("resolution must be even in both dimensions for H.264")
        self.output_dir = Path(self.output_dir)
        self.scripts_dir = Path(self.scripts_dir)
        self.cache_dir = Path(self.cache_dir)

    @property
    def preset_data(self) -> Dict[str, object]:
        return PRESETS[self.preset]


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SCENE_RE = re.compile(
    r"^##\s+Scene\s+(?P<index>\d+)\s*(?:[—–-]\s*(?P<duration>[\d.]+)\s*s)?\s*$",
    re.MULTILINE,
)
_FIELD_RE = re.compile(
    r"^\*\*(?P<key>Visual|Narration|Overlay):\*\*\s*(?P<value>.*)$", re.MULTILINE
)


def _split_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        front = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        LOGGER.warning("malformed frontmatter, ignoring: %s", exc)
        front = {}
    if not isinstance(front, dict):
        front = {}
    return front, text[match.end() :]


def _parse_scenes(body: str) -> List[Scene]:
    """Parse ``## Scene N`` blocks.

    Tolerant by design: a block missing ``**Narration:**`` yields an empty
    narration rather than raising, and unparseable input yields ``[]`` so the
    caller can fall back instead of crashing mid-render.
    """
    scenes: List[Scene] = []
    matches = list(_SCENE_RE.finditer(body))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end]
        fields = {m.group("key"): m.group("value").strip() for m in _FIELD_RE.finditer(block)}
        visual = fields.get("Visual", "").strip()
        narration = fields.get("Narration", "").strip()
        if not visual and not narration:
            continue
        duration = float(match.group("duration") or 0) or MIN_SCENE_SECONDS
        scenes.append(
            Scene(
                index=int(match.group("index")),
                visual=visual,
                narration=narration,
                duration=duration,
                overlay=fields.get("Overlay", ""),
            )
        )
    return scenes


def _slugify(text: str, max_len: int = 32) -> str:
    """Filename-safe slug, truncated on a word boundary rather than mid-word."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        # Drop the trailing partial word, unless that would leave nothing.
        head = slug[:max_len].rsplit("-", 1)[0]
        slug = head or slug[:max_len]
    return slug.strip("-") or "untitled"


def _stable_seed(text: str, salt: int = 0) -> int:
    """Deterministic across processes, unlike ``hash()`` under PYTHONHASHSEED."""
    digest = hashlib.md5(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------- #
# Script store
# --------------------------------------------------------------------------- #


class ScriptStore:
    """Durable, human-editable script archive under ``video_scripts/``."""

    def __init__(self, root: Path = Path("video_scripts")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, script_id: str) -> Path:
        return self.root / f"{script_id}.md"

    def save(self, script: VideoScript) -> Path:
        path = self.path_for(script.id)
        path.write_text(script.to_markdown(), encoding="utf-8")
        self.write_index()
        return path

    def load(self, script_id: str) -> VideoScript:
        path = Path(script_id) if str(script_id).endswith(".md") else self.path_for(script_id)
        if not path.exists():
            raise FileNotFoundError(f"no script at {path}")
        return VideoScript.from_markdown(path.read_text(encoding="utf-8"))

    def list_scripts(self) -> List[VideoScript]:
        scripts: List[VideoScript] = []
        for path in sorted(self.root.glob("*.md")):
            if path.name == "INDEX.md":
                continue
            try:
                scripts.append(VideoScript.from_markdown(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
                LOGGER.warning("skipping unreadable script %s: %s", path.name, exc)
        scripts.sort(key=lambda s: s.created, reverse=True)
        return scripts

    def accept(self, script: VideoScript, discard_siblings: bool = True) -> Path:
        """Mark ``script`` accepted; demote the other candidates from its batch."""
        script.status = "accepted"
        batch = script.config.get("batch")
        if discard_siblings and batch:
            for other in self.list_scripts():
                if (
                    other.id != script.id
                    and other.config.get("batch") == batch
                    and other.status == "candidate"
                ):
                    other.status = "rejected"
                    self.path_for(other.id).write_text(other.to_markdown(), encoding="utf-8")
        return self.save(script)

    def record_render(self, script: VideoScript, record: Dict[str, object]) -> Path:
        script.renders.append(record)
        script.status = "rendered"
        return self.save(script)

    def write_index(self) -> Path:
        rows = [
            "# Video script index",
            "",
            "Regenerated automatically on every save. One row per script, so both",
            "you and the model can scan the whole history without opening files.",
            "",
            "| id | title | variant | status | scenes | est. | renders |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for script in self.list_scripts():
            rows.append(
                f"| [`{script.id}`]({script.id}.md) | {script.title} | {script.variant} "
                f"| {script.status} | {len(script.scenes)} "
                f"| {script.estimated_duration:.0f}s | {len(script.renders)} |"
            )
        path = self.root / "INDEX.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Script generation (deterministic backbone + optional LLM enrichment)
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here it its of to in on at
    for with from by as is are was were be been being do does did done have has had having i you
    he she they we me him her them us my your his their our not no nor so very just also can could
    would should will shall may might must about into over under after before while when where who
    whom which what how why all any both each few more most other some such only own same too
    one every anyway still until again already always never ever yet once now then said says say
    went get got make made take took come came go going know knew think thought seem seemed
    """.split()
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“‘(])|\n{2,}")


class StoryScripter:
    """Turns a prose story into scene lists.

    The deterministic path always runs and always produces well-formed scenes.
    An optional local LLM only *rewrites the visual prompts*; if it is missing,
    slow, or emits nonsense, the deterministic text stands.  Keeping the model
    off the critical path is what makes generation instant and offline-safe.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._llm = None
        self._llm_failed = False

    # -- public ------------------------------------------------------------ #

    def generate_candidates(
        self, story: str, variants: Optional[Sequence[str]] = None
    ) -> List[VideoScript]:
        story = story.strip()
        if not story:
            raise ValueError("story is empty")
        batch = datetime.now().strftime("%Y%m%d-%H%M%S")
        title = self._derive_title(story)
        slug = _slugify(title)
        scripts: List[VideoScript] = []
        for name in variants or list(VARIANTS):
            scenes = self._build_scenes(story, name)
            if self.config.use_llm:
                scenes = self._enrich(scenes, title, name)
            suffix = hashlib.md5(f"{batch}{name}".encode()).hexdigest()[:4]
            scripts.append(
                VideoScript(
                    id=f"{datetime.now():%Y-%m-%d}-{slug}-{suffix}",
                    title=title,
                    variant=name,
                    scenes=scenes,
                    source_story=story,
                    config={
                        "preset": self.config.preset,
                        "seed": self.config.seed,
                        "batch": batch,
                        "llm_enriched": bool(self.config.use_llm and not self._llm_failed),
                    },
                )
            )
        return scripts

    # -- deterministic backbone -------------------------------------------- #

    @staticmethod
    def split_sentences(story: str) -> List[str]:
        raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(story) if s and s.strip()]
        # Very long sentences are split on clause boundaries so that a single
        # run-on sentence cannot collapse the whole story into one scene.
        out: List[str] = []
        for sentence in raw:
            if len(sentence.split()) <= 34:
                out.append(sentence)
                continue
            clauses = re.split(
                r",\s+(?=(?:and|but|then|while|as|until|before|after)\b)", sentence
            )
            buf = ""
            for clause in clauses:
                buf = f"{buf}, {clause}" if buf else clause
                if len(buf.split()) >= 14:
                    out.append(buf.strip(" ,"))
                    buf = ""
            if buf.strip(" ,"):
                out.append(buf.strip(" ,"))
        return out or [story.strip()]

    @staticmethod
    def group_into_beats(sentences: Sequence[str], count: int) -> List[str]:
        """Even-ish contiguous grouping; never drops or reorders sentences."""
        count = max(1, min(count, len(sentences)))
        base, extra = divmod(len(sentences), count)
        beats: List[str] = []
        cursor = 0
        for i in range(count):
            size = base + (1 if i < extra else 0)
            beats.append(" ".join(sentences[cursor : cursor + size]).strip())
            cursor += size
        return [b for b in beats if b]

    def _build_scenes(self, story: str, variant: str) -> List[Scene]:
        spec = VARIANTS[variant]
        lo, hi = spec["scenes"]  # type: ignore[misc]
        sentences = self.split_sentences(story)
        target = max(lo, min(hi, round(len(sentences) / 1.4) or lo))
        beats = self.group_into_beats(sentences, target)
        wpm = float(spec["wpm"])  # type: ignore[arg-type]
        return [
            Scene(
                index=i,
                visual=self._visual_prompt(beat, str(spec["style"]), story),
                narration=beat,
                duration=self.estimate_duration(beat, wpm),
            )
            for i, beat in enumerate(beats, start=1)
        ]

    @staticmethod
    def estimate_duration(text: str, wpm: float) -> float:
        words = max(1, len(text.split()))
        seconds = words / wpm * 60.0 + SCENE_TAIL_PAD
        return round(max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, seconds)), 2)

    @staticmethod
    def keywords(text: str, limit: int = 8) -> List[str]:
        seen: Dict[str, int] = {}
        for word in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
            low = word.lower().strip("'-")
            if len(low) < 3 or low in _STOPWORDS:
                continue
            # Capitalised mid-sentence words are probably proper nouns; weight them.
            seen[low] = seen.get(low, 0) + (2 if word[0].isupper() else 1)
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], text.lower().find(kv[0])))
        return [w for w, _ in ranked[:limit]]

    def _visual_prompt(self, beat: str, style: str, story: str) -> str:
        subject = ", ".join(self.keywords(beat, limit=6))
        context = ", ".join(w for w in self.keywords(story, limit=3) if w not in subject)
        pieces = [p for p in (subject, context) if p]
        return f"{', '.join(pieces)}, {style}" if pieces else style

    @staticmethod
    def _derive_title(story: str) -> str:
        """Title from the story's own opening words.

        Keyword soup ("One Cartographer Spent Life") makes a poor title, so a
        long opening sentence is truncated rather than reassembled.
        """
        first = StoryScripter.split_sentences(story)[0].strip().rstrip(".!?,;:")
        words = first.split()
        if not words:
            return "Untitled Story"
        return " ".join(words[:8] if len(words) > 8 else words)

    # -- optional LLM enrichment ------------------------------------------- #

    def _load_llm(self):  # pragma: no cover - requires model download
        if self._llm is not None or self._llm_failed:
            return self._llm
        if not HAS_TRANSFORMERS:
            self._llm_failed = True
            return None
        try:
            from transformers import pipeline

            self._llm = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype=torch.float32,
                device=-1,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM enrichment unavailable (%s); using deterministic prompts", exc)
            self._llm_failed = True
            self._llm = None
        return self._llm

    def _enrich(self, scenes: List[Scene], title: str, variant: str) -> List[Scene]:
        # pragma: no cover - requires model download
        llm = self._load_llm()
        if llm is None:
            return scenes
        style = str(VARIANTS[variant]["style"])
        for scene in scenes:
            prompt = (
                "Rewrite the following into a single vivid image-generation prompt. "
                "Describe only what is visible: subject, setting, lighting, framing. "
                "No narration, no quotes, under 40 words.\n\n"
                f"Story title: {title}\nScene text: {scene.narration}\nAnswer:"
            )
            try:
                out = llm(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=64,
                    do_sample=False,
                    return_full_text=False,
                )
                text = out[0]["generated_text"].strip().strip('"')
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("LLM enrichment failed on scene %s: %s", scene.index, exc)
                continue
            # Reject degenerate output rather than poisoning the prompt.
            if 12 <= len(text) <= 400 and "\n" not in text.strip():
                scene.visual = f"{text}, {style}"
        return scenes


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


def _system_tts_command() -> Optional[str]:
    """Locate an offline OS speech synthesiser, if one exists."""
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for candidate in ("espeak-ng", "espeak"):
        if shutil.which(candidate):
            return candidate
    return None


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 [-1, 1] audio, preferring soundfile, else stdlib."""
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_SOUNDFILE:
        sf.write(str(path), samples, sample_rate)
        return
    import wave

    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

ProgressCallback = Callable[[float, str], None]


class FreeTextToVideoGenerator:
    """The pipeline: story -> candidate scripts -> accepted script -> MP4."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.progress: ProgressCallback = progress or (lambda pct, msg: None)
        self.store = ScriptStore(self.config.scripts_dir)
        self.scripter = StoryScripter(self.config)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir = self.config.cache_dir / "images"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        # Scratch space for intermediate audio only.  Nothing the user keeps is
        # ever written here -- the original wrote the finished MP4 into the temp
        # dir and then deleted that directory on exit.
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_gen_"))
        self._cleaned = False

        self.device = self._select_device()
        self._image_pipe = None
        self._image_pipe_failed = False
        self._tts = None
        self._tts_failed = False

        if HAS_TORCH and self.device == "cpu":  # pragma: no cover
            torch.set_num_threads(max(1, os.cpu_count() or 1))

        atexit.register(self.cleanup)
        LOGGER.info(
            "device=%s preset=%s resolution=%dx%d capabilities=%s",
            self.device,
            self.config.preset,
            self.config.resolution[0],
            self.config.resolution[1],
            capability_report(),
        )

    # -- lifecycle --------------------------------------------------------- #

    @staticmethod
    def _select_device() -> str:
        if not HAS_TORCH:
            return "cpu"
        try:  # pragma: no cover - hardware dependent
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def cleanup(self) -> None:
        """Idempotent teardown.  Only ever removes the scratch directory."""
        if self._cleaned:
            return
        self._cleaned = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def unload_models(self) -> None:
        self._image_pipe = None
        self._tts = None
        if HAS_TORCH and self.device == "cuda":  # pragma: no cover
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "FreeTextToVideoGenerator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    # -- scripting --------------------------------------------------------- #

    def propose_scripts(self, story: str) -> List[VideoScript]:
        """Generate candidates and persist them all as ``status: candidate``."""
        scripts = self.scripter.generate_candidates(story)
        for script in scripts:
            self.store.save(script)
        return scripts

    def accept_script(self, script: VideoScript) -> Path:
        return self.store.accept(script)

    # -- image generation -------------------------------------------------- #

    @property
    def image_pipe(self):
        if self._image_pipe is None and not self._image_pipe_failed:
            self._image_pipe = self._load_image_model()
            self._image_pipe_failed = self._image_pipe is None
        return self._image_pipe

    def _load_image_model(self):  # pragma: no cover - requires model download
        if not HAS_DIFFUSERS:
            LOGGER.info("diffusers/torch unavailable; using procedural scene cards")
            return None
        model_id = str(self.config.preset_data["model"])
        try:
            from diffusers import AutoPipelineForText2Image

            # fp16 only helps on CUDA; on CPU it is emulated and *slower*, and
            # MPS needs fp32 for several of the SD ops.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.enable_attention_slicing()
                if self.device == "cpu":
                    pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001 - optional memory optimisations
                pass
            LOGGER.info("loaded image model %s on %s (%s)", model_id, self.device, dtype)
            return pipe
        except Exception as exc:  # noqa: BLE001
            # Non-fatal but *loud*: the original swallowed this silently and
            # then rendered black frames for the entire video.
            LOGGER.warning(
                "could not load image model %s (%s); falling back to procedural scene "
                "cards -- run `pip install -r requirements-ml.txt` for AI visuals",
                model_id,
                exc,
            )
            return None

    def generate_image(self, prompt: str, seed_salt: int = 0) -> Image.Image:
        """One still per scene.  Disk-cached across runs by (prompt, preset, seed)."""
        preset = self.config.preset_data
        size = int(preset["image_size"])  # type: ignore[arg-type]
        key = hashlib.sha1(
            f"{prompt}|{self.config.preset}|{self.config.seed}|{seed_salt}|{size}".encode("utf-8")
        ).hexdigest()
        cached = self.image_cache_dir / f"{key}.png"
        if cached.exists():
            try:
                with Image.open(cached) as handle:
                    return handle.convert("RGB")
            except Exception:  # noqa: BLE001 - corrupt cache entry
                cached.unlink(missing_ok=True)

        seed = (self.config.seed + _stable_seed(prompt, seed_salt)) % (2**31 - 1)
        pipe = self.image_pipe
        image: Optional[Image.Image] = None
        if pipe is not None:  # pragma: no cover - requires model download
            try:
                kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": int(preset["steps"]),  # type: ignore[arg-type]
                    "guidance_scale": float(preset["guidance"]),  # type: ignore[arg-type]
                    "width": size,
                    "height": size,
                    # A per-call generator is both reproducible and thread-safe,
                    # unlike the global torch.manual_seed the original used.
                    "generator": torch.Generator(device="cpu").manual_seed(seed),
                }
                # Turbo checkpoints run with CFG disabled, where a negative
                # prompt is not merely useless -- the pipeline rejects it.
                if float(preset["guidance"]) > 1.0 and preset["negative_prompt"]:
                    kwargs["negative_prompt"] = preset["negative_prompt"]
                image = pipe(**kwargs).images[0]
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("image generation failed (%s); using scene card", exc)
                image = None

        if image is None:
            # The palette seed is per-video, so all the cards in one render
            # share a colour family instead of jumping hue at every cut.
            image = self.render_scene_card(
                prompt, seed, (size, size), palette_seed=self.config.seed
            )

        try:
            image.save(cached)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("could not cache image: %s", exc)
        return image

    # -- procedural fallback visuals --------------------------------------- #

    @staticmethod
    def render_scene_card(
        prompt: str,
        seed: int,
        size: Tuple[int, int],
        palette_seed: Optional[int] = None,
    ) -> Image.Image:
        """A composed abstract card, used when no image model is loaded.

        This is what makes the pipeline genuinely runnable with zero downloads.
        It is not photoreal, but it is deliberately designed: a graded sky, a
        soft horizon with layered atmospheric ridges, a light source, grain and
        a vignette.

        ``palette_seed`` is shared by every scene in one video, and each scene
        only shifts the hue slightly from it.  Without that, consecutive scenes
        pick unrelated hues and the cuts read as glitches rather than as a
        deliberate look.
        """
        width, height = size
        rng = np.random.default_rng(seed)
        palette_rng = np.random.default_rng(seed if palette_seed is None else palette_seed)

        def hsv_rgb(h: float, s: float, v: float) -> np.ndarray:
            swatch = Image.new("HSV", (1, 1), (int(h * 255) % 256, int(s), int(v)))
            return np.array(swatch.convert("RGB").getpixel((0, 0)), dtype=np.float32)

        # One family hue for the whole video; scenes vary within a narrow arc.
        hue = float(palette_rng.uniform(0, 1)) + float(rng.uniform(-0.055, 0.055))
        warm = hue + 0.5 + float(palette_rng.uniform(-0.05, 0.05))  # complement

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        v_norm = yy / max(1, height - 1)

        # Sky: bright near the horizon, deeper overhead.
        c_high = hsv_rgb(hue, palette_rng.uniform(150, 210), 120)
        c_low = hsv_rgb(hue - 0.04, palette_rng.uniform(90, 150), 225)
        horizon = float(rng.uniform(0.52, 0.72))
        sky_ramp = np.clip(v_norm / horizon, 0, 1)[..., None] ** 1.4
        canvas = c_high[None, None, :] * (1 - sky_ramp) + c_low[None, None, :] * sky_ramp

        # A low sun near the horizon anchors the composition.
        sun_x = float(rng.uniform(0.2, 0.8)) * width
        sun_y = horizon * height - float(rng.uniform(0.02, 0.12)) * height
        sun_r = float(rng.uniform(0.10, 0.20)) * max(width, height)
        glow = np.exp(-(((xx - sun_x) ** 2 + (yy - sun_y) ** 2) / (2 * sun_r**2)))
        canvas += glow[..., None] * hsv_rgb(warm, 90, 255)[None, None, :] * 0.55

        # Layered ridges below the horizon: each darker and closer to the eye.
        for layer in range(int(rng.integers(2, 4))):
            depth = (layer + 1) / 4.0
            crest = horizon + depth * float(rng.uniform(0.05, 0.13))
            wobble = (
                np.sin(xx / width * np.pi * float(rng.uniform(1.0, 2.6)) + rng.uniform(0, 6.28))
                * float(rng.uniform(0.012, 0.035))
            )
            edge = (crest + wobble) * height
            # Clipped so the sigmoid saturates cleanly instead of overflowing.
            ramp = np.clip((yy - edge) / max(1.0, height * 0.006), -60.0, 60.0)
            mask = 1.0 / (1.0 + np.exp(-ramp))
            shade = hsv_rgb(hue + 0.02 * layer, 170, 90 - 22 * layer)
            canvas = canvas * (1 - mask[..., None] * 0.92) + shade[None, None, :] * (
                mask[..., None] * 0.92
            )

        # Vignette, then fine grain so flat areas do not band after H.264.
        dx = (xx / width - 0.5) * 2
        dy = (yy / height - 0.5) * 2
        canvas *= (1.0 - 0.40 * np.clip(dx**2 + dy**2, 0, 1))[..., None]
        canvas += rng.normal(0, 3.2, canvas.shape).astype(np.float32)

        image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        return image.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 500.0))

    # -- typography -------------------------------------------------------- #

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
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

    def build_overlay(
        self, text: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
        """Pre-render a lower-third caption once per scene.

        Returns ``(rgb, alpha, bbox)`` limited to the caption's bounding box, so
        per-frame compositing touches a slice rather than the whole frame.
        """
        text = " ".join(text.split())
        if not text:
            return None
        width, height = self.config.resolution
        font_size = max(16, int(height * 0.042))
        font = self._load_font(font_size)

        # Wrap to ~86% of frame width using real glyph metrics.
        max_width = int(width * 0.86)
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
        if len(lines) > 3:  # keep captions to three lines, ellipsise the rest
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip(" ,.") + "…"

        line_height = int(font_size * 1.32)
        pad = int(font_size * 0.62)
        box_h = min(height, line_height * len(lines) + pad * 2)
        box_w = min(width, max(font.getbbox(l)[2] for l in lines) + pad * 2)
        x0 = max(0, (width - box_w) // 2)
        y0 = max(0, height - box_h - int(height * 0.055))

        layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            [0, 0, box_w - 1, box_h - 1], radius=int(font_size * 0.35), fill=(8, 10, 14, 170)
        )
        for i, line in enumerate(lines):
            tx = (box_w - font.getbbox(line)[2]) // 2
            ty = pad + i * line_height
            draw.text((tx + 2, ty + 2), line, font=font, fill=(0, 0, 0, 190))  # shadow
            draw.text((tx, ty), line, font=font, fill=(245, 246, 250, 255))

        arr = np.asarray(layer, dtype=np.float32)
        return arr[..., :3], arr[..., 3:4] / 255.0, (x0, y0, box_w, box_h)

    # -- Ken Burns --------------------------------------------------------- #

    def make_scene_clip(
        self, image: Image.Image, duration: float, seed: int, overlay_text: str = ""
    ) -> VideoClip:
        """A standalone clip for one scene (used by tests and single-scene renders)."""
        duration = max(0.1, float(duration))
        return VideoClip(
            frame_function=self.make_scene_renderer(image, duration, seed, overlay_text),
            duration=duration,
        )

    def make_scene_renderer(
        self, image: Image.Image, duration: float, seed: int, overlay_text: str = ""
    ) -> Callable[[float], np.ndarray]:
        """Animate one still with a slow crop-and-zoom (Ken Burns).

        This replaces the original five-diffusion-calls-plus-pixel-interpolation
        approach: one inference per scene, and the motion costs a crop and a
        resize per frame rather than another model call.

        Returns the bare frame function rather than a clip so that the whole
        video can be assembled as a *single* ``VideoClip``.  Routing scenes
        through moviepy's ``concatenate_videoclips(method="compose")`` instead
        costs 4x more wall-clock, because it drives its mask-blitting path for
        every frame and not just the ones inside a crossfade.
        """
        out_w, out_h = self.config.resolution
        zoom = 1.18  # oversample factor -- the slack we pan and zoom inside

        big_w, big_h = int(out_w * zoom), int(out_h * zoom)
        base = self._cover_resize(image, (big_w, big_h))

        rng = np.random.default_rng(seed)
        s0, s1 = (zoom, 1.0) if rng.integers(0, 2) else (1.0, zoom)
        # Pan endpoints as normalised positions within the slack area.
        p0 = (float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
        p1 = (float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))

        overlay = self.build_overlay(overlay_text) if self.config.overlays else None
        duration = max(0.1, float(duration))

        def frame_function(t: float) -> np.ndarray:
            # Smoothstep easing: no visible jerk at the scene boundaries.
            u = min(1.0, max(0.0, t / duration))
            eased = u * u * (3.0 - 2.0 * u)

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
            frame = np.asarray(window, dtype=np.uint8)

            if overlay is not None:
                rgb, alpha, (x0, y0, bw, bh) = overlay
                frame = frame.copy()
                region = frame[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
                frame[y0 : y0 + bh, x0 : x0 + bw] = (
                    region * (1.0 - alpha) + rgb * alpha
                ).astype(np.uint8)
            return frame

        return frame_function

    @staticmethod
    def _cover_resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Scale to fully cover ``size``, centre-cropping the overflow."""
        target_w, target_h = size
        src_w, src_h = image.size
        scale = max(target_w / src_w, target_h / src_h)
        new = image.convert("RGB").resize(
            (
                max(target_w, int(math.ceil(src_w * scale))),
                max(target_h, int(math.ceil(src_h * scale))),
            ),
            Image.LANCZOS,
        )
        left = (new.width - target_w) // 2
        top = (new.height - target_h) // 2
        return new.crop((left, top, left + target_w, top + target_h))

    # -- speech ------------------------------------------------------------ #

    @property
    def tts(self):
        if self._tts is None and not self._tts_failed:
            self._tts = self._load_tts()
            self._tts_failed = self._tts is None
        return self._tts

    def _load_tts(self):  # pragma: no cover - requires model download
        if not HAS_TRANSFORMERS:
            return None
        try:
            from transformers import AutoTokenizer, VitsModel

            model = VitsModel.from_pretrained("facebook/mms-tts-eng").to("cpu").eval()
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            LOGGER.info("loaded neural TTS facebook/mms-tts-eng")
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("neural TTS unavailable (%s); trying system voice", exc)
            return None

    def generate_speech(self, text: str, out_path: Path) -> Tuple[Path, float]:
        """Synthesise narration, returning the file and its *real* duration.

        Fallback chain: MMS-TTS (neural) -> OS synthesiser -> timed silence.
        The measured length is returned so the caller can make it the source of
        truth for scene length instead of guessing from word count.
        """
        text = " ".join(text.split())
        if not text:
            return self._silence(out_path, MIN_SCENE_SECONDS)

        model_pair = self.tts
        if model_pair is not None:  # pragma: no cover - requires model download
            try:
                model, tokenizer = model_pair
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    waveform = model(**inputs).waveform
                samples = waveform.squeeze().cpu().numpy().astype(np.float32)
                # The rate comes from the checkpoint; hardcoding 16 kHz (as the
                # original did) detunes and mistimes every single line.
                rate = int(model.config.sampling_rate)
                samples = self._apply_rate(samples, self.config.voice_rate)
                _write_wav(out_path, self._normalize(samples), rate)
                return out_path, len(samples) / rate
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("neural TTS failed (%s); trying system voice", exc)

        system = self._system_speech(text, out_path)
        if system is not None:
            return system
        # Silence sized to a natural reading pace keeps captions readable even
        # when no synthesiser at all is available.
        return self._silence(out_path, StoryScripter.estimate_duration(text, 150))

    def _system_speech(self, text: str, out_path: Path) -> Optional[Tuple[Path, float]]:
        command = _system_tts_command()
        if command is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if command == "say":
                argv = [
                    "say", "-r", str(int(180 * self.config.voice_rate)),
                    "--data-format=LEI16@22050", "-o", str(out_path), text,
                ]
            else:
                argv = [
                    command, "-s", str(int(165 * self.config.voice_rate)),
                    "-w", str(out_path), text,
                ]
            subprocess.run(argv, check=True, capture_output=True, timeout=180)
            with AudioFileClip(str(out_path)) as clip:
                duration = float(clip.duration)
            if duration <= 0.05:
                return None
            return out_path, duration
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("system TTS (%s) failed: %s", command, exc)
            return None

    def _silence(self, out_path: Path, duration: float) -> Tuple[Path, float]:
        rate = 22050
        _write_wav(out_path, np.zeros(int(rate * duration), dtype=np.float32), rate)
        return out_path, duration

    @staticmethod
    def _apply_rate(samples: np.ndarray, rate: float) -> np.ndarray:
        """Resampling speed change (pitch shifts slightly, as expected)."""
        if abs(rate - 1.0) < 1e-3 or samples.size == 0:
            return samples
        target_len = max(1, int(len(samples) / rate))
        src = np.linspace(0, len(samples) - 1, num=target_len)
        return np.interp(src, np.arange(len(samples)), samples).astype(np.float32)

    @staticmethod
    def _normalize(samples: np.ndarray, peak: float = 0.89) -> np.ndarray:
        if samples.size == 0:
            return samples
        current = float(np.max(np.abs(samples)))
        if current < 1e-6:
            return samples
        return (samples * (peak / current)).astype(np.float32)

    # -- music ------------------------------------------------------------- #

    def generate_music(self, duration: float, seed: int) -> Optional[AudioArrayClip]:
        """A synthesised pad underscore -- no model, no download, no licensing."""
        if not self.config.music or duration <= 0.2:
            return None
        rate = 44100
        rng = np.random.default_rng(seed)
        root = float(rng.choice([110.0, 123.47, 130.81, 146.83, 164.81]))
        # i - VI - III - VII: minor and unresolved, so it stays out of the way
        # of the narration instead of competing with it.
        progression = [
            (1.0, 1.2, 1.5), (0.8, 1.0, 1.2), (1.2, 1.5, 1.8), (0.9, 1.125, 1.35),
        ]
        bar = 4.0
        total = int(rate * duration)
        out = np.zeros(total, dtype=np.float32)
        t_bar = np.arange(int(rate * bar), dtype=np.float32) / rate
        # Slow attack/release so chord changes glide instead of clicking.
        env = np.clip(np.minimum(t_bar / 0.9, np.minimum(1.0, (bar - t_bar) / 1.2)), 0.0, 1.0)

        cursor = 0
        index = 0
        while cursor < total:
            wave = np.zeros_like(t_bar)
            for harmonic, ratio in enumerate(progression[index % len(progression)]):
                freq = root * ratio
                wave += np.sin(2 * np.pi * freq * t_bar) * (0.5**harmonic)
                wave += 0.25 * np.sin(4 * np.pi * freq * t_bar) * (0.5**harmonic)
            wave *= env
            chunk = wave[: total - cursor]
            out[cursor : cursor + len(chunk)] += chunk
            cursor += len(chunk)
            index += 1

        out = self._normalize(out, peak=1.0) * self.config.music_volume
        fade = int(rate * 1.5)  # do not slam the bed on at t=0
        if total > 2 * fade:
            out[:fade] *= np.linspace(0, 1, fade)
            out[-fade:] *= np.linspace(1, 0, fade)
        return AudioArrayClip(np.stack([out, out], axis=1), fps=rate)

    # -- render ------------------------------------------------------------ #

    def render(self, script: VideoScript, output: Optional[Path] = None) -> Path:
        """Render a script to MP4 and record the result back into its file."""
        if not script.scenes:
            raise ValueError("script has no scenes")
        output = Path(output) if output else self.config.output_dir / f"{script.id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        started = time.time()
        per_scene: List[float] = []
        renderers: List[Callable[[float], np.ndarray]] = []
        durations: List[float] = []
        audio_clips: List[AudioFileClip] = []
        total = len(script.scenes)

        try:
            for i, scene in enumerate(script.scenes):
                scene_started = time.time()
                self.progress(i / total * 90.0, f"Scene {i + 1}/{total}: narration")

                wav = self.temp_dir / f"scene_{i:03d}.wav"
                _, narration_seconds = self.generate_speech(scene.narration, wav)
                # Real audio length drives scene length.  The original computed
                # the video duration first and then time-stretched the audio,
                # which accumulated A/V drift across the whole video.
                duration = max(MIN_SCENE_SECONDS, narration_seconds + SCENE_TAIL_PAD)
                scene.duration = round(duration, 2)

                self.progress((i + 0.35) / total * 90.0, f"Scene {i + 1}/{total}: visual")
                image = self.generate_image(scene.visual, seed_salt=scene.index)

                self.progress((i + 0.75) / total * 90.0, f"Scene {i + 1}/{total}: animating")
                renderers.append(
                    self.make_scene_renderer(
                        image,
                        duration,
                        seed=self.config.seed + scene.index,
                        overlay_text=scene.overlay or scene.narration,
                    )
                )
                durations.append(duration)
                narration = AudioFileClip(str(wav))
                if narration.duration > duration:
                    narration = narration.subclipped(0, duration)
                audio_clips.append(narration)
                per_scene.append(round(time.time() - scene_started, 2))

            self.progress(92.0, "Assembling timeline")
            fade = self._crossfade_for(durations)
            video = self.build_timeline(renderers, durations, fade)

            # Each narration is pinned to the start of *its own* scene on the
            # crossfaded timeline.  Naively concatenating the audio instead
            # would drift by `fade` per scene, because every crossfade overlaps
            # two clips and shortens the video by that much.
            tracks = [
                narration.with_effects(
                    [afx.AudioFadeIn(0.05), afx.AudioFadeOut(0.12)]
                ).with_start(start)
                for narration, start in zip(audio_clips, self._scene_starts(durations, fade))
            ]
            music = self.generate_music(video.duration, seed=self.config.seed)
            if music is not None:
                tracks.append(music.with_duration(video.duration))
            video = video.with_audio(CompositeAudioClip(tracks).with_duration(video.duration))
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
        self.store.record_render(
            script,
            {
                "output": str(output),
                "at": datetime.now().isoformat(timespec="seconds"),
                "preset": self.config.preset,
                "resolution": f"{self.config.resolution[0]}x{self.config.resolution[1]}",
                # Real encoded length, which is shorter than the sum of the
                # scene durations by one crossfade per cut.
                "duration_s": rendered_seconds,
                "elapsed_s": elapsed,
                "per_scene_s": per_scene,
                "backends": capability_report(),
            },
        )
        self.progress(100.0, f"Done: {output}")
        return output

    @staticmethod
    def _crossfade_for(durations: Sequence[float]) -> float:
        """Crossfade length, capped so it can never swallow a short scene."""
        if len(durations) < 2:
            return 0.0
        fade = min(CROSSFADE_SECONDS, min(durations) / 3.0)
        return fade if fade > 0.05 else 0.0

    @staticmethod
    def _scene_starts(durations: Sequence[float], fade: float) -> List[float]:
        """Start time of each scene on the assembled (overlapping) timeline."""
        starts: List[float] = []
        cursor = 0.0
        for duration in durations:
            starts.append(cursor)
            cursor += duration - fade
        return starts

    def build_timeline(
        self,
        renderers: Sequence[Callable[[float], np.ndarray]],
        durations: Sequence[float],
        fade: float,
    ) -> VideoClip:
        """Assemble every scene into one clip, crossfading between them.

        Doing the dissolve arithmetic here keeps the cost proportional to the
        overlap: only frames actually inside a crossfade render two scenes.
        """
        starts = self._scene_starts(durations, fade)
        total = starts[-1] + durations[-1]
        last = len(renderers) - 1

        def frame_function(t: float) -> np.ndarray:
            # Index of the last scene that has started by t.
            index = 0
            for i, start in enumerate(starts):
                if t >= start:
                    index = i
                else:
                    break
            index = min(index, last)
            frame = renderers[index](t - starts[index])

            # Inside the overlap the previous scene is still on screen.
            if fade > 0 and index > 0 and t < starts[index - 1] + durations[index - 1]:
                alpha = min(1.0, max(0.0, (t - starts[index]) / fade))
                previous = renderers[index - 1](t - starts[index - 1])
                frame = (
                    previous.astype(np.float32) * (1.0 - alpha)
                    + frame.astype(np.float32) * alpha
                ).astype(np.uint8)
            return frame

        return VideoClip(frame_function=frame_function, duration=total)

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
            # CRF is the quality target.  Passing `bitrate=` as well (as the
            # original did) makes the two settings fight and x264 drops one.
            ffmpeg_params=[
                "-crf", str(self.config.preset_data["crf"]),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )

    # -- convenience ------------------------------------------------------- #

    def generate_from_story(
        self, story: str, variant: str = "Cinematic", output: Optional[Path] = None
    ) -> Tuple[VideoScript, Path]:
        """Non-interactive one-shot: story -> chosen variant -> MP4."""
        scripts = self.propose_scripts(story)
        chosen = next((s for s in scripts if s.variant == variant), scripts[0])
        self.accept_script(chosen)
        return chosen, self.render(chosen, output)

    SELF_TEST_STORY = (
        "The lighthouse keeper had not seen a ship in forty years. "
        "Every night he lit the lamp anyway, and every morning he wrote the date "
        "in his ledger. On the fourth of October a sail appeared on the horizon. "
        "He did not trust his eyes, so he waited until noon. "
        "The ship was still there, and it was coming closer."
    )

    def self_test(self, output: Optional[Path] = None) -> Path:
        """Run the full pipeline on a fixed story with zero network access."""
        script = self.scripter.generate_candidates(
            self.SELF_TEST_STORY, variants=["Cinematic"]
        )[0]
        script.id = "self-test"
        script.title = "Self Test"
        self.store.save(script)
        self.accept_script(script)
        return self.render(script, output or self.config.output_dir / "self_test.mp4")


# --------------------------------------------------------------------------- #
# Interactive wizard
# --------------------------------------------------------------------------- #


def _is_interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def _ask(prompt: str, default: str = "", assume_yes: bool = False) -> str:
    """Prompt helper that degrades to defaults when non-interactive or --yes."""
    if assume_yes or not _is_interactive():
        return default
    try:
        answer = input(f"{prompt} [{default}] " if default else f"{prompt} ").strip()
    except EOFError:
        return default
    return answer or default


def _read_story() -> str:
    print("\nWrite your short story. Finish with an empty line.\n")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _print_candidates(scripts: Sequence[VideoScript]) -> None:
    print("\n" + "=" * 74)
    print("CANDIDATE SCRIPTS — pick the one your video should be based on")
    print("=" * 74)
    for i, script in enumerate(scripts, start=1):
        feel = VARIANTS.get(script.variant, {}).get("feel", "")
        print(
            f"\n[{i}] {script.variant}  ·  {len(script.scenes)} scenes  ·  "
            f"~{script.estimated_duration:.0f}s  ·  {script.id}"
        )
        print(f"    {feel}")
        for scene in script.scenes[:2]:
            print(f"    Scene {scene.index}: {textwrap.shorten(scene.narration, 62)}")
            print(f"       visual: {textwrap.shorten(scene.visual, 62)}")
        if len(script.scenes) > 2:
            print(f"    … {len(script.scenes) - 2} more scene(s)")
    print("\n" + "-" * 74)


def run_wizard(
    config: GenerationConfig, story: str = "", assume_yes: bool = False
) -> Optional[Path]:
    caps = capability_report()
    visuals = (
        "Stable Diffusion"
        if caps["diffusers"]
        else "procedural scene cards (pip install -r requirements-ml.txt for AI visuals)"
    )
    narration = (
        "neural TTS"
        if caps["transformers"]
        else ("system voice" if caps["system_tts"] else "silent + captions")
    )
    print("\n" + "=" * 74)
    print("  VIDEO GENERATOR")
    print("=" * 74)
    print(
        f"  preset    : {config.preset}  "
        f"({config.resolution[0]}x{config.resolution[1]} @ {config.fps}fps)"
    )
    print(f"  visuals   : {visuals}")
    print(f"  narration : {narration}")
    print(f"  scripts   : {config.scripts_dir}/")
    print("=" * 74)

    with FreeTextToVideoGenerator(
        config, progress=lambda pct, msg: print(f"  [{pct:5.1f}%] {msg}", flush=True)
    ) as generator:
        interactive = _is_interactive() and not assume_yes
        existing = generator.store.list_scripts()
        if not story and existing and interactive:
            print(
                f"\n{len(existing)} script(s) already in {config.scripts_dir}/. "
                "Enter an id to re-render one, or press Enter to write a new story."
            )
            for script in existing[:8]:
                print(f"  · {script.id}  [{script.status}]  {script.title} ({script.variant})")
            choice = _ask("\nScript id (or Enter for new):")
            if choice:
                try:
                    return _confirm_and_render(generator, generator.store.load(choice), assume_yes)
                except FileNotFoundError:
                    print(f"  ! no script '{choice}', writing a new story instead")

        if not story and interactive:
            story = _read_story()
        if not story:
            print("No story given; nothing to do.")
            return None

        while True:
            print("\nGenerating candidate scripts…")
            scripts = generator.propose_scripts(story)
            _print_candidates(scripts)
            if not interactive:
                chosen = scripts[0]
                print(f"Non-interactive: accepting [1] {chosen.variant}")
                break
            answer = _ask(
                f"Accept which script? 1-{len(scripts)}, 'v N' to view in full, "
                "'r' to regenerate:",
                default="1",
            ).lower()
            if answer.startswith("v"):
                parts = answer.split()
                index = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
                if 0 <= index < len(scripts):
                    print("\n" + "=" * 74)
                    print(scripts[index].to_markdown())
                    print("=" * 74)
                continue
            if answer == "r":
                continue
            if answer.isdigit() and 1 <= int(answer) <= len(scripts):
                chosen = scripts[int(answer) - 1]
                break
            print("  ! not a valid choice")

        generator.accept_script(chosen)
        print(f"\n✓ Accepted: {chosen.variant} → {generator.store.path_for(chosen.id)}")
        return _confirm_and_render(generator, chosen, assume_yes)


def _confirm_and_render(
    generator: FreeTextToVideoGenerator, script: VideoScript, assume_yes: bool
) -> Optional[Path]:
    name = _ask("Output filename:", default=f"{script.id}.mp4", assume_yes=assume_yes)
    output = generator.config.output_dir / name

    print(f"\n  {len(script.scenes)} scenes · ~{script.estimated_duration:.0f}s of video")
    print(f"  estimated render time: {_estimate_render_time(generator, script)}")
    if not assume_yes and _is_interactive():
        if _ask("Render now? [Y/n]", default="y").lower().startswith("n"):
            print("Cancelled. The script is saved and can be rendered later with --render.")
            return None

    started = time.time()
    path = generator.render(script, output)
    print(f"\n✓ {path}  ({time.time() - started:.1f}s)")
    print(f"  script + render log: {generator.store.path_for(script.id)}")
    return path


def _estimate_render_time(generator: FreeTextToVideoGenerator, script: VideoScript) -> str:
    if generator.image_pipe is None:
        return f"under a minute ({len(script.scenes)} procedural scenes)"
    steps = int(generator.config.preset_data["steps"])  # type: ignore[arg-type]
    per_image = {"cuda": 0.6, "mps": 3.0, "cpu": 11.0}.get(generator.device, 11.0) * steps
    total = per_image * len(script.scenes) + script.estimated_duration * 0.8
    return f"~{total / 60:.1f} min ({per_image:.0f}s/image on {generator.device})"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_generator",
        description="Turn a short story into a narrated video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              python video_generator.py                     interactive wizard
              python video_generator.py --self-test         offline smoke test
              python video_generator.py --story-file s.txt --yes --preset draft
              python video_generator.py --list              show saved scripts
              python video_generator.py --render <id>       re-render a saved script
            """
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--story", help="story text passed inline")
    parser.add_argument("--story-file", type=Path, help="read the story from a file")
    parser.add_argument("--output", type=Path, help="output MP4 path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="skip the accept step")
    parser.add_argument("--render", metavar="ID", help="render an existing saved script")
    parser.add_argument("--list", action="store_true", help="list saved scripts and exit")
    parser.add_argument("--self-test", action="store_true", help="offline end-to-end check")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-overlays", action="store_true", help="disable burned-in captions")
    parser.add_argument("--llm", action="store_true", help="enrich prompts with a local LLM")
    parser.add_argument("--yes", "-y", action="store_true", help="accept all defaults")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _console_progress(pct: float, message: str) -> None:
    print(f"  [{pct:5.1f}%] {message}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = GenerationConfig(
        preset=args.preset,
        seed=args.seed,
        music=not args.no_music,
        overlays=not args.no_overlays,
        use_llm=args.llm,
    )

    if args.list:
        for script in ScriptStore(config.scripts_dir).list_scripts():
            print(
                f"{script.id}  [{script.status:9}] {script.variant:12} "
                f"{len(script.scenes)} scenes  {script.title}"
            )
        return 0

    if args.self_test:
        with FreeTextToVideoGenerator(config, progress=_console_progress) as generator:
            path = generator.self_test(args.output)
        print(f"\nSelf-test OK: {path}")
        return 0

    if args.render:
        with FreeTextToVideoGenerator(config, progress=_console_progress) as generator:
            _confirm_and_render(generator, generator.store.load(args.render), args.yes)
        return 0

    story = args.story_file.read_text(encoding="utf-8") if args.story_file else (args.story or "")

    if args.variant and story:
        with FreeTextToVideoGenerator(config, progress=_console_progress) as generator:
            _, path = generator.generate_from_story(story, args.variant, args.output)
        print(f"\n✓ {path}")
        return 0

    run_wizard(config, story=story, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
