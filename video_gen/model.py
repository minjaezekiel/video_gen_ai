"""The screenplay data model, and its markdown representation.

A production is a cast plus a shot list, not a flat list of narration lines.
Everything persists as markdown with YAML frontmatter so a script stays
diffable, hand-editable, and readable by the model on a later run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from .config import (
    DEFAULT_ASPECT,
    DEFAULT_FORMAT,
    DEFAULT_PRESET,
    DEFAULT_STYLE,
    MIN_SHOT_SECONDS,
)

NARRATOR = "NARRATOR"


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def slugify(text: str, max_len: int = 32) -> str:
    """Filename-safe slug, truncated on a word boundary rather than mid-word."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        head = slug[:max_len].rsplit("-", 1)[0]
        slug = head or slug[:max_len]
    return slug.strip("-") or "untitled"


def stable_seed(text: str, salt: int = 0) -> int:
    """Deterministic across processes, unlike ``hash()`` under PYTHONHASHSEED."""
    return int(hashlib.md5(f"{salt}:{text}".encode("utf-8")).hexdigest()[:8], 16)


def character_key(name: str) -> str:
    """Canonical lookup key for a character name (case/spacing insensitive)."""
    return re.sub(r"\s+", " ", name).strip().upper()


# --------------------------------------------------------------------------- #
# Cast
# --------------------------------------------------------------------------- #


@dataclass
class VoiceProfile:
    """A cast voice, resolved to a concrete synthesiser voice.

    ``engine``/``voice`` name the backend and its voice; ``pitch`` and ``rate``
    are shaping applied on top, so two characters sharing a voice still sound
    like different people.
    """

    engine: str = "system"  # system | neural | silent
    voice: str = ""
    rate: float = 1.0  # 1.0 = the voice's natural pace
    pitch: float = 0.0  # semitones
    gender: str = "unknown"  # male | female | neutral | unknown
    age: str = "adult"  # child | young | adult | elder

    def to_dict(self) -> Dict[str, object]:
        return {
            "engine": self.engine,
            "voice": self.voice,
            "rate": round(self.rate, 3),
            "pitch": round(self.pitch, 2),
            "gender": self.gender,
            "age": self.age,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "VoiceProfile":
        data = data or {}
        return cls(
            engine=str(data.get("engine", "system")),
            voice=str(data.get("voice", "")),
            rate=float(data.get("rate", 1.0) or 1.0),
            pitch=float(data.get("pitch", 0.0) or 0.0),
            gender=str(data.get("gender", "unknown")),
            age=str(data.get("age", "adult")),
        )


@dataclass
class Character:
    """A cast member. Reusable across productions via ``characters/<id>.md``."""

    name: str
    role: str = ""  # protagonist, narrator, customer, teacher...
    description: str = ""  # who they are; drives casting and prompts
    appearance: str = ""  # what they look like; drives visual conditioning
    voice: VoiceProfile = field(default_factory=VoiceProfile)
    reference_image: Optional[str] = None  # user-supplied photo/art
    identity_ref: Optional[str] = None  # cached identity embedding
    lora: Optional[str] = None  # trained character LoRA
    palette: Optional[List[int]] = None  # procedural staging colour

    @property
    def key(self) -> str:
        return character_key(self.name)

    @property
    def id(self) -> str:
        return slugify(self.name)

    def prompt_fragment(self) -> str:
        """How this character is described to an image model."""
        parts = [p.strip() for p in (self.appearance, self.description) if p and p.strip()]
        return ", ".join(parts) if parts else self.name

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "appearance": self.appearance,
            "voice": self.voice.to_dict(),
        }
        for key in ("reference_image", "identity_ref", "lora"):
            value = getattr(self, key)
            if value:
                data[key] = value
        if self.palette:
            data["palette"] = list(self.palette)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Character":
        palette = data.get("palette")
        return cls(
            name=str(data.get("name", "Unnamed")),
            role=str(data.get("role", "") or ""),
            description=str(data.get("description", "") or ""),
            appearance=str(data.get("appearance", "") or ""),
            voice=VoiceProfile.from_dict(data.get("voice")),  # type: ignore[arg-type]
            reference_image=(str(data["reference_image"]) if data.get("reference_image") else None),
            identity_ref=(str(data["identity_ref"]) if data.get("identity_ref") else None),
            lora=(str(data["lora"]) if data.get("lora") else None),
            palette=[int(c) for c in palette] if isinstance(palette, (list, tuple)) else None,
        )


# --------------------------------------------------------------------------- #
# Shots
# --------------------------------------------------------------------------- #


@dataclass
class Line:
    """One spoken line, attributed to a character."""

    character: str  # character name, or NARRATOR
    text: str
    delivery: str = ""  # parenthetical: (whispering), (excited)

    @property
    def key(self) -> str:
        return character_key(self.character)

    @property
    def is_narration(self) -> bool:
        return self.key == NARRATOR


@dataclass
class Shot:
    """One continuous camera setup."""

    index: int
    shot_type: str = "medium"
    subjects: List[str] = field(default_factory=list)  # character names on screen
    action: str = ""  # what is visible / happening
    camera: str = "static"
    lines: List[Line] = field(default_factory=list)
    duration: float = MIN_SHOT_SECONDS
    visual: str = ""  # resolved image prompt, filled by the screenwriter

    @property
    def speech(self) -> str:
        return " ".join(line.text for line in self.lines).strip()

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for line in self.lines:
            if line.character not in seen:
                seen.append(line.character)
        return seen

    def to_markdown(self) -> str:
        head = f"### Shot {self.index} — {self.shot_type} — {self.duration:.1f}s"
        parts = [head, ""]
        if self.subjects:
            parts.append(f"**With:** {', '.join(self.subjects)}")
        if self.camera and self.camera != "static":
            parts.append(f"**Camera:** {self.camera}")
        if self.action:
            parts.append(f"**Action:** {self.action}")
        if self.visual:
            parts.append(f"**Visual:** {self.visual}")
        for line in self.lines:
            delivery = f" ({line.delivery})" if line.delivery else ""
            parts.append(f"**{line.character.upper()}{delivery}:** {line.text}")
        return "\n".join(parts)


@dataclass
class Scene:
    """A location and time; a container for shots."""

    index: int
    setting: str = ""
    time_of_day: str = "day"
    mood: str = "neutral"
    shots: List[Shot] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(shot.duration for shot in self.shots)

    def to_markdown(self) -> str:
        heading = f"## Scene {self.index} — {self.setting or 'Untitled'}"
        meta = f"*{self.time_of_day} · {self.mood}*"
        body = "\n\n".join(shot.to_markdown() for shot in self.shots)
        return f"{heading}\n\n{meta}\n\n{body}"


# --------------------------------------------------------------------------- #
# Production
# --------------------------------------------------------------------------- #


@dataclass
class Production:
    """A complete film: metadata, cast and shot list."""

    id: str
    title: str
    format: str = DEFAULT_FORMAT
    style: str = DEFAULT_STYLE
    aspect: str = DEFAULT_ASPECT
    logline: str = ""
    audience: str = "general"
    tone: str = "neutral"
    target_seconds: int = 60
    status: str = "draft"  # draft -> accepted | rejected -> rendered
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_prompt: str = ""
    cast: List[Character] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)
    renders: List[Dict[str, object]] = field(default_factory=list)

    # -- access ------------------------------------------------------------ #

    @property
    def shots(self) -> List[Shot]:
        """Every shot in order, flattened across scenes."""
        return [shot for scene in self.scenes for shot in scene.shots]

    @property
    def estimated_duration(self) -> float:
        return sum(shot.duration for shot in self.shots)

    @property
    def speaking_cast(self) -> List[Character]:
        speaking = {line.key for shot in self.shots for line in shot.lines}
        return [c for c in self.cast if c.key in speaking]

    def character(self, name: str) -> Optional[Character]:
        key = character_key(name)
        return next((c for c in self.cast if c.key == key), None)

    def add_character(self, character: Character) -> Character:
        """Add unless already cast; returns whichever instance is now in the cast."""
        existing = self.character(character.name)
        if existing is not None:
            return existing
        self.cast.append(character)
        return character

    # -- markdown ---------------------------------------------------------- #

    def to_markdown(self) -> str:
        front = {
            "id": self.id,
            "title": self.title,
            "format": self.format,
            "style": self.style,
            "aspect": self.aspect,
            "logline": self.logline,
            "audience": self.audience,
            "tone": self.tone,
            "target_seconds": self.target_seconds,
            "status": self.status,
            "created": self.created,
            "source_prompt": self.source_prompt,
            "cast": [c.to_dict() for c in self.cast],
            "config": self.config,
            "renders": self.renders,
        }
        header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=88)
        body = "\n\n".join(scene.to_markdown() for scene in self.scenes)
        cast_block = "\n".join(
            f"- **{c.name}**"
            + (f" — {c.role}" if c.role else "")
            + (f" · voice: {c.voice.voice}" if c.voice.voice else "")
            for c in self.cast
        )
        return (
            f"---\n{header}---\n\n"
            f"# {self.title}\n\n"
            + (f"> {self.logline}\n\n" if self.logline else "")
            + (f"## Cast\n\n{cast_block}\n\n" if cast_block else "")
            + f"{body}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "Production":
        front, body = split_frontmatter(text)
        cast_data = front.get("cast") or []
        production = cls(
            id=str(front.get("id", "unknown")),
            title=str(front.get("title", "Untitled")),
            format=str(front.get("format", DEFAULT_FORMAT)),
            style=str(front.get("style", DEFAULT_STYLE)),
            aspect=str(front.get("aspect", DEFAULT_ASPECT)),
            logline=str(front.get("logline", "") or ""),
            audience=str(front.get("audience", "general")),
            tone=str(front.get("tone", "neutral")),
            target_seconds=int(front.get("target_seconds", 60) or 60),
            status=str(front.get("status", "draft")),
            created=str(front.get("created", "")),
            source_prompt=str(front.get("source_prompt", "") or ""),
            cast=[Character.from_dict(c) for c in cast_data if isinstance(c, dict)],
            scenes=parse_scenes(body),
            config=dict(front.get("config") or {}),
            renders=list(front.get("renders") or []),
        )
        return production


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SCENE_RE = re.compile(r"^##\s+Scene\s+(?P<index>\d+)\s*(?:[—–-]\s*(?P<setting>.*?))?\s*$", re.M)
_SHOT_RE = re.compile(
    r"^###\s+Shot\s+(?P<index>\d+)\s*"
    r"(?:[—–-]\s*(?P<type>[a-zA-Z]+)\s*)?"
    r"(?:[—–-]\s*(?P<duration>[\d.]+)\s*s)?\s*$",
    re.M,
)
_META_RE = re.compile(r"^\*(?P<time>[^·*]+?)\s*·\s*(?P<mood>[^*]+?)\*\s*$", re.M)
# Named fields (**With:**) and dialogue (**MAYA (whispering):**) share a shape,
# so one pattern captures both and the caller decides which it is.
_FIELD_RE = re.compile(r"^\*\*(?P<key>[^:*]+?):\*\*\s*(?P<value>.*)$", re.M)
_KNOWN_FIELDS = {"with", "camera", "action", "visual"}


def split_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        front = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        front = {}
    return (front if isinstance(front, dict) else {}), text[match.end() :]


def _parse_shot(block: str, index: int, shot_type: str, duration: float) -> Optional[Shot]:
    shot = Shot(index=index, shot_type=shot_type or "medium", duration=duration)
    for match in _FIELD_RE.finditer(block):
        raw_key = match.group("key").strip()
        value = match.group("value").strip()
        key = raw_key.lower()
        if key == "with":
            shot.subjects = [s.strip() for s in value.split(",") if s.strip()]
        elif key == "camera":
            shot.camera = value or "static"
        elif key == "action":
            shot.action = value
        elif key == "visual":
            shot.visual = value
        elif key not in _KNOWN_FIELDS and value:
            # Anything else in bold with a colon is a speaker, optionally with
            # a parenthetical delivery note: **MAYA (whispering):** text
            speaker = raw_key
            delivery = ""
            paren = re.match(r"^(?P<name>.+?)\s*\((?P<delivery>[^)]*)\)\s*$", raw_key)
            if paren:
                speaker = paren.group("name").strip()
                delivery = paren.group("delivery").strip()
            shot.lines.append(Line(character=speaker.strip(), text=value, delivery=delivery))
    if not (shot.lines or shot.action or shot.visual or shot.subjects):
        return None
    if not shot.subjects and shot.lines:
        shot.subjects = shot.speakers
    return shot


def parse_scenes(body: str) -> List[Scene]:
    """Parse ``## Scene`` / ``### Shot`` blocks.

    Tolerant by design: unparseable input yields ``[]`` so a caller can fall
    back rather than crash mid-render, and a malformed shot is skipped instead
    of taking its scene down with it.
    """
    scenes: List[Scene] = []
    scene_matches = list(_SCENE_RE.finditer(body))
    for i, match in enumerate(scene_matches):
        start = match.end()
        end = scene_matches[i + 1].start() if i + 1 < len(scene_matches) else len(body)
        block = body[start:end]

        scene = Scene(index=int(match.group("index")), setting=(match.group("setting") or "").strip())
        meta = _META_RE.search(block)
        if meta:
            scene.time_of_day = meta.group("time").strip()
            scene.mood = meta.group("mood").strip()

        shot_matches = list(_SHOT_RE.finditer(block))
        for j, shot_match in enumerate(shot_matches):
            shot_start = shot_match.end()
            shot_end = (
                shot_matches[j + 1].start() if j + 1 < len(shot_matches) else len(block)
            )
            duration = float(shot_match.group("duration") or 0) or MIN_SHOT_SECONDS
            shot = _parse_shot(
                block[shot_start:shot_end],
                index=int(shot_match.group("index")),
                shot_type=(shot_match.group("type") or "medium").strip().lower(),
                duration=duration,
            )
            if shot is not None:
                scene.shots.append(shot)
        if scene.shots:
            scenes.append(scene)
    return scenes
