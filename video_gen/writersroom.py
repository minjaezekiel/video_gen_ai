"""Dialogue written by a model, and the gate that decides whether to keep it.

The deterministic screenwriter can structure a story, stage it, and lift quoted
or reported speech into lines. What it cannot do is *invent* speech: given "a 20
second advert for a coffee roastery" there is nothing to lift, so the narrator
ends up reading the brief back. That is the gap this module closes.

Two design rules, both learned by measurement rather than assumed:

**Narrow schemas, several calls.** A 1B model asked for beats of five fields
(setting, time, action, character, text) put the action in `setting` and left
half the `text` fields empty. The same model asked only for `{character, text}`
pairs produced usable advertising copy every time. So the work is split: the
model writes *dialogue*, and the deterministic screenwriter keeps doing the
staging, framing and timing it is already good at.

**Nothing the model says is trusted.** Every line goes through `clean_line`,
every speaker is resolved against the real cast, and a draft that survives with
too few lines is discarded whole. The deterministic result is always still
there to fall back to, so the worst case of a bad model is the behaviour we had
before it was installed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import FORMATS, MAX_SHOT_SECONDS
from .llm import LLMError, cast_names_hint, discover_llm
from .model import NARRATOR, Character, Line, character_key

LOGGER = logging.getLogger("video_gen.writersroom")

# Long enough to be worth speaking, short enough to fit a shot. MAX_SHOT_SECONDS
# at a normal 150 wpm is about 30 words, so anything past that would be cut off
# by the shot clock anyway. The lower bound is characters rather than words
# because "Morning." and "Exactly!" are real lines while "Go" is usually debris.
MIN_LINE_LETTERS = 4
MAX_LINE_WORDS = int(MAX_SHOT_SECONDS / 60.0 * 150)

# A draft with fewer than this is not worth the disruption of replacing the
# deterministic version.
MIN_DRAFT_LINES = 3

# Above this a prompt is a story, not a brief: it contains the film's actual
# content, and inventing dialogue to replace it would discard the user's work.
BRIEF_MAX_WORDS = 60

_SPEAKER_PREFIX_RE = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z0-9 .'\-]{0,24}?)\s*"
                                r"(?:\((?P<delivery>[^)]{0,40})\))?\s*:\s*")
_STAGE_MARKERS = ("INT.", "EXT.", "CUT TO", "FADE IN", "FADE OUT", "V.O.", "O.S.", "SFX")
# Whether a description already says who someone is, so a stated gender is only
# added when it would otherwise be missing.
_GENDERED_RE = re.compile(r"\b(he|him|his|man|boy|male|she|her|woman|girl|female)\b", re.I)
_STOP = frozenset(
    """a an the and or but of to in on at for with from by as is are was were be it its this that
    you your we our they their he she his her i my me not no so if then than there here""".split()
)


@dataclass
class Draft:
    """What the room produced, and what it cost to get it."""

    lines: List[Line] = field(default_factory=list)
    cast: List[Character] = field(default_factory=list)
    setting: str = ""
    time_of_day: str = ""
    model: str = ""
    seconds: float = 0.0
    rejected: List[str] = field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for line in self.lines:
            if line.character not in seen and not line.is_narration:
                seen.append(line.character)
        return seen

    def summary(self) -> str:
        who = ", ".join(self.speakers) or "narrator only"
        return f"{len(self.lines)} line(s) for {who} — {self.model} in {self.seconds:.1f}s"


# --------------------------------------------------------------------------- #
# Validation — the part that makes an unreliable model safe to use
# --------------------------------------------------------------------------- #


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in _STOP]


def echoes(text: str, brief: str) -> bool:
    """True when a "line" is really the brief read back at us.

    This is the exact failure the module exists to fix, so it must not be
    reintroduced by a model that paraphrases its instructions instead of
    answering them.
    """
    words = _content_words(text)
    if len(words) < 4:
        return False
    haystack = set(_content_words(brief))
    if not haystack:
        return False
    shared = sum(1 for w in words if w in haystack)
    return shared / len(words) >= 0.85


def clean_line(raw: str) -> Optional[tuple]:
    """Reduce a model's output to speakable words, or reject it.

    Returns ``(text, delivery)``. Rejection is not a failure -- it is the whole
    point -- so callers should expect ``None`` routinely.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return None
    # Checked before the speaker prefix is stripped, because "CUT TO: the street"
    # looks exactly like an attributed line once the prefix is gone.
    if any(marker in text.upper() for marker in _STAGE_MARKERS):
        return None

    # Models re-emit the speaker even when told not to: "AMA: Morning."
    delivery = ""
    prefix = _SPEAKER_PREFIX_RE.match(text)
    if prefix:
        delivery = (prefix.group("delivery") or "").strip()
        text = text[prefix.end():].strip()

    # A leading parenthetical is a delivery note, not speech.
    lead = re.match(r"^\((?P<note>[^)]{1,40})\)\s*", text)
    if lead:
        delivery = delivery or lead.group("note").strip()
        text = text[lead.end():].strip()

    text = text.strip('"“”\'')
    # Strip trailing stage directions the model appended to the speech itself.
    text = re.sub(r"\s*[\(\[][^)\]]{0,60}[\)\]]\s*$", "", text).strip()
    text = re.sub(r"\*+", "", text).strip()

    if not text:
        return None
    if text.startswith(("#", "-", "*", ">")):
        return None
    # An all-caps line is a slugline or a heading, not something a person says.
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 8 and all(c.isupper() for c in letters):
        return None
    if len(letters) < MIN_LINE_LETTERS:
        return None

    words = text.split()
    if len(words) > MAX_LINE_WORDS:
        # Trim on a sentence boundary so the line still ends like a line.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept: List[str] = []
        for sentence in sentences:
            if len(" ".join(kept + [sentence]).split()) > MAX_LINE_WORDS:
                break
            kept.append(sentence)
        text = " ".join(kept) if kept else " ".join(words[:MAX_LINE_WORDS])
        if len([c for c in text if c.isalpha()]) < MIN_LINE_LETTERS:
            return None

    if text and text[-1] not in ".!?…":
        text += "."
    return text[:1].upper() + text[1:], delivery


def resolve_speaker(name: str, cast: Sequence[Character]) -> Optional[str]:
    """Map whatever the model called someone onto an actual cast member.

    Exact key first, then a first-name match, because a model given "Ama" will
    sometimes answer "Ama the barista". An unrecognised name is rejected rather
    than added: a character with no voice, description or staging is worse than
    one fewer speaker.
    """
    key = character_key(name)
    if not key:
        return None
    if key == NARRATOR:
        return NARRATOR
    for member in cast:
        if member.key == key:
            return member.name
    first = key.split()[0] if key.split() else ""
    for member in cast:
        if member.key.split() and member.key.split()[0] == first:
            return member.name
    return None


def validate_lines(
    raw: Sequence[Any], cast: Sequence[Character], brief: str, rejected: Optional[List[str]] = None
) -> List[Line]:
    """Turn the model's `{character, text}` list into lines that can be filmed."""
    rejected = rejected if rejected is not None else []
    lines: List[Line] = []
    for item in raw:
        if not isinstance(item, dict):
            rejected.append("not an object")
            continue
        cleaned = clean_line(str(item.get("text", "")))
        if cleaned is None:
            rejected.append(f"unspeakable: {str(item.get('text', ''))[:48]!r}")
            continue
        text, delivery = cleaned
        if echoes(text, brief):
            rejected.append(f"echoes the brief: {text[:48]!r}")
            continue
        speaker = resolve_speaker(str(item.get("character", "")), cast)
        if speaker is None:
            rejected.append(f"unknown speaker: {str(item.get('character', ''))[:24]!r}")
            continue
        # A model that repeats itself produces a film that stutters.
        if lines and lines[-1].text.lower() == text.lower():
            rejected.append("duplicate line")
            continue
        lines.append(Line(character=speaker, text=text, delivery=delivery))
    return lines


def validate_cast(raw: Sequence[Any], limit: int = 3) -> List[Character]:
    """Accept invented characters only if they are usable as characters."""
    out: List[Character] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = re.sub(r"[^A-Za-z '\-]", "", str(item.get("name", ""))).strip()
        name = " ".join(name.split()[:2]).title()
        if not name or len(name) < 2 or character_key(name) in seen:
            continue
        if character_key(name) == NARRATOR:
            continue
        description = re.sub(r"\s+", " ", str(item.get("description", ""))).strip()[:180]
        # `casting.infer_traits` reads the description, so the stated gender goes
        # in as a word it already recognises rather than into a hidden field.
        # That keeps it visible and hand-editable in `characters/<id>.md`.
        gender = str(item.get("gender", "")).strip().lower()
        if gender in {"male", "female"} and not _GENDERED_RE.search(description):
            article = "A man" if gender == "male" else "A woman"
            description = f"{article}. {description}".strip()
        seen.add(character_key(name))
        out.append(Character(name=name, description=description, role=str(item.get("role", "")).strip()[:40]))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def dialogue_schema(minimum: int, maximum: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": max(1, minimum),
                "maxItems": max(minimum, maximum),
                "items": {
                    "type": "object",
                    "properties": {"character": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["character", "text"],
                },
            }
        },
        "required": ["lines"],
    }


CAST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    # Asked for outright rather than inferred: a model writing
                    # "the owner, with a kind smile" leaves nothing for
                    # `infer_traits` to read, and an invented name is unlikely
                    # to be in its given-name table either, so both roles in a
                    # two-hander ended up with the same gender of voice.
                    "gender": {"type": "string", "enum": ["male", "female"]},
                    "description": {"type": "string"},
                },
                "required": ["name", "gender", "description"],
            },
        }
    },
    "required": ["characters"],
}

SETTING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "setting": {"type": "string"},
        "time_of_day": {"type": "string", "enum": ["day", "night"]},
    },
    "required": ["setting", "time_of_day"],
}


# --------------------------------------------------------------------------- #
# The room
# --------------------------------------------------------------------------- #


class WritersRoom:
    """Asks a local model for dialogue and refuses most of what comes back."""

    def __init__(self, client: Optional[object] = None, seed: int = 0) -> None:
        self._client = client
        self._searched = client is not None
        self.seed = seed
        self.last_error = ""

    @property
    def client(self):
        if not self._searched:
            self._client = discover_llm()
            self._searched = True
        return self._client

    def available(self) -> bool:
        client = self.client
        return client is not None and client.available()

    # -- individual asks --------------------------------------------------- #

    def invent_cast(self, brief: str, meta, count: int = 2) -> List[Character]:
        """Characters for a brief that names none.

        A film with no cast has nothing but a narrator, and "a voice for each
        character" is the point of the whole program.
        """
        client = self.client
        if client is None:
            return []
        audience = getattr(meta, "audience", "general")
        try:
            reply = client.complete(
                "You cast short films. A description is a short phrase giving age, job and "
                "manner — never a sentence of story. Use ordinary first names.",
                f"Film: {brief.strip()[:600]}\n"
                f"Format: {getattr(meta, 'format', 'short')}. Audience: {audience}.\n"
                f"Invent {count} contrasting characters who could carry this on screen. "
                "Give each a first name, a gender, and a short description.",
                schema=CAST_SCHEMA,
                seed=self.seed,
                temperature=0.7,
                max_tokens=300,
            )
            payload = reply.json()
        except LLMError as exc:
            self.last_error = str(exc)
            LOGGER.info("cast invention failed (%s); staying with the deterministic cast", exc)
            return []
        characters = payload.get("characters") if isinstance(payload, dict) else None
        return validate_cast(characters or [], limit=count)

    def find_setting(self, brief: str) -> tuple:
        """One location for the film. Long answers are hallucinated specifics."""
        client = self.client
        if client is None:
            return "", ""
        try:
            reply = client.complete(
                "You are a location scout. 'setting' is two to four words naming a place.",
                f"Film: {brief.strip()[:600]}\nName the single location this is filmed in.",
                schema=SETTING_SCHEMA,
                seed=self.seed,
                temperature=0.5,
                max_tokens=80,
            )
            payload = reply.json()
        except LLMError as exc:
            self.last_error = str(exc)
            return "", ""
        if not isinstance(payload, dict):
            return "", ""
        setting = re.sub(r"[^A-Za-z0-9 ,'\-]", "", str(payload.get("setting", ""))).strip()
        # Asked for a place, a small model will happily name a real university.
        # More than five words means it invented specifics nobody asked for.
        if not setting or len(setting.split()) > 5:
            return "", ""
        time_of_day = str(payload.get("time_of_day", "")).strip().lower()
        return setting.title(), (time_of_day if time_of_day in {"day", "night"} else "")

    def write_dialogue(
        self, brief: str, meta, cast: Sequence[Character], wanted: int
    ) -> Draft:
        """The main ask: `wanted` alternating spoken lines for this cast."""
        draft = Draft(cast=list(cast))
        client = self.client
        if client is None:
            self.last_error = "no local LLM"
            return draft

        speaking = [c for c in cast if c.key != NARRATOR]
        roster = "\n".join(
            f"- {c.name}" + (f" ({c.description})" if c.description else "") for c in speaking
        ) or "- Narrator (an unseen voice)"
        tone = getattr(meta, "tone", "") or "neutral"
        fmt = getattr(meta, "format", "short")
        spec = FORMATS.get(fmt, FORMATS["short"])
        # Each format already declares its own narrative shape ("hook, problem,
        # product, proof, call to action"). Handing the model that structure is
        # what stops a 30-second advert from being six unrelated nice sentences.
        structure = " → ".join(str(b) for b in spec.get("beats", []))  # type: ignore[union-attr]
        closing = (
            "End on a call to action."
            if spec.get("needs_cta")
            else "End on a line that lands."
        )

        try:
            reply = client.complete(
                "You write film dialogue. Output only the words a person says out loud. "
                "Never write a character name inside the text, never write stage directions, "
                "never describe the camera. One idea per line.",
                f"Film: {brief.strip()[:900]}\n"
                f"Format: {spec.get('label', fmt)}. Tone: {tone}. "
                f"Audience: {getattr(meta, 'audience', 'general')}.\n"
                + (f"Shape it as: {structure}.\n" if structure else "")
                + f"Characters:\n{roster}\n\n"
                f"Write {wanted} lines of dialogue, alternating between "
                f"{cast_names_hint([c.name for c in speaking]) or 'the narrator'}. "
                f"Each line under 16 words. {closing}",
                # Capped at `wanted`, not above it: the schema is the only hard
                # limit on runtime, since a model told "six lines" will happily
                # write eight and the film then overshoots its target length.
                schema=dialogue_schema(max(2, wanted - 2), wanted),
                seed=self.seed,
                temperature=0.8,
                max_tokens=180 + 60 * wanted,
            )
            payload = reply.json()
        except LLMError as exc:
            self.last_error = str(exc)
            LOGGER.info("dialogue generation failed (%s); keeping the deterministic script", exc)
            return draft

        draft.model = reply.model
        draft.seconds = reply.seconds
        raw = payload.get("lines") if isinstance(payload, dict) else None
        draft.lines = validate_lines(raw or [], cast, brief, draft.rejected)
        if draft.rejected:
            LOGGER.debug("rejected %d line(s): %s", len(draft.rejected), "; ".join(draft.rejected[:4]))
        return draft

    # -- the whole job ----------------------------------------------------- #

    def develop(self, brief: str, meta, cast: Sequence[Character]) -> Optional[Draft]:
        """Cast if needed, then write dialogue. `None` means "keep what you had".

        Returning `None` rather than a partial draft is deliberate: a caller
        should never have to decide whether half a draft is worth using.
        """
        if not self.available():
            return None

        speaking = [c for c in cast if c.key != NARRATOR]
        invented: List[Character] = []
        if not speaking:
            invented = self.invent_cast(brief, meta)
            speaking = invented
        if not speaking:
            LOGGER.info("no cast to write for; keeping the deterministic script")
            return None

        wanted = self.line_budget(meta)
        draft = self.write_dialogue(brief, meta, speaking, wanted)
        draft.cast = invented
        if len(draft.lines) < MIN_DRAFT_LINES:
            LOGGER.info(
                "draft rejected: %d usable line(s) of %d requested (%s)",
                len(draft.lines),
                wanted,
                "; ".join(draft.rejected[:3]) or "model returned nothing",
            )
            return None
        draft.setting, draft.time_of_day = self.find_setting(brief)
        return draft

    @staticmethod
    def line_budget(meta) -> int:
        """How many lines fill the target runtime.

        Derived from the format's own words-per-minute rather than a constant,
        so an advert (fast, punchy) and an explainer (measured) do not get the
        same shot count for the same duration.
        """
        target = int(getattr(meta, "resolved_target", lambda: 30)())
        spec = FORMATS.get(getattr(meta, "format", "short"), {})
        low, high = spec.get("shot_seconds", (3.0, 6.0))  # type: ignore[misc]
        per_line = max(2.5, (float(low) + float(high)) / 2.0)
        return max(3, min(12, int(round(target / per_line))))
