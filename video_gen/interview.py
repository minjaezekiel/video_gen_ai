"""The clarifying questions asked before anything is generated.

Every answer has a default inferred from the prompt itself, so the interview is
a confirmation rather than an interrogation -- and so the whole thing collapses
to nothing when running non-interactively. The questions exist to catch the
choices that are expensive to get wrong: format, length, style, aspect and who
is in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .config import ASPECTS, DEFAULT_ASPECT, DEFAULT_FORMAT, DEFAULT_STYLE, FORMATS, STYLE_TIERS
from .screenwriter import Screenwriter, StoryMeta

AskFn = Callable[[str, str], str]

# Prompt phrasing -> the answer it implies. Ordered most specific first.
_FORMAT_HINTS = (
    ("ad", r"\b(advert|advertis\w*|commercial|promo|campaign|brand|product launch)\b"),
    ("educational", r"\b(teach|educat\w*|lesson|lecture|course|classroom|students?|tutorial)\b"),
    ("explainer", r"\b(explain\w*|how it works|onboarding|walkthrough|demo|guide)\b"),
    ("short", r"\b(short film|film|story|drama|scene|screenplay|movie|anime|episode)\b"),
)
_STYLE_HINTS = (
    ("2d", r"\b(anime|manga|cartoon|2d|cel[- ]?shad\w*|illustrat\w*|hand[- ]?drawn)\b"),
    ("3d", r"\b(3d|pixar|cgi|rendered|claymation|blender)\b"),
    ("realistic", r"\b(realistic|photoreal\w*|live[- ]action|documentary|cinematic film|footage)\b"),
)
_ASPECT_HINTS = (
    # "shorts" only in the plural: "a short film" is a format, not an aspect
    # ratio, and matching it made every short film render vertical.
    ("9:16", r"\b(tiktok|reels?|shorts|vertical|portrait|phone)\b"),
    ("1:1", r"\b(square|instagram feed|feed post)\b"),
    ("16:9", r"\b(youtube|widescreen|landscape|tv|cinema)\b"),
)
_DURATION_RE = re.compile(
    r"\b(?P<value>\d{1,3})\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m)\b", re.I
)
_AUDIENCE_HINTS = (
    ("children", r"\b(kids?|children|toddlers?|preschool|young audience)\b"),
    ("teens", r"\b(teens?|teenagers?|high school|ya)\b"),
    ("professionals", r"\b(professionals?|developers?|engineers?|business|enterprise|b2b)\b"),
    ("students", r"\b(students?|learners?|undergrad\w*|classroom)\b"),
)
_TONE_HINTS = (
    ("playful", r"\b(funny|humou?rous|playful|light[- ]hearted|comedy|silly)\b"),
    ("tense", r"\b(tense|thriller|suspense|dark|scary|horror)\b"),
    ("warm", r"\b(warm|heartfelt|touching|wholesome|cosy|cozy)\b"),
    ("urgent", r"\b(urgent|fast|energetic|punchy|action)\b"),
    ("serious", r"\b(serious|formal|professional|corporate)\b"),
)


def _first_hint(text: str, hints: Sequence, fallback: str) -> str:
    for value, pattern in hints:
        if re.search(pattern, text, re.I):
            return value
    return fallback


def infer_meta(prompt: str) -> StoryMeta:
    """Best guess at every answer, read straight out of the prompt."""
    text = prompt or ""
    meta = StoryMeta(
        format=_first_hint(text, _FORMAT_HINTS, DEFAULT_FORMAT),
        style=_first_hint(text, _STYLE_HINTS, DEFAULT_STYLE),
        aspect=_first_hint(text, _ASPECT_HINTS, DEFAULT_ASPECT),
        audience=_first_hint(text, _AUDIENCE_HINTS, "general"),
        tone=_first_hint(text, _TONE_HINTS, ""),
    )

    match = _DURATION_RE.search(text)
    if match:
        value = int(match.group("value"))
        unit = match.group("unit").lower()
        seconds = value * 60 if unit.startswith("m") and unit != "s" else value
        meta.target_seconds = max(5, min(1800, seconds))

    writer = Screenwriter()
    if not writer.looks_like_script(text):
        meta.character_names = [c.name for c in writer.extract_characters(text)]
    meta.title = writer.derive_title(text)
    return meta


@dataclass
class Question:
    """One thing to confirm, with a default good enough to just accept."""

    key: str
    prompt: str
    default: str
    choices: Optional[List[str]] = None
    help: str = ""

    def render(self) -> str:
        if self.choices:
            return f"{self.prompt} ({'/'.join(self.choices)})"
        return self.prompt

    def coerce(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return self.default
        if not self.choices:
            return answer
        lowered = answer.lower()
        for choice in self.choices:
            if choice.lower() == lowered or choice.lower().startswith(lowered):
                return choice
        return self.default


class Interview:
    """Asks the few questions worth asking, then hands back a `StoryMeta`."""

    def __init__(self, ask: Optional[AskFn] = None, assume_yes: bool = False) -> None:
        self.ask = ask
        self.assume_yes = assume_yes
        self.answers: Dict[str, str] = {}

    def questions(self, meta: StoryMeta) -> List[Question]:
        target = meta.resolved_target()
        cast = ", ".join(meta.character_names)
        return [
            Question(
                "title", "Title", meta.title or "Untitled",
                help="Used for the filename and the script record.",
            ),
            Question(
                "format", "What kind of video is this?", meta.format, sorted(FORMATS),
                help="Drives pacing, structure and how dialogue-heavy it is.",
            ),
            Question(
                "style", "Visual style?", meta.style, sorted(STYLE_TIERS),
                help="2d = anime/cartoon, 3d = rendered look, realistic = live action.",
            ),
            Question(
                "aspect", "Aspect ratio?", meta.aspect, sorted(ASPECTS),
                help="9:16 for phone-first, 16:9 for everything else.",
            ),
            Question("target_seconds", "Roughly how long, in seconds?", str(target)),
            Question("audience", "Who is it for?", meta.audience or "general"),
            Question("tone", "Tone?", meta.tone or "neutral"),
            Question(
                "cast", "Characters (comma separated)", cast,
                help="Leave blank for narration only. Detected names are pre-filled.",
            ),
        ]

    def run(self, prompt: str, overrides: Optional[Dict[str, object]] = None) -> StoryMeta:
        """Infer, optionally confirm with the user, and apply explicit overrides.

        Overrides always win: a flag passed on the command line is a decision
        already made, and re-asking it would be noise.
        """
        meta = infer_meta(prompt)
        overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}

        if self.ask is not None and not self.assume_yes:
            for question in self.questions(meta):
                if question.key in overrides:
                    continue
                raw = self.ask(question.render(), question.default)
                self.answers[question.key] = question.coerce(raw)
            self._apply(meta, self.answers)

        self._apply(meta, {k: str(v) for k, v in overrides.items()})
        return meta

    @staticmethod
    def _apply(meta: StoryMeta, answers: Dict[str, str]) -> None:
        if "title" in answers:
            meta.title = answers["title"].strip() or meta.title
        if "format" in answers and answers["format"] in FORMATS:
            meta.format = answers["format"]
        if "style" in answers and answers["style"] in STYLE_TIERS:
            meta.style = answers["style"]
        if "aspect" in answers and answers["aspect"] in ASPECTS:
            meta.aspect = answers["aspect"]
        if "target_seconds" in answers:
            digits = re.sub(r"[^\d]", "", str(answers["target_seconds"]))
            if digits:
                meta.target_seconds = max(5, min(1800, int(digits)))
        if "audience" in answers:
            meta.audience = answers["audience"].strip() or meta.audience
        if "tone" in answers:
            meta.tone = answers["tone"].strip() or meta.tone
        if "cast" in answers:
            names = [n.strip() for n in str(answers["cast"]).split(",") if n.strip()]
            if names:
                meta.character_names = names

    def summary(self, meta: StoryMeta) -> str:
        """One block confirming what will be made, before anything is generated."""
        spec = FORMATS.get(meta.format, FORMATS[DEFAULT_FORMAT])
        cast = ", ".join(meta.character_names) or "narration only"
        return (
            f"  title    : {meta.title}\n"
            f"  format   : {meta.format} ({spec['label']})\n"
            f"  style    : {meta.style} ({STYLE_TIERS[meta.style]['label']})\n"
            f"  aspect   : {meta.aspect}\n"
            f"  length   : ~{meta.resolved_target()}s\n"
            f"  audience : {meta.audience}\n"
            f"  tone     : {meta.tone or 'neutral'}\n"
            f"  cast     : {cast}"
        )
