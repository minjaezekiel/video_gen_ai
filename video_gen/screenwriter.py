"""Prompt or script in, shot list out.

The deterministic path always runs and always produces a well-formed, castable
screenplay: it finds characters, attributes dialogue, groups beats into scenes,
assigns shot types and writes an image prompt per shot. An optional local LLM
only *rewrites visual prompts*; if it is missing, slow, or emits nonsense, the
deterministic text stands. Keeping the model off the critical path is what makes
this instant and offline-safe.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (
    DEFAULT_ASPECT,
    DEFAULT_FORMAT,
    DEFAULT_STYLE,
    FORMATS,
    MAX_SHOT_SECONDS,
    MIN_SHOT_SECONDS,
    SHOT_TAIL_PAD,
    SHOT_TYPES,
    STYLE_TIERS,
)
from .model import (
    NARRATOR,
    Character,
    Line,
    Production,
    Scene,
    Shot,
    character_key,
    slugify,
    stable_seed,
)
from .llm import discover_llm
from .music import mood_for
from .writersroom import BRIEF_MAX_WORDS, WritersRoom

LOGGER = logging.getLogger("video_gen.screenwriter")

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

# NAME: dialogue  /  NAME (delivery): dialogue -- the common screenplay shorthand.
_SCRIPT_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Z][A-Za-z0-9 .'\-]{0,28}?)\s*"
    r"(?:\((?P<delivery>[^)]{0,40})\))?\s*:\s*(?P<text>\S.*)$"
)
# Slugline: INT./EXT. LOCATION - TIME
_SLUGLINE_RE = re.compile(
    r"^\s*(?P<where>INT\.?|EXT\.?|INT/EXT\.?)\s*(?P<location>[^-\n]*?)\s*(?:-\s*(?P<time>.+))?$",
    re.I,
)
_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{2,300})[\"“”]")
_ATTRIB_AFTER = re.compile(r"^[,\s]*(?:said|asked|replied|shouted|whispered|answered|added)\s+([A-Z][a-z]+)")
_ATTRIB_BEFORE = re.compile(r"([A-Z][a-z]+)\s+(?:said|asked|replied|shouted|whispered|answered|added)[,\s]*$")
# Explicit introductions: "named Dev", "Ama, a barista", "Dev the commuter".
_INTRODUCED_RE = re.compile(
    r"(?:\bnamed\s+|\bcalled\s+)([A-Z][a-z]+)"
    r"|\b([A-Z][a-z]+),\s+(?:a|an|the)\s+[a-z]",
)
# Reported speech in a brief: "Ama tells Dev that the first cup is on the house".
_REPORTED_RE = re.compile(
    r"\b(?P<speaker>[A-Z][a-z]+)\b"
    # An appositive role often sits between the name and the verb:
    # "Ama, a barista, tells ...".
    r"(?:\s*,\s*(?:a|an|the)\s+[^,]{1,50},)?"
    r"\s+(?:tells?|told|says?|said|asks?|explains?|replies|reminds?|promises?|announces?)\s+"
    r"(?P<rest>[^.!?]{4,240})"
)
# Trailing style notes ("Warm and friendly, anime style.") are directions to the
# generator, not lines for the narrator to read out.
_DIRECTIVE_WORDS = frozenset(
    """
    style anime cartoon cinematic realistic photoreal warm friendly playful tense serious
    upbeat energetic calm moody bright dark colourful colorful minimal modern retro tone
    mood aesthetic vibe look feel palette lighting vertical horizontal square seconds
    """.split()
)


def _is_directive(sentence: str) -> bool:
    """A style note rather than story content: short, and mostly style words."""
    words = [w.strip(".,!?;:").lower() for w in sentence.split()]
    words = [w for w in words if w and w not in _STOPWORDS]
    if not words or len(words) > 7:
        return False
    hits = sum(1 for w in words if w in _DIRECTIVE_WORDS)
    return hits >= max(1, len(words) // 2)
# A brief ("a 20 second vertical advert for X") is an instruction, not a title.
_BRIEF_PREFIX_RE = re.compile(
    r"^\s*(?:a|an|the)?\s*\d+\s*(?:second|sec|minute|min)s?\s+"
    r"(?:[a-z-]+\s+){0,3}?(?:advert|advertisement|ad|commercial|video|film|clip|explainer|short)\s+"
    r"(?:for|about|on)\s+",
    re.I,
)

_NON_NAMES = frozenset(
    """
    The A An And But If Then When Where While Because However Meanwhile Later Finally
    Suddenly Inside Outside Today Tomorrow Yesterday After Before During Once Now Next
    It He She They We You I This That These Those There His Her Their Our My Your Its
    Int Ext Fade Cut Scene Shot Title Narrator Voice
    """.split()
    # A brief usually opens with an imperative, and a capitalised first word is
    # otherwise indistinguishable from a name: "Explain photosynthesis..." cast
    # a character called Explain, which then blocked the writers' room from
    # inventing anyone real, because the film already had a "cast".
    + """
    Explain Describe Show Tell Teach Introduce Demonstrate Advertise Promote Announce
    Sell Pitch Create Make Write Generate Produce Build Film Shoot Record Design
    Summarise Summarize Compare Convince Persuade Illustrate Present Walk Give Help
    """.split()
)


def split_sentences(text: str) -> List[str]:
    """Sentence split that also breaks run-on sentences on clause boundaries."""
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    out: List[str] = []
    for sentence in raw:
        if len(sentence.split()) <= 34:
            out.append(sentence)
            continue
        clauses = re.split(r",\s+(?=(?:and|but|then|while|as|until|before|after)\b)", sentence)
        buffer = ""
        for clause in clauses:
            buffer = f"{buffer}, {clause}" if buffer else clause
            if len(buffer.split()) >= 14:
                out.append(buffer.strip(" ,"))
                buffer = ""
        if buffer.strip(" ,"):
            out.append(buffer.strip(" ,"))
    return out or ([text.strip()] if text.strip() else [])


def keywords(text: str, limit: int = 8) -> List[str]:
    """Salient words, proper nouns weighted double."""
    seen: Dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        low = word.lower().strip("'-")
        if len(low) < 3 or low in _STOPWORDS:
            continue
        seen[low] = seen.get(low, 0) + (2 if word[0].isupper() else 1)
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], text.lower().find(kv[0])))
    return [w for w, _ in ranked[:limit]]


def estimate_duration(text: str, wpm: float) -> float:
    words = max(1, len(str(text).split()))
    seconds = words / wpm * 60.0 + SHOT_TAIL_PAD
    return round(max(MIN_SHOT_SECONDS, min(MAX_SHOT_SECONDS, seconds)), 2)


def _tidy_dialogue(text: str) -> str:
    """Trim a quote pulled out of prose into a speakable line."""
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:] if text else text


def _tidy_narration(text: str) -> str:
    """Remove the debris left behind after lifting dialogue out of prose."""
    # Dangling attributions ("she said,", ", said Maya") once the quote is gone.
    text = re.sub(
        r"\b(?:he|she|they|[A-Z][a-z]+)\s+(?:said|asked|replied|shouted|whispered|answered|added)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b(?:said|asked|replied|shouted|whispered|answered|added)\s+[A-Z][a-z]+\b", " ", text
    )
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.")
    return f"{text}." if text and text[-1] not in ".!?" else text


@dataclass
class Beat:
    """One unit of story: a place, a moment, some action and some lines."""

    action: str = ""
    lines: List[Line] = field(default_factory=list)
    setting: str = ""
    time_of_day: str = ""

    @property
    def speech(self) -> str:
        return " ".join(line.text for line in self.lines)


@dataclass
class StoryMeta:
    """The answers the interview collects; every field has a workable default."""

    title: str = ""
    format: str = DEFAULT_FORMAT
    style: str = DEFAULT_STYLE
    aspect: str = DEFAULT_ASPECT
    target_seconds: int = 0
    audience: str = "general"
    tone: str = ""
    character_names: List[str] = field(default_factory=list)
    logline: str = ""

    def resolved_target(self) -> int:
        if self.target_seconds:
            return self.target_seconds
        return int(FORMATS.get(self.format, FORMATS[DEFAULT_FORMAT])["target_seconds"])  # type: ignore[arg-type]


class Screenwriter:
    """Turns a prompt (idea, story or script) into a `Production`."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[object] = None,
        llm_model: str = "",
        variation: int = 0,
    ) -> None:
        self.use_llm = use_llm or llm_client is not None
        self._llm_client = llm_client
        self._llm_model = llm_model
        # Bumped by the caller to get a different draft of the same prompt;
        # without it "regenerate" would return the identical script forever.
        self.variation = variation
        self._room: Optional[WritersRoom] = None
        # Seeds the model so the same prompt writes the same script; set per call.
        self._room_seed_text = ""
        # Set when a model actually contributed, so the CLI can say so and the
        # screenplay can record who wrote it.
        self.written_by = ""

    # -- entry point ------------------------------------------------------- #

    def write(self, prompt: str, meta: Optional[StoryMeta] = None) -> Production:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is empty")
        meta = meta or StoryMeta()
        spec = FORMATS.get(meta.format, FORMATS[DEFAULT_FORMAT])
        self._room_seed_text = prompt

        title = meta.title or self.derive_title(prompt)
        production = Production(
            id=f"{datetime.now():%Y-%m-%d}-{slugify(title)}-{stable_seed(prompt) % 0x10000:04x}",
            title=title,
            format=meta.format,
            style=meta.style,
            aspect=meta.aspect,
            logline=meta.logline or self.derive_logline(prompt),
            audience=meta.audience,
            tone=meta.tone or "neutral",
            target_seconds=meta.resolved_target(),
            source_prompt=prompt,
        )

        if self.looks_like_script(prompt):
            beats = self._beats_from_script(prompt, production)
        else:
            beats = self._beats_from_prose(prompt, production, meta)
            beats = self._maybe_write_dialogue(prompt, production, meta, beats)

        self._build_scenes(production, beats, spec)
        if self.written_by:
            production.config["written_by"] = self.written_by
        if self.use_llm:
            self._enrich(production)
        return production

    # -- script input ------------------------------------------------------ #

    @staticmethod
    def looks_like_script(text: str) -> bool:
        """True when the input already reads as a screenplay."""
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        speaker_lines = sum(1 for line in lines if _SCRIPT_LINE_RE.match(line))
        sluglines = sum(1 for line in lines if _SLUGLINE_RE.match(line.strip()))
        return speaker_lines >= 2 or (speaker_lines >= 1 and sluglines >= 1)

    def _beats_from_script(self, text: str, production: Production) -> List[Beat]:
        """Parse an existing script into beats, honouring sluglines."""
        beats: List[Beat] = []
        setting = ""
        time_of_day = ""
        action: List[str] = []
        lines: List[Line] = []

        def flush() -> None:
            if lines or action:
                beats.append(
                    Beat(
                        action=" ".join(action).strip(),
                        lines=list(lines),
                        setting=setting,
                        time_of_day=time_of_day,
                    )
                )
            action.clear()
            lines.clear()

        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            slug = _SLUGLINE_RE.match(stripped)
            if slug and slug.group("where"):
                flush()
                setting = (slug.group("location") or "").strip(" .-").title()
                # The slugline is the authority on time of day; guessing from
                # the action text below would contradict it.
                time_of_day = (slug.group("time") or "").strip().lower()
                continue
            match = _SCRIPT_LINE_RE.match(raw)
            if match:
                name = match.group("name").strip()
                if name.lower() in {"title", "fade in", "fade out", "cut to"}:
                    continue
                production.add_character(Character(name=name.title()))
                lines.append(
                    Line(
                        character=name.title(),
                        text=match.group("text").strip(),
                        delivery=(match.group("delivery") or "").strip(),
                    )
                )
                continue
            # Unattributed prose inside a script is stage action.
            if lines:
                flush()
            action.append(stripped)
        flush()
        return beats or [Beat(action=text.strip())]

    # -- prose input ------------------------------------------------------- #

    def extract_characters(self, text: str, known: Sequence[str] = ()) -> List[Character]:
        """Find the proper nouns that behave like people.

        The decisive signal is not how often a word appears but whether it is
        *ever* written in lower case. A real name never is, so "Maya" survives a
        single sentence-opening mention while "Banks" or "Rain" -- which show up
        lower case elsewhere, or read as common nouns -- do not. Counting
        occurrences instead loses the protagonist of a short story, who is often
        named only once.
        """
        capital: Dict[str, int] = {}
        lowered: Dict[str, int] = {}
        compound: set = set()
        for sentence in split_sentences(text):
            words = sentence.split()
            cleaned = [w.strip(".,!?;:\"'“”()[]") for w in words]
            for index, word in enumerate(cleaned):
                if len(word) < 3 or not word.isalpha():
                    continue
                key = word.lower()
                if word[0].isupper():
                    if word in _NON_NAMES or key in _STOPWORDS:
                        continue
                    # Adjacent capitalised words are an entity -- a brand, a
                    # company, a place ("Kestrel Coffee") -- not two people.
                    neighbour = cleaned[index + 1] if index + 1 < len(cleaned) else ""
                    previous = cleaned[index - 1] if index else ""
                    for other, at in ((neighbour, index + 1), (previous, index - 1)):
                        if (
                            other
                            and len(other) > 2
                            and other.isalpha()
                            and other[0].isupper()
                            and other not in _NON_NAMES
                            and at != 0  # a sentence-opening capital proves nothing
                            and other.lower() not in _STOPWORDS
                        ):
                            compound.add(word)
                            compound.add(other)
                    # "Banks of monitors" is a common noun, not a character.
                    if neighbour.lower() in {"of", "and"} and index == 0:
                        continue
                    capital[word] = capital.get(word, 0) + (1 if index == 0 else 3)
                else:
                    lowered[key] = lowered.get(key, 0) + 1

        # An explicit introduction outranks everything: "named Dev", "Ama, a barista".
        # The appositive form also matches ordinary sentences ("Outside, the
        # rain..."), so the same exclusions still apply.
        introduced = {g for m in _INTRODUCED_RE.finditer(text) for g in m.groups() if g}
        for name in introduced:
            if name in _NON_NAMES or name.lower() in _STOPWORDS or lowered.get(name.lower()):
                continue
            # Note this does *not* clear `compound`: "Kestrel Coffee, a small
            # independent roastery" matches the appositive form exactly like
            # "Ama, a barista" does, and letting an introduction outrank the
            # two-capitals rule cast the brand's second word as a character.
            capital[name] = capital.get(name, 0) + 8

        # Attribution is the strongest evidence of all: "said Maya" / "Maya said".
        for pattern in (_ATTRIB_AFTER, _ATTRIB_BEFORE):
            for hit in pattern.finditer(text):
                name = hit.group(1)
                capital[name] = capital.get(name, 0) + 6

        names = [
            name
            for name, score in sorted(capital.items(), key=lambda kv: -kv[1])
            if score >= 1 and not lowered.get(name.lower()) and name not in compound
        ]
        for name in reversed(list(known)):
            if name and character_key(name) not in {character_key(n) for n in names}:
                names.insert(0, name)

        return [Character(name=name, description=self._describe(name, text)) for name in names[:6]]

    @staticmethod
    def _describe(name: str, text: str) -> str:
        """The clause around a character's first mention, as their description."""
        for sentence in split_sentences(text):
            if re.search(rf"\b{re.escape(name)}\b", sentence):
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                return cleaned[:220]
        return ""

    def extract_dialogue(
        self, text: str, cast: Sequence[str] = ()
    ) -> List[Tuple[Optional[str], str, str]]:
        """Quoted speech, its speaker, and the raw span to strip from narration.

        Attribution is resolved in three steps: an explicit name next to the
        quote, then a pronoun resolved to the nearest preceding cast member, and
        only then giving up. Skipping the pronoun step puts the wrong character's
        words in their mouth, which is worse than leaving it unattributed.
        """
        results: List[Tuple[Optional[str], str, str]] = []
        names = [n for n in cast if n and character_key(n) != NARRATOR]
        for match in _QUOTE_RE.finditer(text):
            quote = match.group(1).strip()
            after = text[match.end() : match.end() + 60]
            before = text[max(0, match.start() - 80) : match.start()]

            speaker: Optional[str] = None
            span = match.group(0)
            hit = _ATTRIB_AFTER.match(after)
            if hit:
                speaker = hit.group(1)
                span = span + after[: hit.end()]
            else:
                hit = _ATTRIB_BEFORE.search(before.rstrip())
                if hit:
                    speaker = hit.group(1)
                    span = before[hit.start() :] + span

            if speaker is None:
                # A pronoun attribution ("she said") points at whoever was named
                # most recently before the quote.
                pronoun = re.match(r"^[,\s]*(?:he|she|they)\s+\w+", after, re.I)
                preceding = text[: match.start()]
                nearest, position = None, -1
                for name in names:
                    found = preceding.rfind(name)
                    if found > position:
                        nearest, position = name, found
                if nearest is not None:
                    speaker = nearest
                if pronoun:
                    span = span + after[: pronoun.end()]

            results.append((speaker, quote.strip(" ,;"), span))
        return results

    def reported_speech(
        self, chunk: str, cast: Sequence[str]
    ) -> List[Tuple[Optional[str], str, str]]:
        """Turn "Ama tells Dev that X" into Ama actually saying X.

        A brief describes speech rather than quoting it. Without this the
        narrator reads the instructions aloud, which is the most obvious way a
        generated advert looks unfinished.
        """
        known = {character_key(n) for n in cast}
        results: List[Tuple[Optional[str], str, str]] = []
        for match in _REPORTED_RE.finditer(chunk):
            speaker = match.group("speaker")
            if known and character_key(speaker) not in known:
                continue
            rest = match.group("rest").strip(" ,;")
            # "tells <listener> that <content>" -- everything before "that" is
            # who is being addressed, which is staging, not speech.
            split = re.split(r"\bthat\s+", rest, maxsplit=1)
            content = re.sub(r"^\b(?:to|about)\s+", "", (split[1] if len(split) > 1 else rest))
            if len(content.split()) < 2:
                continue
            results.append((speaker, _tidy_dialogue(content), match.group(0)))
        return results

    def _beats_from_prose(
        self, text: str, production: Production, meta: StoryMeta
    ) -> List[Beat]:
        """Segment prose into beats, promoting described speech to dialogue."""
        for character in self.extract_characters(text, meta.character_names):
            production.add_character(character)

        # Drop the instruction sentence of a brief: it describes the deliverable
        # and must never be spoken as part of it.
        all_sentences = split_sentences(text)
        if all_sentences and _BRIEF_PREFIX_RE.match(all_sentences[0]):
            remainder = [s for s in all_sentences[1:] if not _is_directive(s)]
            if len(" ".join(remainder).split()) >= 6:
                text = " ".join(remainder).strip()

        spec = FORMATS.get(meta.format, FORMATS[DEFAULT_FORMAT])
        target = meta.resolved_target()
        shot_low, shot_high = spec["shot_seconds"]  # type: ignore[misc]
        wanted = max(3, min(24, int(target / ((shot_low + shot_high) / 2))))

        sentences = split_sentences(text)
        beats_text = self._group(sentences, wanted)

        cast_names = [c.name for c in production.cast]
        beats: List[Beat] = []
        for chunk in beats_text:
            lines: List[Line] = []
            narration = chunk
            # Resolve against the whole text, not just this beat: the speaker of
            # a quote is often named in an earlier beat.
            offset = text.find(chunk)
            context = text[: offset + len(chunk)] if offset >= 0 else chunk
            spoken = self.extract_dialogue(context, cast_names)
            if not any(quote in chunk for _, quote, _ in spoken):
                # No quoted speech here, so fall back to described speech.
                spoken = self.reported_speech(chunk, cast_names)
            for speaker, quote, span in spoken:
                if quote not in chunk and span not in chunk:
                    continue  # belongs to an earlier beat
                # Strip the quote *and* its attribution, so the narrator does not
                # also read the dialogue or say "she said".
                narration = narration.replace(span, " ")
                if span not in chunk:
                    narration = narration.replace(f'"{quote}"', " ").replace(f"“{quote}”", " ")
                name = speaker or (cast_names[0] if cast_names else NARRATOR)
                if speaker:
                    production.add_character(Character(name=speaker))
                lines.append(Line(character=name, text=_tidy_dialogue(quote)))
            narration = _tidy_narration(narration)
            if narration and (not lines or len(narration.split()) > 4):
                lines.insert(0, Line(character=NARRATOR, text=narration))
            if lines:
                beats.append(Beat(action=chunk, lines=lines))
        if not beats:
            beats = [Beat(action=text, lines=[Line(character=NARRATOR, text=text)])]

        if any(line.is_narration for beat in beats for line in beat.lines):
            production.add_character(
                Character(name=NARRATOR, role="narrator", description="Unseen voice of the film.")
            )
        return beats

    @staticmethod
    def _group(sentences: Sequence[str], count: int) -> List[str]:
        """Even contiguous grouping; never drops or reorders sentences."""
        count = max(1, min(count, len(sentences)))
        base, extra = divmod(len(sentences), count)
        groups: List[str] = []
        cursor = 0
        for i in range(count):
            size = base + (1 if i < extra else 0)
            groups.append(" ".join(sentences[cursor : cursor + size]).strip())
            cursor += size
        return [g for g in groups if g]

    # -- written dialogue -------------------------------------------------- #

    def _writers_room(self) -> WritersRoom:
        seed = stable_seed(self._room_seed_text, salt=self.variation)
        if self._room is None:
            client = self._llm_client
            if client is None and self._llm_model:
                client = discover_llm(model=self._llm_model)
            self._room = WritersRoom(client=client, seed=seed)
        self._room.seed = seed
        return self._room

    def _maybe_write_dialogue(
        self, prompt: str, production: Production, meta: StoryMeta, beats: List[Beat]
    ) -> List[Beat]:
        """Let a model write the dialogue a brief does not contain.

        Two conditions, both necessary:

        * **Nobody speaks yet.** If the deterministic pass already attributed a
          line to a character, that speech came from the user's own text and is
          not ours to rewrite.
        * **The prompt is a brief, not a story.** Below `BRIEF_MAX_WORDS` there
          is no content to lose; above it the narration *is* the film, and
          replacing 300 words of someone's story with six invented lines would
          be rewriting their work rather than finishing it.
        """
        if not self.use_llm:
            return beats
        if any(not line.is_narration for beat in beats for line in beat.lines):
            return beats
        if len(prompt.split()) > BRIEF_MAX_WORDS:
            LOGGER.info("prompt is a story, not a brief; leaving its narration alone")
            return beats

        draft = self._writers_room().develop(prompt, meta, production.cast)
        if draft is None:
            return beats

        for member in draft.cast:
            production.add_character(member)
        self.written_by = draft.model
        LOGGER.info("writers' room: %s", draft.summary())
        return self._beats_from_draft(draft, beats)

    @staticmethod
    def _beats_from_draft(draft, staged: Sequence[Beat]) -> List[Beat]:
        """One beat per written line, reusing whatever staging was worked out.

        The model wrote words; the deterministic pass already decided where the
        film happens. Spreading the lines across those locations in order keeps
        the scene progression it derived from the prompt instead of throwing it
        away for a single flat location.
        """
        places = [
            (beat.setting or draft.setting, beat.time_of_day or draft.time_of_day, beat.action)
            for beat in staged
        ] or [(draft.setting, draft.time_of_day, "")]

        beats: List[Beat] = []
        for index, line in enumerate(draft.lines):
            # Monotonic, so scenes still run in the order they were planned.
            setting, time_of_day, action = places[index * len(places) // max(1, len(draft.lines))]
            beats.append(
                Beat(action=action, lines=[line], setting=setting, time_of_day=time_of_day)
            )
        return beats

    # -- assembly ---------------------------------------------------------- #

    @staticmethod
    def _split_by_speaker(lines: Sequence[Line]) -> List[List[Line]]:
        """Group lines so each shot has at most one speaking character.

        Two characters talking in one shot means the camera can never cut
        between them and the staging cannot show who is speaking, so a beat with
        an exchange becomes a shot/reverse-shot pair. Narration attaches to the
        run that follows it.
        """
        groups: List[List[Line]] = []
        current: List[Line] = []
        current_speaker: Optional[str] = None
        for line in lines:
            if line.is_narration:
                if current_speaker is not None:
                    groups.append(current)
                    current, current_speaker = [], None
                current.append(line)
                continue
            if current_speaker is not None and line.key != current_speaker:
                groups.append(current)
                current = []
            current.append(line)
            current_speaker = line.key
        if current:
            groups.append(current)
        return groups or [[]]

    def _build_scenes(
        self, production: Production, beats: Sequence[Beat], spec: Dict[str, object]
    ) -> None:
        """Turn beats into scenes of shots, with framing that varies."""
        wpm = float(spec["wpm"])  # type: ignore[arg-type]
        style_suffix = str(STYLE_TIERS.get(production.style, {}).get("prompt_suffix", ""))
        cast_names = {c.key for c in production.cast}
        story_context = ", ".join(
            w for w in keywords(production.source_prompt, limit=4) if w.upper() not in cast_names
        )

        # Expand beats into per-speaker shot groups first, so shot framing can be
        # chosen against the real shot count rather than the beat count.
        planned: List[Tuple[Beat, List[Line]]] = []
        for beat in beats:
            for group in self._split_by_speaker(beat.lines):
                if group or beat.action:
                    planned.append((beat, group))

        scenes: List[Scene] = []
        scene: Optional[Scene] = None
        last_setting = object()  # sentinel: the first beat always opens a scene
        shot_index = 0

        for position, (beat, lines) in enumerate(planned):
            if beat.setting != last_setting or scene is None:
                text = f"{beat.action} {beat.speech}"
                scene = Scene(
                    index=len(scenes) + 1,
                    setting=beat.setting or self._setting_from(text, cast_names),
                    time_of_day=beat.time_of_day
                    or ("night" if re.search(r"\bnight\b", text, re.I) else "day"),
                    mood=mood_for(text, production.tone),
                )
                scenes.append(scene)
                last_setting = beat.setting

            shot_index += 1
            speakers = [line.character for line in lines if not line.is_narration]
            subjects = list(dict.fromkeys(speakers)) or self._subjects_for(beat.action, production)
            shot_type = self._shot_type(position, len(planned), bool(speakers), len(subjects))

            shot = Shot(
                index=shot_index,
                shot_type=shot_type,
                subjects=subjects,
                action=re.sub(r"\s+", " ", beat.action).strip()[:240],
                camera=self._camera(shot_type, shot_index),
                lines=list(lines),
                duration=estimate_duration(" ".join(l.text for l in lines), wpm),
            )
            shot.visual = self._visual_prompt(shot, production, style_suffix, story_context)
            scene.shots.append(shot)

        production.scenes = [s for s in scenes if s.shots]

    @staticmethod
    def _setting_from(text: str, cast_names: Optional[set] = None) -> str:
        """Name a scene after its location words, not after who is standing in it."""
        cast_names = cast_names or set()
        words = [w for w in keywords(text, limit=6) if w.upper() not in cast_names]
        return " ".join(w.capitalize() for w in words[:3]) if words else "Untitled"

    @staticmethod
    def _subjects_for(action: str, production: Production) -> List[str]:
        """Cast members actually named in this beat."""
        return [c.name for c in production.cast if re.search(rf"\b{re.escape(c.name)}\b", action, re.I)]

    @staticmethod
    def _shot_type(index: int, total: int, has_dialogue: bool, subject_count: int) -> str:
        """Vary framing so a film does not sit on one lens the whole way through."""
        if index == 0:
            return "establishing"
        if not has_dialogue:
            return "wide" if index % 2 else "insert"
        if subject_count > 1:
            return "medium"
        # Alternate medium/close on single-speaker dialogue; end tight.
        if index == total - 1:
            return "close"
        return "close" if index % 2 else "medium"

    @staticmethod
    def _camera(shot_type: str, index: int) -> str:
        if shot_type == "establishing":
            return "push in"
        if shot_type == "insert":
            return "static"
        return ("static", "push in", "pan right", "pull out", "pan left")[index % 5]

    @staticmethod
    def _visual_prompt(
        shot: Shot, production: Production, style_suffix: str, story_context: str
    ) -> str:
        """The image prompt for a shot: who, where, framed how, in what style."""
        framing = str(SHOT_TYPES.get(shot.shot_type, SHOT_TYPES["medium"])["framing"])
        who = []
        for name in shot.subjects:
            member = production.character(name)
            fragment = member.prompt_fragment() if member else name
            who.append(fragment)
        subject = "; ".join(who)
        setting = ", ".join(keywords(shot.action or shot.speech, limit=5))
        pieces = [p for p in (subject, setting, story_context, framing, style_suffix) if p]
        return ", ".join(pieces)

    # -- titles ------------------------------------------------------------ #

    @staticmethod
    def derive_title(text: str) -> str:
        """Title from the text's own opening words rather than keyword soup."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                return stripped.split(":", 1)[1].strip() or "Untitled"
        first = (split_sentences(text) or [text])[0].strip()
        # "A 20 second vertical advert for Kestrel Coffee" is a brief; the title
        # is what it is *for*, not the instruction wrapped around it.
        stripped = _BRIEF_PREFIX_RE.sub("", first).strip().rstrip(".!?,;:")
        words = (stripped or first.rstrip(".!?,;:")).split()
        if not words:
            return "Untitled"
        return " ".join(words[:8])

    @staticmethod
    def derive_logline(text: str) -> str:
        """First real sentence, skipping screenplay furniture.

        Splitting a script naively yields "INT." as its first sentence, because
        the abbreviation's full stop looks like a sentence boundary.
        """
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or _SLUGLINE_RE.match(stripped) and re.match(r"^(INT|EXT)", stripped, re.I):
                continue
            if _SCRIPT_LINE_RE.match(line):
                continue
            sentences = split_sentences(stripped)
            if sentences and len(sentences[0].split()) >= 3:
                return sentences[0][:200]
        sentences = [s for s in split_sentences(text) if len(s.split()) >= 3]
        return sentences[0][:200] if sentences else ""

    # -- optional LLM enrichment ------------------------------------------- #

    def _enrich(self, production: Production) -> None:
        """Rewrite each shot's image prompt, keeping the deterministic one on failure.

        This runs through the same client as the dialogue, which is the whole
        reason it now works at all: the old implementation went straight to a
        `transformers` pipeline, and torch cannot be installed on this machine.
        """
        room = self._writers_room()
        client = room.client
        if client is None:
            return
        style_suffix = str(STYLE_TIERS.get(production.style, {}).get("prompt_suffix", ""))
        schema = {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        }
        for shot in production.shots:
            try:
                reply = client.complete(
                    "You write image-generation prompts. Describe only what is visible: "
                    "subject, setting, lighting, framing. Never dialogue, never quotes.",
                    f"Film: {production.title}\nShot: {shot.shot_type}\n"
                    f"Action: {shot.action or shot.speech}\n"
                    "Write one vivid prompt under 40 words.",
                    schema=schema,
                    seed=stable_seed(shot.visual or shot.action, salt=shot.index),
                    temperature=0.6,
                    max_tokens=96,
                )
                text = str(reply.json().get("prompt", "")).strip().strip('"')
            except Exception as exc:  # noqa: BLE001 - enrichment is never critical
                LOGGER.debug("prompt enrichment failed on shot %s: %s", shot.index, exc)
                continue
            if 12 <= len(text) <= 400 and "\n" not in text:
                shot.visual = f"{text}, {style_suffix}" if style_suffix else text
