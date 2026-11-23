"""Subtitle sidecars built from the render's own measured timings.

Burned-in captions suit a finished film; a sidecar suits everything else --
searching, translating, accessibility, re-cutting, and an agent handing a video
to a person who wants the text. The timings are not estimated: they are the
speech windows the renderer measured after synthesis, offset by each shot's
place on the timeline, which is the same clock the audio is mixed on.

Word timings *are* an estimate, and the only one here: a line's measured span
is divided across its words in proportion to their length, with a floor so a
one-letter word still gets a readable moment. A real forced aligner would be
better and would mean shipping an acoustic model, which this project exists to
avoid. The estimate is good enough to drive karaoke-style highlighting and is
documented as an estimate wherever it surfaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

#: Minimum share of a line a single word may occupy, as a fraction.
_MIN_WORD_SHARE = 0.02

#: A cue shorter than this is extended; anything briefer cannot be read.
MIN_CUE_SECONDS = 0.6

#: How long a cue is held past the end of its audio.
CUE_TAIL_SECONDS = 0.35


@dataclass
class Word:
    """One word and the slice of its line's span it is estimated to occupy."""

    text: str
    start: float
    end: float


@dataclass
class Cue:
    """One subtitle: a span, who said it, and the text."""

    index: int
    start: float
    end: float
    text: str
    speaker: str = ""
    narration: bool = False
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _clock(seconds: float, comma: bool) -> str:
    """`01:02:03,456` for SRT, `01:02:03.456` for WebVTT."""
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:  # rounding may carry
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def estimate_words(text: str, start: float, end: float) -> List[Word]:
    """Split a line's span across its words, weighted by length.

    Weighting by characters rather than dividing evenly matters: "I" and
    "extraordinarily" do not take the same time to say, and an even split makes
    highlighting visibly lag the voice on any line with a long word in it.
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    span = max(0.0, end - start)
    if not tokens or span <= 0:
        return []

    # Letters only: punctuation is not spoken, so it should not buy time.
    weights = [max(1.0, len(re.sub(r"[^\w']", "", token)) or 1) for token in tokens]
    total = sum(weights)
    floor = _MIN_WORD_SHARE * total
    weights = [max(w, floor) for w in weights]
    total = sum(weights)

    words: List[Word] = []
    cursor = start
    for token, weight in zip(tokens, weights):
        share = span * (weight / total)
        words.append(Word(token, round(cursor, 3), round(min(end, cursor + share), 3)))
        cursor += share
    # Absorb float drift into the last word rather than leaving a gap.
    if words:
        words[-1].end = round(end, 3)
    return words


def build_cues(
    spoken: Sequence[Tuple[float, float, str, str, bool]],
    with_words: bool = True,
) -> List[Cue]:
    """Turn `(start, end, character, text, narration)` tuples into cues.

    Input is in timeline seconds. Overlaps are resolved by trimming the earlier
    cue, because two subtitles on screen at once is worse than one ending a
    fraction early.
    """
    ordered = sorted(
        (s for s in spoken if str(s[3]).strip()), key=lambda s: (s[0], s[1])
    )
    cues: List[Cue] = []
    for start, end, character, text, narration in ordered:
        start = max(0.0, float(start))
        end = max(start, float(end)) + CUE_TAIL_SECONDS
        if end - start < MIN_CUE_SECONDS:
            end = start + MIN_CUE_SECONDS
        cue = Cue(
            index=len(cues) + 1,
            start=round(start, 3),
            end=round(end, 3),
            text=" ".join(str(text).split()),
            speaker="" if narration else str(character),
            narration=bool(narration),
        )
        if cues and cue.start < cues[-1].end:
            cues[-1].end = round(max(cues[-1].start + 0.05, cue.start - 0.02), 3)
        if with_words:
            cue.words = estimate_words(cue.text, cue.start, cue.end)
        cues.append(cue)
    return cues


def to_srt(cues: Iterable[Cue], speaker_labels: bool = True) -> str:
    """SRT, the format every player and every transcription tool reads."""
    blocks: List[str] = []
    for position, cue in enumerate(cues, start=1):
        label = f"{cue.speaker.upper()}: " if (speaker_labels and cue.speaker) else ""
        blocks.append(
            f"{position}\n"
            f"{_clock(cue.start, True)} --> {_clock(cue.end, True)}\n"
            f"{label}{cue.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(cues: Iterable[Cue], speaker_labels: bool = True, words: bool = False) -> str:
    """WebVTT, with optional inline word timings for karaoke highlighting.

    `<00:00:01.250>` between words is the standard cue-timing mechanism, so a
    browser can highlight each word as it is spoken with no extra data.
    """
    out = ["WEBVTT", ""]
    for position, cue in enumerate(cues, start=1):
        out.append(str(position))
        out.append(f"{_clock(cue.start, False)} --> {_clock(cue.end, False)}")
        label = ""
        if speaker_labels and cue.speaker:
            # <v Name> is WebVTT's own voice span; players show or style it.
            label = f"<v {cue.speaker}>"
        if words and cue.words:
            body = []
            for index, word in enumerate(cue.words):
                if index:
                    body.append(f"<{_clock(word.start, False)}>")
                body.append(word.text)
            out.append(label + " ".join(body).replace("> ", ">"))
        else:
            out.append(f"{label}{cue.text}")
        out.append("")
    return "\n".join(out)


def write_sidecars(
    output: Path,
    cues: Sequence[Cue],
    speaker_labels: bool = True,
    word_timings: bool = True,
) -> List[Path]:
    """Write `<output>.srt` and `<output>.vtt` beside a rendered film."""
    output = Path(output)
    written: List[Path] = []
    if not cues:
        return written
    targets = [
        (output.with_suffix(".srt"), to_srt(cues, speaker_labels)),
        (output.with_suffix(".vtt"), to_vtt(cues, speaker_labels, word_timings)),
    ]
    for path, body in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def parse_srt(text: str) -> List[Cue]:
    """Read SRT back. Used by the tests, and by anyone editing a transcript."""

    def seconds(stamp: str) -> float:
        stamp = stamp.strip().replace(",", ".")
        hours, minutes, rest = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(rest)

    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        match = re.match(r"(.+?)\s*-->\s*(.+)", lines[1])
        if not match:
            continue
        body = " ".join(lines[2:]).strip()
        speaker = ""
        label = re.match(r"^([A-Z][A-Z0-9 '\-]*):\s*(.*)$", body)
        if label:
            speaker, body = label.group(1).title(), label.group(2)
        cues.append(
            Cue(
                index=int(lines[0].strip() or len(cues) + 1),
                start=seconds(match.group(1)),
                end=seconds(match.group(2)),
                text=body,
                speaker=speaker,
                narration=not speaker,
            )
        )
    return cues
