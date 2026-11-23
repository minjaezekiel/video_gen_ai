"""The always-available score: synthesised, offline, instant.

This is the original behaviour, moved behind the backend seam without changing
a sample of what it produces. It is the floor the hosted tier falls back to, so
it must never be unavailable and never raise.
"""

from __future__ import annotations

from typing import Optional, Set

from ..music import DEFAULT_MOOD, SAMPLE_RATE, score_timeline
from .base import BED, CUE, ScoreRequest, ScoreResult


class ProceduralMusic:
    """Mood-driven sine pads with crossfaded per-scene cues."""

    name = "procedural"
    priority = 10

    def __init__(self, config=None) -> None:
        self.config = config

    def available(self) -> bool:
        return True

    def capabilities(self) -> Set[str]:
        # No PROMPT: it reads a mood keyword, not free text.
        return {BED, CUE}

    def render(self, request: ScoreRequest) -> ScoreResult:
        rate = request.sample_rate or SAMPLE_RATE
        segments = request.segments or [(0.0, request.total_duration, DEFAULT_MOOD)]
        samples = score_timeline(
            segments,
            request.total_duration,
            seed=request.seed,
            rate=rate,
        )
        return ScoreResult(
            backend=self.name,
            samples=samples,
            sample_rate=rate,
            detail=f"{len(segments)} cue(s), moods: {', '.join(request.moods) or DEFAULT_MOOD}",
            used={BED, CUE},
        )

    def describe(self) -> str:
        return "synthesised pads, per-scene mood, no model and no network"
