"""The seam between the film pipeline and whatever can write music.

This mirrors ``video_gen.visuals`` on purpose. The renderer asks a registry for
a bed and gets samples back; it never knows whether they came from a few sine
waves or from a hosted model, and it never branches on which.

The one rule that is not negotiable: **a score must never fail a render.** A
hosted provider that times out, rate-limits, returns something undecodable or
runs out of credit falls back to the procedural bed and the film still comes
out. That is why ``MusicRegistry.render`` takes a fallback chain rather than a
single backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

import numpy as np

LOGGER = logging.getLogger("video_gen.score")

#: Capability vocabulary.
BED = "bed"  # can fill a whole timeline
CUE = "cue"  # can vary the music per scene rather than looping one mood
PROMPT = "prompt"  # conditioned on free text, not just a mood keyword

ALL_CAPABILITIES = (BED, CUE, PROMPT)


@dataclass
class ScoreRequest:
    """One film's worth of music, fully resolved."""

    #: ``(start, end, mood)`` per scene, in timeline order.
    segments: List[Tuple[float, float, str]] = field(default_factory=list)
    total_duration: float = 0.0
    seed: int = 0
    sample_rate: int = 44100
    style: str = "2d"
    format: str = "short"
    title: str = ""
    #: Free text for models that take a prompt. Built by ``prompt_for`` when
    #: nothing better is supplied.
    prompt: str = ""

    @property
    def moods(self) -> List[str]:
        seen: List[str] = []
        for _, _, mood in self.segments:
            if mood and mood not in seen:
                seen.append(mood)
        return seen

    def prompt_for(self) -> str:
        """A text description of the cue a hosted model should write."""
        if self.prompt:
            return self.prompt
        moods = ", ".join(self.moods) or "neutral"
        bits = [
            f"instrumental score for a {self.format}",
            f"mood: {moods}",
            "no vocals, no lyrics",
            "sits under spoken dialogue, leaves the midrange clear",
        ]
        if self.title:
            bits.insert(1, f"titled {self.title!r}")
        return "; ".join(bits)


@dataclass
class ScoreResult:
    """Mono float32 in [-1, 1] at ``sample_rate``, however it was made."""

    backend: str
    samples: np.ndarray
    sample_rate: int
    detail: str = ""
    used: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32)
        if self.samples.ndim > 1:
            self.samples = self.samples.mean(axis=1).astype(np.float32)

    def fitted(self, duration: float, sample_rate: int) -> np.ndarray:
        """Exactly ``duration`` seconds at ``sample_rate``.

        A hosted model returns whatever length it feels like, so the bed is
        resampled, then looped or trimmed to the timeline. Looping is seamless
        enough under dialogue at -11 dB, and a short bed is far better than a
        silent tail.
        """
        samples = self.samples
        if samples.size == 0:
            return np.zeros(max(1, int(duration * sample_rate)), dtype=np.float32)

        if self.sample_rate != sample_rate:
            target = max(1, int(round(len(samples) * sample_rate / self.sample_rate)))
            samples = np.interp(
                np.linspace(0, len(samples) - 1, num=target),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)

        want = max(1, int(round(duration * sample_rate)))
        if len(samples) >= want:
            return samples[:want].astype(np.float32)

        # Crossfade the loop point so a repeat is a transition, not a click.
        blend = min(len(samples) // 4, int(sample_rate * 0.75))
        out = np.zeros(want, dtype=np.float32)
        cursor = 0
        while cursor < want:
            piece = samples
            if cursor and blend > 1:
                head = piece[:blend] * np.linspace(0.0, 1.0, blend, dtype=np.float32)
                start = max(0, cursor - blend)
                tail = min(want, start + blend)
                if tail > start:
                    span = tail - start
                    out[start:tail] *= np.linspace(1.0, 0.0, blend, dtype=np.float32)[:span]
                    out[start:tail] += head[:span]
                piece = piece[blend:]
            if piece.size == 0:
                break
            end = min(want, cursor + len(piece))
            out[cursor:end] += piece[: end - cursor]
            cursor = end
        return out.astype(np.float32)


@runtime_checkable
class MusicBackend(Protocol):
    """Anything that can write a film's score."""

    name: str
    priority: int

    def available(self) -> bool:
        """Whether this backend can run here, right now."""

    def capabilities(self) -> Set[str]:
        """Which of BED / CUE / PROMPT this backend supports."""

    def render(self, request: ScoreRequest) -> ScoreResult:
        """Produce the score. Must return something usable or raise."""

    def describe(self) -> str:
        """One line for `doctor`, including *why* it is unavailable."""


class MusicRegistry:
    """Chooses a backend, and guarantees a bed comes back regardless."""

    def __init__(self, backends: Sequence[MusicBackend]) -> None:
        self.backends = list(backends)

    def available(self) -> List[MusicBackend]:
        usable: List[MusicBackend] = []
        for backend in self.backends:
            try:
                if backend.available():
                    usable.append(backend)
            except Exception as exc:  # noqa: BLE001 - a broken probe is not fatal
                LOGGER.warning("music backend %s failed its probe: %s", backend.name, exc)
        return sorted(usable, key=lambda b: -b.priority)

    def select(self, prefer: Optional[str] = None) -> MusicBackend:
        usable = self.available()
        if not usable:
            raise RuntimeError("no music backend is available, not even the procedural one")
        if prefer:
            for backend in usable:
                if backend.name == prefer:
                    return backend
            known = ", ".join(b.name for b in usable)
            raise ValueError(f"music backend {prefer!r} is not available; available: {known}")
        return usable[0]

    def chain(self, prefer: Optional[str] = None) -> List[MusicBackend]:
        """The preferred backend first, then every other available one."""
        usable = self.available()
        if not prefer:
            return usable
        head = [b for b in usable if b.name == prefer]
        return head + [b for b in usable if b.name != prefer]

    def render(self, request: ScoreRequest, prefer: Optional[str] = None) -> ScoreResult:
        """First backend that works. A paid provider failing costs a fallback."""
        errors: List[str] = []
        for backend in self.chain(prefer):
            try:
                result = backend.render(request)
                if result.samples.size:
                    if errors:
                        LOGGER.info("score: fell back to %s", backend.name)
                    return result
                errors.append(f"{backend.name}: produced no samples")
            except Exception as exc:  # noqa: BLE001 - try the next one
                errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")
                LOGGER.warning("music backend %s failed: %s", backend.name, exc)
        raise RuntimeError("every music backend failed: " + "; ".join(errors))

    def capabilities(self, prefer: Optional[str] = None) -> Set[str]:
        try:
            return set(self.select(prefer).capabilities())
        except Exception:  # noqa: BLE001
            return set()

    def report(self) -> List[Dict[str, object]]:
        """Per-backend status for the `doctor` command."""
        rows: List[Dict[str, object]] = []
        for backend in sorted(self.backends, key=lambda b: -b.priority):
            try:
                is_available = bool(backend.available())
            except Exception as exc:  # noqa: BLE001
                is_available = False
                LOGGER.debug("probe failed for %s: %s", backend.name, exc)
            rows.append(
                {
                    "name": backend.name,
                    "priority": backend.priority,
                    "available": is_available,
                    "capabilities": sorted(backend.capabilities()) if is_available else [],
                    "detail": backend.describe(),
                }
            )
        return rows


def default_registry(config=None, providers=None) -> MusicRegistry:
    """Procedural always; hosted only when a providers config switches it on."""
    from .procedural import ProceduralMusic

    backends: List[MusicBackend] = []
    try:
        from .hosted import HostedMusic

        backends.append(HostedMusic(config, providers))
    except Exception as exc:  # noqa: BLE001 - importing it must never break the CLI
        LOGGER.debug("hosted music backend unavailable at import: %s", exc)
    backends.append(ProceduralMusic(config))
    return MusicRegistry(backends)
