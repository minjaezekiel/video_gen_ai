"""The seam between the film pipeline and whatever can draw pictures.

Every visual generator implements one Protocol, so the renderer never branches
on hardware. A backend advertises capabilities; the renderer asks for what it
wants and copes with what it gets. A machine that can make stills but not
motion still produces a finished film -- it just uses camera motion instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from PIL import Image

LOGGER = logging.getLogger("video_gen.visuals")

#: Capability vocabulary.
STILL = "still"  # can render one image for a prompt
IDENTITY = "identity"  # can condition on a character reference image
MOTION = "motion"  # can render a moving clip rather than a still
TRAIN = "train"  # can fine-tune a character (LoRA/DreamBooth)

ALL_CAPABILITIES = (STILL, IDENTITY, MOTION, TRAIN)


@dataclass
class CharacterRef:
    """What a backend needs to know to keep a character looking consistent."""

    name: str
    prompt_fragment: str = ""
    reference_image: Optional[Path] = None
    identity_ref: Optional[str] = None
    lora: Optional[str] = None
    palette: Optional[Tuple[int, int, int]] = None
    speaking: bool = False

    @property
    def has_identity(self) -> bool:
        return bool(self.reference_image or self.identity_ref or self.lora)


@dataclass
class ShotRequest:
    """One shot's worth of work, fully resolved -- no lookups needed downstream."""

    prompt: str
    frame_size: Tuple[int, int]
    seed: int
    #: Shared by every shot in one production. Backends that invent their own
    #: colour must key it off this, or consecutive cuts change palette and the
    #: film reads as broken rather than as having a look.
    palette_seed: int = 0
    negative_prompt: str = ""
    image_size: int = 512
    steps: int = 4
    guidance: float = 0.0
    style: str = "2d"
    shot_type: str = "medium"
    camera: str = "static"
    duration: float = 3.0
    fps: int = 24
    characters: List[CharacterRef] = field(default_factory=list)
    want_motion: bool = False

    @property
    def speaking(self) -> List[CharacterRef]:
        return [c for c in self.characters if c.speaking]

    @property
    def needs_identity(self) -> bool:
        return any(c.has_identity for c in self.characters)


@dataclass
class MouthAnchor:
    """Where a staged figure's mouth is, in the plate's own pixels.

    A backend that stages recognisable faces can report these so the renderer
    animates them from the measured audio. Reporting none simply means no lip
    sync for that shot, which is what every model-backed backend does today.
    """

    name: str
    x: float
    y: float
    width: float
    height: float
    speaking: bool = False
    colour: Optional[Tuple[int, int, int]] = None


@dataclass
class ShotVisual:
    """What a backend produced: a still, or a sequence of frames."""

    backend: str
    image: Optional[Image.Image] = None
    frames: Optional[List[Image.Image]] = None
    frame_rate: float = 0.0
    used: Set[str] = field(default_factory=set)
    #: Mouth positions in `image` pixel space, when the backend knows them.
    mouths: List[MouthAnchor] = field(default_factory=list)

    @property
    def is_motion(self) -> bool:
        return bool(self.frames)

    def __post_init__(self) -> None:
        if self.image is None and not self.frames:
            raise ValueError("a ShotVisual must carry either an image or frames")


@runtime_checkable
class VisualBackend(Protocol):
    """Anything that can draw a shot."""

    name: str
    priority: int
    #: Whether `render` may be called from several threads at once. False for
    #: anything holding one model on one device.
    parallel_safe: bool

    def available(self) -> bool:
        """Whether this backend can run here, right now."""

    def capabilities(self) -> Set[str]:
        """Which of STILL / IDENTITY / MOTION / TRAIN this backend supports."""

    def render(self, request: ShotRequest) -> ShotVisual:
        """Produce the shot. Must return something usable or raise."""

    def describe(self) -> str:
        """One line for `doctor`, including *why* it is unavailable."""


class BackendRegistry:
    """Chooses the best available backend and reports honestly on the rest."""

    def __init__(self, backends: Sequence[VisualBackend]) -> None:
        self.backends = list(backends)

    def available(self) -> List[VisualBackend]:
        usable: List[VisualBackend] = []
        for backend in self.backends:
            try:
                if backend.available():
                    usable.append(backend)
            except Exception as exc:  # noqa: BLE001 - a broken probe is not fatal
                LOGGER.warning("backend %s failed its availability probe: %s", backend.name, exc)
        return sorted(usable, key=lambda b: -b.priority)

    def select(self, prefer: Optional[str] = None) -> VisualBackend:
        """Highest-priority available backend, or the one named by ``prefer``."""
        usable = self.available()
        if not usable:
            raise RuntimeError("no visual backend is available, not even the procedural one")
        if prefer:
            for backend in usable:
                if backend.name == prefer:
                    return backend
            known = ", ".join(b.name for b in usable)
            raise ValueError(f"backend {prefer!r} is not available; available: {known}")
        return usable[0]

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


def default_registry(config=None) -> BackendRegistry:
    """The stock backend set: diffusion if possible, procedural always."""
    from .procedural import ProceduralBackend

    backends: List[VisualBackend] = []
    try:
        from .diffusion import DiffusionBackend

        backends.append(DiffusionBackend(config))
    except Exception as exc:  # noqa: BLE001 - importing it must never break the CLI
        LOGGER.debug("diffusion backend unavailable at import: %s", exc)
    backends.append(ProceduralBackend(config))
    return BackendRegistry(backends)
