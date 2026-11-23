"""Music backends: one Protocol, several implementations, one registry."""

from .base import (
    ALL_CAPABILITIES,
    BED,
    CUE,
    PROMPT,
    MusicBackend,
    MusicRegistry,
    ScoreRequest,
    ScoreResult,
    default_registry,
)
from .hosted import HostedMusic, decode_audio
from .mock import MockMusic
from .procedural import ProceduralMusic

__all__ = [
    "ALL_CAPABILITIES",
    "BED",
    "CUE",
    "HostedMusic",
    "MockMusic",
    "MusicBackend",
    "MusicRegistry",
    "PROMPT",
    "ProceduralMusic",
    "ScoreRequest",
    "ScoreResult",
    "decode_audio",
    "default_registry",
]
