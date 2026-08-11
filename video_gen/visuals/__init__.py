"""Visual backends: one Protocol, several implementations, one registry."""

from .base import (
    ALL_CAPABILITIES,
    IDENTITY,
    MOTION,
    STILL,
    TRAIN,
    BackendRegistry,
    CharacterRef,
    ShotRequest,
    ShotVisual,
    VisualBackend,
    default_registry,
)
from .mock import MockBackend
from .procedural import ProceduralBackend, character_palette, render_environment, stage_characters

__all__ = [
    "ALL_CAPABILITIES",
    "IDENTITY",
    "MOTION",
    "STILL",
    "TRAIN",
    "BackendRegistry",
    "CharacterRef",
    "MockBackend",
    "ProceduralBackend",
    "ShotRequest",
    "ShotVisual",
    "VisualBackend",
    "character_palette",
    "default_registry",
    "render_environment",
    "stage_characters",
]
