"""Visual backends: one Protocol, several implementations, one registry."""

from .base import (
    ALL_CAPABILITIES,
    IDENTITY,
    MOTION,
    STILL,
    TRAIN,
    BackendRegistry,
    CharacterRef,
    MouthAnchor,
    ShotRequest,
    ShotVisual,
    VisualBackend,
    default_registry,
)
from .mock import MockBackend
from .procedural import (
    ProceduralBackend,
    character_palette,
    figure_layout,
    render_environment,
    stage_characters,
)

__all__ = [
    "ALL_CAPABILITIES",
    "IDENTITY",
    "MOTION",
    "STILL",
    "TRAIN",
    "BackendRegistry",
    "CharacterRef",
    "MouthAnchor",
    "MockBackend",
    "ProceduralBackend",
    "ShotRequest",
    "ShotVisual",
    "VisualBackend",
    "character_palette",
    "default_registry",
    "figure_layout",
    "render_environment",
    "stage_characters",
]
