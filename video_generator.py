#!/usr/bin/env python3
"""Compatibility shim.

The engine moved from this single file into the ``video_gen`` package when it
grew from a narrated-slideshow generator into a film pipeline (cast, shots,
dialogue, per-character voices, pluggable visual backends). This module keeps
``python video_generator.py`` and ``from video_generator import ...`` working.

New code should import from ``video_gen`` directly:

    from video_gen import FilmRenderer, GenerationConfig, Screenwriter
"""

from __future__ import annotations

import sys

from video_gen import (  # noqa: F401
    ASPECTS,
    FORMATS,
    PRESETS,
    STYLE_TIERS,
    BackendRegistry,
    Character,
    CharacterStore,
    FilmRenderer,
    GenerationConfig,
    Interview,
    Line,
    MockBackend,
    ProceduralBackend,
    Production,
    ProductionStore,
    Scene,
    Screenwriter,
    Shot,
    SpeechSynthesizer,
    StoryMeta,
    VoiceProfile,
    cast_production,
    default_registry,
    discover_voices,
    infer_meta,
)
from video_gen.cli import main  # noqa: F401

__all__ = [
    "ASPECTS",
    "FORMATS",
    "PRESETS",
    "STYLE_TIERS",
    "BackendRegistry",
    "Character",
    "CharacterStore",
    "FilmRenderer",
    "GenerationConfig",
    "Interview",
    "Line",
    "MockBackend",
    "ProceduralBackend",
    "Production",
    "ProductionStore",
    "Scene",
    "Screenwriter",
    "Shot",
    "SpeechSynthesizer",
    "StoryMeta",
    "VoiceProfile",
    "cast_production",
    "default_registry",
    "discover_voices",
    "infer_meta",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
