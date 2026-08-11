"""Offline film generation: prompt or script in, narrated voiced video out."""

from .casting import cast_production, discover_voices
from .config import ASPECTS, FORMATS, PRESETS, STYLE_TIERS, GenerationConfig
from .interview import Interview, infer_meta
from .model import Character, Line, Production, Scene, Shot, VoiceProfile
from .render import FilmRenderer
from .screenwriter import Screenwriter, StoryMeta
from .speech import SpeechSynthesizer
from .store import CharacterStore, ProductionStore
from .visuals import BackendRegistry, MockBackend, ProceduralBackend, default_registry

__version__ = "2.0.0"

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
]
