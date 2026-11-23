"""Offline film generation: prompt or script in, narrated voiced video out."""

from .casting import cast_production, discover_voices
from .config import ASPECTS, FORMATS, PRESETS, STYLE_TIERS, GenerationConfig
from .interview import Interview, infer_meta
from .llm import LLMClient, MockLLM, OllamaClient, discover_llm
from .model import Character, Line, Production, Scene, Shot, VoiceProfile
from .render import FilmRenderer
from .screenwriter import Screenwriter, StoryMeta
from .speech import SpeechSynthesizer
from .store import CharacterStore, ProductionStore
from .visuals import BackendRegistry, MockBackend, ProceduralBackend, default_registry
from .writersroom import WritersRoom

__version__ = "2.1.0"

__all__ = [
    "ASPECTS",
    "BackendRegistry",
    "cast_production",
    "Character",
    "CharacterStore",
    "default_registry",
    "discover_llm",
    "discover_voices",
    "FilmRenderer",
    "FORMATS",
    "GenerationConfig",
    "infer_meta",
    "Interview",
    "Line",
    "LLMClient",
    "MockBackend",
    "MockLLM",
    "OllamaClient",
    "PRESETS",
    "ProceduralBackend",
    "Production",
    "ProductionStore",
    "Scene",
    "Screenwriter",
    "Shot",
    "SpeechSynthesizer",
    "StoryMeta",
    "STYLE_TIERS",
    "VoiceProfile",
    "WritersRoom",
]
