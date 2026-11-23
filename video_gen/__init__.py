"""Offline film generation: prompt or script in, narrated voiced video out."""

from .agent import CommandResult, doctor_payload, production_payload, render_payload
from .casting import cast_production, discover_espeak_voices, discover_voices
from .config import ASPECTS, FORMATS, PRESETS, STYLE_TIERS, GenerationConfig
from .interview import Interview, infer_meta
from .llm import LLMClient, MockLLM, OllamaClient, discover_llm
from .model import Character, Line, Production, Scene, Shot, VoiceProfile
from .render import FilmRenderer
from .providers import ProvidersConfig, ProviderSpec
from .score import (
    HostedMusic,
    MockMusic,
    MusicBackend,
    MusicRegistry,
    ProceduralMusic,
    ScoreRequest,
    ScoreResult,
)
from .screenwriter import Screenwriter, StoryMeta
from .speech import SpeechSynthesizer
from .store import CharacterStore, ProductionStore
from .visuals import BackendRegistry, MockBackend, ProceduralBackend, default_registry
from .writersroom import WritersRoom

__version__ = "3.3.0"

__all__ = [
    "ASPECTS",
    "BackendRegistry",
    "cast_production",
    "Character",
    "CharacterStore",
    "CommandResult",
    "default_registry",
    "discover_espeak_voices",
    "discover_llm",
    "discover_voices",
    "doctor_payload",
    "FilmRenderer",
    "FORMATS",
    "GenerationConfig",
    "HostedMusic",
    "infer_meta",
    "Interview",
    "Line",
    "LLMClient",
    "MockBackend",
    "MockLLM",
    "MockMusic",
    "MusicBackend",
    "MusicRegistry",
    "OllamaClient",
    "PRESETS",
    "ProceduralBackend",
    "ProceduralMusic",
    "Production",
    "production_payload",
    "ProductionStore",
    "ProvidersConfig",
    "ProviderSpec",
    "render_payload",
    "Scene",
    "ScoreRequest",
    "ScoreResult",
    "Screenwriter",
    "Shot",
    "SpeechSynthesizer",
    "StoryMeta",
    "STYLE_TIERS",
    "VoiceProfile",
    "WritersRoom",
]
