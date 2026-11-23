"""Opt-in hosted providers, declared as data rather than code.

Everything else in this project is offline by design. This module is the single
place where that can be relaxed, and it is built so the relaxation is always a
deliberate, visible act:

* **Disabled is the default.** With no config file and no environment variable,
  ``load()`` returns a config where every hosted tier is off, and no socket is
  ever opened.
* **One switch per tier.** Music and speech are enabled separately, so paying
  for a score does not silently start paying for voices too.
* **A provider is a dictionary, not a plugin.** Endpoint, auth header, request
  body and where the audio sits in the response are all declarative, so a new
  service needs a few lines of YAML rather than a code change. Built-in presets
  cover the common ones; ``custom`` covers the rest.
* **Keys live in the environment.** ``api_key_env`` names a variable; the key
  itself is never required to appear in a file that might get committed.

YAML rather than TOML because PyYAML is already a dependency and the project
already stores its screenplays as YAML frontmatter, so there is nothing new for
a user to learn and nothing new to install on Python 3.9.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

LOGGER = logging.getLogger("video_gen.providers")

#: The file name looked for in the project directory and in the user's config
#: directory. ``VIDEO_GEN_PROVIDERS`` overrides both with an explicit path.
CONFIG_NAME = "providers.yaml"
CONFIG_ENV = "VIDEO_GEN_PROVIDERS"

#: Tiers that can be switched on independently.
MUSIC = "music"
SPEECH = "speech"
TIERS = (MUSIC, SPEECH)


# --------------------------------------------------------------------------- #
# Built-in provider presets
#
# These encode the request shape of each service, so a user supplies only a key
# and a switch. Endpoints and field names are the vendors' and can change; the
# `custom` escape hatch exists precisely so a stale preset is never a dead end.
# --------------------------------------------------------------------------- #

BUILTIN_PROVIDERS: Dict[str, Dict[str, Any]] = {
    # Agensi proxies many tools behind one key, which is the whole point of it:
    # one account instead of three. `tool` selects which one.
    "agensi": {
        "tier": MUSIC,
        "endpoint": "https://api.agensi.io/v1/tools/{tool}/invoke",
        "method": "POST",
        "api_key_env": "AGENSI_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "tool": "lyria",
        "body": {
            "input": {
                "prompt": "{prompt}",
                "duration_seconds": "{duration_s:int}",
            }
        },
        "response": "json:output.audio_url",
        "audio_format": "mp3",
        "notes": "one key for Lyria / ElevenLabs Music / MiniMax; set `tool`",
    },
    "lyria": {
        "tier": MUSIC,
        "endpoint": (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "lyria-002:predict"
        ),
        "method": "POST",
        "api_key_env": "GOOGLE_API_KEY",
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
        "body": {
            "instances": [{"prompt": "{prompt}"}],
            "parameters": {"sampleCount": 1},
        },
        "response": "base64:predictions.0.bytesBase64Encoded",
        "audio_format": "wav",
        "notes": "Google Lyria text-to-music",
    },
    "elevenlabs": {
        "tier": MUSIC,
        "endpoint": "https://api.elevenlabs.io/v1/music",
        "method": "POST",
        "api_key_env": "ELEVENLABS_API_KEY",
        "auth_header": "xi-api-key",
        "auth_prefix": "",
        "body": {
            "prompt": "{prompt}",
            "music_length_ms": "{duration_ms:int}",
        },
        "response": "audio",
        "audio_format": "mp3",
        "notes": "ElevenLabs Music",
    },
    "minimax": {
        "tier": MUSIC,
        "endpoint": "https://api.minimax.chat/v1/music_generation",
        "method": "POST",
        "api_key_env": "MINIMAX_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "body": {
            "model": "music-1.5",
            "prompt": "{prompt}",
            "audio_setting": {"sample_rate": 44100, "format": "mp3"},
        },
        "response": "json:data.audio",
        "audio_format": "mp3",
        "notes": "MiniMax Music",
    },
    "elevenlabs-tts": {
        "tier": SPEECH,
        "endpoint": "https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        "method": "POST",
        "api_key_env": "ELEVENLABS_API_KEY",
        "auth_header": "xi-api-key",
        "auth_prefix": "",
        "body": {
            "text": "{text}",
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        "response": "audio",
        "audio_format": "mp3",
        "voices": [
            "21m00Tcm4TlvDq8ikWAM",
            "AZnzlk1XvdvUeBnXmlld",
            "EXAVITQu4vr4xnSDxMaL",
            "ErXwobaYiN019PkySvjV",
            "MF3mGyEYCl7XYWbV9V6O",
            "TxGEqnHWrfWFTfGW9XjX",
            "pNInz6obpgDQGcFmaJgB",
            "yoZ06aMxZJJ28mfd3POQ",
        ],
        "notes": "ElevenLabs text-to-speech; `voices` are cast round-robin",
    },
}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass
class ProviderSpec:
    """One hosted service, fully resolved."""

    name: str
    tier: str = MUSIC
    endpoint: str = ""
    method: str = "POST"
    api_key_env: str = ""
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    headers: Dict[str, str] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    query: Dict[str, Any] = field(default_factory=dict)
    response: str = "audio"
    audio_format: str = "mp3"
    timeout: float = 180.0
    tool: str = ""
    voices: List[str] = field(default_factory=list)
    notes: str = ""

    def key(self) -> str:
        """The API key, from the config or the named environment variable."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def missing_key_reason(self) -> Optional[str]:
        """Why this provider cannot run, or None if it can."""
        if not self.endpoint:
            return "no endpoint configured"
        if self.key():
            return None
        if self.api_key_env:
            return f"${self.api_key_env} is not set"
        return "no api_key or api_key_env configured"

    def redacted(self) -> Dict[str, Any]:
        """Safe to print: says whether a key was found, never what it is."""
        return {
            "name": self.name,
            "tier": self.tier,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self.key()),
            "response": self.response,
            "audio_format": self.audio_format,
            "notes": self.notes,
        }


@dataclass
class ProvidersConfig:
    """Which hosted tiers are on, and the specs they resolve to."""

    enabled: Dict[str, bool] = field(default_factory=lambda: {t: False for t in TIERS})
    selected: Dict[str, str] = field(default_factory=lambda: {t: "" for t in TIERS})
    specs: Dict[str, ProviderSpec] = field(default_factory=dict)
    source: Optional[Path] = None

    def is_enabled(self, tier: str) -> bool:
        return bool(self.enabled.get(tier))

    def spec_for(self, tier: str) -> Optional[ProviderSpec]:
        """The configured provider for a tier, only if that tier is enabled."""
        if not self.is_enabled(tier):
            return None
        name = self.selected.get(tier) or ""
        spec = self.specs.get(name)
        if spec is None:
            LOGGER.warning("%s is enabled but provider %r is unknown", tier, name)
            return None
        if spec.tier != tier:
            LOGGER.warning("provider %r is a %s provider, not %s", name, spec.tier, tier)
            return None
        return spec

    def status(self) -> List[Dict[str, Any]]:
        """One row per tier for `--doctor` and `--providers`."""
        rows: List[Dict[str, Any]] = []
        for tier in TIERS:
            spec = self.specs.get(self.selected.get(tier) or "")
            reason: Optional[str] = None
            if not self.is_enabled(tier):
                reason = "disabled (offline)"
            elif spec is None:
                reason = f"provider {self.selected.get(tier)!r} is not configured"
            else:
                reason = spec.missing_key_reason()
            rows.append(
                {
                    "tier": tier,
                    "enabled": self.is_enabled(tier),
                    "provider": self.selected.get(tier) or "",
                    "ready": self.is_enabled(tier) and spec is not None and reason is None,
                    "detail": reason or "ready",
                }
            )
        return rows


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def config_paths(explicit: Optional[Path] = None) -> List[Path]:
    """Where a providers file may live, most specific first."""
    if explicit:
        return [Path(explicit)]
    from_env = os.environ.get(CONFIG_ENV, "").strip()
    if from_env:
        return [Path(from_env).expanduser()]
    return [
        Path.cwd() / CONFIG_NAME,
        Path.home() / ".config" / "video_gen" / CONFIG_NAME,
    ]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _spec_from(name: str, raw: Mapping[str, Any]) -> ProviderSpec:
    """Build a spec from a preset plus the user's overrides."""
    base = copy.deepcopy(BUILTIN_PROVIDERS.get(name, {}))
    # `preset` lets a user name their provider freely and still inherit a
    # built-in request shape: `mymusic: {preset: elevenlabs, ...}`.
    preset = str(raw.get("preset", "") or "")
    if preset:
        inherited = copy.deepcopy(BUILTIN_PROVIDERS.get(preset, {}))
        inherited.update(base)
        base = inherited
    merged: Dict[str, Any] = {**base, **{k: v for k, v in raw.items() if k != "preset"}}
    known = {f for f in ProviderSpec.__dataclass_fields__ if f != "name"}  # noqa: SLF001
    unknown = sorted(set(merged) - known)
    if unknown:
        LOGGER.debug("provider %s: ignoring unknown keys %s", name, ", ".join(unknown))
    return ProviderSpec(name=name, **{k: v for k, v in merged.items() if k in known})


def load(explicit: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> ProvidersConfig:
    """Read the providers config, or return an all-disabled one.

    Never raises: a malformed file logs a warning and leaves every hosted tier
    off, because a syntax error in an optional config must not stop a render
    that was going to run offline anyway.
    """
    environ = env if env is not None else os.environ
    config = ProvidersConfig()

    # Presets are always known, so `--providers` can list what is available
    # even with no config file at all.
    for name, raw in BUILTIN_PROVIDERS.items():
        config.specs[name] = _spec_from(name, raw)

    data: Dict[str, Any] = {}
    for path in config_paths(explicit):
        try:
            if not path.is_file():
                continue
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                LOGGER.warning("%s: expected a mapping at the top level", path)
                continue
            data = loaded
            config.source = path
            break
        except Exception as exc:  # noqa: BLE001 - an optional file must not be fatal
            LOGGER.warning("could not read %s: %s", path, exc)

    for tier in TIERS:
        section = data.get(tier) or {}
        if not isinstance(section, dict):
            LOGGER.warning("%s: section %r should be a mapping", config.source, tier)
            section = {}
        for name, raw in (section.get("providers") or {}).items():
            if isinstance(raw, dict):
                config.specs[str(name)] = _spec_from(str(name), raw)
        config.enabled[tier] = _as_bool(section.get("enabled"), False)
        config.selected[tier] = str(section.get("provider", "") or "")

    # Environment overrides, so an agent can switch a tier on for one run
    # without writing a file: VIDEO_GEN_MUSIC_PROVIDER=elevenlabs.
    for tier in TIERS:
        chosen = str(environ.get(f"VIDEO_GEN_{tier.upper()}_PROVIDER", "") or "").strip()
        if chosen:
            config.selected[tier] = chosen
            config.enabled[tier] = True
        switch = environ.get(f"VIDEO_GEN_{tier.upper()}_HOSTED")
        if switch is not None:
            config.enabled[tier] = _as_bool(switch, False)

    return config


# --------------------------------------------------------------------------- #
# Request templating
# --------------------------------------------------------------------------- #


def _coerce(raw: str, cast: str) -> Any:
    if cast == "int":
        return int(round(float(raw)))
    if cast == "float":
        return float(raw)
    return raw


def fill(template: Any, values: Mapping[str, Any]) -> Any:
    """Substitute ``{name}`` and ``{name:int}`` through a nested structure.

    A string that is *only* a placeholder takes the value's own type, so
    ``"{duration_ms:int}"`` becomes a JSON number rather than the string
    ``"30000"`` -- which is the difference between a working request and a 400.
    """
    if isinstance(template, str):
        stripped = template.strip()
        if stripped.startswith("{") and stripped.endswith("}") and stripped.count("{") == 1:
            token = stripped[1:-1]
            name, _, cast = token.partition(":")
            if name in values:
                return _coerce(str(values[name]), cast) if cast else values[name]
            return template
        out = template
        for name, value in values.items():
            out = out.replace("{" + name + "}", str(value))
        return out
    if isinstance(template, dict):
        return {key: fill(value, values) for key, value in template.items()}
    if isinstance(template, (list, tuple)):
        return [fill(item, values) for item in template]
    return template


def extract(response: Any, selector: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Pull audio out of a response according to ``selector``.

    ``audio`` means the body is already the audio. ``json:a.b.0.c`` walks a
    decoded JSON body to a URL to fetch. ``base64:a.b`` walks to inline base64.
    Returns ``(bytes, url)`` -- exactly one of which is set.
    """
    kind, _, path = selector.partition(":")
    kind = kind.strip() or "audio"
    if kind == "audio":
        return (response if isinstance(response, bytes) else None), None

    node: Any = response
    for part in [p for p in path.split(".") if p]:
        if isinstance(node, Mapping):
            node = node.get(part)
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None, None
        else:
            return None, None
        if node is None:
            return None, None

    if kind == "base64":
        import base64

        if isinstance(node, str):
            try:
                return base64.b64decode(node), None
            except Exception:  # noqa: BLE001
                return None, None
        return None, None

    if isinstance(node, str):
        # Some services inline a data: URI where a URL is documented.
        if node.startswith("data:"):
            import base64

            _, _, payload = node.partition(",")
            try:
                return base64.b64decode(payload), None
            except Exception:  # noqa: BLE001
                return None, None
        if node.startswith("http://") or node.startswith("https://"):
            return None, node
        # A bare hex blob is how at least one vendor returns audio.
        if len(node) > 64 and all(c in "0123456789abcdefABCDEF" for c in node[:64]):
            try:
                return bytes.fromhex(node), None
            except ValueError:
                return None, None
    return None, None


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #

EXAMPLE_CONFIG = """\
# video_gen hosted providers.
#
# Everything here is OFF by default and the whole pipeline runs offline without
# it. Switch a tier on only if you want that tier to call a paid service.
#
# Put this file at ./providers.yaml, or at ~/.config/video_gen/providers.yaml,
# or point $VIDEO_GEN_PROVIDERS at it.
#
# Keys are read from the environment -- never write one into this file if the
# file is going into git.

music:
  enabled: false          # true = call a hosted model for the score
  provider: elevenlabs    # agensi | lyria | elevenlabs | minimax | <your own>

speech:
  enabled: false          # true = call a hosted model for character voices
  provider: elevenlabs-tts

# Optional. Add or override a provider; `preset` inherits a built-in request
# shape so you only state what differs.
#
# music:
#   providers:
#     my-music:
#       preset: elevenlabs
#       endpoint: https://api.example.com/v1/music
#       api_key_env: MY_MUSIC_KEY
#       auth_header: Authorization
#       auth_prefix: "Bearer "
#       body:
#         prompt: "{prompt}"
#         seconds: "{duration_s:int}"
#       response: json:data.url    # audio | json:<path> | base64:<path>
#       audio_format: mp3
#
# Placeholders available to `body`, `endpoint` and `query`:
#   music   {prompt} {mood} {duration_s} {duration_ms} {seed} {style} {title}
#   speech  {text} {voice} {rate} {pitch} {gender}
# Append :int or :float to send a number instead of a string.
"""


def write_example(path: Path, force: bool = False) -> Path:
    """Drop a commented, all-disabled config next to the user's project."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    return path
