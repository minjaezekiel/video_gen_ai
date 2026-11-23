"""The machine-readable contract: what an agent gets back from `--json`.

Kept in one file on purpose. An agent calling this tool needs a payload whose
shape it can rely on across versions, and a schema spread across the CLI's
print statements is not a contract -- it is whatever the formatting happened to
be that week.

Three rules hold here:

* **Every payload carries ``schema``.** An agent can branch on it instead of
  guessing from which keys are present.
* **Paths are strings, absolute where they name a real file.** An agent's
  working directory is rarely the one the render ran in.
* **Nothing is omitted on failure.** ``ok`` and ``error`` are always present,
  so a caller never has to tell "failed" apart from "key missing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Bumped only when an existing key changes meaning or disappears. Adding a key
#: is not a breaking change and does not bump it.
SCHEMA = 1


# --------------------------------------------------------------------------- #
# Exit codes
#
# An agent reads these before it reads anything else, so they are part of the
# contract and documented in SKILL.md.
# --------------------------------------------------------------------------- #

EXIT_OK = 0
EXIT_USAGE = 1  # nothing to make, bad flag, unknown id
EXIT_NO_AUDIO = 2  # a film came out, but no line was audible
EXIT_FAILED = 3  # no film came out
EXIT_CANCELLED = 4  # the user declined at a prompt

EXIT_MEANINGS = {
    EXIT_OK: "ok",
    EXIT_USAGE: "usage error",
    EXIT_NO_AUDIO: "rendered with no audible dialogue",
    EXIT_FAILED: "render failed",
    EXIT_CANCELLED: "cancelled",
}


@dataclass
class CommandResult:
    """What every CLI command returns: an exit code and a JSON payload."""

    code: int = EXIT_OK
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code == EXIT_OK

    def payload(self, command: str) -> Dict[str, Any]:
        """The complete object written to stdout under ``--json``."""
        body: Dict[str, Any] = {
            "schema": SCHEMA,
            "command": command,
            "ok": self.ok,
            "exit_code": self.code,
            "status": EXIT_MEANINGS.get(self.code, "error"),
            "error": self.data.get("error"),
        }
        # The envelope wins: a command that returned a key of its own by the
        # same name would otherwise redefine `status` or `ok` under the caller.
        reserved = set(body) - {"error"}
        for key, value in self.data.items():
            if key == "error":
                continue
            body[f"data_{key}" if key in reserved else key] = value
        return body


def fail(code: int, message: str, **extra: Any) -> CommandResult:
    return CommandResult(code=code, data={"error": message, **extra})


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #


def _path(value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(Path(value).resolve())
    except Exception:  # noqa: BLE001 - an unresolvable path is still worth reporting
        return str(value)


def voice_payload(voice) -> Dict[str, Any]:
    data = {
        "engine": voice.engine,
        "voice": voice.voice,
        "rate": round(voice.rate, 3),
        "pitch": round(voice.pitch, 2),
        "gender": voice.gender,
        "age": voice.age,
    }
    if getattr(voice, "hosted_voice", ""):
        data["hosted_voice"] = voice.hosted_voice
    return data


def character_payload(character) -> Dict[str, Any]:
    return {
        "name": character.name,
        "key": character.key,
        "role": character.role,
        "description": character.description,
        "appearance": getattr(character, "appearance", ""),
        "reference_image": _path(getattr(character, "reference_image", None)),
        "voice": voice_payload(character.voice),
    }


def line_payload(line) -> Dict[str, Any]:
    return {
        "character": line.character,
        "text": line.text,
        "delivery": line.delivery,
        "narration": line.is_narration,
    }


def shot_payload(shot) -> Dict[str, Any]:
    return {
        "index": shot.index,
        "shot_type": shot.shot_type,
        "camera": shot.camera,
        "subjects": list(shot.subjects),
        "action": shot.action,
        "visual": shot.visual,
        "duration_s": round(shot.duration, 2),
        "lines": [line_payload(line) for line in shot.lines],
    }


def scene_payload(scene) -> Dict[str, Any]:
    return {
        "index": scene.index,
        "setting": scene.setting,
        "time_of_day": scene.time_of_day,
        "mood": scene.mood,
        "duration_s": round(scene.duration, 2),
        "shots": [shot_payload(shot) for shot in scene.shots],
    }


def production_payload(production, screenplay: Optional[Path] = None) -> Dict[str, Any]:
    """A whole screenplay, enough to review or edit it without opening the file."""
    return {
        "id": production.id,
        "title": production.title,
        "logline": production.logline,
        "format": production.format,
        "style": production.style,
        "aspect": production.aspect,
        "audience": production.audience,
        "tone": production.tone,
        # Not "status": that is the command's outcome in the envelope, and one
        # key cannot mean two things in the same object.
        "production_status": production.status,
        "created": production.created,
        "target_seconds": production.target_seconds,
        "estimated_duration_s": round(production.estimated_duration, 2),
        "scene_count": len(production.scenes),
        "shot_count": len(production.shots),
        "written_by": production.config.get("written_by") or None,
        "screenplay": _path(screenplay),
        "cast": [character_payload(c) for c in production.cast],
        "scenes": [scene_payload(scene) for scene in production.scenes],
        "renders": list(production.renders),
    }


def production_summary(production, screenplay: Optional[Path] = None) -> Dict[str, Any]:
    """One row for a listing: no scenes, no shots."""
    return {
        "id": production.id,
        "title": production.title,
        "format": production.format,
        "style": production.style,
        "aspect": production.aspect,
        "production_status": production.status,
        "cast_count": len(production.cast),
        "shot_count": len(production.shots),
        "estimated_duration_s": round(production.estimated_duration, 2),
        "render_count": len(production.renders),
        "screenplay": _path(screenplay),
    }


def render_payload(
    production,
    output: Optional[Path],
    elapsed_s: float,
    voiced_lines: int,
    silent_lines: int,
    screenplay: Optional[Path] = None,
    config=None,
    score_backend: str = "",
    speech_engine: str = "",
    visual_backend: str = "",
    subtitle_files: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """The result of a render: where the file is, and whether it has sound."""
    last = production.renders[-1] if production.renders else {}
    data: Dict[str, Any] = {
        "id": production.id,
        "title": production.title,
        "output": _path(output),
        "screenplay": _path(screenplay),
        "duration_s": last.get("duration_s"),
        "elapsed_s": round(elapsed_s, 2),
        "scenes": len(production.scenes),
        "shots": len(production.shots),
        "voiced_lines": voiced_lines,
        "silent_lines": silent_lines,
        "format": production.format,
        "style": production.style,
        "aspect": production.aspect,
        "written_by": production.config.get("written_by") or None,
        "subtitles": [_path(p) for p in (subtitle_files or [])] or last.get("subtitles") or [],
        "audio_cache_hits": last.get("audio_cache_hits", 0),
        "content_hash": last.get("content_hash"),
        "visual_backend": visual_backend or last.get("backend"),
        "score_backend": score_backend or last.get("score_backend"),
        "speech_engine": speech_engine or last.get("speech_engine"),
        "cast": [
            {
                "name": c.name,
                "role": c.role,
                "voice": c.voice.voice,
                "hosted_voice": getattr(c.voice, "hosted_voice", ""),
            }
            for c in production.speaking_cast
        ],
    }
    if config is not None:
        data["resolution"] = list(config.resolution)
        data["fps"] = config.fps
        data["preset"] = config.preset
        data["music"] = bool(config.music)
        data["captions"] = bool(config.captions)
    return data


def doctor_payload(
    config,
    voices: Sequence[Any],
    speech_ok: bool,
    speech_detail: str,
    llm_describe: str,
    llm_probe: Optional[Sequence[Any]],
    visual_rows: List[Dict[str, Any]],
    music_rows: List[Dict[str, Any]],
    provider_rows: List[Dict[str, Any]],
    selected_visual: str,
    selected_music: str,
    accelerator: Optional[Dict[str, Any]] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Everything an agent needs to decide what it can ask for on this machine.

    This is the call an agent should make first. It answers, in one object:
    can this box speak, can it write dialogue, what can it draw, and is any
    paid tier switched on.
    """
    import sys

    from .config import ASPECTS, FORMATS, PRESETS, STYLE_TIERS

    from . import __version__

    usable = [v for v in voices if not getattr(v, "novelty", False)]
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "resolution": list(config.resolution),
        "fps": config.fps,
        "preset": config.preset,
        "voices": {
            "total": len(voices),
            "castable": len(usable),
            "female": sum(1 for v in usable if v.gender == "female"),
            "male": sum(1 for v in usable if v.gender == "male"),
            "accents": sorted({v.accent.upper() for v in usable}),
            "names": [v.name for v in usable],
        },
        "speech": {"ok": speech_ok, "detail": speech_detail},
        "writers_room": {
            "available": llm_describe != "none",
            "detail": llm_describe,
            "probe_ok": (bool(llm_probe[0]) if llm_probe else None),
            "probe_detail": (str(llm_probe[1]) if llm_probe else None),
        },
        "visual_backends": visual_rows,
        "music_backends": music_rows,
        "selected": {"visual": selected_visual, "music": selected_music},
        "accelerator": accelerator or {"available": False},
        "shot_workers": workers,
        "hosted_providers": provider_rows,
        "capabilities": {
            "render": True,
            "speech": speech_ok,
            "write_dialogue": llm_describe != "none",
            "hosted_music": any(r["tier"] == "music" and r["ready"] for r in provider_rows),
            "hosted_speech": any(r["tier"] == "speech" and r["ready"] for r in provider_rows),
            "native_accelerator": bool((accelerator or {}).get("available")),
            "lip_sync": True,
            "subtitles": True,
        },
        "formats": sorted(FORMATS),
        "styles": sorted(STYLE_TIERS),
        "aspects": sorted(ASPECTS),
        "presets": sorted(PRESETS),
        "exit_codes": {str(k): v for k, v in EXIT_MEANINGS.items()},
    }
