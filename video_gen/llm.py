"""A local language model, if this machine happens to have one.

Writing dialogue is the one part of screenwriting that cannot be done with
regular expressions. Everything else in this package is deterministic; this
module is the single place where a model gets a say, and it is deliberately
built so that the model can be absent, slow, or wrong without the pipeline
noticing.

Three rules hold everywhere below:

1. **stdlib only.** Ollama speaks HTTP, so `urllib` is enough. Adding `requests`
   or an SDK for this would put a dependency in the critical path of a feature
   that is optional by design.
2. **Offline means offline.** Ollama can proxy cloud models; those are filtered
   out of discovery. A film that silently needed the internet would break the
   one promise this project actually makes.
3. **Structured output, not prose parsing.** Small models emit malformed JSON
   often enough that scraping their prose is a losing game -- measured here, a
   1B model failed on a nested request in plain JSON mode but succeeded every
   time against a schema, because the schema constrains decoding rather than
   merely asking nicely.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

LOGGER = logging.getLogger("video_gen.llm")

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 120.0

# This host is a CPU. A 30B model answers, eventually, but a six-line exchange
# would take minutes and the interview would feel broken. Anything above this
# is only used when the user names it explicitly.
AUTO_MODEL_MAX_BYTES = 6 * 1024**3

# Ranked by how well they follow a "reply with JSON in this shape" instruction
# at small sizes. Unlisted models still work; they just sort after these.
_PREFERRED = ("llama3.2", "qwen2.5", "qwen3", "gemma3", "gemma2", "phi4", "phi3", "mistral")


class LLMError(RuntimeError):
    """The model was asked something and did not answer usefully."""


@dataclass
class LLMReply:
    """A completion plus what it cost, so callers can report honestly."""

    text: str
    model: str
    seconds: float = 0.0
    tokens: int = 0

    def json(self) -> Any:
        """Parse the reply as JSON, raising `LLMError` rather than returning junk."""
        try:
            return json.loads(self.text)
        except (ValueError, TypeError) as exc:
            raise LLMError(f"model did not return JSON: {exc}") from exc


class LLMClient(Protocol):
    """What the writers' room needs from a model. Deliberately tiny."""

    name: str
    model: str

    def available(self) -> bool: ...

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> LLMReply: ...


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def _http_json(url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - localhost
        return json.loads(response.read().decode("utf-8"))


@dataclass
class OllamaModel:
    name: str
    size: int = 0

    @property
    def is_cloud(self) -> bool:
        """Cloud-proxied models need the network, so they are not offline."""
        return self.name.endswith(":cloud") or self.size == 0

    @property
    def rank(self) -> tuple:
        """Sort key: known-good families first, then smallest (== fastest here)."""
        family = next((i for i, p in enumerate(_PREFERRED) if self.name.startswith(p)), len(_PREFERRED))
        return (family, self.size)


class OllamaClient:
    """Talks to a local `ollama serve`. The only LLM path verified on this machine."""

    name = "ollama"

    def __init__(
        self,
        model: str = "",
        host: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if "://" not in self.host:  # OLLAMA_HOST is often bare "127.0.0.1:11434"
            self.host = f"http://{self.host}"
        self.timeout = timeout
        self.model = model or os.environ.get("VIDEO_GEN_LLM_MODEL", "")
        self._explicit = bool(self.model)
        self._models: Optional[List[OllamaModel]] = None
        self.detail = ""

    # -- discovery --------------------------------------------------------- #

    def models(self) -> List[OllamaModel]:
        if self._models is not None:
            return self._models
        try:
            data = _http_json(f"{self.host}/api/tags", None, timeout=4.0)
        except Exception as exc:  # noqa: BLE001 - any failure means "no server"
            self.detail = f"no server at {self.host} ({type(exc).__name__})"
            self._models = []
            return self._models
        self._models = [
            OllamaModel(name=str(m.get("name", "")), size=int(m.get("size", 0) or 0))
            for m in data.get("models", [])
            if m.get("name")
        ]
        return self._models

    def resolve_model(self) -> str:
        """Pick a model: the named one, else the best small local one."""
        if self.model:
            return self.model
        local = [m for m in self.models() if not m.is_cloud]
        if not local:
            if self.models():
                self.detail = "only cloud models installed; that is not offline"
            return ""
        usable = [m for m in local if m.size <= AUTO_MODEL_MAX_BYTES] or local
        chosen = sorted(usable, key=lambda m: m.rank)[0]
        if chosen.size > AUTO_MODEL_MAX_BYTES:
            self.detail = f"{chosen.name} is large for a CPU host; expect slow drafts"
        self.model = chosen.name
        return self.model

    def available(self) -> bool:
        if self._explicit:
            # A named model still has to exist, or every call fails at generate
            # time with an unhelpful 404 halfway through an interview.
            names = {m.name for m in self.models()}
            if names and self.model not in names:
                self.detail = f"model {self.model!r} is not installed ({len(names)} available)"
                return False
            return bool(names) or not self.models()
        return bool(self.resolve_model())

    # -- generation -------------------------------------------------------- #

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> LLMReply:
        model = self.resolve_model()
        if not model:
            raise LLMError(self.detail or "no local ollama model")
        payload: Dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": float(temperature),
                # A fixed seed makes the same prompt produce the same film, which
                # matters because every other stage of this pipeline is already
                # deterministic and a re-render should not rewrite the script.
                "seed": int(seed) & 0x7FFFFFFF,
                "num_predict": int(max_tokens),
            },
        }
        if schema is not None:
            payload["format"] = schema
        started = time.time()
        try:
            data = _http_json(f"{self.host}/api/chat", payload, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # pragma: no cover - server-dependent
            raise LLMError(f"ollama returned {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"ollama unreachable: {exc}") from exc
        text = str((data.get("message") or {}).get("content", "")).strip()
        if not text:
            raise LLMError("ollama returned an empty message")
        return LLMReply(
            text=text,
            model=model,
            seconds=time.time() - started,
            tokens=int(data.get("eval_count", 0) or 0),
        )


# --------------------------------------------------------------------------- #
# transformers (present for completeness; cannot run on this machine)
# --------------------------------------------------------------------------- #


class TransformersClient:
    """A local HF pipeline.

    ⚠️ **Never executed on the development machine** -- torch publishes no macOS
    x86_64 wheel past Python 3.12. It is kept because a machine that *can* run
    diffusion can run this too, and because it is the fallback when Ollama is not
    installed. Treat it as unverified.
    """

    name = "transformers"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model = model
        self.detail = ""
        self._pipe = None
        self._failed = False

    def _load(self):  # pragma: no cover - needs torch
        if self._pipe is not None or self._failed:
            return self._pipe
        try:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation", model=self.model, torch_dtype=torch.float32, device=-1
            )
        except Exception as exc:  # noqa: BLE001
            self.detail = f"{type(exc).__name__}: {exc}"
            self._failed = True
        return self._pipe

    def available(self) -> bool:
        try:
            import importlib.util

            for module in ("torch", "transformers"):
                if importlib.util.find_spec(module) is None:
                    self.detail = f"{module} is not installed"
                    return False
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            self.detail = str(exc)
            return False
        return True

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> LLMReply:  # pragma: no cover - needs torch
        pipe = self._load()
        if pipe is None:
            raise LLMError(self.detail or "transformers pipeline unavailable")
        # No grammar constraint here, so the schema can only be *requested*. The
        # caller validates anyway, which is why that is survivable.
        instruction = user
        if schema is not None:
            instruction = f"{user}\n\nReply with JSON matching this schema:\n{json.dumps(schema)}"
        started = time.time()
        out = pipe(
            [{"role": "system", "content": system}, {"role": "user", "content": instruction}],
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(0.01, temperature),
            return_full_text=False,
        )
        text = str(out[0]["generated_text"]).strip()
        if schema is not None:
            text = extract_json(text) or text
        return LLMReply(text=text, model=self.model, seconds=time.time() - started)


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #


@dataclass
class MockLLM:
    """A scripted model, so the writers' room is tested without one installed.

    Same role as `MockBackend` in `visuals/`: the point is to exercise the
    orchestration -- prompting, validation, rejection, fallback -- on a machine
    where the real thing may be absent.
    """

    replies: List[str] = field(default_factory=list)
    name: str = "mock"
    model: str = "mock"
    is_available: bool = True
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def available(self) -> bool:
        return self.is_available

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> LLMReply:
        self.calls.append({"system": system, "user": user, "schema": schema, "seed": seed})
        if not self.replies:
            raise LLMError("mock ran out of scripted replies")
        return LLMReply(text=self.replies.pop(0), model=self.model, seconds=0.0)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

_JSON_SPAN_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull the JSON out of a reply that wrapped it in prose or a code fence."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    span = _JSON_SPAN_RE.search(text)
    return span.group(0) if span else ""


def discover_llm(model: str = "", host: str = "", timeout: float = DEFAULT_TIMEOUT):
    """The best local model this machine can actually run, or None.

    Ollama first: it is a native binary with no Python wheel to satisfy, so it
    runs on hosts where `torch` cannot be installed at all -- which is exactly
    the case on the development machine.
    """
    ollama = OllamaClient(model=model, host=host, timeout=timeout)
    if ollama.available():
        return ollama
    hf = TransformersClient(model) if model and "/" in model else TransformersClient()
    if hf.available():
        return hf
    LOGGER.info("no local LLM (%s)", ollama.detail or "ollama not running")
    return None


def probe(client: Optional[object] = None) -> tuple:
    """Actually generate something and check it came back in the right shape.

    A reachable server is not the same as a working writers' room: the model can
    be installed but incapable of holding a JSON schema. Only a real round trip
    tells the truth, which is the same reason `SpeechSynthesizer.probe` exists.
    """
    client = client or discover_llm()
    if client is None:
        return False, "no local LLM found (install ollama, or requirements-ml.txt)"
    schema = {
        "type": "object",
        "properties": {"line": {"type": "string"}},
        "required": ["line"],
    }
    try:
        reply = client.complete(
            "You write film dialogue. Reply with JSON only.",
            'Write one short line of dialogue for a barista greeting a customer. Shape: {"line": "..."}',
            schema=schema,
            seed=7,
            temperature=0.4,
            max_tokens=64,
        )
        payload = reply.json()
    except LLMError as exc:
        return False, f"{getattr(client, 'name', '?')}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{getattr(client, 'name', '?')}: {type(exc).__name__}: {exc}"
    line = str(payload.get("line", "")) if isinstance(payload, dict) else ""
    if not line.strip():
        return False, f"{client.name}/{client.model}: valid JSON but no dialogue in it"
    return True, f"{client.name}/{client.model} wrote {len(line.split())} words in {reply.seconds:.1f}s"


def describe(client: Optional[object] = None) -> str:
    """One line for `--doctor`, without generating anything."""
    client = client if client is not None else discover_llm()
    if client is None:
        return "none — dialogue will be deterministic only"
    extra = getattr(client, "detail", "")
    return f"{client.name}/{client.model}" + (f" ({extra})" if extra else "")


def cast_names_hint(names: Sequence[str]) -> str:
    """Render a cast list the way a prompt reads best."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"
