"""The opt-in hosted score: a paid model writes the music.

Off unless a providers config says otherwise, and built so switching it on is
cheap to live with:

* **Cached by request.** The cache key is the prompt, duration, provider and
  seed, so re-rendering an edited screenplay does not pay for the same cue
  twice. This matters more here than for images, because each call costs money
  rather than seconds.
* **Never fatal.** Every failure path -- no key, HTTP error, undecodable body,
  timeout -- raises, and the registry falls back to the procedural bed. A film
  always comes out.
* **No surprise network.** ``available()`` is False unless the tier is enabled
  *and* a key resolves, so a misconfigured provider degrades to offline rather
  than hanging a render on a connection attempt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np

from .. import providers as providers_mod
from ..music import SAMPLE_RATE
from .base import BED, CUE, PROMPT, ScoreRequest, ScoreResult

LOGGER = logging.getLogger("video_gen.score.hosted")

#: A hosted cue longer than this is almost certainly a provider bug, and
#: decoding it would blow memory for no benefit.
MAX_AUDIO_BYTES = 96 * 1024 * 1024

USER_AGENT = "video_gen/score"


def decode_audio(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any container the bundled ffmpeg understands -> mono float32.

    ffmpeg ships with ``imageio-ffmpeg``, which is already required for
    encoding, so this needs no new dependency and no system install.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"no ffmpeg available to decode hosted audio: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "cue.bin"
        source.write_bytes(data)
        argv = [
            exe,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(source),
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(int(sample_rate)),
            "-",
        ]
        done = subprocess.run(argv, capture_output=True, timeout=300)
        if done.returncode != 0 or not done.stdout:
            detail = done.stderr.decode("utf-8", "replace").strip()[:300]
            raise RuntimeError(f"ffmpeg could not decode the hosted audio: {detail or 'empty output'}")
        samples = np.frombuffer(done.stdout, dtype="<f4").astype(np.float32)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples


class HostedMusic:
    """Calls whatever service the providers config names."""

    name = "hosted"
    priority = 50  # above procedural, so it wins when it is switched on

    def __init__(self, config=None, providers=None) -> None:
        self.config = config
        self._providers = providers
        self._reason = ""

    # -- config ------------------------------------------------------------ #

    @property
    def providers(self) -> "providers_mod.ProvidersConfig":
        if self._providers is None:
            self._providers = providers_mod.load()
        return self._providers

    @property
    def spec(self) -> Optional["providers_mod.ProviderSpec"]:
        return self.providers.spec_for(providers_mod.MUSIC)

    def available(self) -> bool:
        spec = self.spec
        if spec is None:
            self._reason = "hosted music is disabled (offline by default)"
            return False
        reason = spec.missing_key_reason()
        if reason:
            self._reason = f"{spec.name}: {reason}"
            return False
        self._reason = ""
        return True

    def capabilities(self) -> Set[str]:
        return {BED, CUE, PROMPT}

    def describe(self) -> str:
        spec = self.spec
        if not self.available():
            return self._reason or "hosted music is disabled (offline by default)"
        assert spec is not None
        note = f" — {spec.notes}" if spec.notes else ""
        return f"{spec.name} via {urllib.parse.urlsplit(spec.endpoint).netloc}{note}"

    # -- cache ------------------------------------------------------------- #

    def _cache_path(self, key: str) -> Optional[Path]:
        root = getattr(self.config, "cache_dir", None)
        if root is None:
            return None
        directory = Path(root) / "music"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.f32"

    @staticmethod
    def _cache_key(spec: "providers_mod.ProviderSpec", request: ScoreRequest, prompt: str) -> str:
        blob = json.dumps(
            {
                "provider": spec.name,
                "endpoint": spec.endpoint,
                "prompt": prompt,
                "duration": round(request.total_duration, 2),
                "seed": request.seed,
                "rate": request.sample_rate,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- http -------------------------------------------------------------- #

    def _request(self, spec: "providers_mod.ProviderSpec", values: dict) -> Tuple[bytes, str]:
        """Perform the call and return ``(audio bytes, how)``."""
        endpoint = providers_mod.fill(spec.endpoint, values)
        query = providers_mod.fill(spec.query, values) if spec.query else {}
        if query:
            joiner = "&" if urllib.parse.urlsplit(endpoint).query else "?"
            endpoint = f"{endpoint}{joiner}{urllib.parse.urlencode(query)}"

        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        headers.update({str(k): str(providers_mod.fill(v, values)) for k, v in spec.headers.items()})
        key = spec.key()
        if key and spec.auth_header:
            headers[spec.auth_header] = f"{spec.auth_prefix}{key}"

        body = providers_mod.fill(spec.body, values) if spec.body else None
        payload = json.dumps(body).encode("utf-8") if body is not None else None

        http = urllib.request.Request(  # noqa: S310 - the endpoint is the user's own config
            endpoint,
            data=payload,
            headers=headers,
            method=(spec.method or "POST").upper(),
        )
        try:
            with urllib.request.urlopen(http, timeout=spec.timeout) as response:  # noqa: S310
                raw = response.read(MAX_AUDIO_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", "replace").strip()
            raise RuntimeError(f"{spec.name} returned HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{spec.name} unreachable: {exc.reason}") from exc

        if len(raw) > MAX_AUDIO_BYTES:
            raise RuntimeError(f"{spec.name} returned more than {MAX_AUDIO_BYTES} bytes")

        decoded: object = raw
        if "json" in content_type.lower() or spec.response.startswith(("json:", "base64:")):
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{spec.name} did not return readable JSON: {exc}") from exc

        audio, url = providers_mod.extract(decoded, spec.response)
        if audio:
            return audio, f"{spec.name} inline"
        if url:
            fetch = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
            try:
                with urllib.request.urlopen(fetch, timeout=spec.timeout) as response:  # noqa: S310
                    audio = response.read(MAX_AUDIO_BYTES + 1)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{spec.name}: could not fetch {url}: {exc}") from exc
            if len(audio) > MAX_AUDIO_BYTES:
                raise RuntimeError(f"{spec.name} audio exceeded {MAX_AUDIO_BYTES} bytes")
            return audio, f"{spec.name} url"

        preview = raw[:200].decode("utf-8", "replace") if isinstance(raw, bytes) else ""
        raise RuntimeError(
            f"{spec.name}: no audio found at response selector {spec.response!r} ({preview})"
        )

    # -- render ------------------------------------------------------------ #

    def render(self, request: ScoreRequest) -> ScoreResult:
        spec = self.spec
        if spec is None:
            raise RuntimeError("hosted music is disabled")
        reason = spec.missing_key_reason()
        if reason:
            raise RuntimeError(f"{spec.name}: {reason}")

        rate = request.sample_rate or SAMPLE_RATE
        prompt = request.prompt_for()
        cache = self._cache_path(self._cache_key(spec, request, prompt))

        if cache is not None and cache.is_file():
            try:
                cached = np.fromfile(cache, dtype="<f4").astype(np.float32)
                if cached.size:
                    return ScoreResult(
                        backend=self.name,
                        samples=cached,
                        sample_rate=rate,
                        detail=f"{spec.name} (cached)",
                        used={BED, CUE, PROMPT},
                    )
            except Exception as exc:  # noqa: BLE001 - a bad cache entry just means a refetch
                LOGGER.debug("ignoring unreadable music cache %s: %s", cache, exc)

        values = {
            "prompt": prompt,
            "mood": ", ".join(request.moods) or "neutral",
            "duration_s": max(1, int(round(request.total_duration))),
            "duration_ms": max(1000, int(round(request.total_duration * 1000))),
            "seed": request.seed,
            "style": request.style,
            "format": request.format,
            "title": request.title,
            "tool": spec.tool or spec.name,
            "sample_rate": rate,
        }
        audio, how = self._request(spec, values)
        samples = decode_audio(audio, rate)
        if not samples.size:
            raise RuntimeError(f"{spec.name} returned audio that decoded to nothing")

        if cache is not None:
            try:
                samples.astype("<f4").tofile(cache)
            except Exception as exc:  # noqa: BLE001 - caching is an optimisation
                LOGGER.debug("could not cache hosted music: %s", exc)

        return ScoreResult(
            backend=self.name,
            samples=samples,
            sample_rate=rate,
            detail=f"{how}, {samples.size / rate:.1f}s",
            used={BED, CUE, PROMPT},
        )
