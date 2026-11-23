"""Per-character speech synthesis.

Fallback chain per line: neural TTS -> the host synthesiser -> timed silence.
Whatever produces the audio, the *measured* duration is returned, because shot
length is derived from it -- guessing from word count is what made the original
pipeline drift out of sync.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .casting import system_tts_command
from .model import VoiceProfile

LOGGER = logging.getLogger("video_gen.speech")

try:  # pragma: no cover - optional
    import soundfile as sf

    HAS_SOUNDFILE = True
except Exception:  # noqa: BLE001
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False

DEFAULT_RATE = 22050
MIN_LINE_SECONDS = 0.35


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 [-1, 1] audio, preferring soundfile, else stdlib."""
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_SOUNDFILE:
        sf.write(str(path), samples, sample_rate)
        return
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767.0).astype("<i2").tobytes())


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a wav back as mono float32."""
    if HAS_SOUNDFILE:
        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(rate)
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        channels = handle.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    peak = float(np.iinfo(dtype).max)
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / peak
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def normalize(samples: np.ndarray, peak: float = 0.89) -> np.ndarray:
    if samples.size == 0:
        return samples
    current = float(np.max(np.abs(samples)))
    if current < 1e-6:
        return samples
    return (samples * (peak / current)).astype(np.float32)


def apply_rate(samples: np.ndarray, rate: float) -> np.ndarray:
    """Resampling speed change (pitch shifts slightly, as expected)."""
    if abs(rate - 1.0) < 1e-3 or samples.size == 0:
        return samples
    target_len = max(1, int(len(samples) / rate))
    source = np.linspace(0, len(samples) - 1, num=target_len)
    return np.interp(source, np.arange(len(samples)), samples).astype(np.float32)


def apply_pitch(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch by resampling.

    Duration changes as a side effect, and that is deliberate: shot length comes
    from the *measured* audio duration after synthesis, so nothing downstream
    depends on the pre-shift length. Resampling is exact and artefact-free,
    unlike the phase-vocoder tricks used to hold duration constant -- and since
    speaking rate is already controlled properly by the synthesiser's own rate
    flag, there is nothing to gain from preserving it here.
    """
    if abs(semitones) < 0.05 or samples.size < 32:
        return samples
    return apply_rate(samples, float(2.0 ** (semitones / 12.0)))


class SpeechSynthesizer:
    """Renders a line of dialogue in a specific character's voice."""

    def __init__(self, allow_neural: bool = True, config=None) -> None:
        self.allow_neural = allow_neural
        self._neural = None
        self._neural_failed = not allow_neural
        self.command = system_tts_command()
        self.config = config
        # A film with captions but no audible dialogue is a silent failure --
        # it looks finished and is completely broken. These counters let the
        # renderer notice and say so instead of shipping a mute video.
        self.voiced_lines = 0
        self.silent_lines = 0
        self.failures: List[str] = []
        #: Which engine actually spoke, recorded into the render log.
        self.engine_name = ""
        self._hosted = None
        self._hosted_failed = False
        # Shots can render in parallel, so the counters and the failure list are
        # shared mutable state. A film reporting the wrong number of silent
        # lines is worse than useless: that count is what exit code 2 rests on.
        self._lock = threading.Lock()
        self.cache_hits = 0

    # -- cache ------------------------------------------------------------- #

    @property
    def cache_dir(self) -> Optional[Path]:
        """Where synthesised lines are kept between renders, if anywhere.

        Re-rendering an edited screenplay re-speaks every unchanged line
        otherwise, and TTS was 25% of a measured render -- the single largest
        cost after encoding.
        """
        root = getattr(self.config, "cache_dir", None)
        if root is None:
            return None
        try:
            directory = Path(root) / "audio"
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        except OSError as exc:  # noqa: PERF203 - an unwritable cache is not fatal
            LOGGER.debug("no audio cache: %s", exc)
            return None

    @property
    def engine_tag(self) -> str:
        """A stable name for the engine this synthesiser prefers.

        Derived from configuration only, never from what has already been
        tried. `_neural_failed` flips the first time a neural attempt fails, so
        including it meant a line was looked up under one key and stored under
        another -- a cache that could never hit on a later run.

        A line cached when neural TTS worked and read back when it does not is
        still correct audio for that text and voice, which is why the preferred
        engine is a good enough tag and the realised one is not needed.
        """
        if self.hosted is not None:
            return f"hosted:{self.hosted.name}"
        if self.allow_neural:
            return "neural"
        return f"system:{self.command or 'none'}"

    def cache_key(self, text: str, voice: VoiceProfile) -> str:
        """Identity of a spoken line: the text and everything shaping it."""
        blob = "|".join(
            [
                " ".join(str(text).split()),
                voice.engine,
                voice.voice,
                f"{voice.rate:.4f}",
                f"{voice.pitch:.4f}",
                voice.hosted_voice,
                self.command or "",
                self.engine_tag,
            ]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]

    # -- hosted (opt-in, paid) --------------------------------------------- #

    @property
    def hosted(self):
        """The hosted TTS provider, or None when the tier is off.

        Resolved lazily and cached, including the negative case, so a run with
        no providers config never reads one and never retries a dead provider
        once per line.
        """
        if self._hosted_failed:
            return None
        if self._hosted is None:
            providers = getattr(self.config, "providers", None)
            if providers is None:
                self._hosted_failed = True
                return None
            from .providers import SPEECH

            spec = providers.spec_for(SPEECH)
            if spec is None or spec.missing_key_reason():
                self._hosted_failed = True
                return None
            self._hosted = spec
        return self._hosted

    @property
    def available(self) -> bool:
        """Whether any synthesiser at all can produce audible speech."""
        return (
            self.command is not None
            or (self.allow_neural and not self._neural_failed)
            or self.hosted is not None
        )

    def probe(self, out_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Actually synthesise a phrase and confirm it is audible.

        Checking that `say` is on PATH is not the same as checking that it
        produces sound, and the difference is the whole bug this guards.
        """
        import tempfile

        target = out_path or Path(tempfile.mkdtemp()) / "probe.wav"
        try:
            path, duration = self.speak("Testing one two three.", VoiceProfile(), target)
            samples, _ = read_wav(path)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if peak < 1e-4:
                return False, f"synthesiser produced {duration:.1f}s of silence"
            return True, f"{duration:.1f}s at peak {peak:.2f} via {self.command or 'neural'}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    # -- neural (only where torch exists) ---------------------------------- #

    def _load_neural(self):  # pragma: no cover - requires model download
        if self._neural is not None or self._neural_failed:
            return self._neural
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, VitsModel

            model = VitsModel.from_pretrained("facebook/mms-tts-eng").to("cpu").eval()
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self._neural = (model, tokenizer)
            LOGGER.info("loaded neural TTS facebook/mms-tts-eng")
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("neural TTS unavailable (%s); using system voices", exc)
            self._neural_failed = True
            self._neural = None
        return self._neural

    def _speak_neural(self, text: str, voice: VoiceProfile, out_path: Path):
        # pragma: no cover - requires model download
        pair = self._load_neural()
        if pair is None:
            return None
        try:
            import torch

            model, tokenizer = pair
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                waveform = model(**inputs).waveform
            samples = waveform.squeeze().cpu().numpy().astype(np.float32)
            rate = int(model.config.sampling_rate)
            samples = apply_pitch(apply_rate(samples, voice.rate), voice.pitch)
            write_wav(out_path, normalize(samples), rate)
            return out_path, len(samples) / rate
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("neural TTS failed (%s); using system voice", exc)
            return None

    # -- system ------------------------------------------------------------ #

    def _speak_system(self, text: str, voice: VoiceProfile, out_path: Path):
        if self.command is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = out_path.with_suffix(".raw.wav")
        try:
            if self.command == "say":
                argv = ["say", "-r", str(int(180 * voice.rate))]
                if voice.voice:
                    argv += ["-v", voice.voice]
                argv += ["--data-format=LEI16@22050", "-o", str(raw), text]
            else:
                argv = [
                    self.command,
                    "-s", str(int(165 * voice.rate)),
                    "-p", str(int(np.clip(50 + voice.pitch * 6, 0, 99))),
                ]
                # Without this, every character on Linux speaks in espeak's
                # default voice no matter what casting decided.
                if voice.voice:
                    argv += ["-v", voice.voice]
                argv += ["-w", str(raw), text]
            subprocess.run(argv, check=True, capture_output=True, timeout=180)
            samples, rate = read_wav(raw)
            if samples.size < 8:
                return None
            # `say` has no pitch flag, so shaping happens here. espeak already
            # applied -p, so it only needs level matching.
            if self.command == "say":
                samples = apply_pitch(samples, voice.pitch)
            write_wav(out_path, normalize(samples), rate)
            return out_path, len(samples) / rate
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("system TTS (%s) failed: %s", self.command, exc)
            return None
        finally:
            raw.unlink(missing_ok=True)

    # -- hosted ------------------------------------------------------------ #

    def _speak_hosted(self, text: str, voice: VoiceProfile, out_path: Path):
        """A paid provider speaks the line. Off unless a config says otherwise."""
        spec = self.hosted
        if spec is None:
            return None

        from . import providers as providers_mod
        from .score.hosted import decode_audio

        # A character carries its own hosted voice id when casting assigned
        # one; otherwise the provider's first listed voice keeps output
        # deterministic rather than random per line.
        voice_id = voice.hosted_voice or (spec.voices[0] if spec.voices else "")
        values = {
            "text": text,
            "voice": voice_id,
            "rate": round(voice.rate, 3),
            "pitch": round(voice.pitch, 2),
            "gender": voice.gender or "",
        }
        try:
            from .score.hosted import HostedMusic

            audio, _ = HostedMusic(self.config)._request(spec, values)  # noqa: SLF001
            samples = decode_audio(audio, DEFAULT_RATE)
            if samples.size < 8:
                raise RuntimeError("decoded to nothing")
            write_wav(out_path, normalize(samples), DEFAULT_RATE)
            self.engine_name = f"hosted:{spec.name}"
            return out_path, len(samples) / DEFAULT_RATE
        except Exception as exc:  # noqa: BLE001 - fall through to a local engine
            LOGGER.warning("hosted TTS (%s) failed: %s", spec.name, exc)
            self.failures.append(f"hosted {spec.name}: {exc}")
            # One failure disables the tier for the rest of the render: paying
            # for a provider that is down, once per line, is worse than
            # finishing the film with system voices.
            with self._lock:
                self._hosted_failed = True
            return None

    # -- public ------------------------------------------------------------ #

    def speak(self, text: str, voice: VoiceProfile, out_path: Path) -> Tuple[Path, float]:
        """Synthesise one line, returning the file and its measured duration."""
        text = " ".join(str(text).split())
        if not text:
            return self.silence(out_path, MIN_LINE_SECONDS)

        cached = self._from_cache(text, voice, out_path)
        if cached is not None:
            return cached

        result = self._synthesise(text, voice, out_path)
        self._to_cache(text, voice, result[0])
        return result

    # -- cache plumbing ---------------------------------------------------- #

    def _from_cache(
        self, text: str, voice: VoiceProfile, out_path: Path
    ) -> Optional[Tuple[Path, float]]:
        directory = self.cache_dir
        if directory is None:
            return None
        entry = directory / f"{self.cache_key(text, voice)}.wav"
        if not entry.is_file():
            return None
        try:
            samples, rate = read_wav(entry)
            if samples.size < 8 or rate <= 0:
                raise ValueError("cached line is empty")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, out_path)
        except Exception as exc:  # noqa: BLE001 - a bad entry just means re-speaking
            LOGGER.debug("ignoring unusable cached line %s: %s", entry, exc)
            entry.unlink(missing_ok=True)
            return None
        with self._lock:
            self.voiced_lines += 1
            self.cache_hits += 1
            self.engine_name = self.engine_name or "cache"
        return out_path, samples.size / rate

    def _to_cache(self, text: str, voice: VoiceProfile, produced: Path) -> None:
        directory = self.cache_dir
        if directory is None or not Path(produced).is_file():
            return
        entry = directory / f"{self.cache_key(text, voice)}.wav"
        if entry.exists():
            return
        try:
            # Written beside, then renamed: two shots speaking in parallel must
            # never leave a half-written entry for the next render to read.
            scratch = entry.with_suffix(f".{threading.get_ident():x}.part")
            shutil.copyfile(produced, scratch)
            scratch.replace(entry)
        except Exception as exc:  # noqa: BLE001 - caching is an optimisation
            LOGGER.debug("could not cache a spoken line: %s", exc)

    def _synthesise(self, text: str, voice: VoiceProfile, out_path: Path) -> Tuple[Path, float]:
        """The fallback chain, with no cache in the way."""
        if self.hosted is not None:
            result = self._speak_hosted(text, voice, out_path)
            if result is not None:
                with self._lock:
                    self.voiced_lines += 1
                return result

        if self.allow_neural:
            result = self._speak_neural(text, voice, out_path)
            if result is not None:
                with self._lock:
                    self.voiced_lines += 1
                    self.engine_name = self.engine_name or "neural"
                return result
        # A stored profile can say "silent" because discovery failed once on
        # some other machine. If a synthesiser exists now, use it anyway rather
        # than honouring a stale decision and muting the character forever.
        if self.command is not None:
            result = self._speak_system(text, voice, out_path)
            if result is not None:
                with self._lock:
                    self.voiced_lines += 1
                    self.engine_name = self.engine_name or self.command
                return result
            with self._lock:
                self.failures.append(
                    f"{voice.voice or 'default voice'}: {self.command} produced nothing"
                )
        else:
            with self._lock:
                if not self.failures:
                    self.failures.append("no speech synthesiser found on this system")

        # Silence paced like speech keeps captions readable with no synthesiser.
        with self._lock:
            self.silent_lines += 1
        words = max(1, len(text.split()))
        return self.silence(out_path, max(MIN_LINE_SECONDS, words / 150.0 * 60.0))

    def silence(self, out_path: Path, duration: float) -> Tuple[Path, float]:
        write_wav(out_path, np.zeros(int(DEFAULT_RATE * duration), dtype=np.float32), DEFAULT_RATE)
        return out_path, duration
