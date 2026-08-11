"""Per-character speech synthesis.

Fallback chain per line: neural TTS -> the host synthesiser -> timed silence.
Whatever produces the audio, the *measured* duration is returned, because shot
length is derived from it -- guessing from word count is what made the original
pipeline drift out of sync.
"""

from __future__ import annotations

import logging
import subprocess
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

    def __init__(self, allow_neural: bool = True) -> None:
        self.allow_neural = allow_neural
        self._neural = None
        self._neural_failed = not allow_neural
        self.command = system_tts_command()
        # A film with captions but no audible dialogue is a silent failure --
        # it looks finished and is completely broken. These counters let the
        # renderer notice and say so instead of shipping a mute video.
        self.voiced_lines = 0
        self.silent_lines = 0
        self.failures: List[str] = []

    @property
    def available(self) -> bool:
        """Whether any synthesiser at all can produce audible speech."""
        return self.command is not None or (self.allow_neural and not self._neural_failed)

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
                    "-w", str(raw), text,
                ]
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

    # -- public ------------------------------------------------------------ #

    def speak(self, text: str, voice: VoiceProfile, out_path: Path) -> Tuple[Path, float]:
        """Synthesise one line, returning the file and its measured duration."""
        text = " ".join(str(text).split())
        if not text:
            return self.silence(out_path, MIN_LINE_SECONDS)

        if self.allow_neural:
            result = self._speak_neural(text, voice, out_path)
            if result is not None:
                self.voiced_lines += 1
                return result
        # A stored profile can say "silent" because discovery failed once on
        # some other machine. If a synthesiser exists now, use it anyway rather
        # than honouring a stale decision and muting the character forever.
        if self.command is not None:
            result = self._speak_system(text, voice, out_path)
            if result is not None:
                self.voiced_lines += 1
                return result
            self.failures.append(f"{voice.voice or 'default voice'}: {self.command} produced nothing")
        elif not self.failures:
            self.failures.append("no speech synthesiser found on this system")

        # Silence paced like speech keeps captions readable with no synthesiser.
        self.silent_lines += 1
        words = max(1, len(text.split()))
        return self.silence(out_path, max(MIN_LINE_SECONDS, words / 150.0 * 60.0))

    def silence(self, out_path: Path, duration: float) -> Tuple[Path, float]:
        write_wav(out_path, np.zeros(int(DEFAULT_RATE * duration), dtype=np.float32), DEFAULT_RATE)
        return out_path, duration
