"""Synthesised score, and the dialogue mix.

The bed is generated rather than sampled: no model, no download, no licensing.
Mood selects the mode and tempo, so a tense scene and a warm one do not share a
cue. Ducking is what actually makes dialogue intelligible -- a fixed low volume
either buries speech or leaves the music inaudible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 44100

#: Mood -> (scale degrees as frequency ratios, seconds per bar, brightness).
#: Minor and unresolved for tension; major and slower for warmth.
MOODS: Dict[str, Dict[str, object]] = {
    "neutral": {"chords": [(1.0, 1.2, 1.5), (0.89, 1.06, 1.33)], "bar": 4.0, "bright": 0.5},
    "tense": {"chords": [(1.0, 1.19, 1.41), (0.94, 1.12, 1.33)], "bar": 2.4, "bright": 0.8},
    "warm": {"chords": [(1.0, 1.25, 1.5), (0.84, 1.05, 1.26)], "bar": 5.0, "bright": 0.35},
    "sad": {"chords": [(1.0, 1.2, 1.5), (0.75, 0.9, 1.12)], "bar": 5.5, "bright": 0.25},
    "hopeful": {"chords": [(1.0, 1.26, 1.5), (1.12, 1.41, 1.68)], "bar": 4.5, "bright": 0.6},
    "urgent": {"chords": [(1.0, 1.19, 1.5), (1.06, 1.26, 1.59)], "bar": 2.0, "bright": 0.9},
    "playful": {"chords": [(1.0, 1.26, 1.5), (1.33, 1.68, 2.0)], "bar": 3.0, "bright": 0.7},
}

DEFAULT_MOOD = "neutral"

#: Roots low enough to sit under speech without masking it.
_ROOTS = (98.0, 110.0, 123.47, 130.81, 146.83)


def _adsr(length: int, attack: float, release: float, rate: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / rate
    total = length / rate
    envelope = np.minimum(t / max(attack, 1e-3), np.minimum(1.0, (total - t) / max(release, 1e-3)))
    return np.clip(envelope, 0.0, 1.0)


def render_bed(
    duration: float,
    mood: str = DEFAULT_MOOD,
    seed: int = 0,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """A mono pad for one scene's mood."""
    spec = MOODS.get(mood, MOODS[DEFAULT_MOOD])
    rng = np.random.default_rng(seed)
    root = float(_ROOTS[seed % len(_ROOTS)])
    bar = float(spec["bar"])  # type: ignore[arg-type]
    bright = float(spec["bright"])  # type: ignore[arg-type]
    chords: Sequence[Tuple[float, ...]] = spec["chords"]  # type: ignore[assignment]

    total = max(1, int(rate * duration))
    out = np.zeros(total, dtype=np.float32)
    bar_len = max(1, int(rate * bar))
    t_bar = np.arange(bar_len, dtype=np.float32) / rate
    envelope = _adsr(bar_len, attack=bar * 0.22, release=bar * 0.3, rate=rate)

    cursor = 0
    index = 0
    while cursor < total:
        wave = np.zeros(bar_len, dtype=np.float32)
        for harmonic, ratio in enumerate(chords[index % len(chords)]):
            freq = root * ratio
            wave += np.sin(2 * np.pi * freq * t_bar) * (0.55**harmonic)
            # Brighter moods get more upper harmonic; darker ones stay muted.
            wave += bright * 0.3 * np.sin(4 * np.pi * freq * t_bar) * (0.55**harmonic)
        wave *= envelope
        chunk = wave[: total - cursor]
        out[cursor : cursor + len(chunk)] += chunk
        cursor += len(chunk)
        index += 1

    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out /= peak
    # A touch of noise keeps the pad from sounding like a test tone.
    out += rng.normal(0, 0.004, total).astype(np.float32)
    return out.astype(np.float32)


def score_timeline(
    segments: Sequence[Tuple[float, float, str]],
    total_duration: float,
    seed: int = 0,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build one continuous bed from per-scene ``(start, end, mood)`` segments.

    Adjacent cues are crossfaded so a mood change is a transition rather than a
    cut, which is what makes a multi-scene film sound scored rather than
    assembled.
    """
    total = max(1, int(rate * total_duration))
    out = np.zeros(total, dtype=np.float32)
    if not segments:
        return out

    blend = int(rate * 1.0)
    for index, (start, end, mood) in enumerate(segments):
        length = max(0.0, end - start)
        if length <= 0.01:
            continue
        # Overlap into the previous cue so the crossfade has material.
        pad = blend if index else 0
        begin = max(0, int(rate * start) - pad)
        piece = render_bed(length + pad / rate, mood=mood, seed=seed + index, rate=rate)
        if pad:
            piece[:pad] *= np.linspace(0.0, 1.0, pad, dtype=np.float32)
            out[begin : begin + pad] *= np.linspace(1.0, 0.0, pad, dtype=np.float32)
        end_index = min(total, begin + len(piece))
        out[begin:end_index] += piece[: end_index - begin]

    fade = int(rate * 1.5)
    if total > 2 * fade:
        out[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return out


def duck(
    bed: np.ndarray,
    speech_windows: Sequence[Tuple[float, float]],
    rate: int = SAMPLE_RATE,
    depth_db: float = -11.0,
    attack: float = 0.25,
    release: float = 0.6,
) -> np.ndarray:
    """Drop the bed under dialogue.

    A gain envelope is built at 1.0 everywhere and ``depth_db`` across each
    speech window, then smoothed so the level glides instead of stepping. This
    is why speech stays intelligible without the music being inaudible between
    lines.
    """
    if bed.size == 0 or not speech_windows:
        return bed

    gain = np.ones(len(bed), dtype=np.float32)
    floor = float(10.0 ** (depth_db / 20.0))
    for start, end in speech_windows:
        a = max(0, int(start * rate))
        b = min(len(bed), int(end * rate))
        if b > a:
            gain[a:b] = floor

    # Box-smooth the envelope; the window is the mean of attack and release, so
    # one cheap pass approximates a proper attack/release curve closely enough.
    window = max(1, int(rate * (attack + release) / 2))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.concatenate([np.ones(window, dtype=np.float32), gain, np.ones(window, dtype=np.float32)])
        gain = np.convolve(padded, kernel, mode="same")[window : window + len(bed)]
    return (bed * gain).astype(np.float32)


def to_stereo(mono: np.ndarray, volume: float = 1.0) -> np.ndarray:
    """Shape (n, 2) float32, which is what AudioArrayClip expects."""
    scaled = np.clip(mono * volume, -1.0, 1.0).astype(np.float32)
    return np.stack([scaled, scaled], axis=1)


def mood_for(text: str, fallback: str = DEFAULT_MOOD) -> str:
    """Pick a mood from a scene's own words."""
    lowered = f" {text.lower()} "
    table = {
        "tense": ("tense", "danger", "threat", "fear", "afraid", "dark", "alarm", "warning"),
        "urgent": ("urgent", "run", "chase", "hurry", "escape", "race", "emergency", "fight"),
        "sad": ("sad", "grief", "loss", "lonely", "alone", "mourn", "cry", "gone", "empty"),
        "warm": ("warm", "home", "family", "gentle", "comfort", "love", "quiet", "kind"),
        "hopeful": ("hope", "dawn", "begin", "new", "future", "rise", "discover", "solve"),
        "playful": ("play", "funny", "joke", "laugh", "game", "silly", "fun", "kids"),
    }
    best, best_hits = fallback, 0
    for mood, words in table.items():
        hits = sum(1 for word in words if f" {word}" in lowered)
        if hits > best_hits:
            best, best_hits = mood, hits
    return best
