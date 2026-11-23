"""The optional native accelerator, and the pure-Python path it replaces.

Profiling a 43.8 s render put 20 % of it in one function -- the Ken Burns crop
and resize, called once per frame -- and another 4 % in the caption composite.
Those two, plus the lip-sync envelope, are what `video_gen_core` implements in
Rust.

Three rules keep it honest:

* **It is optional.** With the extension absent, every function here falls back
  to Pillow and NumPy and the pipeline behaves identically. `pip install -r
  requirements.txt` stays a pure-Python install with no toolchain.
* **It adds no capability.** Anything only the fast path could do would make the
  fallback a second-class renderer, and then the default install would be the
  broken one.
* **Both paths are compared in the test suite**, not assumed equal. `resize`
  agrees with Pillow to within 1 per channel (Pillow convolves in fixed point,
  this in f64) and `blend` agrees exactly.

Frames travel as ``bytes`` rather than arrays, so a frame can go from resize to
composite without a conversion at each hop; the caller does one
``np.frombuffer`` at the end.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("video_gen.fastpath")

#: Set ``VIDEO_GEN_NO_FASTPATH=1`` to force the pure-Python path. The test suite
#: uses it to exercise the fallback on a machine where the extension is built.
DISABLE_ENV = "VIDEO_GEN_NO_FASTPATH"

_core = None
_reason = "not built; `pip install ./rust` to enable"

if os.environ.get(DISABLE_ENV, "").strip() not in ("", "0", "false", "no"):
    _reason = f"disabled by ${DISABLE_ENV}"
else:
    try:
        import video_gen_core as _core  # type: ignore[no-redef]
    except ImportError as exc:
        _reason = f"not built ({exc.__class__.__name__}); `pip install ./rust` to enable"
    except Exception as exc:  # noqa: BLE001 - a broken build must not stop a render
        _core = None
        _reason = f"unusable: {exc}"
        LOGGER.warning("video_gen_core is installed but failed to load: %s", exc)


def available() -> bool:
    """Whether the native accelerator is in use."""
    return _core is not None


def version() -> str:
    return getattr(_core, "__version__", "") if _core is not None else ""


def threads() -> int:
    """How many threads the accelerator will use, or 1 for the Python path."""
    if _core is None:
        return 1
    try:
        return int(_core.threads())
    except Exception:  # noqa: BLE001
        return 1


def describe() -> str:
    """One line for `doctor`, including *why* it is unavailable."""
    if _core is None:
        return f"pure Python (Pillow + NumPy) — {_reason}"
    return (
        f"native video_gen_core {version()} on {threads()} thread(s) — "
        "~2.9x on frame resize, ~3.7x on caption composite"
    )


# --------------------------------------------------------------------------- #
# Ken Burns: crop a source image and resample it to the frame size
# --------------------------------------------------------------------------- #


def resize_crop(
    src: bytes,
    src_w: int,
    src_h: int,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
) -> bytes:
    """RGB8 in, RGB8 out: the crop box resampled to ``out_w`` x ``out_h``.

    Identical in meaning to ``Image.resize(out, Image.BILINEAR, box=...)``,
    which is what the fallback actually calls.
    """
    if _core is not None:
        return _core.resize_crop(
            src, src_w, src_h, x0, y0, crop_w, crop_h, out_w, out_h
        )
    image = Image.frombuffer("RGB", (src_w, src_h), src, "raw", "RGB", 0, 1)
    window = image.resize(
        (out_w, out_h), Image.BILINEAR, box=(x0, y0, x0 + crop_w, y0 + crop_h)
    )
    return window.tobytes()


# --------------------------------------------------------------------------- #
# Caption composite
# --------------------------------------------------------------------------- #


def blend_box(
    frame: bytes,
    frame_w: int,
    frame_h: int,
    x0: int,
    y0: int,
    box_w: int,
    box_h: int,
    rgba: bytes,
) -> bytes:
    """Alpha-composite an RGBA overlay onto an RGB frame, returning a new frame."""
    if _core is not None:
        return _core.blend_box(frame, frame_w, frame_h, x0, y0, box_w, box_h, rgba)

    buffer = np.frombuffer(frame, dtype=np.uint8).reshape(frame_h, frame_w, 3).copy()
    overlay = np.frombuffer(rgba, dtype=np.uint8).reshape(box_h, box_w, 4)
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    colour = overlay[..., :3].astype(np.float32)
    region = buffer[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float32)
    # astype truncates, and the Rust path truncates to match.
    buffer[y0 : y0 + box_h, x0 : x0 + box_w] = (
        region * (1.0 - alpha) + colour * alpha
    ).astype(np.uint8)
    return buffer.tobytes()


# --------------------------------------------------------------------------- #
# Lip sync envelope
# --------------------------------------------------------------------------- #

DEFAULT_ATTACK = 0.65
DEFAULT_RELEASE = 0.25


def mouth_openness(
    samples: Sequence[float],
    sample_rate: float,
    fps: float,
    frames: int,
    attack: float = DEFAULT_ATTACK,
    release: float = DEFAULT_RELEASE,
) -> List[float]:
    """Per-frame mouth openness in ``[0, 1]`` from a mono waveform.

    A short-window RMS normalised to the loudest window, square-rooted so quiet
    speech still moves the mouth, then given a fast attack and a slower release
    so it snaps open and eases shut. Without that asymmetry it reads as flicker
    rather than as speech.
    """
    if frames <= 0:
        return []
    array = np.asarray(samples, dtype=np.float32)
    if array.size == 0 or sample_rate <= 0 or fps <= 0:
        return [0.0] * frames

    if _core is not None:
        return list(
            _core.mouth_openness(
                array.tolist(), float(sample_rate), float(fps), int(frames),
                float(attack), float(release),
            )
        )

    per_frame = max(1.0, sample_rate / fps)
    raw = np.zeros(frames, dtype=np.float32)
    for index in range(frames):
        start = int(index * per_frame)
        end = min(array.size, int((index + 1) * per_frame))
        if start >= end:
            continue
        window = array[start:end].astype(np.float64)
        raw[index] = float(np.sqrt(np.mean(window * window)))

    peak = float(raw.max()) if raw.size else 0.0
    if peak > 1e-6:
        raw = np.clip(np.sqrt(raw / peak), 0.0, 1.0).astype(np.float32)

    out = np.zeros(frames, dtype=np.float32)
    level = 0.0
    attack = float(np.clip(attack, 0.0, 1.0))
    release = float(np.clip(release, 0.0, 1.0))
    for index in range(frames):
        target = float(raw[index])
        coeff = attack if target > level else release
        level += (target - level) * coeff
        out[index] = min(1.0, max(0.0, level))
    return [float(v) for v in out]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report() -> dict:
    """Status for `--doctor`, in the same shape as a backend row."""
    return {
        "name": "video_gen_core",
        "available": available(),
        "version": version(),
        "threads": threads(),
        "detail": describe(),
    }
