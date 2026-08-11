#!/usr/bin/env python3
"""Prove -- or disprove -- the diffusion backend on real hardware.

The diffusion code in ``video_gen/visuals/diffusion.py`` has never been executed:
the development machine is an Intel Mac on Python 3.13, where neither torch nor
onnxruntime publishes a wheel. This script is how that code gets its first real
run. Take it to a machine with a GPU (or Apple Silicon) and:

    pip install -r requirements.txt -r requirements-ml.txt
    python verify_backends.py            # probe + still render
    python verify_backends.py --motion   # also try a motion clip
    python verify_backends.py --full     # also render a complete short film

It writes samples to ``verify_out/`` and prints a pass/fail line per capability.
Report failures against the tier they belong to -- a machine without the VRAM
for AnimateDiff is expected to fail MOTION and still pass STILL and IDENTITY.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

OUT = Path("verify_out")


def check(label: str, fn):
    """Run one probe, printing a single pass/fail line and never raising."""
    print(f"\n--- {label} ---")
    started = time.time()
    try:
        detail = fn()
        print(f"PASS  {label}  ({time.time() - started:.1f}s)  {detail or ''}")
        return True
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not die
        print(f"FAIL  {label}  ({time.time() - started:.1f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion", action="store_true", help="also try a motion clip")
    parser.add_argument("--full", action="store_true", help="also render a whole short film")
    parser.add_argument("--preset", default="draft")
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    from video_gen import GenerationConfig
    from video_gen.visuals import CharacterRef, ShotRequest, default_registry
    from video_gen.visuals.diffusion import DiffusionBackend

    config = GenerationConfig(preset=args.preset, output_dir=OUT, cache_dir=OUT / ".cache")
    results = {}

    print("=" * 70)
    print("  BACKEND VERIFICATION")
    print("=" * 70)
    print(f"  python  : {sys.version.split()[0]} on {sys.platform}")
    for row in default_registry(config).report():
        print(f"  {row['name']:<12} available={row['available']}  {row['capabilities']}")
        print(f"      {row['detail']}")

    backend = DiffusionBackend(config)
    if not backend.available():
        print("\nDiffusion backend is unavailable here; nothing further to verify.")
        print(f"  reason: {backend.describe()}")
        return 1

    def request(**kwargs) -> ShotRequest:
        base = dict(
            prompt="a weather-beaten lighthouse at dusk, cold blue light, cinematic",
            frame_size=config.resolution,
            seed=1234,
            image_size=int(config.preset_data["image_size"]),
            steps=int(config.preset_data["steps"]),
            guidance=float(config.preset_data["guidance"]),
            negative_prompt=str(config.preset_data["negative_prompt"]),
        )
        base.update(kwargs)
        return ShotRequest(**base)

    def still():
        visual = backend.render(request())
        path = OUT / "still.png"
        visual.image.save(path)
        return f"-> {path} ({visual.image.size[0]}x{visual.image.size[1]})"

    results["STILL"] = check("STILL: text to image", still)

    def determinism():
        a = backend.render(request(seed=99)).image
        b = backend.render(request(seed=99)).image
        import numpy as np

        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise AssertionError("same seed produced different images")
        return "same seed reproduces the same frame"

    results["DETERMINISM"] = check("DETERMINISM: seeded reproducibility", determinism)

    def identity():
        reference = OUT / "reference.png"
        if not reference.exists():
            raise FileNotFoundError(
                f"put a face/character image at {reference} to verify identity conditioning"
            )
        visual = backend.render(
            request(
                prompt="the same person standing on a pier at sunrise, cinematic",
                characters=[CharacterRef("Ref", reference_image=reference, speaking=True)],
            )
        )
        path = OUT / "identity.png"
        visual.image.save(path)
        from video_gen.visuals import IDENTITY

        if IDENTITY not in visual.used:
            raise AssertionError("IP-Adapter did not engage; check the load warning above")
        return f"-> {path}"

    results["IDENTITY"] = check("IDENTITY: IP-Adapter conditioning", identity)

    if args.motion:
        def motion():
            visual = backend.render(request(want_motion=True, duration=2.0, fps=8))
            if not visual.is_motion:
                raise AssertionError("backend returned a still; MOTION is not really available")
            for index, frame in enumerate(visual.frames):
                frame.save(OUT / f"motion_{index:02d}.png")
            return f"-> {len(visual.frames)} frames in {OUT}/"

        results["MOTION"] = check("MOTION: AnimateDiff clip", motion)

    if args.full:
        def film():
            from video_gen import FilmRenderer, Screenwriter, StoryMeta, cast_production
            from video_gen.casting import discover_voices

            script = (
                "INT. LIGHTHOUSE - NIGHT\n\nThe lamp turns.\n\n"
                "MAYA: Forty years and not one ship.\n"
                "JONAS: Then what is that on the horizon?\n"
            )
            production = Screenwriter().write(script, StoryMeta(title="Verify"))
            cast_production(production.cast, discover_voices())
            with FilmRenderer(config) as renderer:
                path = renderer.render(production, OUT / "film.mp4")
            return f"-> {path}"

        results["FILM"] = check("FILM: full render through the diffusion backend", film)

    print("\n" + "=" * 70)
    for label, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
