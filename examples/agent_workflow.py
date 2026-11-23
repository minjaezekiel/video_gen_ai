#!/usr/bin/env python3
"""The full agent loop: probe, plan, write, inspect, render, verify.

This is the shape an agent should use. It never parses console output and never
assumes a render worked because a file appeared -- it reads the JSON payload and
the exit code, because a video with captions and no sound exits 2 while looking
completely finished on disk.

    python examples/agent_workflow.py
    python examples/agent_workflow.py --prompt-file examples/02_prose.txt

Nothing here needs a key or a network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# From video_gen.agent; repeated here so this file stands alone as a reference.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_AUDIO = 2
EXIT_FAILED = 3


class CallFailed(RuntimeError):
    """A call came back non-zero. ``payload`` carries the reason."""

    def __init__(self, payload: Dict[str, Any], code: int) -> None:
        super().__init__(payload.get("error") or f"exit {code}")
        self.payload = payload
        self.code = code


def call(*argv: str, allow: Optional[List[int]] = None) -> Dict[str, Any]:
    """Run video_gen with --json and return the payload.

    stdout is the object and nothing else; stderr is the human log, which is
    worth keeping for diagnostics but must never be parsed.
    """
    command = [sys.executable, "-m", "video_gen", "--json", *argv]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        tail = (done.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"no JSON on stdout (exit {done.returncode}): {exc}\n" + "\n".join(tail)
        ) from exc

    if done.returncode not in (allow or [EXIT_OK]):
        raise CallFailed(payload, done.returncode)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, default=Path(__file__).parent / "01_screenplay.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "agent_workflow.mp4")
    parser.add_argument("--format", default="short")
    parser.add_argument("--aspect", default="16:9")
    args = parser.parse_args()

    # 1. What can this machine do? Always first: it decides what to ask for.
    print("→ probing capabilities")
    doctor = call("--doctor")
    able = doctor["capabilities"]
    print(f"  version {doctor['version']} on {doctor['platform']}, python {doctor['python']}")
    print(f"  {doctor['voices']['castable']} castable voices, "
          f"accents: {', '.join(doctor['voices']['accents']) or 'none'}")
    print(f"  speech={able['speech']}  write_dialogue={able['write_dialogue']}  "
          f"hosted_music={able['hosted_music']}")

    if not able["speech"]:
        # Worth stopping for: the render would succeed and be unusable.
        print("\n✗ no speech synthesiser on this machine.")
        print(f"  {doctor['speech']['detail']}")
        print("  Linux: apt-get install espeak-ng. macOS: `say` should exist.")
        return 1

    prompt_file = args.prompt_file
    words = len(prompt_file.read_text(encoding="utf-8").split())

    # 2. A short brief with nobody speaking needs a model to invent dialogue.
    #    A script or prose does not, and is better off without one.
    use_llm = words < 25 and able["write_dialogue"]
    if words < 25 and not able["write_dialogue"]:
        print("\n  note: short brief and no local model, so the narrator will")
        print("  read it back. Supply a screenplay for a better result.")

    # 3. Write the screenplay, but stop before rendering so it can be read.
    print(f"→ writing a screenplay from {prompt_file.name} ({words} words)")
    plan = call(
        "--prompt-file", str(prompt_file),
        "--format", args.format,
        "--aspect", args.aspect,
        "--no-render",
        *(["--llm"] if use_llm else []),
    )
    production_id = plan["id"]
    print(f"  {plan['title']}  [{production_id}]")
    print(f"  {plan['scene_count']} scene(s), {plan['shot_count']} shot(s), "
          f"~{plan['estimated_duration_s']:.0f}s")
    if plan.get("written_by"):
        print(f"  dialogue written by {plan['written_by']}")

    # 4. Inspect it the way a person would, before spending time on a render.
    full = call("--show", production_id)
    for scene in full["scenes"]:
        print(f"  SCENE {scene['index']} — {scene['setting']} ({scene['mood']})")
        for shot in scene["shots"]:
            for line in shot["lines"]:
                who = "NARRATOR" if line["narration"] else line["character"].upper()
                print(f"    {shot['index']:>2}. {who}: {line['text'][:58]}")

    speaking = [c for c in full["cast"] if c["voice"]]
    if len(speaking) < 2 and plan["shot_count"] > 2:
        # Not an error, but the thing a user usually wants to know.
        print("  note: only one voice is speaking; the film is a monologue.")

    # 5. Render. Exit 2 is accepted here so the silent case can be reported
    #    properly rather than raising as if no file had been produced.
    print("→ rendering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = call(
        "--render", production_id,
        "--output", str(args.output),
        allow=[EXIT_OK, EXIT_NO_AUDIO],
    )

    if result["exit_code"] == EXIT_NO_AUDIO:
        print(f"\n⚠ {result['output']}")
        print(f"  {result['error']}")
        print("  The file plays subtitles over silence. Do not hand it over as finished.")
        return EXIT_NO_AUDIO

    print(f"\n✓ {result['output']}")
    print(f"  {result['duration_s']}s · {result['shots']} shots · "
          f"{result['resolution'][0]}x{result['resolution'][1]} · "
          f"rendered in {result['elapsed_s']}s")
    print(f"  {result['voiced_lines']} line(s) voiced via {result['speech_engine']}, "
          f"score: {result['score_backend']}")
    for member in result["cast"]:
        print(f"    {member['name']:<14} {member['voice']}")
    print(f"  screenplay: {result['screenplay']}")
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CallFailed as exc:
        print(f"\n✗ {exc} (exit {exc.code})", file=sys.stderr)
        sys.exit(exc.code)
