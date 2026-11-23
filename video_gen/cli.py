"""The command line: interview -> screenplay -> accept -> render."""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Sequence

from . import llm
from .casting import cast_production, describe_pool, discover_voices
from .config import ASPECTS, DEFAULT_PRESET, FORMATS, PRESETS, STYLE_TIERS, GenerationConfig
from .interview import Interview, infer_meta
from .model import Production
from .render import FilmRenderer
from .screenwriter import Screenwriter
from .speech import SpeechSynthesizer
from .store import CharacterStore, ProductionStore
from .visuals import default_registry

LOGGER = logging.getLogger("video_gen.cli")


# --------------------------------------------------------------------------- #
# Prompting helpers
# --------------------------------------------------------------------------- #


def is_interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def ask(prompt: str, default: str = "") -> str:
    """Prompt that degrades to its default when there is no terminal."""
    if not is_interactive():
        return default
    try:
        answer = input(f"{prompt} [{default}] " if default else f"{prompt} ").strip()
    except EOFError:
        return default
    return answer or default


def read_multiline(banner: str) -> str:
    print(f"\n{banner}\n")
    if not is_interactive():
        return ""
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def console_progress(pct: float, message: str) -> None:
    print(f"  [{pct:5.1f}%] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def print_screenplay(production: Production, full: bool = False) -> None:
    print("\n" + "=" * 76)
    print(f"  {production.title}   ({production.format} · {production.style} · {production.aspect})")
    if production.logline:
        print(f"  {textwrap.shorten(production.logline, 72)}")
    print("=" * 76)
    print(
        f"  {len(production.scenes)} scene(s) · {len(production.shots)} shots · "
        f"~{production.estimated_duration:.0f}s"
    )
    if production.config.get("written_by"):
        print(f"  dialogue written by {production.config['written_by']}")
    if production.cast:
        print("\n  CAST")
        for member in production.cast:
            voice = member.voice.voice or "uncast"
            print(f"    {member.name:<16} {voice:<24} {member.role}")
    limit = None if full else 6
    print()
    for scene in production.scenes:
        print(f"  SCENE {scene.index} — {scene.setting}  ({scene.time_of_day}, {scene.mood})")
        for shot in scene.shots[:limit]:
            who = ", ".join(shot.subjects) or "—"
            print(f"    Shot {shot.index:>2} {shot.shot_type:<13}{shot.duration:>5.1f}s  {who}")
            for line in shot.lines:
                tag = "narration" if line.is_narration else line.character.upper()
                print(f"        {tag}: {textwrap.shorten(line.text, 58)}")
        if limit and len(scene.shots) > limit:
            print(f"    … {len(scene.shots) - limit} more shot(s)")
    print("-" * 76)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def command_doctor(config: GenerationConfig) -> int:
    """Report honestly on what this machine can actually do."""
    print("\n" + "=" * 76)
    print("  CAPABILITY REPORT")
    print("=" * 76)
    print(f"  python      : {sys.version.split()[0]} on {sys.platform}")
    print(f"  resolution  : {config.resolution[0]}x{config.resolution[1]} @ {config.fps}fps")
    print(f"  voices      : {describe_pool()}")

    # Probe for real: `say` being on PATH does not prove it makes sound, and
    # the difference is a film with subtitles and no voices.
    speech = SpeechSynthesizer()
    ok, detail = speech.probe()
    print(f"  speech      : {'OK  ' if ok else 'FAIL'} {detail}")
    if not ok:
        print("        -> renders will have subtitles but NO audible dialogue.")
        print("           macOS: `say` should already exist. Linux: install espeak-ng.")
        print("           Or install requirements-ml.txt for neural TTS.")

    # Same rule as speech: a reachable server is not a working writers' room,
    # so this generates a line of dialogue and checks it came back usable.
    print(f"  writers room: {llm.describe()}")
    if config.use_llm:
        ok, detail = llm.probe()
        print(f"                {'OK  ' if ok else 'FAIL'} {detail}")
    else:
        print("                (pass --llm to use it, or --doctor --llm to test it)")

    print("\n  VISUAL BACKENDS")
    registry = default_registry(config)
    for row in registry.report():
        mark = "OK " if row["available"] else "-- "
        caps = ", ".join(row["capabilities"]) or "unavailable"
        print(f"    {mark}{row['name']:<12} {caps}")
        print(f"        {row['detail']}")

    try:
        chosen = registry.select(config.backend)
        print(f"\n  selected    : {chosen.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  selected    : none ({exc})")

    print("\n  NOTE Diffusion needs torch, which has no macOS x86_64 wheel beyond")
    print("       Python 3.12. On this machine the procedural backend is the only")
    print("       one that can run; see decision.md for the full explanation.")
    print("=" * 76)
    return 0


def command_characters(config: GenerationConfig) -> int:
    store = CharacterStore(config.characters_dir)
    members = store.list_characters()
    if not members:
        print(f"No characters yet in {config.characters_dir}/")
        return 0
    print(f"\n{len(members)} character(s) in {config.characters_dir}/\n")
    for member in members:
        reference = " [ref image]" if member.reference_image else ""
        print(f"  {member.name:<18} {member.voice.voice:<24} {member.role}{reference}")
    return 0


def command_list(config: GenerationConfig) -> int:
    for production in ProductionStore(config.scripts_dir).list_productions():
        print(
            f"{production.id}  [{production.status:9}] {production.format:<12}"
            f"{len(production.shots):>3} shots  {production.title}"
        )
    return 0


def command_cast(config: GenerationConfig, production_id: str) -> int:
    """Re-cast a saved production's voices and show the result."""
    store = ProductionStore(config.scripts_dir)
    characters = CharacterStore(config.characters_dir)
    production = store.load(production_id)
    for member in production.cast:
        member.voice.voice = ""  # force a fresh casting pass
    cast_production(production.cast, discover_voices())
    store.save(production)
    characters.remember(production)
    print(f"\nRecast {production.title}:")
    for member in production.cast:
        print(f"  {member.name:<18} {member.voice.voice:<24} "
              f"rate={member.voice.rate:.2f} pitch={member.voice.pitch:+.1f}")
    return 0


def confirm_and_render(
    config: GenerationConfig,
    production: Production,
    store: ProductionStore,
    assume_yes: bool,
    output: Optional[Path] = None,
) -> Optional[Path]:
    name = output or ask("Output filename:", f"{production.id}.mp4") if not assume_yes else None
    target = Path(output) if output else config.output_dir / (name or f"{production.id}.mp4")

    print(f"\n  {len(production.shots)} shots · ~{production.estimated_duration:.0f}s of video")
    if not assume_yes and is_interactive():
        if ask("Render now? [Y/n]", "y").lower().startswith("n"):
            print("Cancelled. The screenplay is saved; render it later with --render.")
            return None

    started = time.time()
    with FilmRenderer(config, progress=console_progress) as renderer:
        path = renderer.render(production, target)
        voiced, silent = renderer.speech.voiced_lines, renderer.speech.silent_lines
    store.save(production)
    CharacterStore(config.characters_dir).remember(production)
    print(f"\n✓ {path}  ({time.time() - started:.1f}s)")
    if voiced:
        cast = ", ".join(f"{c.name}={c.voice.voice}" for c in production.speaking_cast)
        print(f"  {voiced} line(s) voiced  ·  {cast}")
    if silent:
        # Impossible to miss: this is the difference between a finished film and
        # one that plays subtitles over silence.
        print(f"  ⚠ {silent} line(s) had NO AUDIO — run `--doctor` to see why")
    print(f"  screenplay + render log: {store.path_for(production.id)}")
    return path


def command_create(config: GenerationConfig, args: argparse.Namespace) -> int:
    """The main flow: prompt -> interview -> screenplay -> accept -> render."""
    prompt = args.prompt or ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not prompt:
        prompt = read_multiline(
            "Describe the video you want, or paste a script. Finish with an empty line."
        )
    if not prompt.strip():
        print("Nothing to make.")
        return 1

    overrides = {
        "format": args.format,
        "style": args.style,
        "aspect": args.aspect,
        "target_seconds": args.seconds,
        "title": args.title,
    }
    interview = Interview(ask=ask, assume_yes=args.yes)
    print("\nA few questions so this comes out right:\n" if not args.yes else "")
    meta = interview.run(prompt, overrides)

    print("\n" + "=" * 76)
    print("  MAKING")
    print("=" * 76)
    print(interview.summary(meta))
    print("=" * 76)

    # The interview owns aspect and style; rebuild the config so the renderer
    # and the screenplay cannot disagree about the frame size.
    config = GenerationConfig(
        preset=config.preset,
        aspect=meta.aspect,
        style=meta.style,
        seed=config.seed,
        music=config.music,
        captions=config.captions,
        backend=config.backend,
        use_llm=config.use_llm,
        llm_model=config.llm_model,
        output_dir=config.output_dir,
        scripts_dir=config.scripts_dir,
        characters_dir=config.characters_dir,
        cache_dir=config.cache_dir,
    )

    store = ProductionStore(config.scripts_dir)
    characters = CharacterStore(config.characters_dir)

    writer = Screenwriter(use_llm=config.use_llm, llm_model=config.llm_model)
    if config.use_llm:
        print(f"\n  writers' room: {llm.describe()}")

    while True:
        production = writer.write(prompt, meta)
        writer.variation += 1  # so [r]egenerate does not return the same draft
        characters.merge_into(production)  # reuse stored appearance/voice
        cast_production(production.cast, discover_voices())
        store.save(production)
        print_screenplay(production)

        if args.yes or not is_interactive():
            break
        answer = ask("Accept this screenplay? [Y]es / [v]iew full / [r]egenerate / [q]uit:", "y").lower()
        if answer.startswith("v"):
            print_screenplay(production, full=True)
            continue
        if answer.startswith("r"):
            continue
        if answer.startswith("q"):
            print(f"Saved as {store.path_for(production.id)} without rendering.")
            return 0
        break

    production.status = "accepted"
    store.save(production)
    characters.remember(production)
    print(f"\n✓ Accepted → {store.path_for(production.id)}")
    confirm_and_render(config, production, store, args.yes, args.output)
    return 0


def command_render(config: GenerationConfig, args: argparse.Namespace) -> int:
    store = ProductionStore(config.scripts_dir)
    production = store.load(args.render)
    CharacterStore(config.characters_dir).merge_into(production)
    print_screenplay(production)
    confirm_and_render(config, production, store, args.yes, args.output)
    return 0


def command_self_test(config: GenerationConfig, output: Optional[Path]) -> int:
    """Full pipeline, two characters, zero network access."""
    prompt = (
        "INT. CONTROL ROOM - NIGHT\n\n"
        "Banks of monitors glow in the dark. Rain streaks the window.\n\n"
        "MAYA: The signal is back. Same rhythm as last night.\n"
        "JONAS (wary): Then it was never noise.\n"
        "MAYA: We are out of time. We go tonight.\n\n"
        "EXT. CAR PARK - NIGHT\n\n"
        "Empty tarmac under sodium light.\n\n"
        "JONAS: Drive carefully.\n"
    )
    meta = infer_meta(prompt)
    meta.title = "Self Test"
    production = Screenwriter().write(prompt, meta)
    production.id = "self-test"
    cast_production(production.cast, discover_voices())

    store = ProductionStore(config.scripts_dir)
    store.save(production)
    print_screenplay(production)
    with FilmRenderer(config, progress=console_progress) as renderer:
        path = renderer.render(production, output or config.output_dir / "self_test.mp4")
        voiced, silent = renderer.speech.voiced_lines, renderer.speech.silent_lines
    store.save(production)
    cast = ", ".join(f"{c.name}={c.voice.voice}" for c in production.speaking_cast)
    print(f"\nSelf-test OK: {path}")
    print(f"  {voiced} line(s) voiced  ·  {cast}")
    if silent:
        print(f"  ⚠ {silent} line(s) had NO AUDIO — run `--doctor` to see why")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_gen",
        description="Turn a prompt or a script into a narrated, voiced film.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              python -m video_gen                          interview and make a film
              python -m video_gen --prompt "a 30s advert for a bike shop" --yes
              python -m video_gen --prompt-file script.txt --style 3d --aspect 9:16
              python -m video_gen --self-test              offline end-to-end check
              python -m video_gen --doctor                 what can this machine do?
              python -m video_gen --list                   saved productions
              python -m video_gen --render <id>            re-render a saved one
            """
        ),
    )
    parser.add_argument("prompt", nargs="?", help="the idea, story or script")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file")
    parser.add_argument("--title")
    parser.add_argument("--format", choices=sorted(FORMATS))
    parser.add_argument("--style", choices=sorted(STYLE_TIERS))
    parser.add_argument("--aspect", choices=sorted(ASPECTS))
    parser.add_argument("--seconds", type=int, help="target length")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--backend", help="force a visual backend (see --doctor)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--render", metavar="ID", help="render a saved production")
    parser.add_argument("--cast", metavar="ID", help="re-cast a saved production's voices")
    parser.add_argument("--list", action="store_true", help="list saved productions")
    parser.add_argument("--characters", action="store_true", help="list saved characters")
    parser.add_argument("--doctor", action="store_true", help="report machine capabilities")
    parser.add_argument("--self-test", action="store_true", help="offline end-to-end check")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument(
        "--llm", action="store_true", help="let a local LLM write dialogue and image prompts"
    )
    parser.add_argument("--llm-model", metavar="NAME", default="", help="e.g. llama3.2:1b")
    parser.add_argument("--yes", "-y", action="store_true", help="accept all defaults")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = GenerationConfig(
        preset=args.preset,
        aspect=args.aspect or "16:9",
        style=args.style or "2d",
        seed=args.seed,
        music=not args.no_music,
        captions=not args.no_captions,
        use_llm=args.llm or bool(args.llm_model),
        llm_model=args.llm_model,
        backend=args.backend,
    )

    if args.doctor:
        return command_doctor(config)
    if args.list:
        return command_list(config)
    if args.characters:
        return command_characters(config)
    if args.cast:
        return command_cast(config, args.cast)
    if args.self_test:
        return command_self_test(config, args.output)
    if args.render:
        return command_render(config, args)
    return command_create(config, args)
