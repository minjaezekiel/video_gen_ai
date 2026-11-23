"""The command line: interview -> screenplay -> accept -> render.

Two callers, one code path. A person gets the formatted report; an agent passes
``--json`` and gets a documented object on stdout while every human line goes
to stderr. Commands therefore return a :class:`CommandResult` -- an exit code
and a payload -- rather than printing and returning an int, so the two renderings
can never drift apart.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Sequence

from . import llm
from .agent import (
    EXIT_CANCELLED,
    EXIT_FAILED,
    EXIT_NO_AUDIO,
    EXIT_OK,
    EXIT_USAGE,
    CommandResult,
    doctor_payload,
    fail,
    production_payload,
    production_summary,
    render_payload,
)
from .casting import cast_production, describe_pool, discover_voices
from .config import ASPECTS, DEFAULT_PRESET, FORMATS, PRESETS, STYLE_TIERS, GenerationConfig
from .interview import Interview, infer_meta
from .model import Production
from .render import FilmRenderer
from .screenwriter import Screenwriter
from .score import default_registry as default_music_registry
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


def command_doctor(config: GenerationConfig) -> CommandResult:
    """Report honestly on what this machine can actually do.

    For an agent this is the call to make first: it answers what this box can
    speak, write and draw, and whether any paid tier is on, in one object.
    """
    voices = discover_voices()
    speech = SpeechSynthesizer(config=config)
    # Probe for real: `say` being on PATH does not prove it makes sound, and
    # the difference is a film with subtitles and no voices.
    speech_ok, speech_detail = speech.probe()
    llm_detail = llm.describe()
    llm_probe = llm.probe() if config.use_llm else None

    registry = default_registry(config)
    music = default_music_registry(config, config.providers)
    visual_rows = registry.report()
    music_rows = music.report()
    provider_rows = config.providers.status()

    try:
        selected_visual = registry.select(config.backend).name
    except Exception as exc:  # noqa: BLE001
        selected_visual = f"none ({exc})"
    try:
        selected_music = music.select(config.music_backend).name
    except Exception as exc:  # noqa: BLE001
        selected_music = f"none ({exc})"

    print("\n" + "=" * 76)
    print("  CAPABILITY REPORT")
    print("=" * 76)
    print(f"  python      : {sys.version.split()[0]} on {sys.platform}")
    print(f"  resolution  : {config.resolution[0]}x{config.resolution[1]} @ {config.fps}fps")
    print(f"  voices      : {describe_pool(voices)}")
    print(f"  speech      : {'OK  ' if speech_ok else 'FAIL'} {speech_detail}")
    if not speech_ok:
        print("        -> renders will have subtitles but NO audible dialogue.")
        print("           macOS: `say` should already exist. Linux: install espeak-ng.")
        print("           Or install requirements-ml.txt for neural TTS.")

    # Same rule as speech: a reachable server is not a working writers' room,
    # so this generates a line of dialogue and checks it came back usable.
    print(f"  writers room: {llm_detail}")
    if llm_probe is not None:
        print(f"                {'OK  ' if llm_probe[0] else 'FAIL'} {llm_probe[1]}")
    else:
        print("                (pass --llm to use it, or --doctor --llm to test it)")

    print("\n  VISUAL BACKENDS")
    for row in visual_rows:
        mark = "OK " if row["available"] else "-- "
        caps = ", ".join(row["capabilities"]) or "unavailable"
        print(f"    {mark}{row['name']:<12} {caps}")
        print(f"        {row['detail']}")

    print("\n  MUSIC BACKENDS")
    for row in music_rows:
        mark = "OK " if row["available"] else "-- "
        caps = ", ".join(row["capabilities"]) or "unavailable"
        print(f"    {mark}{row['name']:<12} {caps}")
        print(f"        {row['detail']}")

    print("\n  HOSTED PROVIDERS   (opt-in; everything above runs offline)")
    for row in provider_rows:
        mark = "OK " if row["ready"] else "-- "
        label = row["provider"] or "none"
        print(f"    {mark}{row['tier']:<12} {label:<18} {row['detail']}")
    source = config.providers.source
    print(f"        config: {source if source else 'none — run --init-providers to create one'}")

    print(f"\n  selected    : visual={selected_visual}  music={selected_music}")
    print("\n  NOTE Diffusion needs torch, which has no macOS x86_64 wheel beyond")
    print("       Python 3.12. On this machine the procedural backend is the only")
    print("       one that can run; see decision.md for the full explanation.")
    print("=" * 76)

    return CommandResult(
        EXIT_OK,
        doctor_payload(
            config=config,
            voices=voices,
            speech_ok=speech_ok,
            speech_detail=speech_detail,
            llm_describe=llm_detail,
            llm_probe=llm_probe,
            visual_rows=visual_rows,
            music_rows=music_rows,
            provider_rows=provider_rows,
            selected_visual=selected_visual,
            selected_music=selected_music,
        ),
    )


def command_providers(config: GenerationConfig) -> CommandResult:
    """Show which hosted tiers are on, and what could be switched on."""
    providers = config.providers
    from .providers import BUILTIN_PROVIDERS

    print("\n" + "=" * 76)
    print("  HOSTED PROVIDERS")
    print("=" * 76)
    print(f"  config file : {providers.source or 'none'}")
    print("  Everything in this project runs offline. These tiers are opt-in,")
    print("  cost money per call, and fall back to the offline path on failure.\n")

    for row in providers.status():
        mark = "OK " if row["ready"] else "-- "
        print(f"    {mark}{row['tier']:<10} {row['provider'] or 'none':<18} {row['detail']}")

    print("\n  AVAILABLE PRESETS")
    for name, spec in sorted(BUILTIN_PROVIDERS.items()):
        resolved = providers.specs.get(name)
        key_state = "key set" if resolved and resolved.key() else f"needs ${spec.get('api_key_env', '?')}"
        print(f"    {name:<16} {spec.get('tier', 'music'):<8} {key_state:<24} {spec.get('notes', '')}")

    print("\n  To switch one on:")
    print("    python -m video_gen --init-providers      # writes providers.yaml")
    print("    $EDITOR providers.yaml                    # enabled: true")
    print("    export ELEVENLABS_API_KEY=...             # or whichever key")
    print("    python -m video_gen --doctor              # confirm it is ready")
    print("=" * 76)

    return CommandResult(
        EXIT_OK,
        {
            "config_file": str(providers.source) if providers.source else None,
            "tiers": providers.status(),
            "presets": [spec.redacted() for spec in sorted(providers.specs.values(), key=lambda s: s.name)],
        },
    )


def command_init_providers(config: GenerationConfig, path: Optional[Path], force: bool) -> CommandResult:
    """Write a commented, all-disabled providers config."""
    from .providers import CONFIG_NAME, write_example

    target = Path(path) if path else Path.cwd() / CONFIG_NAME
    try:
        written = write_example(target, force=force)
    except FileExistsError:
        print(f"\n{target} already exists. Pass --force to overwrite it.")
        return fail(EXIT_USAGE, f"{target} already exists", path=str(target))
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not write {target}: {exc}")
        return fail(EXIT_FAILED, f"could not write {target}: {exc}")

    print(f"\n✓ wrote {written}")
    print("  Every tier is disabled. Edit the file, export the key it names,")
    print("  then run `--doctor` to confirm the provider is ready.")
    return CommandResult(EXIT_OK, {"path": str(written.resolve()), "enabled": False})


def command_characters(config: GenerationConfig) -> CommandResult:
    store = CharacterStore(config.characters_dir)
    members = store.list_characters()
    if not members:
        print(f"No characters yet in {config.characters_dir}/")
        return CommandResult(EXIT_OK, {"characters": [], "directory": str(config.characters_dir)})
    print(f"\n{len(members)} character(s) in {config.characters_dir}/\n")
    for member in members:
        reference = " [ref image]" if member.reference_image else ""
        print(f"  {member.name:<18} {member.voice.voice:<24} {member.role}{reference}")

    from .agent import character_payload

    return CommandResult(
        EXIT_OK,
        {
            "directory": str(config.characters_dir),
            "characters": [character_payload(m) for m in members],
        },
    )


def command_list(config: GenerationConfig) -> CommandResult:
    store = ProductionStore(config.scripts_dir)
    productions = store.list_productions()
    for production in productions:
        print(
            f"{production.id}  [{production.status:9}] {production.format:<12}"
            f"{len(production.shots):>3} shots  {production.title}"
        )
    return CommandResult(
        EXIT_OK,
        {
            "directory": str(config.scripts_dir),
            "productions": [
                production_summary(p, store.path_for(p.id)) for p in productions
            ],
        },
    )


def command_show(config: GenerationConfig, production_id: str) -> CommandResult:
    """The full screenplay as JSON, so an agent can read it without parsing markdown."""
    store = ProductionStore(config.scripts_dir)
    try:
        production = store.load(production_id)
    except Exception as exc:  # noqa: BLE001
        print(f"No such production {production_id!r}: {exc}")
        return fail(EXIT_USAGE, f"no such production {production_id!r}")
    print_screenplay(production, full=True)
    return CommandResult(EXIT_OK, production_payload(production, store.path_for(production.id)))


def command_cast(config: GenerationConfig, production_id: str) -> CommandResult:
    """Re-cast a saved production's voices and show the result."""
    store = ProductionStore(config.scripts_dir)
    characters = CharacterStore(config.characters_dir)
    try:
        production = store.load(production_id)
    except Exception as exc:  # noqa: BLE001
        print(f"No such production {production_id!r}: {exc}")
        return fail(EXIT_USAGE, f"no such production {production_id!r}")

    for member in production.cast:
        member.voice.voice = ""  # force a fresh casting pass
        member.voice.hosted_voice = ""
    cast_production(production.cast, discover_voices(), hosted_voices=_hosted_voices(config))
    store.save(production)
    characters.remember(production)
    print(f"\nRecast {production.title}:")
    for member in production.cast:
        print(f"  {member.name:<18} {member.voice.voice:<24} "
              f"rate={member.voice.rate:.2f} pitch={member.voice.pitch:+.1f}")

    from .agent import character_payload

    return CommandResult(
        EXIT_OK,
        {
            "id": production.id,
            "title": production.title,
            "screenplay": str(store.path_for(production.id).resolve()),
            "cast": [character_payload(c) for c in production.cast],
        },
    )


def _hosted_voices(config: GenerationConfig) -> List[str]:
    """The paid provider's voice ids, or nothing when the tier is off."""
    from .providers import SPEECH

    spec = config.providers.spec_for(SPEECH)
    if spec is None or spec.missing_key_reason():
        return []
    return list(spec.voices)


def confirm_and_render(
    config: GenerationConfig,
    production: Production,
    store: ProductionStore,
    assume_yes: bool,
    output: Optional[Path] = None,
) -> CommandResult:
    name = output or ask("Output filename:", f"{production.id}.mp4") if not assume_yes else None
    target = Path(output) if output else config.output_dir / (name or f"{production.id}.mp4")

    print(f"\n  {len(production.shots)} shots · ~{production.estimated_duration:.0f}s of video")
    if not assume_yes and is_interactive():
        if ask("Render now? [Y/n]", "y").lower().startswith("n"):
            print("Cancelled. The screenplay is saved; render it later with --render.")
            return CommandResult(
                EXIT_CANCELLED,
                {
                    "id": production.id,
                    "screenplay": str(store.path_for(production.id).resolve()),
                    "output": None,
                },
            )

    started = time.time()
    try:
        with FilmRenderer(config, progress=console_progress) as renderer:
            path = renderer.render(production, target)
            voiced, silent = renderer.speech.voiced_lines, renderer.speech.silent_lines
            score_backend = renderer.score_backend
            speech_engine = renderer.speech.engine_name
            visual_backend = renderer.backend.name
    except Exception as exc:  # noqa: BLE001 - report it, do not traceback at an agent
        LOGGER.error("render failed: %s", exc, exc_info=True)
        print(f"\n✗ render failed: {type(exc).__name__}: {exc}")
        return fail(
            EXIT_FAILED,
            f"{type(exc).__name__}: {exc}",
            id=production.id,
            screenplay=str(store.path_for(production.id).resolve()),
            output=None,
        )

    elapsed = time.time() - started
    store.save(production)
    CharacterStore(config.characters_dir).remember(production)
    print(f"\n✓ {path}  ({elapsed:.1f}s)")
    if voiced:
        cast = ", ".join(f"{c.name}={c.voice.voice}" for c in production.speaking_cast)
        print(f"  {voiced} line(s) voiced  ·  {cast}")
    if score_backend and score_backend != "procedural":
        print(f"  score: {score_backend}")
    if silent:
        # Impossible to miss: this is the difference between a finished film and
        # one that plays subtitles over silence.
        print(f"  ⚠ {silent} line(s) had NO AUDIO — run `--doctor` to see why")
    print(f"  screenplay + render log: {store.path_for(production.id)}")

    data = render_payload(
        production,
        output=path,
        elapsed_s=elapsed,
        voiced_lines=voiced,
        silent_lines=silent,
        screenplay=store.path_for(production.id),
        config=config,
        score_backend=score_backend,
        speech_engine=speech_engine,
        visual_backend=visual_backend,
    )
    # A film nobody can hear is not a success, and an agent must be able to see
    # that from the exit code alone rather than by reading a warning line.
    if voiced == 0 and silent > 0:
        data["error"] = f"all {silent} line(s) were silent; the video has captions but no voices"
        return CommandResult(EXIT_NO_AUDIO, data)
    return CommandResult(EXIT_OK, data)


def command_create(config: GenerationConfig, args: argparse.Namespace) -> CommandResult:
    """The main flow: prompt -> interview -> screenplay -> accept -> render."""
    prompt = args.prompt or ""
    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Could not read {args.prompt_file}: {exc}")
            return fail(EXIT_USAGE, f"could not read {args.prompt_file}: {exc}")
    if not prompt:
        prompt = read_multiline(
            "Describe the video you want, or paste a script. Finish with an empty line."
        )
    if not prompt.strip():
        print("Nothing to make.")
        return fail(EXIT_USAGE, "no prompt given")

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
        music_backend=config.music_backend,
        providers_file=config.providers_file,
        use_llm=config.use_llm,
        llm_model=config.llm_model,
        output_dir=config.output_dir,
        scripts_dir=config.scripts_dir,
        characters_dir=config.characters_dir,
        cache_dir=config.cache_dir,
    )

    store = ProductionStore(config.scripts_dir)
    characters = CharacterStore(config.characters_dir)
    hosted = _hosted_voices(config)

    writer = Screenwriter(use_llm=config.use_llm, llm_model=config.llm_model)
    if config.use_llm:
        print(f"\n  writers' room: {llm.describe()}")

    while True:
        production = writer.write(prompt, meta)
        writer.variation += 1  # so [r]egenerate does not return the same draft
        characters.merge_into(production)  # reuse stored appearance/voice
        cast_production(production.cast, discover_voices(), hosted_voices=hosted)
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
            return CommandResult(
                EXIT_CANCELLED,
                production_payload(production, store.path_for(production.id)),
            )
        break

    production.status = "accepted"
    store.save(production)
    characters.remember(production)
    print(f"\n✓ Accepted → {store.path_for(production.id)}")

    if args.no_render:
        print("  --no-render: stopping before the render.")
        return CommandResult(
            EXIT_OK, production_payload(production, store.path_for(production.id))
        )
    return confirm_and_render(config, production, store, args.yes, args.output)


def command_render(config: GenerationConfig, args: argparse.Namespace) -> CommandResult:
    store = ProductionStore(config.scripts_dir)
    try:
        production = store.load(args.render)
    except Exception as exc:  # noqa: BLE001
        print(f"No such production {args.render!r}: {exc}")
        return fail(EXIT_USAGE, f"no such production {args.render!r}")
    CharacterStore(config.characters_dir).merge_into(production)
    print_screenplay(production)
    return confirm_and_render(config, production, store, args.yes, args.output)


def command_self_test(config: GenerationConfig, output: Optional[Path]) -> CommandResult:
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

    started = time.time()
    try:
        with FilmRenderer(config, progress=console_progress) as renderer:
            path = renderer.render(production, output or config.output_dir / "self_test.mp4")
            voiced, silent = renderer.speech.voiced_lines, renderer.speech.silent_lines
            score_backend = renderer.score_backend
            speech_engine = renderer.speech.engine_name
            visual_backend = renderer.backend.name
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("self-test failed: %s", exc, exc_info=True)
        print(f"\n✗ self-test failed: {type(exc).__name__}: {exc}")
        return fail(EXIT_FAILED, f"{type(exc).__name__}: {exc}")

    elapsed = time.time() - started
    store.save(production)
    cast = ", ".join(f"{c.name}={c.voice.voice}" for c in production.speaking_cast)
    print(f"\nSelf-test OK: {path}")
    print(f"  {voiced} line(s) voiced  ·  {cast}")

    data = render_payload(
        production,
        output=path,
        elapsed_s=elapsed,
        voiced_lines=voiced,
        silent_lines=silent,
        screenplay=store.path_for(production.id),
        config=config,
        score_backend=score_backend,
        speech_engine=speech_engine,
        visual_backend=visual_backend,
    )
    if silent:
        print(f"  ⚠ {silent} line(s) had NO AUDIO — run `--doctor` to see why")
        data["error"] = f"{silent} line(s) had no audio"
        return CommandResult(EXIT_NO_AUDIO, data)
    return CommandResult(EXIT_OK, data)


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
              python -m video_gen "a 30s advert for a bike shop" --yes
              python -m video_gen --prompt "a 30s advert for a bike shop" --yes
              python -m video_gen --prompt-file script.txt --style 3d --aspect 9:16
              python -m video_gen --self-test              offline end-to-end check
              python -m video_gen --doctor                 what can this machine do?
              python -m video_gen --list                   saved productions
              python -m video_gen --render <id>            re-render a saved one

            for agents (machine-readable, never interactive):
              python -m video_gen --doctor --json          capabilities as JSON
              python -m video_gen --prompt "..." --json     render, result as JSON
              python -m video_gen --show <id> --json       a screenplay as JSON

            exit codes:
              0 ok · 1 usage · 2 rendered but no audible dialogue · 3 failed
              4 cancelled
            """
        ),
    )
    parser.add_argument("prompt_positional", nargs="?", metavar="PROMPT",
                        help="the idea, story or script")
    # Documented in --help since v1 but never actually defined, so argparse
    # prefix-matched it to --prompt-file and died on a missing file.
    parser.add_argument("--prompt", metavar="TEXT", help="same as the positional PROMPT")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file")
    parser.add_argument("--title")
    parser.add_argument("--format", choices=sorted(FORMATS))
    parser.add_argument("--style", choices=sorted(STYLE_TIERS))
    parser.add_argument("--aspect", choices=sorted(ASPECTS))
    parser.add_argument("--seconds", type=int, help="target length")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--backend", help="force a visual backend (see --doctor)")
    parser.add_argument("--music-backend", help="force a music backend (see --doctor)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--render", metavar="ID", help="render a saved production")
    parser.add_argument("--show", metavar="ID", help="print a saved screenplay")
    parser.add_argument("--cast", metavar="ID", help="re-cast a saved production's voices")
    parser.add_argument("--list", action="store_true", help="list saved productions")
    parser.add_argument("--characters", action="store_true", help="your cast library")
    parser.add_argument("--doctor", action="store_true", help="report machine capabilities")
    parser.add_argument("--self-test", action="store_true", help="offline end-to-end check")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-render", action="store_true",
                        help="write the screenplay and stop")
    parser.add_argument(
        "--llm", action="store_true", help="let a local LLM write dialogue and image prompts"
    )
    parser.add_argument("--llm-model", metavar="NAME", default="", help="e.g. llama3.2:1b")
    parser.add_argument("--providers", action="store_true",
                        help="show opt-in hosted providers and their status")
    parser.add_argument("--init-providers", nargs="?", const="", metavar="PATH",
                        help="write a providers.yaml (all tiers disabled)")
    parser.add_argument("--providers-file", type=Path, metavar="PATH",
                        help="use this providers config instead of the usual places")
    parser.add_argument("--force", action="store_true",
                        help="overwrite when writing a providers file")
    parser.add_argument(
        "--json", action="store_true",
        help="emit one JSON object on stdout; implies --yes and sends all other output to stderr",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="accept all defaults")
    parser.add_argument("--verbose", "-v", action="store_true")

    from . import __version__

    parser.add_argument("--version", action="version", version=f"video_gen {__version__}")
    return parser


def dispatch(args: argparse.Namespace) -> CommandResult:
    """Build the config and run exactly one command."""
    try:
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
            music_backend=args.music_backend,
            providers_file=args.providers_file,
        )
    except ValueError as exc:
        print(f"Bad configuration: {exc}")
        return fail(EXIT_USAGE, str(exc))

    if args.init_providers is not None:
        path = Path(args.init_providers) if args.init_providers else None
        return command_init_providers(config, path, args.force)
    if args.providers:
        return command_providers(config)
    if args.doctor:
        return command_doctor(config)
    if args.list:
        return command_list(config)
    if args.characters:
        return command_characters(config)
    if args.show:
        return command_show(config, args.show)
    if args.cast:
        return command_cast(config, args.cast)
    if args.self_test:
        return command_self_test(config, args.output)
    if args.render:
        return command_render(config, args)
    return command_create(config, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # One prompt, three ways to give it. The positional wins because that is
    # what the quick start has always used.
    args.prompt = args.prompt_positional or args.prompt
    # An agent reading JSON cannot answer a prompt, so JSON is never
    # interactive. Saying so here means no command has to care.
    if args.json:
        args.yes = True

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.json:
        return dispatch(args).code

    # Human output goes to stderr so stdout carries nothing but the object.
    # Redirecting is used rather than threading a stream through every print,
    # because a single missed print is a corrupted payload at the caller.
    command = _command_name(args)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(args)
    except KeyboardInterrupt:
        result = fail(EXIT_CANCELLED, "interrupted")
    except Exception as exc:  # noqa: BLE001 - an agent gets JSON, never a traceback
        LOGGER.error("unhandled error: %s", exc, exc_info=True)
        result = fail(EXIT_FAILED, f"{type(exc).__name__}: {exc}")

    json.dump(result.payload(command), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return result.code


def _command_name(args: argparse.Namespace) -> str:
    for flag, name in (
        ("init_providers", "init-providers"),
        ("providers", "providers"),
        ("doctor", "doctor"),
        ("list", "list"),
        ("characters", "characters"),
        ("show", "show"),
        ("cast", "cast"),
        ("self_test", "self-test"),
        ("render", "render"),
    ):
        value = getattr(args, flag, None)
        if value not in (None, False, ""):
            return name
    return "create"
