---
name: video-gen
description: Turn a prompt, a story or a screenplay into a narrated MP4 with a cast, a shot list, captions, a score and a different voice for every character. Runs offline with no API keys. Use when asked to make, generate or render a video, advert, short film, explainer or educational clip from text, to write a screenplay and render it, or to re-render an edited screenplay.
license: MIT
---

# video-gen

Text in, finished MP4 out. A prompt becomes a screenplay, every character gets
its own voice, and the result is rendered with captions, camera moves and a
score that ducks under the dialogue.

Everything runs locally. No API key is needed and nothing is uploaded.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Nothing else. ffmpeg ships with `imageio-ffmpeg`; there are no models to
download. On Linux also `apt-get install -y espeak-ng`, or every character will
be silent — see **Linux and containers**.

## Always do this first

```bash
python -m video_gen --doctor --json
```

It reports what this machine can actually do, and it probes rather than guesses:
it synthesises a real phrase and, with `--llm`, writes a real line of dialogue.
Read `capabilities` from the result before planning any work:

| key | if false |
|---|---|
| `capabilities.speech` | the film will have captions but **no voices**. Say so before rendering. |
| `capabilities.write_dialogue` | a bare topic cannot become dialogue. Supply a script or prose instead. |
| `capabilities.hosted_music` | the score is synthesised locally. Normal, and the default. |

`voices.castable` is how many distinct voices exist here: 26 on macOS, up to 36
on Linux with espeak-ng, 0 with no synthesiser at all.

## The contract

**Pass `--json` for every call.** One JSON object on stdout, all human output on
stderr, never interactive — `--json` implies `--yes`, so nothing can block
waiting for an answer.

Every payload has `schema`, `command`, `ok`, `exit_code`, `status` and `error`
(`null` when fine). `schema` is `1`; new keys get added without bumping it, so
read the keys you need and ignore the rest.

### Exit codes

Check these before parsing anything.

| code | meaning | what to do |
|---|---|---|
| 0 | ok | use `output` |
| 1 | usage error — no prompt, unknown id, bad flag | fix the call |
| 2 | **an MP4 exists but no line was audible** | the file plays captions over silence. Report it; run `--doctor --json` to find out why |
| 3 | render failed, no file | read `error` |
| 4 | cancelled at a prompt | only possible without `--json` |

Code 2 is the one that matters. It is a real file of the right length with real
subtitles and no sound, so an agent that only checks "did a file appear" will
hand over a broken video.

## Making a video

```bash
python -m video_gen --prompt "<text>" --json --output out.mp4
```

Useful flags:

| flag | values |
|---|---|
| `--format` | `ad` · `short` · `educational` · `explainer` — pacing and structure |
| `--style` | `2d` · `3d` · `realistic` |
| `--aspect` | `16:9` · `9:16` · `1:1` · `4:5` · `21:9` |
| `--preset` | `draft` (480p) · `standard` (720p) · `quality` (1080p) |
| `--seconds N` | target length — advisory, not exact |
| `--seed N` | reproducible visuals, palette and staging |
| `--llm` | let a local model invent dialogue (see below) |
| `--no-music`, `--no-captions` | drop the score / burned-in subtitles |
| `--no-render` | write the screenplay and stop |
| `--title`, `--output` | name the film and the file |

The result carries `output` (absolute path), `duration_s`, `shots`,
`voiced_lines`, `silent_lines`, `screenplay` and the `cast` with each
character's voice.

### Choose the input shape deliberately

This is the one decision that determines whether the output is any good.

**A screenplay** — parsed directly, including parentheticals. Use this whenever
you know what should be said. Most reliable path by a wide margin.

```
INT. CONTROL ROOM - NIGHT

Banks of monitors glow in the dark.

MAYA: The signal is back. Same rhythm as last night.
JONAS (wary): Then it was never noise.
```

**Prose** — sentences become beats, quoted speech becomes attributed dialogue,
and pronouns resolve to the right character.

```
Maya had run the night shift for three years. "We are out of time," she said.
```

**A one-line brief** — structured and staged, and reported speech is lifted into
real dialogue, but **no new dialogue is invented** without `--llm`.

A bare topic such as `Explain photosynthesis to kids` with no `--llm` produces
one shot of the narrator reading that sentence back. That is working as
designed, not a bug. Either add `--llm`, or write the script yourself and pass
it — which, for a short piece, is usually the better video.

### Writing dialogue with a local model

```bash
python -m video_gen --prompt "Explain photosynthesis to primary school kids" \
    --format educational --seconds 30 --llm --json
```

Needs Ollama running locally with a model pulled (`ollama pull llama3.2:1b`).
Check `capabilities.write_dialogue` first. Still entirely offline: cloud-proxied
models are refused on purpose.

Only consulted when a brief left nobody speaking *and* the prompt is short
enough to be a brief — above ~60 words your prose *is* the film, and inventing
speech would be rewriting the user's work.

Quality is the model's ceiling. `llama3.2:1b` writes serviceable copy and
sometimes drifts off-brief; it will not reliably produce a two-character
conversation. Any larger Ollama model is a drop-in upgrade via `--llm-model`.

## Reviewing and re-rendering

```bash
python -m video_gen --list --json              # saved productions
python -m video_gen --show <id> --json         # full screenplay: scenes, shots, lines
python -m video_gen --render <id> --json       # re-render after editing
python -m video_gen --cast <id> --json         # re-roll the voices
python -m video_gen --characters --json        # the reusable cast library
```

The screenplay is markdown with YAML frontmatter in `video_scripts/<id>.md`.
To change a line, edit that file and `--render <id>`; images are cached by
prompt, preset and seed, so only what changed regenerates.

A good loop for a request that must be right: `--no-render` first, read the
screenplay from `--show`, fix what is wrong in the file, then `--render`.

## Hosted providers (opt-in, paid)

Off by default, and the whole pipeline works without them. Switch one on only
when a better score or better voices is worth money per call.

```bash
python -m video_gen --init-providers     # writes providers.yaml, all tiers off
python -m video_gen --providers --json   # what is configured, what is ready
```

Then in `providers.yaml` set `enabled: true` and a `provider`, and export the
key it names. Presets exist for `agensi`, `lyria`, `elevenlabs`, `minimax` and
`elevenlabs-tts`; any other HTTP service can be described in the same file
without code.

Two guarantees worth relying on: a provider that fails falls back to the offline
path rather than failing the render, and results are cached by request so
re-rendering an edited screenplay does not pay twice.

**Never switch a paid tier on without the user asking.** Check
`capabilities.hosted_music` / `hosted_speech` to see what the user already
chose.

## Linux and containers

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng
```

Without `espeak-ng` there is no synthesiser, every line falls back to timed
silence, and renders exit **2**. With it, `--doctor` reports up to 36 castable
voices built from espeak's own accents and variants.

## Limits worth stating up front

- **This is not Sora, Veo or Kling.** Those are datacenter models. Without the
  optional model stack the visuals are composed environments and staged
  geometric figures — stylised, deliberately, with no lip sync and no limb
  animation. Say this before delivering, so nobody expects photoreal footage.
- **"3D" is a look, not geometry.** The style tiers are prompt conditioning on
  one image pipeline.
- `--seconds` is a target the pacing aims at, not a guarantee; measured audio
  owns the clock, so the real length is whatever the dialogue takes to speak.
- The Stable Diffusion path needs `pip install -r requirements-ml.txt` and has
  never been executed on the development machine. Treat it as untested until
  `verify_backends.py` passes on your hardware.

## Worked examples

[`examples/`](examples/) has runnable scripts for each input shape, each with
the command, the expected JSON keys and what the output should look like.
`examples/agent_workflow.py` is the full plan → inspect → render loop.
