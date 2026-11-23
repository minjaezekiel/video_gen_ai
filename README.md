# video_gen_ai

Turn a prompt or a script into a narrated film — with a cast, a shot list, and a
different voice for every character. Entirely offline.

```bash
python -m video_gen
```

You describe what you want. It asks a few questions, writes a screenplay, casts a
voice per character, and renders an MP4 with spoken dialogue, captions, camera
moves and a score that ducks under the speech. The screenplay stays on disk as
markdown you can edit and re-render.

For ads, short films, explainers and educational video. No API keys, no network,
no GPU required.

---

## Contents

- [Install](#install) · [Quick start](#quick-start)
- [How it works](#how-it-works) — the pipeline, stage by stage
- [What it has](#what-it-has) — features and where they live
- [How to use it](#how-to-use-it) — commands, flags, input formats
- [Writing dialogue with a local LLM](#writing-dialogue-with-a-local-llm)
- [Characters and voices](#characters-and-voices)
- [Screenplays on disk](#screenplays-on-disk)
- [What runs where](#what-runs-where) · [Troubleshooting](#troubleshooting)
- [Honest limits](#honest-limits)

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That is everything. No system ffmpeg (it ships with `imageio-ffmpeg`), no models
to download.

```bash
python -m video_gen --doctor       # what can this machine do? does audio work?
python -m video_gen --self-test    # full offline render, ~15s
```

`--doctor` actually synthesises a test phrase rather than just checking that a
synthesiser exists — see [Troubleshooting](#troubleshooting).

### Optional: AI visuals

```bash
pip install -r requirements-ml.txt   # torch, diffusers, transformers (~3 GB)
```

Swaps the procedural visuals for Stable Diffusion, and adds character identity
conditioning and motion clips where the hardware supports them.

> ⚠️ **The diffusion path has never been executed.** PyTorch publishes no macOS
> x86_64 wheel beyond Python 3.12, so it cannot run on the machine this was built
> on. It is written against the documented APIs and gated behind capability
> probes, but treat it as untested until
> [`verify_backends.py`](verify_backends.py) passes on your hardware. Everything
> else in this README is measured.

### Optional: written dialogue

```bash
# https://ollama.com — a native binary, no Python wheels involved
ollama pull llama3.2:1b
python -m video_gen --doctor --llm     # confirms it can actually write a line
```

Lets a **local** model write dialogue for a one-line brief, instead of the
narrator reading your brief back. Nothing leaves your machine: cloud-proxied
models are refused. See
[Writing dialogue with a local LLM](#writing-dialogue-with-a-local-llm).

## Quick start

```bash
# Interactive: it asks, you press Enter a lot
python -m video_gen

# One shot, no questions
python -m video_gen "Ama, a barista, tells a tired commuter named Dev
  that the first cup is on the house." --format ad --aspect 9:16 --yes

# Feed it a real screenplay
python -m video_gen --prompt-file script.txt --style 3d
```

---

## How it works

```
      your prompt
           │
   ┌───────▼────────┐   infers format, style, aspect, length,
   │   INTERVIEW    │   audience, tone and cast from your words,
   └───────┬────────┘   then asks you to confirm  (interview.py)
           │
   ┌───────▼────────┐   detects script / prose / brief and builds a
   │  SCREENWRITER  │   Production: scenes → shots → attributed lines
   └───────┬────────┘   (screenwriter.py)
           │
   ┌───────▼────────┐   --llm only, and only when a brief left nobody
   │ WRITERS' ROOM  │   speaking: a local model invents a cast and writes
   └───────┬────────┘   their lines; every line is validated (writersroom.py)
           │
   ┌───────▼────────┐   one distinct voice per character, from the
   │    CASTING     │   host's voice pool, persisted so it never drifts
   └───────┬────────┘   (casting.py)
           │
     ┌─────▼─────┐  you accept the screenplay  ──► video_scripts/<id>.md
           │
   ┌───────▼────────┐   per shot: synthesise each line, ask a visual
   │    RENDERER    │   backend for the picture, build captions
   └───────┬────────┘   (render.py)
           │
   ┌───────▼────────┐   one VideoClip, hand-rolled crossfades,
   │    TIMELINE    │   dialogue pinned to its own shot, score ducked
   └───────┬────────┘
           │
       output/<id>.mp4  +  render log appended to the screenplay
```

### The parts that matter

**Measured audio owns the clock.** A shot lasts exactly as long as its dialogue
takes to speak. The screenplay's duration is a planning estimate; the renderer
overwrites it with what the synthesiser actually produced. Computing picture
first and stretching audio to fit is what makes sound drift away from vision over
a long timeline.

**One speaker per shot.** A beat with an exchange becomes a shot/reverse-shot
pair, because two people talking in a single shot means the camera can never cut
between them and the staging cannot show who is speaking.

**The whole film is one clip.** Shots are composited by hand rather than with
`concatenate_videoclips(method="compose")`, which runs moviepy's mask blitter on
every frame instead of only inside a crossfade — measured 4× slower.

**Ducking, not a fixed level.** The score drops 11 dB across each speech window
and glides back. A fixed low volume either buries the dialogue or makes the music
inaudible.

**The model is never on the critical path.** The deterministic screenwriter runs
first and always produces a filmable screenplay. A local LLM can then improve it,
but every line it writes is validated — stage directions stripped, unknown
speakers refused, the brief-read-back caught — and a draft that fails is
discarded whole. The worst case of a bad model is the result you would have had
without one.

**Visual generation sits behind one interface.** `VisualBackend` has four
capabilities — `still`, `identity`, `motion`, `train` — and the renderer asks for
what it wants and copes with what it gets. A machine that can make stills but not
motion still produces a finished film; it just uses camera moves instead.

---

## What it has

| | |
|---|---|
| **Interview** | 8 questions, every default inferred from your prompt |
| **Three input formats** | screenplay, prose story, or a one-line brief |
| **Cast + shot list** | scenes, shot types, camera moves, attributed dialogue |
| **A voice per character** | 26 castable voices on macOS, matched by gender/age/accent |
| **Speaker-labelled captions** | each name in that character's own colour |
| **Score with ducking** | per-scene mood, crossfaded cues, −11 dB under speech |
| **Formats** | ad · short film · educational · explainer |
| **Styles** | 2D animation · 3D animated · realistic |
| **Aspects** | 16:9 · 9:16 · 1:1 · 4:5 · 21:9 |
| **Character library** | reusable across productions, with your own reference art |
| **Written dialogue** | optional local LLM invents a cast and writes their lines |
| **Render history** | every render appends real timings to the screenplay |
| **176 tests** | all offline, ~8s |

Shot types (`establishing`, `wide`, `medium`, `close`, `insert`) drive both the
image prompt and how large characters are staged, so framing genuinely changes
across a film.

---

## How to use it

### Commands

```bash
python -m video_gen                    # interview → screenplay → accept → render
python -m video_gen --list             # saved productions
python -m video_gen --render <id>      # re-render one (edit the .md first)
python -m video_gen --characters       # your cast library
python -m video_gen --cast <id>        # re-roll the voices for a production
python -m video_gen --doctor           # capabilities + a real audio probe
python -m video_gen --doctor --llm     # ...and a real dialogue probe
python -m video_gen --self-test        # offline end-to-end check
```

`python video_generator.py` also works — it is a shim over the `video_gen`
package.

### Flags

| flag | meaning |
|---|---|
| `--format ad\|short\|educational\|explainer` | pacing, structure, dialogue density |
| `--style 2d\|3d\|realistic` | anime / rendered / live-action look |
| `--aspect 16:9\|9:16\|1:1\|4:5\|21:9` | 9:16 renders a true 720×1280 |
| `--preset draft\|standard\|quality` | 480p / 720p / 1080p on the short side |
| `--seconds N` | target length |
| `--seed N` | reproducible visuals, palette and staging |
| `--title`, `--output` | name the film and the file |
| `--backend NAME` | force a visual backend (see `--doctor`) |
| `--no-music`, `--no-captions` | drop the score / burned-in subtitles |
| `--llm` | let a local LLM write dialogue and image prompts |
| `--llm-model NAME` | pick the model, e.g. `llama3.2:1b` (implies `--llm`) |
| `--yes` / `-y` | accept every default; automatic when stdin is not a TTY |
| `--verbose` | debug logging |

### The three input formats

**A screenplay** — sluglines and `NAME:` lines are parsed directly, including
parentheticals. The slugline sets the scene, its setting and its time of day.

```
INT. CONTROL ROOM - NIGHT

Banks of monitors glow in the dark.

MAYA: The signal is back. Same rhythm as last night.
JONAS (wary): Then it was never noise.
```

**A prose story** — sentences become beats, quoted speech becomes attributed
dialogue, and pronouns resolve to the right character.

```
Maya had run the night shift for three years. "We are out of time,"
she said, watching the trace climb. Jonas leaned over the console.
```
→ `MAYA: We are out of time.` (not the narrator, and not Jonas)

**A brief** — the instruction is stripped and reported speech becomes real
dialogue.

```
A 20 second vertical advert for Kestrel Coffee. Ama, a barista, tells
a tired commuter named Dev that the first cup is on the house.
```
→ title `Kestrel Coffee`, cast `Ama` + `Dev` (not "Kestrel" and "Coffee"),
and `AMA: The first cup is on the house.`

> On its own the engine structures and stages a brief but will not *invent* ad
> copy, so a brief with no speech in it becomes narration. Add `--llm` and it
> writes the dialogue — next section.

---

## Writing dialogue with a local LLM

Everything above is deterministic: no model, instant, identical every run. The
one thing regular expressions genuinely cannot do is **invent speech**. Without
help, `Explain photosynthesis to primary school kids` becomes a single shot of
the narrator reading that sentence back at you.

```bash
ollama pull llama3.2:1b     # 1.3 GB, and fast enough on a CPU
python -m video_gen "Explain photosynthesis to primary school kids" \
    --format educational --seconds 30 --llm --yes
```

```
  CAST
    Maya             Flo (English (US))
    Jax              Junior

  SCENE 1 — Green Valley Farm  (day, playful)
    Shot  1 establishing   1.6s   MAYA: What's photosynthesis?
    Shot  2 close          2.5s   JAX:  Is it how plants grow?
    Shot  3 medium         1.6s   MAYA: Exactly, Jax!
    Shot  4 close          3.4s   JAX:  I thought it was just eating leaves.
    Shot  5 medium         4.2s   MAYA: That's close, but plants make food from sunlight too.
```

Two characters that did not exist, with different voices, having a conversation.

**It still runs entirely offline.** Ollama is a local binary; cloud-proxied
models (`:cloud`) are excluded from selection on purpose, and the generation path
is tested to contact `127.0.0.1` and nothing else.

**It is asked only when it is needed** — when a brief left nobody speaking, and
only for prompts short enough to be a brief rather than a story. Above ~60 words
your narration *is* the film, and replacing it with invented lines would be
rewriting your work rather than finishing it.

**Nothing it writes is trusted.** Every line is validated: re-emitted speaker
prefixes (`AMA: Morning.`) and stage directions are stripped, all-caps cards and
sluglines rejected, unknown speakers refused rather than added to the cast, and
the brief-read-back caught. A draft that comes back too thin is discarded whole
and the deterministic screenplay stands. **The worst case is exactly the result
you would have had without a model.**

The same prompt writes the same script; pressing `r` to regenerate asks for a
genuinely different draft.

| | |
|---|---|
| Model chosen automatically | the smallest capable local one; `--llm-model` overrides |
| Dialogue for a 30s film | 8–20 s on an Intel Mac CPU |
| Without Ollama installed | prints `writers' room: none` and falls back |
| Also improves | image prompts, one per shot |

Quality is the model's ceiling, not the engine's: `llama3.2:1b` writes
serviceable copy and drifts off-brief sometimes. Any larger Ollama model is a
straight upgrade with no code change.

---

## Characters and voices

Every character gets their own voice, chosen from what the host offers — on macOS
that is 26 castable English voices across 6 accents, with distinct male, female,
child and elderly options. Casting reads the description first ("an older man,
cautious"), falls back to a given-name table, then applies per-character rate and
pitch so two roles never sound like the same performer.

It is deterministic and persistent: a character keeps their voice across every
render and every production they appear in.

```yaml
# characters/maya.md
---
name: Maya
role: engineer
description: A sharp engineer. She runs the night shift.
appearance: short red hair, grey jacket
voice: {engine: system, voice: Kathy, rate: 0.97, pitch: -0.9, gender: female}
reference_image: characters/maya.png
---
```

Edit the file to override anything — a hand-set voice is never overwritten.
`--cast <id>` re-rolls a production's voices from scratch.

---

## Screenplays on disk

Each production is markdown with YAML frontmatter in `video_scripts/`, with a
regenerated `INDEX.md`.

```markdown
---
id: 2026-08-11-the-signal-a3f1
title: The Signal
format: short
style: 2d
aspect: 16:9
status: rendered
cast:
  - {name: Maya, role: engineer, voice: {voice: Kathy, pitch: -0.9}}
  - {name: Jonas, role: technician, voice: {voice: Fred, pitch: 2.7}}
renders:
  - {output: output/the-signal.mp4, duration_s: 9.72, elapsed_s: 15.1,
     shots: 4, voiced_lines: 4, silent_lines: 0}
---

## Scene 1 — Control Room

*night · tense*

### Shot 1 — establishing — 4.3s

**With:** Maya
**Camera:** push in
**Action:** Banks of monitors glow in the dark.
**MAYA:** The signal is back. Same rhythm as last night.
```

Edit a shot, re-render with `--render <id>`, and only what changed regenerates —
images are cached by prompt, preset and seed. `voiced_lines` / `silent_lines`
record whether the dialogue was actually audible.

---

## What runs where

| | procedural | SD stills | identity | motion | LoRA training |
|---|---|---|---|---|---|
| **Any CPU** (incl. Intel Mac) | ✅ | slow¹ | slow¹ | ✗ | ✗ |
| **Apple Silicon** | ✅ | ✅ | ✅ | slow | slow |
| **NVIDIA ≥ 8 GB** | ✅ | ✅ | ✅ | ✅ | ✗² |
| **This Intel Mac / py3.13** | ✅ | ✗³ | ✗³ | ✗³ | ✗³ |

¹ seconds to a minute per image · ² not implemented · ³ no torch wheel exists

**A complete film always comes out.** Only fidelity changes:

| stage | with models | without |
|---|---|---|
| visuals | Stable Diffusion | composed environments + staged figures |
| character likeness | IP-Adapter from your reference image | stable per-character colour and staging |
| motion | AnimateDiff / SVD clips | camera moves over stills |
| voices | neural TTS | **system voices — fully working, 26 on macOS** |
| dialogue writing | **local LLM via Ollama — working, no GPU needed** | deterministic screenwriter |

The writers' room is the one model-backed feature that needs **no GPU and no
torch**: Ollama is a native binary, so `--llm` works on the same Intel Mac where
the diffusion tier cannot even import.

### Speed

Intel Mac, 16 GB, CPU-only, procedural backend:

| | |
|---|---|
| self-test — 2 characters, 4 shots → 9.7 s of 720p | **15.1 s** |
| vertical ad, 720×1280 | 15.5 s |
| dialogue for a 30 s film (`--llm`, llama3.2:1b) | 8–20 s |
| test suite (176, offline) | ~8 s |

Faster than real time without a GPU.

---

## Troubleshooting

### The video has subtitles but no voices

Run:

```bash
python -m video_gen --doctor
```

The `speech:` line synthesises a real phrase and reports whether it was audible.
Every render also prints how many lines were voiced:

```
✓ output/self_test.mp4  (15.1s)
  4 line(s) voiced  ·  Maya=Kathy, Jonas=Fred
```

and warns loudly if any were not:

```
  ⚠ 3 line(s) had NO AUDIO — run `--doctor` to see why
```

Causes, in order of likelihood:

- **No synthesiser found.** macOS ships `say`. On Linux `apt install espeak-ng`.
  Otherwise install `requirements-ml.txt` for neural TTS.
- **A character has no voice.** Check `characters/<id>.md`; run `--cast <id>` to
  re-roll, or set `voice:` by hand.
- **Your player.** Confirm the file has an audio stream:
  `ffprobe output/<file>.mp4` should list an `Audio: aac` stream.

Every render records `voiced_lines` and `silent_lines` in the screenplay, so you
can tell after the fact whether a given file was ever going to have sound.

### Other

- **Only one shot came out, and the narrator read my prompt back** — the prompt
  was a topic or a one-line brief with no speech in it. Add `--llm`, or give it a
  story or a screenplay.
- **`--llm` did nothing** — check `--doctor --llm`. If it says
  `writers' room: none`, Ollama is not running or has no local model
  (`ollama pull llama3.2:1b`). If your only model ends in `:cloud` it is refused
  on purpose: it is not offline.
- **The dialogue drifted off-brief** — press `r` to regenerate, or use a larger
  model with `--llm-model`. A 1B model is the floor, not the ceiling.
- **A brand became a character** — put the real cast in the interview's
  `Characters` answer, or use `--yes` with names in the prompt ("a barista named
  Ama").
- **Diffusion says unavailable** — expected without `requirements-ml.txt`, and
  unavoidable on Intel macOS with Python 3.13. `--doctor` prints the reason.

---

## Tests

```bash
pytest test.py            # 172 fast offline tests
pytest test.py -m slow    # end-to-end renders + one real-model draft
pytest test.py -m ''      # everything
```

The identity and motion code paths are covered by a mock backend, and the
writers' room by a mock LLM, so both are tested even on a machine that cannot
load a model. The one test that drives a real LLM skips cleanly without one.

---

## Honest limits

- **This is not Sora, Veo, Kling or Seedance.** Those are datacenter models. The
  best open models that run on consumer hardware (Wan 2.x, LTX-Video, CogVideoX,
  SVD, AnimateDiff) produce 2–5 second clips at 480–720p with visible artifacts,
  and multi-shot character consistency is an open research problem.
- **"3D" is a look, not geometry.** The style tiers are checkpoint and prompt
  conditioning on one image pipeline. There is no 3D engine.
- **A brief cannot become a full script without an LLM.** Deterministically the
  engine structures and stages what you give it and lifts reported speech into
  dialogue, but it will not invent ad copy. `--llm` closes this, and needs a
  local model installed.
- **Written dialogue is only as good as your local model.** `llama3.2:1b` writes
  serviceable copy and drifts off-brief; larger Ollama models are a drop-in
  upgrade.
- **A long prose story gains no invented dialogue.** Above ~60 words the writers'
  room is deliberately not asked, because adding speech would mean replacing the
  narration you wrote.
- **The procedural tier is stylised**, deliberately: geometric figures, no lip
  sync, no limb animation.

[decision.md](decision.md) — why each part is built the way it is, plus the audit
of the 16 bugs fixed in v1.
[roadmap.md](roadmap.md) — what is done and what is next.
[IMPLEMENTATION.md](IMPLEMENTATION.md) — current build state and file map.

## Requirements

Python 3.9+. Any OS. ~200 MB installed, or ~3 GB with the optional model stack.

## License

See [LICENSE](LICENSE).
