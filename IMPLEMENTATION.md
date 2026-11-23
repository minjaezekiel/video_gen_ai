# Implementation log

**Read this first when picking the work up cold.** It is the running state of
the build: what is done, what is not, where everything lives, and what cannot be
verified on this machine.

Last updated: 2026-08-23 · v3 (writers' room)

---

## Where things stand

| Step | State | Notes |
|---|---|---|
| 1. Screenplay model + stores | ✅ done | `Production/Character/Scene/Shot/Line`, markdown round-trip is byte-stable |
| 2. Interview | ✅ done | 8 questions, every default inferred from the prompt |
| 3. Screenwriter | ✅ done | script, prose and brief inputs all handled |
| 4. Casting | ✅ done | 26 castable voices found on this machine, deterministic assignment |
| 5. Speech | ✅ done | per-character voice, rate and pitch shaping |
| 6. Music + ducking | ✅ done | per-scene mood, bed ducks −11 dB under dialogue |
| 7. Visual backend seam | ✅ done | Protocol + registry + procedural + mock |
| 8. Render | ✅ done | single-clip timeline, speaker captions, motion-clip support |
| 9. CLI | ✅ done | create / render / list / characters / cast / doctor / self-test |
| 10. Diffusion backend | ⚠️ **written, never executed** | see [Unverified](#unverified-on-this-machine) |
| 11. Writers' room | ✅ done | a local LLM writes dialogue; **verified here** against Ollama |

**Tests: 176 passing** (172 fast in ~8 s, 4 slow: 3 renders + one real-model draft).

### Step 11 — the writers' room (v3)

The engine could stage a story but not *invent* speech, so a brief
("Explain photosynthesis to primary school kids") produced one shot of the
narrator reading the instruction back. That was the largest gap that did not
need a GPU, and it is now closed.

**Ollama, not `transformers`.** The old `--llm` went straight to a HF pipeline
and could therefore never run here — same torch wall as diffusion. Ollama is a
native binary with no Python wheel to satisfy, so it works on this machine, and
`llm.py` speaks to it over HTTP with nothing but `urllib`. `TransformersClient`
is kept as the fallback for hosts that have torch, and is still unverified.

**Narrow schemas, several calls.** Measured, not assumed: asked for beats of
five fields, `llama3.2:1b` put the action in `setting` and left half the `text`
fields empty. Asked only for `{character, text}` pairs, the same model wrote
usable advertising copy every time. So the division of labour is: the model
writes *dialogue*, and the deterministic screenwriter keeps doing the staging,
framing and timing it is already good at. Ollama's JSON-**schema** mode
constrains decoding, which plain JSON mode does not — a 1B model emits malformed
JSON often enough that the difference matters.

**Nothing the model says is trusted.** `clean_line` strips re-emitted speaker
prefixes and stage directions, `echoes` catches the brief being read back,
`resolve_speaker` refuses to invent cast members, and a draft with fewer than
`MIN_DRAFT_LINES` usable lines is discarded whole. The deterministic result is
always still there, so the worst case of a bad model is the behaviour from
before one was installed.

**Two conditions before the room is asked at all** — both necessary:
nobody speaks yet (any attributed line came from the user's own text), and the
prompt is under `BRIEF_MAX_WORDS`. Above that the narration *is* the film, and
replacing 300 words of someone's story with six invented lines would be
rewriting their work rather than finishing it.

Bugs this surfaced and fixed along the way:

- `Explain photosynthesis…` cast a character called **Explain**. A capitalised
  first word is indistinguishable from a name, and the fake cast member then
  stopped the room inventing anyone real. Instruction verbs joined `_NON_NAMES`.
- `Kestrel Coffee, a small independent roastery` cast **Coffee**: the
  "Ama, a barista" introduction rule was clearing the two-capitals brand guard.
- Both roles in a two-hander got the same gender of voice — invented names miss
  the given-name table and model descriptions carry no cue. The cast schema now
  asks for gender outright and states it in the description, which is what
  `infer_traits` already reads.

Verified on this machine: `--doctor --llm` reports `ollama/llama3.2:1b`; a brief
now yields two invented characters with distinct voices trading real lines; and
a socket guard confirms the whole path touches **127.0.0.1 only**, so "entirely
offline" still holds. Cloud-proxied Ollama models are excluded from auto-selection
for the same reason.

### Audio diagnostics (added after a "no voice, only subtitles" report)

The silent-fallback path was **invisible**: if no synthesiser was found, if `say`
failed, or if a stored `VoiceProfile` had `engine: silent`, every line rendered
as pure silence while captions still appeared — and nothing said so. The film
looked finished and was mute.

Fixed by making it impossible to miss:

- `SpeechSynthesizer` counts `voiced_lines` / `silent_lines` and records
  `failures` explaining each fallback.
- `SpeechSynthesizer.probe()` synthesises a real phrase and measures its peak;
  `--doctor` reports `speech: OK/FAIL` from it. Checking that `say` is on `PATH`
  is not the same as checking that it makes sound.
- `render()` logs an ERROR if a production with dialogue produced no audible
  lines, and a WARNING for partial failures.
- The CLI prints `N line(s) voiced · Maya=Kathy, Jonas=Fred` after every render,
  and `⚠ N line(s) had NO AUDIO` when relevant; `--self-test` exits non-zero.
- `voiced_lines` / `silent_lines` are recorded in the render history.
- A stale `engine: silent` profile no longer mutes a character: if a synthesiser
  exists now, it is used regardless of what an earlier machine decided.

Verified on this machine: the rendered MP4's audio cross-correlates with the
spoken line at **0.98** (envelope 0.999), so dialogue is genuinely present in the
file; all 41 discovered voices work with the exact `say` invocation used.

---

## Run it

```bash
.venv/bin/python -m video_gen --doctor        # what can this machine do?
.venv/bin/python -m video_gen --doctor --llm  # ...and actually test the model
.venv/bin/python -m video_gen --self-test     # offline end-to-end, ~15 s
.venv/bin/python -m pytest test.py            # fast tests
.venv/bin/python -m pytest test.py -m ''      # + end-to-end renders
.venv/bin/python -m video_gen "a 30s advert for a bike shop" --yes
.venv/bin/python -m video_gen "a 30s advert for a bike shop" --yes --llm
```

The writers' room needs `ollama serve` and one small local model
(`ollama pull llama3.2:1b`). Without it `--llm` prints
`writers' room: none` and everything falls back to the deterministic path.

`python video_generator.py` still works — it is a re-export shim over `video_gen`.

---

## File map

```
video_gen/
  config.py        PRESETS, ASPECTS, STYLE_TIERS, FORMATS, SHOT_TYPES, GenerationConfig
  model.py         Production > Scene > Shot > Line, Character, VoiceProfile
                   + markdown writer/parser (parse_scenes, split_frontmatter)
  store.py         ProductionStore (video_scripts/), CharacterStore (characters/)
  interview.py     infer_meta() + Interview: the clarifying questions
  llm.py           OllamaClient / TransformersClient / MockLLM, discover_llm(), probe()
  writersroom.py   WritersRoom: LLM dialogue + the validation gate that guards it
  screenwriter.py  Screenwriter.write() -> Production. The deterministic core.
  casting.py       discover_voices(), infer_traits(), cast_voice(), cast_production()
  speech.py        SpeechSynthesizer, apply_rate/apply_pitch/normalize, wav IO
  music.py         MOODS, render_bed(), score_timeline(), duck(), mood_for()
  render.py        FilmRenderer: shots -> frames -> timeline -> mux -> MP4
  cli.py           argparse + all commands
  visuals/
    base.py        VisualBackend Protocol, ShotRequest, ShotVisual, BackendRegistry
    procedural.py  ProceduralBackend: environments + staged figures (always available)
    mock.py        MockBackend: deterministic fake claiming every capability
    diffusion.py   DiffusionBackend: SD + IP-Adapter + AnimateDiff  ⚠️ UNVERIFIED

video_generator.py  compatibility shim
verify_backends.py  run this on GPU hardware to test diffusion.py for the first time
test.py             the whole suite
```

### Key call path

```
cli.command_create
  └ Interview.run(prompt)                  -> StoryMeta
  └ Screenwriter.write(prompt, meta)       -> Production
       ├ looks_like_script? _beats_from_script : _beats_from_prose
       ├ extract_characters / extract_dialogue / reported_speech
       ├ _maybe_write_dialogue (--llm, and only if nobody speaks yet)
       │    └ WritersRoom.develop -> invent_cast / write_dialogue / find_setting
       │         └ validate_cast / validate_lines -> clean_line, echoes, resolve_speaker
       └ _build_scenes -> _split_by_speaker -> Shot(shot_type, camera, visual)
  └ cast_production(cast, discover_voices())
  └ FilmRenderer.render(production)
       ├ per shot: render_shot -> speech.speak per line, backend.render, build_caption
       ├ build_timeline(shots, fade)       -> one VideoClip
       ├ build_audio(...)                  -> dialogue at shot starts + ducked bed
       └ _encode                           -> MP4
```

---

## Decisions made during the build

- **The preset number is the short side, not the height.** Treating it as height
  made 9:16 render 406×720 instead of 720×1280. See `GenerationConfig.resolution_for`.
- **Shot length comes from measured audio, never from an estimate.** The
  screenplay's `duration` is a planning figure; `render_shot` overwrites it with
  what the synthesiser actually produced.
- **One speaker per shot.** `_split_by_speaker` breaks a beat into shot/reverse-shot,
  because two people talking in one shot means the camera can never cut between
  them and the staging cannot show who is speaking.
- **Pitch shifting resamples and lets duration change.** Duration is measured
  afterwards, so nothing needs it preserved, and resampling has no artefacts.
- **Palette seed is per-production, not per-shot.** Otherwise consecutive shots
  pick unrelated hues and the cuts read as glitches.
- **The procedural backend renders at the frame's aspect ratio**, unlike
  diffusion which needs square latents. A square plate got cropped and sliced the
  staged figures off.
- **Adjacent capitalised words are an entity, not two characters.** "Kestrel
  Coffee" was being cast as two speaking roles.
- **A brief is not content.** The instruction sentence ("a 20 second vertical
  advert for X") and trailing style notes are stripped, and reported speech
  ("Ama tells Dev that Y") is converted into an actual spoken line.

---

## Unverified on this machine

**Everything in `video_gen/visuals/diffusion.py` has never run.** Not a single
line of it has been executed, because:

- torch publishes no macOS x86_64 wheel beyond Python 3.12 (last: torch 2.2.2, cp312);
- onnxruntime publishes none for cp313 either, which also rules out Piper/Kokoro;
- this machine has only Python 3.13.7 and 3.14.

So `import torch` cannot succeed here, and every consumer of it is dead code
locally. What compensates:

1. `MockBackend` claims `STILL/IDENTITY/MOTION/TRAIN` and drives those code paths
   in the test suite, so the *orchestration* around diffusion is genuinely tested —
   frame-sequence playback, identity plumbing, capability degradation, and the
   fallback when a backend raises.
2. `verify_backends.py` exists to give `diffusion.py` its first real execution on
   hardware that can run it.

**Treat as untested until `verify_backends.py` passes somewhere real:** model
loading, SD-Turbo/SD1.5 stills, IP-Adapter identity conditioning, AnimateDiff
motion, and every timing estimate for those paths. `DiffusionBackend.train_character`
raises `NotImplementedError` on purpose — LoRA training cannot be written blind.

---

## Known gaps

- **A brief cannot become a full script *deterministically*.** Without `--llm`
  the engine still cannot invent 20 seconds of ad copy from one sentence — it
  structures, stages and lifts reported speech, and the narrator carries the
  rest. `--llm` is what closes this, and it needs a local model installed.
- **Dialogue quality is the model's ceiling, not the engine's.** `llama3.2:1b`
  writes serviceable ad copy but drifts off-brief; it is simply the largest model
  that answers in seconds on a CPU. `--llm-model` takes anything Ollama has.
- **A prose story still gets no invented dialogue.** Above `BRIEF_MAX_WORDS` the
  room is deliberately not asked, because it would have to replace the user's
  narration in order to add speech. Adding dialogue *alongside* narration is the
  open design question.
- **Gender inference falls back to a name table** (~130 names). Unknown names
  cast as neutral, so an LLM-invented cast is asked for its genders outright.
  Override by editing the character's description or `voice:` in
  `characters/<id>.md`, or re-roll with `--cast <id>`.
- **The procedural tier is stylised, not photoreal**, by design. Figures are
  geometric silhouettes; there is no lip sync and no limb animation.
- `psutil` is referenced by the MPS VRAM probe but is not a dependency; the probe
  degrades to `False` if it is missing.

---

## Next

See [roadmap.md](roadmap.md). The highest-value next steps, in order:

1. Run `verify_backends.py` on GPU hardware and fix whatever it finds. This is
   now the only remaining item that cannot be started on this machine.
2. Lip sync for the procedural tier — audio-driven mouth states, no model needed,
   and the largest remaining animation win that is fully runnable here.
3. `--import-character name image.png`: `CharacterStore.import_reference` already
   works, it is simply not wired into the CLI or the interview.
4. LoRA character training, once there is hardware to develop it against.
