# Implementation log

**Read this first when picking the work up cold.** It is the running state of
the build: what is done, what is not, where everything lives, and what cannot be
verified on this machine.

Last updated: 2026-08-11 · v2 (film engine)

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

**Tests: 128 passing** (125 fast in ~7 s, 3 slow end-to-end renders).

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
.venv/bin/python -m video_gen --self-test     # offline end-to-end, ~15 s
.venv/bin/python -m pytest test.py            # fast tests
.venv/bin/python -m pytest test.py -m ''      # + end-to-end renders
.venv/bin/python -m video_gen "a 30s advert for a bike shop" --yes
```

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

- **A brief cannot become a full script deterministically.** The engine can
  structure and stage a story, and can lift reported speech into dialogue, but it
  cannot *invent* 20 seconds of ad copy from one sentence. That is an LLM job —
  `--llm` currently only rewrites visual prompts, not dialogue.
- **Gender inference falls back to a name table** (~130 names). Unknown names
  cast as neutral. Override by editing the character's `voice:` in
  `characters/<id>.md`, or re-roll with `--cast <id>`.
- **The procedural tier is stylised, not photoreal**, by design. Figures are
  geometric silhouettes; there is no lip sync and no limb animation.
- `psutil` is referenced by the MPS VRAM probe but is not a dependency; the probe
  degrades to `False` if it is missing.

---

## Next

See [roadmap.md](roadmap.md). The highest-value next steps, in order:

1. Run `verify_backends.py` on GPU hardware and fix whatever it finds.
2. LLM dialogue generation (not just prompt enrichment) so briefs produce real scripts.
3. LoRA character training, once there is hardware to develop it against.
