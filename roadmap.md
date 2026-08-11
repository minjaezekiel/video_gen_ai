# Roadmap

Where this has got to, and what is next. Rationale for everything below is in
[decision.md](decision.md); current build state is in
[IMPLEMENTATION.md](IMPLEMENTATION.md).

---

## v1 — Slideshow engine ✅ complete

A project that could not run at all became a fast, offline, CPU-friendly
narrated-video generator.

- [x] **Installable**: dropped the `+cu118` pins that made `pip install` fail on
      every non-NVIDIA machine; split the model stack into `requirements-ml.txt`;
      migrated moviepy 1.x → 2.x; bundled ffmpeg via `imageio-ffmpeg`; fixed the
      missing macOS root certificates with `certifi`
- [x] **Correct**: 16 bugs audited and fixed — output written into the directory
      cleanup deletes, every scene narrating simultaneously, narration always
      parsed as empty, the TTS literally speaking the words "female voice",
      A/V drift, `PYTHONHASHSEED`-dependent seeds, a cache that had never once
      been a cache ([full table](decision.md))
- [x] **Fast**: one diffusion call per scene instead of five; Ken Burns motion
      instead of pixel interpolation; the whole video assembled as a single
      `VideoClip` — **42.8 s → 18.5 s**
- [x] **Durable**: markdown script store with render history and a regenerated index
- [x] **Offline**: every stage degrades instead of failing; `--self-test` needs
      no network

## v2 — Film engine ✅ complete

From "narrated slideshow" to "a cast, a shot list and a voice each".

- [x] **Screenplay model**: `Production → Scene → Shot → Line` plus `Character`
      and `VoiceProfile`; markdown round-trip is byte-stable
- [x] **Character registry**: `characters/<id>.md` beside reference art; reusable
      across productions; fills gaps but never overrides a script
- [x] **Interview**: 8 clarifying questions, every default inferred from the
      prompt (format, style, aspect, length, audience, tone, cast)
- [x] **Screenwriter**: screenplays, prose stories and briefs; dialogue
      attribution through pronouns; brand names kept out of the cast; reported
      speech lifted into real dialogue; one speaker per shot
- [x] **Casting**: 26 castable system voices discovered; traits from description
      then a name table; deterministic, distinct, order-independent, never
      overrides a hand-set voice
- [x] **Per-character speech**: rate and pitch shaping so each role is a
      different performer
- [x] **Score + ducking**: per-scene mood with crossfaded cues; bed drops −11 dB
      under dialogue instead of sitting at a fixed level
- [x] **Visual backend seam**: one `VisualBackend` Protocol, a registry that
      degrades per capability, `ProceduralBackend` (always available) and
      `MockBackend` (claims everything, for testing what cannot run here)
- [x] **Render**: speaker-labelled captions in each character's colour, camera
      moves driven by the shot list, motion-clip playback, single-clip timeline
- [x] **Formats and aspects**: ad / short / educational / explainer × 16:9, 9:16,
      1:1, 4:5, 21:9 — the preset sets the short side, so 9:16 is 720×1280
- [x] **CLI**: `create`, `--render`, `--list`, `--characters`, `--cast`,
      `--doctor`, `--self-test`
- [x] **124 tests**, all offline, including mock-driven identity and motion paths

---

## Not done — known gaps

**Run `verify_backends.py` on real hardware.** Everything in
`video_gen/visuals/diffusion.py` — SD stills, IP-Adapter identity, AnimateDiff
motion — is **written but has never been executed**. torch has no macOS x86_64
wheel past Python 3.12 and onnxruntime none for 3.13, so no neural code can run
on the development machine at all. This is the single highest-value next step.

**LLM dialogue generation.** `--llm` currently only rewrites visual prompts. A
brief ("a 20 second advert for X") can be structured and staged, and reported
speech is lifted into lines, but the engine cannot *invent* ad copy. Writing
dialogue is the one part of screenwriting that genuinely needs a model.

**LoRA character training.** `DiffusionBackend.train_character` raises
`NotImplementedError` on purpose — a training loop cannot be written blind
against hardware that has never run it. The seam exists; `Character.lora` is
already carried through the model, the store and `ShotRequest`.

---

## Next

### Visual fidelity
- **Verify, then tune the presets.** Every SD timing in decision.md §2.4 is an
  estimate. Measure, then adjust default resolution if `standard` is too slow.
- **Character consistency across shots**: IP-Adapter is wired but unverified;
  add InstantID/PhotoMaker as alternatives once there is hardware to compare on.
- **img2img shot continuity**: seed each shot from the previous one at low
  strength so a scene holds its look. Cheap, and a large perceived win.
- **LTX-Video / Wan 2.x** as motion backends alongside AnimateDiff — both are
  more consumer-GPU-friendly than SVD.
- **Style-tier checkpoints**: an anime checkpoint for `2d` and a photoreal one
  for `realistic`, rather than prompt conditioning alone.

### Characters
- **Upload → character**: `--import-character name image.png` is half-built
  (`CharacterStore.import_reference` works); wire it into the CLI and the
  interview so a user can cast their own art.
- **Character sheets**: generate a reference turnaround once per character and
  condition every shot on it, rather than on a single photo.
- **Lip sync**: audio-driven mouth states for the procedural tier — a real
  animation win that needs no model at all.

### Audio
- **MusicGen-small** for generated score, cached per mood.
- **Word-level timing** from the TTS to drive per-word caption highlighting.
- **Sound effects** keyed off shot actions.

### Output
- **`.srt` sidecar** — timings are already computed, this is nearly free.
- **Batch render** every `accepted` production.
- **Resume**: skip shots whose cached image and audio are both current.

### Workflow
- **Style references**: seed a new production's look from a past accepted one.
- **`--edit`**: open the screenplay in `$EDITOR` before rendering.
- **Web UI**, if the CLI stops being enough.
