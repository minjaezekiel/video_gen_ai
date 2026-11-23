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

## v3 — Writers' room ✅ complete

The engine could stage a story but not invent speech. A brief produced one shot
of the narrator reading the instruction back; now it produces a cast having a
conversation.

- [x] **Local LLM client**: `OllamaClient` over plain `urllib` — a native binary
      needs no Python wheel, so this runs on the machine where torch cannot be
      installed at all. `TransformersClient` kept for hosts that have torch;
      `MockLLM` drives every path in the tests
- [x] **Offline enforced**: cloud-proxied Ollama models are excluded from
      auto-selection, and a socket guard confirms the whole path touches
      127.0.0.1 only
- [x] **Dialogue generation**: a brief now yields invented characters trading
      real lines, shaped by the format's own beat structure (hook → problem →
      product → proof → call to action)
- [x] **Cast invention**: name, gender and description per character, so two
      roles in a two-hander get audibly different voices
- [x] **A validation gate**: re-emitted speaker prefixes and stage directions
      stripped, the brief-read-back caught, unknown speakers refused, and a thin
      draft discarded whole — the deterministic script always stands behind it
- [x] **Never rewrites the user's work**: the room is asked only when nobody
      speaks yet *and* the prompt is a brief rather than a story
- [x] **Deterministic**: the same prompt writes the same script; `[r]egenerate`
      asks for a genuinely different draft
- [x] **`--doctor --llm`** generates a real line and checks it came back usable,
      the same way the speech probe does
- [x] **176 tests**, including one that drives a real local model and skips
      cleanly without one


## v3.1 — the agent interface ✅ complete

The engine was complete and only a person could drive it. Everything a caller
that is not a person needs, plus the gaps that made it hard to adopt.

- [x] **`--json` on every command**: one object on stdout, all human output on
      stderr, never interactive (`--json` implies `--yes`). The contract lives
      in `agent.py` alone, so the human and machine renderings cannot drift
- [x] **Honest exit codes**: 0 ok · 1 usage · 2 **rendered with no audible
      dialogue** · 3 failed · 4 cancelled. Code 2 is the one that matters: the
      file is the right length, has real subtitles and plays silence, so a
      caller checking only "did a file appear" could not see it
- [x] **`--prompt` fixed**: documented in `--help` since v1 and never defined,
      so argparse prefix-matched it to `--prompt-file` and died on a missing
      file. A test now asserts every flag the epilogue advertises exists
- [x] **Installable**: `pyproject.toml`, a `video-gen` entry point, `[ml]` and
      `[dev]` extras, and five dependencies dropped that nothing imported
      (`requests`, `beautifulsoup4`, `tqdm`, `scipy`; `certifi` moved to the ml
      extra where the model downloads that need it live)
- [x] **`--show`** returns a whole screenplay as JSON, so an agent can review a
      script without parsing markdown; **`--no-render`** stops after writing it
- [x] **Linux voice parity**: the espeak pool is built from espeak's own accent
      and variant listings (up to 36 voices, both genders) instead of one
      hardcoded "default", and `_speak_system` now actually passes `-v` — without
      it every character on Linux spoke in the same voice
- [x] **`SKILL.md`**, **`examples/`** with a verified example per input shape and
      a working agent loop, and a **`Dockerfile`** that fails the build rather
      than shipping an image whose voices do not work
- [x] **Two defects found while writing the examples**: every screenplay was
      titled "INT" (the abbreviation's full stop ends the first sentence), and
      prose wrapped at 80 columns carried its line breaks into the captions
- [x] **293 tests**, 265 of them offline in ~13 s

## v3.2 — the music seam and the hosted tier ✅ complete

Visuals, speech and the LLM each sat behind an interface. Music was a hardcoded
sine-pad generator the renderer called directly, which made it the one thing
that could not be upgraded without editing the renderer.

- [x] **`MusicBackend`**, mirroring `VisualBackend`: a Protocol, a registry, and
      capabilities (`bed`, `cue`, `prompt`). `ProceduralMusic` wraps the existing
      synth unchanged, `MockMusic` drives the hosted path in the tests
- [x] **The registry takes a chain, not a backend.** A paid provider that times
      out, rate-limits, returns something undecodable or runs out of credit falls
      back to the procedural bed, and the film still comes out. This is the
      guarantee the whole tier rests on
- [x] **Opt-in hosted providers**, off by default: with no config file no socket
      is opened. One switch per tier, so paying for a score does not start paying
      for voices. Presets for Agensi, Lyria, ElevenLabs Music, MiniMax and
      ElevenLabs TTS
- [x] **A provider is data, not code**: endpoint, auth header, request body and
      where the audio sits in the response are all declarative, with `preset`
      inheritance, so a new service is a few lines of YAML. `:int` coercion on a
      lone placeholder, because sending `"30000"` where a number is expected is
      a 400
- [x] **Keys from the environment only**, `providers.yaml` in `.gitignore`, and
      no key printed, logged or included in a payload
- [x] **Results cached** by prompt, duration, provider and seed — each call costs
      money, so re-rendering an edited screenplay must not pay twice
- [x] **`--init-providers`** scaffolds a commented, all-disabled config;
      **`--providers`** reports what is configured and what is ready
- [x] **Hosted speech** as a third engine in `SpeechSynthesizer`, with hosted
      voice ids cast round-robin alongside a system voice — so switching the tier
      off leaves the cast audible rather than silent. One failure disables the
      tier for the rest of the render


## v3.3 — performance, and the roadmap's model-free items ✅ complete

Profiling an instrumented 43.8 s render first, then building only against what
it pointed at: encoding 33 %, TTS 25 % and fully serialised, the Ken Burns crop
and resize 20 %, PNG cache writes 15 %, caption compositing 4 %.

- [x] **A native accelerator**, optional: `rust/` is a PyO3 + rayon crate
      implementing the three per-frame hot loops. 2.9x on the crop-and-resize,
      3.7x on the caption composite, measured in isolation. `video_gen/fastpath.py`
      falls back to Pillow and NumPy, so `pip install -r requirements.txt` stays
      pure Python with no toolchain
- [x] **The two frame paths are compared, not trusted.** Pillow's exact triangle
      filter is reimplemented, so the Rust agrees to 1 per channel across eight
      crop and scale cases; the composite agrees exactly. Every accelerator test
      runs twice, once with the extension forced off
- [x] **Shots render in parallel** (`--jobs`), gated on a `parallel_safe` flag so
      a backend holding one model on one device stays serial. A test asserts
      `-j 1` and `-j 4` produce an identical film
- [x] **Spoken lines are cached**, keyed on the text and everything shaping the
      voice, which is what makes a re-render after an edit cheap
- [x] **PNG cache writes at compression level 1** instead of Pillow's 6: a cache
      file is read back once, and zlib was 15 % of a render
- [x] **`.srt` and `.vtt` sidecars** from the render's own measured timings, with
      estimated word timings for karaoke-style highlighting
- [x] **Lip sync** on the procedural tier: mouth positions from the same function
      that staged the figures, openness from a short-window RMS with a fast
      attack and a slower release. `MouthAnchor` is part of the backend seam
- [x] **`--resume`** via a content hash of everything that changes a pixel — not
      a timestamp, so a comment edit does not force a re-render and a changed
      `--seed` does. **`--batch`** renders a library, repeatably when paired
- [x] **`--import-character`** and **`--edit`**, both previously half-built
- [x] Four bugs the new tests caught: the audio cache key depended on state that
      mutated during the first call so re-renders never hit; `content_hash`
      hashed a speaker name that markdown round-trips as `MAYA`, so `--resume`
      could never match; `config.style` was missing from the hash; and the
      lip-sync envelope was parked on `self`, which parallel shots would stomp
- [x] A rejected LLM draft now records *why*, instead of returning `None`
      indistinguishable from having no model at all
- [x] **395 tests**, 353 of them offline in ~36 s

**Deliberately not built: a text-to-video model, in Rust or otherwise.**
Training one needs thousands of GPU-hours and a curated dataset at WebVid-10M
scale; Rust has no mature training stack (Candle and burn are inference-first);
and it would contradict the premise that this runs offline on a CPU with no
downloads. The `VisualBackend` seam is where a *pretrained* model belongs, and
`diffusion.py` is already written against it. Rust earns its place in the
deterministic pixel loops, where profiling put the time — not in the model
layer, where the native work is already native.

---

## Not done — known gaps

**Run `verify_backends.py` on real hardware.** Everything in
`video_gen/visuals/diffusion.py` — SD stills, IP-Adapter identity, AnimateDiff
motion — is **written but has never been executed**. torch has no macOS x86_64
wheel past Python 3.12 and onnxruntime none for 3.13, so no neural code can run
on the development machine at all. This is the single highest-value next step.

**Dialogue quality is capped by the model, not the engine.** `llama3.2:1b` is
simply the largest model that answers in seconds on a CPU; it writes serviceable
ad copy but drifts off-brief. `--llm-model` takes anything Ollama has, and a 7B
model on better hardware is a straight upgrade with no code change.

**A prose story still gets no invented dialogue.** Above `BRIEF_MAX_WORDS` the
writers' room is deliberately not asked, because adding speech would mean
replacing the narration the user actually wrote. Adding dialogue *alongside*
narration is the open design question.

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
- **Character sheets**: generate a reference turnaround once per character and
  condition every shot on it, rather than on a single photo.
- **Visemes rather than amplitude** for lip sync: mouth shapes driven by
  phonemes, so a vowel and a consonant differ. Needs the TTS to expose
  phonemes, which `say` does not and espeak does.

### Audio
- **Verify the hosted providers against real accounts.** Every request shape in
  `providers.py` is written from vendor documentation and has never been sent.
  The config layer, response parsing and all the failure paths are tested with
  `urlopen` replaced, but a renamed field would only show up against a live key.
- **MusicGen-small** as a *local* `MusicBackend`, cached per mood — the seam now
  exists, so this is an added file rather than a change to the renderer.
- **Force-aligned word timings**, replacing the length-weighted estimate the
  `.vtt` carries now. Needs an acoustic model.
- **Sound effects** keyed off shot actions.

### Output
- **Frame prefetch**: ffmpeg is still fed one frame at a time, which is the
  largest remaining serial cost. Rendering frames ahead into a bounded queue is
  the next real speedup, and bigger than anything left in the per-frame loops.
- **Burn the sidecar's word timings into the captions**, so the on-screen
  subtitle highlights as it is spoken rather than appearing whole.

### Writing
- **Dialogue alongside narration**, so a prose story can gain speech without
  losing the narration the user wrote.
- **A second pass over a draft** — ask the model to tighten its own lines against
  the target runtime, rather than capping the count in the schema.
- **Per-scene settings from the room**, so a film can move between locations
  instead of holding one.

### Workflow
- **Run the suite on Linux and on Python 3.9–3.11.** Both are declared and
  neither is exercised: espeak is covered by a stub binary and a fixture, and the
  container is checked by reading rather than by building.
- **Style references**: seed a new production's look from a past accepted one.
- **Web UI**, if the CLI stops being enough.
