# Engineering decisions

The record of what was broken, what was rebuilt, and why. `roadmap.md` covers
what comes next; `IMPLEMENTATION.md` is the current build state.

This document has two parts:

- **[Part I — v1](#part-i--v1-repairing-the-slideshow-generator)**: repairing a
  project that could not run at all, and making it fast.
- **[Part II — v2](#part-ii--v2-from-slideshow-to-film-engine)**: turning it into
  a film engine with a cast, shots, dialogue and per-character voices.

---

# Part I — v1: repairing the slideshow generator

---

## 1. Starting state: why nothing ran

The project did not fail for one reason. It failed for several independent
ones, each individually fatal, which is why "it doesn't work at all" was
literally true rather than an exaggeration.

| # | Problem | Effect |
|---|---------|--------|
| 1 | `requirements.txt` pinned `torch==2.7.1+cu118`, `torchaudio`/`torchvision` `+cu118` | No CUDA wheels exist for macOS. `pip install` failed outright, so **nothing was installed** |
| 2 | `moviepy==1.0.3` on Python 3.13 | `moviepy.editor` was removed in moviepy 2.x, and 1.0.3 does not build on 3.13. Every video import in the file was against the deleted 1.x API |
| 3 | Output written to `self.temp_dir` | `cleanup()` — called from `atexit`, `__del__`, *and* explicitly by `test.py` — deleted the directory. **The finished MP4 was destroyed on every run** |
| 4 | 5 diffusion calls per scene at 768px/25 steps | Hours per video on CPU |

Even with those fixed, the pipeline could not have produced correct output.
The audit below lists what was found and what replaced it.

### 1.1 Correctness bugs found and fixed

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | `_render_final_video` | Output MP4 written inside `temp_dir`, deleted on exit | Writes to `output/`; `temp_dir` holds intermediate audio only |
| 2 | `_render_final_video` | `CompositeAudioClip(...)` starts every clip at t=0 — **all scenes narrate simultaneously** | Each narration pinned to its own scene's start time |
| 3 | `process_single_scene` | `speedx` changed audio length *after* video length was computed → cumulative A/V drift | Measured audio duration is now the source of truth for scene length |
| 4 | `_load_image_model` | `safety_checker=None if not X else None` — both branches identical | Removed; the flag was a no-op |
| 5 | `_load_image_model` | `device_map="auto"` (invalid for diffusers), `variant="fp16"` on repos without it, `enable_model_cpu_offload()` after `.to(device)` — all swallowed by bare `except: pass` | Correct load path; failures log loudly and fall back to a *visible* result, never a black frame |
| 6 | `_load_tts_model` | `facebook/vits-tts` is not a real HF repo — always threw | Single known-good checkpoint, `facebook/mms-tts-eng` |
| 7 | `generate_speech` | Prepended a literal `[female voice]` — the TTS **spoke the words "female voice"** | Removed; voice shaping via rate |
| 8 | `generate_speech` | Hardcoded 16 kHz output | Reads `model.config.sampling_rate` |
| 9 | `_parse_script` | Regex `(.*?)\s*(.*?)(?=...)` — the second group always matched empty, so **narration was always `""`** | Explicit per-field parsing, covered by tests |
| 10 | `cleanup` | Non-recursive `os.listdir` + `os.rmdir`, failed once a subdirectory existed | `shutil.rmtree`, idempotent |
| 11 | `__del__` + `atexit` | Both called `cleanup()` — double-teardown race | `atexit` only, guarded by a `_cleaned` flag |
| 12 | `_render_final_video` | `bitrate="8000k"` **and** `-crf 18` — contradictory; x264 ignores one | CRF only, per preset |
| 13 | imports | Duplicate `AutoTokenizer`; unused `Queue`, `Thread`, `fadein`, `fadeout`, `resample` | Removed |
| 14 | `generate_image` | Seeds derived from `hash(prompt)`, which varies per process with `PYTHONHASHSEED` | `md5`-derived seeds — reproducible across runs |
| 15 | `generate_image` | Disk cache lived in `temp_dir`, wiped on exit — **it had never once been a cache** | Persistent `.cache/images/` |
| 16 | `_pipeline_processing` | `ThreadPoolExecutor` fan-out over one `StableDiffusionPipeline`, which is **not thread-safe** | Removed; the CPU is saturated by one pipeline anyway |

---

## 2. Speed

Target: fast enough to be usable on a CPU-only machine, offline.

### 2.1 One diffusion call per scene, not five

The original generated 5 keyframes per scene and interpolated between them with
`scipy.interp1d` over flattened 768×768 pixel arrays. A pixel-space lerp between
two independently sampled diffusion images is not motion — it is a cross-dissolve
between two unrelated pictures, at 5× the inference cost.

**Replaced with a Ken Burns move:** one still per scene, animated by slowly
translating and zooming a crop window across a slightly oversampled image
(`zoom = 1.18`). Cost is a crop and a bilinear resize per frame — about 20 ms at
720p — instead of another model call. It also reads as real camera motion.

Easing is a smoothstep (`u²(3−2u)`) so scenes do not visibly jerk at their
boundaries.

### 2.2 Not building one clip object per frame

`create_video` built `[ImageClip(f).set_duration(1/24) for f in frames]` — over
700 objects for a 30-second video — then concatenated them. The Ken Burns
`VideoClip` replaces the whole construct.

### 2.3 Assembling the video as a single clip

This was the biggest measured win, and it was not the diffusion model.

With scenes as separate clips joined by `concatenate_videoclips(method="compose")`,
moviepy drives its mask-blitting compositor for **every** frame, not only the
frames inside a crossfade. Measured on a 16-second 720p render:

| assembly | wall clock |
|---|---|
| `method="chain"` (no crossfades) | 8.5 s |
| `method="compose"` (crossfades) | 34.3 s |

x264 was *not* the bottleneck — `medium`, `faster` and `veryfast` all rendered
in 41 s, a 0.1 s spread — so the cost was entirely moviepy's compositing layer.

**`build_timeline()` therefore emits one `VideoClip` for the whole video** and
does the dissolve arithmetic itself: it finds the active scene at time `t`, and
only when `t` falls inside an overlap does it render the previous scene too and
blend. Crossfade cost is now proportional to the overlap rather than to the
whole runtime.

End-to-end self-test (4 scenes, 14.9 s of 720p video, no models):

| version | wall clock |
|---|---|
| composited timeline | 42.8 s |
| single-clip timeline | **18.5 s** |

That is faster than real time on an Intel CPU with no GPU.

### 2.4 Preset table

Turbo checkpoints are distilled to run in 1–4 steps with **classifier-free
guidance disabled**. Passing the conventional `guidance_scale=7.5` and a
negative prompt — as the original always did — produces washed-out garbage on
them, and the pipeline rejects a negative prompt at CFG 0. Guidance and negative
prompts are therefore per-preset, not global.

| preset | model | steps | CFG | image | video | CRF |
|---|---|---|---|---|---|---|
| `draft` | `stabilityai/sd-turbo` | 1 | 0.0 | 384 | 854×480 @24 | 23 |
| `standard` *(default)* | `stabilityai/sd-turbo` | 4 | 0.0 | 512 | 1280×720 @24 | 20 |
| `quality` | `stable-diffusion-v1-5` | 25 | 7.5 | 768 | 1920×1080 @30 | 18 |

Other speed and correctness measures:

- Per-call `torch.Generator(device).manual_seed(seed)` instead of global
  `torch.manual_seed()` — the global is both racy and non-reproducible.
- `torch.set_num_threads(os.cpu_count())` on the CPU path.
- fp16 gated to CUDA only. On CPU it is emulated and *slower*; MPS needs fp32
  for several SD ops.
- Persistent image cache keyed on `(prompt, preset, seed, size)`, so re-rendering
  an edited script only re-generates the scenes that actually changed.

---

## 3. Running offline, and degrading instead of failing

Every stage has a zero-download fallback, so the pipeline always produces a
complete video. Fidelity varies; functionality does not.

| stage | preferred | fallback |
|---|---|---|
| visuals | SD-Turbo via diffusers | procedurally composed scene cards |
| narration | `facebook/mms-tts-eng` (VITS) | OS synthesiser (`say` on macOS, `espeak-ng` on Linux) → timed silence |
| script prompts | Qwen2.5-0.5B-Instruct | deterministic beat segmentation |
| music | — | synthesised chord pad |

This is not merely defensive. It is what makes `--self-test` a real regression
gate: it exercises the entire scene → audio → animation → mux path with **no
network access at all**, in under 20 seconds, which is fast enough to run on
every change.

**The scene cards are designed, not placeholders.** A graded sky, a low sun near
the horizon, layered atmospheric ridges, grain and a vignette. All scenes in one
video share a palette seed and vary the hue only slightly — the first version
drew an independent hue per scene, and consecutive cuts jumping purple → green →
purple read as a bug rather than as a look.

### 3.1 The torch situation on this machine

PyTorch publishes macOS **x86_64** wheels only up to torch 2.2.2 (cp312). This
machine has Python 3.13 and 3.14 only, so the intersection is empty: **no
version of PyTorch can be installed here**, which is why `pip index versions
torch` reports `from versions: none`.

Consequently:

- The model stack moved to a separate `requirements-ml.txt`. `requirements.txt`
  alone is sufficient to run everything and installs cleanly on this machine.
- `torch`, `diffusers` and `transformers` are optional imports behind capability
  flags (`HAS_TORCH`, `HAS_DIFFUSERS`, `HAS_TRANSFORMERS`).
- **The measurements in this document are from the procedural path.** The
  SD-Turbo numbers in §2.4 have *not* been measured on this hardware and remain
  estimates. On an Intel Mac, verifying them needs Python ≤ 3.12.

`certifi` is a hard dependency and `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` are set
at import: the macOS python.org build ships no root certificate store, so every
HTTPS call — including Hugging Face downloads — otherwise fails with
`CERTIFICATE_VERIFY_FAILED`.

---

## 4. Script generation: deterministic first

The original asked FLAN-T5 to emit `Scene N: [visual] | [narration]`. Small
seq2seq models do not reliably produce structured formats, and the parser that
consumed the output was itself broken (bug #9), so failure was silent.

**The backbone is deterministic** and always runs:

1. Split the story into sentences; break run-on sentences on clause boundaries
   so one long sentence cannot collapse into a single scene.
2. Group sentences into N contiguous beats, evenly. Nothing is dropped or
   reordered — a property that is unit-tested.
3. Derive a visual prompt per beat from weighted keywords (proper nouns counted
   double) plus the variant's cinematography suffix.
4. Estimate duration from word count at the variant's words-per-minute.

This is instant, always well-formed, and needs no model. **An optional LLM pass
only rewrites the visual prompts**; if the model is absent, slow, or emits
something degenerate, the deterministic text stands. Keeping the model strictly
off the critical path is what makes generation both fast and offline-safe.

### 4.1 Candidates vary pacing, not sampling temperature

Three samples from one small model give three near-identical outputs, which
makes the accept step meaningless. The three candidates are therefore genuinely
different *treatments* of the same story:

| variant | scenes | wpm | feel |
|---|---|---|---|
| Cinematic | 4–5 | 130 | wide establishing shots, longer holds |
| Fast-cut | 7–8 | 165 | punchy, high scene turnover |
| Documentary | 5–6 | 145 | measured, evenly weighted |

---

## 5. Script store

Scripts live in `video_scripts/<id>.md` as markdown with YAML frontmatter —
durable, diffable, human-editable, and directly readable by the AI on a later
run.

```markdown
---
id: 2026-08-10-the-lighthouse-keeper-a3f1
title: The Lighthouse Keeper
variant: Cinematic
status: candidate          # candidate -> accepted | rejected -> rendered
created: '2026-08-10T18:02:11'
source_story: |
  The keeper had not seen a ship in forty years...
config: {preset: standard, seed: 1234, batch: 20260810-180211}
renders:
  - output: output/lighthouse.mp4
    at: '2026-08-10T18:14:02'
    duration_s: 14.88
    elapsed_s: 18.5
    per_scene_s: [1.71, 1.45, 1.51, 1.49]
    backends: {torch: false, diffusers: false, system_tts: true}
---

## Scene 1 — 7.1s

**Visual:** lighthouse, keeper, ship, cinematic wide shot, film grain
**Narration:** The keeper had not seen a ship in forty years.
```

- `status` tracks the lifecycle. Accepting one candidate marks its batch
  siblings `rejected` rather than deleting them — rejected treatments are still
  useful references.
- Every render **appends** to `renders:` with real measured timings and the
  backends that were actually active. This is the "keep track of scripts and
  details" requirement, and it doubles as a benchmark log.
- `video_scripts/INDEX.md` is regenerated on every save, one row per script, so
  the whole history can be scanned without opening files.
- `duration_s` records the *encoded* length, which is shorter than the sum of
  scene durations by one crossfade per cut.

Scripts are re-renderable (`--render <id>`) and hand-editable: change a
`**Visual:**` line, re-render, and the image cache regenerates only that scene.

---

## 6. Interface

A single interactive wizard (`python video_generator.py`) walks the whole flow:
story → 3 candidates → **accept one** → confirm → render, with per-scene
progress. Prompts go through an `_ask()` helper that falls back to defaults when
`--yes` is passed or stdin is not a TTY, so the same code path is scriptable and
CI-safe.

Flags cover the non-interactive cases: `--self-test`, `--list`, `--render <id>`,
`--variant` (skips the accept step), `--preset`, `--seed`, `--no-music`,
`--no-overlays`, `--llm`.

## 7. Structure

The engine stays in **one file**, as decided. At ~1,750 lines it is navigable
via section banners, and a single file keeps the tool copy-pasteable. The
internal seams (`ScriptStore`, `StoryScripter`, `GenerationConfig`,
`FreeTextToVideoGenerator`) are drawn where a package split would put them, so
splitting later is mechanical if it is ever warranted.

## 8. What this deliberately does not do

Real video-diffusion models (AnimateDiff, SVD) are out of scope: they are not
viable on a 16 GB CPU-only host. **"Animation" here means Ken Burns motion and
crossfades over generated stills**, and this document says so plainly rather
than repeating the original README's overstated claims. Background music is
synthesised, not generated. `roadmap.md` lists both as later options.

---

## 9. Measured results

Intel Mac, 16 GB, CPU-only, Python 3.13, no models installed.

| measurement | result |
|---|---|
| `pip install -r requirements.txt` | succeeds (previously failed outright) |
| self-test, 4 scenes → 14.9 s of 720p | **18.5 s** wall clock |
| per scene (narration + visual + animation) | 1.4–1.7 s |
| 5-scene story at `--preset draft`, 480p | 11.6 s |
| unit tests (63, offline) | 4.8 s |
| output size, 14.9 s @ 720p CRF 20 | ~1.5 MB |

Verified end to end: the MP4 survives cleanup; narration is sequential, not
overlapping; the encoded duration matches the recorded duration exactly; and
`video_scripts/<id>.md` gains a `renders:` entry with `status: rendered`.

SD-Turbo timings remain unmeasured on this hardware for the reason in §3.1.

---

# Part II — v2: from slideshow to film engine

The v1 engine produced narrated slideshows: one still per scene, one narrator,
captions, a music bed. The actual goal is a film — characters that act, a
distinct voice each, in 2D / 3D / realistic styles, for ads, short films and
educational content, entirely offline.

The gap was never the image model. It was that there was no notion of a
**production**: no cast, no shots, no dialogue, no interview. A `Scene` held a
`visual` string and a `narration` string, which cannot express two people
talking.

## 10. The platform wall, and what it forced

Verified, not assumed:

- **torch** publishes no macOS x86_64 wheel beyond Python 3.12 (last: 2.2.2, cp312).
- **onnxruntime** publishes none for cp313 either — which also kills Piper and
  Kokoro, the obvious torch-free neural TTS escape hatches.
- `diffusers`, `peft`, `transformers`, `insightface` are all pure-Python and
  install fine, but every one needs torch or onnxruntime underneath.

This machine runs Python 3.13/3.14 only, so **no neural inference of any kind can
execute here**. That is not a temporary inconvenience to code around; it is a
constraint that shapes the architecture:

1. All visual generation sits behind **one Protocol** (`VisualBackend`), so the
   pipeline never branches on hardware.
2. A deterministic **`MockBackend`** claims every capability and renders obviously
   synthetic frames, so the identity, motion and degradation paths are exercised
   by the test suite even though no model is present.
3. `--doctor` reports what a machine can actually do, and says *why* when it
   cannot.
4. `verify_backends.py` exists to give the diffusion code its first real run on
   hardware that can load it.

**What this honestly means:** the diffusion tier is written, reviewed and typed,
and has never been executed. See §16.

### 10.1 The 177 voices that make this work anyway

macOS ships 177 offline voices via `say`, of which **26 are castable** for
English — 11 female, 11 male, across US, GB, AU, IE, IN and ZA accents, plus
child (`Junior`) and elder (`Grandma`/`Grandpa`) voices. Per-character voice, the
single most important part of "a voice for each character", is therefore fully
solvable on this machine today, with no model and no download.

## 11. Structure: package, not one file

The v1 "keep one file" decision was right at 1,750 lines and stopped being right
here. The engine is now `video_gen/`, with `video_generator.py` kept as a
re-export shim so existing imports and `python video_generator.py` still work.
The seams (`model`, `store`, `casting`, `screenwriter`, `render`, `visuals/`) sit
where a package split would naturally put them.

## 12. Data model

```
Production  meta (format, style, aspect, tone, audience, target length)
  ├ cast    Character[]  name, role, description, appearance,
  │                      VoiceProfile, reference_image?, identity_ref?, lora?
  └ scenes  Scene[]      setting, time_of_day, mood
      └ shots  Shot[]    type, subjects[], action, camera, lines[]
          └ Line         character, text, delivery
```

Persisted as markdown + YAML frontmatter, round-trip byte-stable (a test asserts
it). Cast members also live in `characters/<id>.md` beside their reference art,
so a character is reusable across productions and hand-editable. `merge_into`
fills gaps from the store but **never overrides** what a script states — a stored
character must not silently rewrite the script in front of it.

## 13. Screenwriting, deterministically

Three input shapes, one output:

| input | detection | handling |
|---|---|---|
| screenplay | `NAME:` lines and/or sluglines | parsed directly; sluglines set scene, setting and time of day |
| prose story | neither | sentences → beats; quoted speech → dialogue |
| brief | brief preamble regex | instruction stripped; reported speech → dialogue |

Notable choices:

- **A name is a word that is never written lower case.** Counting occurrences
  loses the protagonist of a short story, who is often named exactly once
  ("Maya had run the night shift..."). Case consistency finds her; "Banks" and
  "Rain" are excluded because they appear lower case elsewhere or are common nouns.
- **Adjacent capitalised words are an entity, not two people.** "Kestrel Coffee"
  was being cast as two speaking characters with their own voices.
- **Pronoun attribution is resolved to the nearest preceding cast member.**
  `"We are out of time," she said` was previously credited to the wrong
  character — worse than leaving it unattributed.
- **Attribution debris is stripped from narration**, so the narrator does not
  also say "she said, watching the trace climb".
- **A brief is not content.** The instruction sentence and trailing style notes
  ("Warm and friendly, anime style.") are removed, and *reported* speech is
  converted into actual dialogue: "Ama tells Dev that the first cup is on the
  house" becomes `AMA: The first cup is on the house.`
- **One speaker per shot** (`_split_by_speaker`), because two people talking in a
  single shot means the camera can never cut between them and the staging cannot
  show who is speaking.

An optional LLM still only rewrites *visual prompts* and stays off the critical
path. It does not write dialogue — see §17.

## 14. Casting

`discover_voices()` enumerates the host pool at runtime and annotates it. Traits
come from the description first, then from a ~130-name given-name table. The
table exists because a screenplay often supplies nothing but `MAYA:` — and
casting Maya with a male voice is a glaring, immediately audible error, which is
exactly what happened before it was added.

Distinctness is enforced twice: the best-fitting *unused* voice is preferred, and
per-character rate (±10%) and pitch (±4 semitones) are applied on top, so even a
reused voice does not sound like the same performer. Casting is deterministic,
independent of the order characters were discovered in, never overrides a voice
already set by hand, and is written back to the character file so it cannot drift
between renders.

## 15. Audio and picture

- **Measured audio owns the clock.** A shot lasts as long as its dialogue
  actually takes to speak. This is the v1 A/V-drift fix, preserved.
- **Dialogue is pinned to its own shot's start** on the crossfaded timeline.
  Concatenating it instead drifts by one fade per cut.
- **Ducking, not a fixed low volume.** A gain envelope drops the bed −11 dB
  across each speech window and is smoothed so the level glides. A fixed level
  either buries speech or makes the music inaudible. Measured: 0.28× under
  dialogue, 1.00× between lines.
- **Score follows scene mood**, with cues crossfaded so a mood change is a
  transition rather than a cut.
- **Captions carry a speaker label in that character's colour**, which is what
  makes a two-hander readable.
- **Pitch shifting resamples** and lets duration change, because duration is
  measured afterwards. Preserving it with a phase vocoder added artefacts for no
  benefit; the first attempt was also inverted (raising pitch lowered it) and is
  now covered by an FFT test asserting the shifted peak within 2%.
- **The whole film is still one `VideoClip`** with hand-rolled crossfades — the
  v1 finding that `concatenate_videoclips(method="compose")` costs 4× still holds.

## 16. Unverified

Everything in `video_gen/visuals/diffusion.py`: model loading, SD-Turbo and SD1.5
stills, IP-Adapter identity conditioning, AnimateDiff motion, and every timing
estimate for those paths. It has never been executed (§10). `train_character`
deliberately raises `NotImplementedError` — a real training loop cannot be
written blind against hardware that has never run it.

`MockBackend` covers the orchestration around all of it. `verify_backends.py`
covers the rest, on a machine that can.

## 17. Honest limits

- **Frontier video quality is not reproducible offline.** Seedance, Sora, Veo and
  Kling are datacenter models. The consumer-runnable open equivalents — Wan 2.x,
  LTX-Video, CogVideoX, SVD, AnimateDiff — give roughly 2–5 second clips at
  480–720p, and multi-shot character consistency remains an open research
  problem rather than a setting.
- **"2D / 3D / realistic" are style tiers, not three engines** — the same
  pipeline with different checkpoints and prompt conditioning. There is no 3D
  geometry anywhere; Blender-class rendering is not viable on a CPU host.
- **A brief cannot become a full script deterministically.** The engine
  structures and stages what it is given, and lifts reported speech into
  dialogue, but it cannot invent 20 seconds of ad copy from one sentence. That
  needs an LLM writing dialogue, which is on the roadmap and not built.
- **The procedural tier is stylised, not photoreal**, by design: geometric
  figures, no lip sync, no limb animation.

## 18. Measured results (v2)

Intel Mac, 16 GB, CPU-only, Python 3.13, procedural backend, no models.

| measurement | result |
|---|---|
| `--self-test`: 2 characters, 4 shots → 9.7 s of 720p | **15.1 s** wall clock |
| vertical ad, 9:16 720×1280 | 15.5 s |
| castable voices discovered | 26 (11F / 11M, 6 accents) |
| ducking depth under dialogue | 0.28× (−11 dB), 1.00× between lines |
| test suite (124, offline) | ~7 s fast, ~25 s with end-to-end renders |

Verified by watching the output: two characters with audibly different voices,
dialogue sequential and in sync to the last shot, speaker-labelled captions in
each character's colour, consistent palette across cuts, shot framing varying
from establishing through close, and 9:16 rendering at a correct 720×1280.
