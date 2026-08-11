# Engineering decisions

The record of what was broken, what was rebuilt, and why. `roadmap.md` covers
what comes next.

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
