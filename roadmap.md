# Roadmap

Status of the rebuild, and what comes next. Rationale for everything in
Phases 0–5 is in [decision.md](decision.md).

---

## Phase 0 — Installable ✅

- [x] Drop the `+cu118` pins that made `pip install` fail on every non-NVIDIA machine
- [x] Split the model stack into `requirements-ml.txt`; core install needs no torch
- [x] Migrate moviepy 1.x → 2.x (`moviepy.editor` removed, `.set_*` → `.with_*`, effects API)
- [x] Bundle ffmpeg via `imageio-ffmpeg` — no system ffmpeg required
- [x] `certifi` + `SSL_CERT_FILE` so HTTPS works on the macOS python.org build
- [x] Device selection `cuda → mps → cpu`, fp16 gated to CUDA
- [x] Verified: clean venv install succeeds on Intel Mac / Python 3.13

## Phase 1 — Pipeline correctness ✅

- [x] Write output to `output/`, not into the temp dir that cleanup deletes
- [x] Sequence narration per scene instead of starting every clip at t=0
- [x] Make measured audio duration the source of truth for scene length
- [x] Fix scene parsing (narration was always empty)
- [x] Fix TTS: real checkpoint, real sample rate, no spoken `[female voice]` tag
- [x] Idempotent recursive cleanup; drop the `__del__`/`atexit` double-teardown
- [x] CRF-only encoding (was fighting a contradictory `bitrate=`)
- [x] Model-load failures log loudly and degrade visibly instead of `except: pass` → black frames
- [x] Full audit table — 16 bugs — in [decision.md §1.1](decision.md)

## Phase 2 — Speed ✅

- [x] One diffusion call per scene; delete keyframe generation + pixel interpolation
- [x] Ken Burns crop/zoom animation with smoothstep easing
- [x] Stop building one `ImageClip` per frame
- [x] Assemble the whole video as a **single** `VideoClip` with hand-rolled crossfades
      (moviepy's `method="compose"` cost 4×) — **42.8 s → 18.5 s**
- [x] SD-Turbo preset table with CFG disabled on turbo checkpoints
- [x] Per-call generator seeding; `md5` seeds instead of `PYTHONHASHSEED`-dependent `hash()`
- [x] Persistent image cache outside the temp dir
- [x] Remove the thread pool around the (not thread-safe) diffusion pipeline

## Phase 3 — Script store ✅

- [x] `video_scripts/<id>.md`, YAML frontmatter + markdown scenes, round-trippable
- [x] `ScriptStore.save/load/list/accept/record_render`
- [x] Lifecycle `candidate → accepted | rejected → rendered`
- [x] Render history appended with measured timings and active backends
- [x] `INDEX.md` regenerated on every save
- [x] Re-render saved scripts (`--render <id>`); hand-edits respected

## Phase 4 — Story → candidates → accept → render ✅

- [x] Deterministic beat segmentation backbone (instant, offline, always well-formed)
- [x] Three candidate treatments varying pacing, not sampling temperature
- [x] Optional LLM prompt enrichment, strictly off the critical path
- [x] Interactive wizard with the accept step; `v N` to view a candidate in full, `r` to regenerate
- [x] Non-interactive/TTY-safe prompts (`--yes`) so the same paths work in CI

## Phase 5 — Tests & docs ✅

- [x] `test.py` → 63 pytest tests, fully offline, 4.8 s
- [x] `--self-test`: full pipeline, zero downloads, primary regression gate
- [x] Regression tests for each fixed bug class (parser, cleanup, cache, A/V drift, seeds)
- [x] `decision.md`, `roadmap.md`, rewritten `README.md`, `.gitignore`

---

## Not done — known gaps

**Measure SD-Turbo on real hardware.** The preset timings in
[decision.md §2.4](decision.md) are estimates. PyTorch has no macOS x86_64 wheel
for Python 3.13, so the model path cannot run on this machine at all
([§3.1](decision.md)). Verifying it needs Python ≤ 3.12 here, or any Linux /
Apple Silicon / Windows box. The procedural path is fully measured.

**Tune `standard` after measuring.** If a 5-scene `standard` render exceeds
~5 minutes on a typical CPU, drop the image size to 448 px and re-measure.

---

## Later

### Visual quality
- **LCM / SDXL-Turbo** as an additional preset once base timings are measured.
- **AnimateDiff or SVD** for genuine motion. Not viable on a 16 GB CPU host —
  gate behind a `--preset motion` that requires CUDA and says so.
- **img2img scene continuity**: seed each scene from the previous one at low
  strength so a story keeps a consistent look. Cheap, and a large perceived win.
- **Character consistency** via a fixed character description prefix plus a
  locked seed per named character.

### Audio
- **MusicGen-small** for real background music, cached per mood. ~2 GB, minutes
  per clip on CPU — optional, off by default.
- **Ducking**: drop the music bed a few dB under narration instead of using a
  fixed low volume.
- **Word-level timing** from the TTS to drive per-line caption timing.

### Output
- **Subtitle sidecar** (`.srt`) alongside the burned-in captions — scene
  timings are already computed.
- **Aspect presets**: 9:16 and 1:1 for social, with the caption box and Ken
  Burns framing adapted per aspect.
- **Batch mode**: render every `accepted` script in `video_scripts/`.
- **Resume**: skip scenes whose cached image and audio are both current.

### Workflow
- **Style references**: let the wizard seed a new script's look from a past
  accepted script — the store already keeps everything needed.
- **`--edit`**: open the accepted script in `$EDITOR` before rendering.
- **Web UI**, if the CLI ever stops being enough.
