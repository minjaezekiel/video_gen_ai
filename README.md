# video_gen_ai

Turn a short story into a narrated video, on a CPU, offline.

You write a story. It proposes three different scripts. You pick one. It renders
an MP4 with narration, captions, camera motion and a music bed — and keeps the
script on disk as markdown you can edit and re-render.

```
python video_generator.py
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That is everything needed. No system ffmpeg (it ships with `imageio-ffmpeg`),
no GPU, no model downloads, no API keys.

### Optional: AI visuals and neural narration

```bash
pip install -r requirements-ml.txt   # torch, diffusers, transformers (~3 GB)
```

This swaps the procedural scene cards for Stable Diffusion Turbo and the system
voice for neural TTS. Models download on first use and are then cached locally.

> **Intel Mac:** PyTorch publishes macOS x86_64 wheels only up to torch 2.2.2
> (cp312), so this file needs Python ≤ 3.12 there. Apple Silicon, Linux and
> Windows are unaffected. Without it everything still works — see
> [Fallbacks](#fallbacks).

> **NVIDIA:** install CUDA torch first, then the rest:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Check it works

```bash
python video_generator.py --self-test
```

Runs the full pipeline offline and writes `output/self_test.mp4`.
Takes about 18 seconds on a CPU-only Intel Mac.

---

## Use

### The wizard

```bash
python video_generator.py
```

1. Write your story (finish with a blank line).
2. It generates three candidate scripts — different pacings, not three
   samples of the same thing:

   | variant | scenes | feel |
   |---|---|---|
   | **Cinematic** | 4–5 | wide establishing shots, longer holds |
   | **Fast-cut** | 7–8 | punchy, high scene turnover |
   | **Documentary** | 5–6 | measured, evenly weighted |

3. **You accept one.** `1`/`2`/`3` to pick, `v 2` to read one in full,
   `r` to regenerate.
4. Confirm the filename, and it renders with per-scene progress.

### Non-interactive

```bash
# Whole flow, no prompts
python video_generator.py --story-file story.txt --yes --preset draft

# Skip the accept step and pick a variant directly
python video_generator.py --story "..." --variant Fast-cut

python video_generator.py --list            # every saved script
python video_generator.py --render <id>     # re-render a saved script
```

| flag | meaning |
|---|---|
| `--preset draft\|standard\|quality` | speed/quality tradeoff (default `standard`) |
| `--seed N` | reproducible visuals and palette |
| `--variant NAME` | skip the accept step |
| `--no-music`, `--no-overlays` | drop the music bed / burned-in captions |
| `--llm` | enrich visual prompts with a local LLM (needs `requirements-ml.txt`) |
| `--yes` | accept all defaults; also the default when stdin is not a TTY |

---

## Presets

| preset | model | steps | image | video |
|---|---|---|---|---|
| `draft` | sd-turbo | 1 | 384px | 854×480 @24 |
| `standard` | sd-turbo | 4 | 512px | 1280×720 @24 |
| `quality` | SD 1.5 | 25 | 768px | 1920×1080 @30 |

**Timings.** The procedural path renders a 4-scene, 15-second 720p video in
about **18 seconds** on a CPU-only Intel Mac — faster than real time. With
Stable Diffusion the render is dominated by image generation: budget roughly
10–15 s per image per step on a CPU, seconds total on a GPU. `quality` on a CPU
takes minutes per image; use a GPU for it.

Generated images are cached in `.cache/images/`, keyed by prompt, preset and
seed. Edit one scene's `**Visual:**` line and re-render, and only that scene
regenerates.

---

## Scripts on disk

Every script is a markdown file in `video_scripts/`, with a regenerated
`INDEX.md` listing them all.

```markdown
---
id: 2026-08-10-the-lighthouse-keeper-a3f1
title: The Lighthouse Keeper
variant: Cinematic
status: rendered            # candidate -> accepted | rejected -> rendered
config: {preset: standard, seed: 1234}
renders:
  - output: output/lighthouse.mp4
    duration_s: 14.88
    elapsed_s: 18.5
    per_scene_s: [1.71, 1.45, 1.51, 1.49]
---

## Scene 1 — 7.1s

**Visual:** a weather-beaten lighthouse at dusk, cold blue light, film grain

**Narration:** The keeper had not seen a ship in forty years.
```

Edit them by hand, re-render with `--render <id>`, and every render appends its
real timings. Rejected candidates are kept, not deleted.

---

## Fallbacks

Every stage degrades instead of failing, so a video always comes out. Fidelity
changes; functionality does not.

| stage | with models | without |
|---|---|---|
| visuals | Stable Diffusion Turbo | composed scene cards (graded sky, horizon, grain) |
| narration | `facebook/mms-tts-eng` | `say` (macOS) / `espeak-ng` (Linux) → timed silence |
| visual prompts | Qwen2.5-0.5B | deterministic keyword extraction |
| music | — | synthesised chord pad |

The wizard prints which backends are active at startup.

---

## Tests

```bash
pytest test.py            # 63 offline unit tests, ~5s
pytest test.py -m slow    # end-to-end renders
pytest test.py -m ''      # everything
```

---

## How it works

Story → sentence split → contiguous beats → one visual prompt and one narration
line per beat. Each scene generates **one** still, animated by a slow crop-and-zoom
(Ken Burns) rather than by more model calls. Narration is synthesised first and
its measured length sets the scene length, so audio and video cannot drift. The
whole video is assembled as a single clip with hand-rolled crossfades — going
through moviepy's compositor instead costs 4× the wall clock.

**What this is not:** there is no video diffusion here. "Animation" means Ken
Burns motion and crossfades over generated stills. Real motion models
(AnimateDiff, SVD) are not viable on a CPU-only machine.

[decision.md](decision.md) — why each component is built the way it is, plus the
audit of the 16 bugs fixed during the rebuild.
[roadmap.md](roadmap.md) — what is done and what is next.

## Requirements

Python 3.9+. Any OS. ~200 MB installed, or ~3 GB with the optional model stack.

## License

See [LICENSE](LICENSE).
