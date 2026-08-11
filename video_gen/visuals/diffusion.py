"""Stable Diffusion visuals, character identity, and video motion.

    ⚠ UNVERIFIED. None of the code below has ever been executed.

PyTorch publishes no macOS x86_64 wheel beyond Python 3.12, and onnxruntime
publishes none for 3.13 either, so on the development machine `import torch`
cannot succeed and this whole module is dead. It is written against the
documented diffusers APIs and gated behind capability probes so that it stays
inert until a machine can actually load it. Every path here should be treated
as untested until `verify_backends.py` passes somewhere real.

Capability tiers, in the order they degrade:

  TRAIN     LoRA character training                     CUDA, ~8-12 GB VRAM
  MOTION    AnimateDiff / SVD short clips               >= 8 GB VRAM in practice
  IDENTITY  IP-Adapter conditioning on a reference      any device that runs SD
  STILL     plain text-to-image                         any device that runs SD
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

from PIL import Image

from .base import IDENTITY, MOTION, STILL, TRAIN, ShotRequest, ShotVisual

LOGGER = logging.getLogger("video_gen.visuals.diffusion")

#: Motion models, cheapest first. AnimateDiff rides on an SD1.5 checkpoint and
#: is by far the most forgiving; SVD is image-to-video and needs more memory.
MOTION_MODELS = {
    "animatediff": "guoyww/animatediff-motion-adapter-v1-5-2",
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt",
}

#: IP-Adapter gives cross-shot character likeness without any training.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter_sd15.bin"


def _probe():
    """Import torch/diffusers once, returning ``(torch, diffusers)`` or None."""
    try:
        import torch
        import diffusers

        return torch, diffusers
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("diffusion stack unavailable: %s", exc)
        return None


def select_device(torch) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class DiffusionBackend:
    """Text-to-image, identity conditioning and motion, where hardware allows."""

    name = "diffusion"
    priority = 100

    def __init__(self, config=None, enable_motion: bool = True) -> None:
        self.config = config
        self.enable_motion = enable_motion
        self._probe = _probe()
        self._pipe = None
        self._motion_pipe = None
        self._ip_adapter_loaded = False
        self._failed = False
        self.device = select_device(self._probe[0]) if self._probe else "cpu"

    # -- capability -------------------------------------------------------- #

    def available(self) -> bool:
        return self._probe is not None and not self._failed

    def capabilities(self) -> Set[str]:
        if not self.available():
            return set()
        caps = {STILL, IDENTITY}
        # Motion and training are gated on real VRAM, not on the import working:
        # offering them on a 4 GB card produces an OOM mid-render instead of a
        # graceful downgrade to stills.
        if self.enable_motion and self._has_vram(8):
            caps.add(MOTION)
        if self.device == "cuda" and self._has_vram(10):
            caps.add(TRAIN)
        return caps

    def _has_vram(self, gigabytes: float) -> bool:
        if not self._probe:
            return False
        torch, _ = self._probe
        try:
            if self.device == "cuda":
                total = torch.cuda.get_device_properties(0).total_memory
                return total / 1e9 >= gigabytes
            if self.device == "mps":
                # Unified memory: assume usable if the machine is not tiny.
                import psutil  # type: ignore

                return psutil.virtual_memory().total / 1e9 >= gigabytes * 2
        except Exception:  # noqa: BLE001
            return False
        return False

    def describe(self) -> str:
        if self._probe is None:
            return (
                "unavailable: torch/diffusers not importable "
                "(no macOS x86_64 wheel beyond py3.12) — pip install -r requirements-ml.txt"
            )
        return f"Stable Diffusion on {self.device}; capabilities {sorted(self.capabilities())}"

    # -- loading ----------------------------------------------------------- #

    def _load(self):
        if self._pipe is not None or self._failed or not self._probe:
            return self._pipe
        torch, _ = self._probe
        model_id = str((self.config.preset_data["model"] if self.config else "stabilityai/sd-turbo"))
        try:
            from diffusers import AutoPipelineForText2Image

            # fp16 only helps on CUDA: on CPU it is emulated and slower, and MPS
            # needs fp32 for several of the SD ops.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
            ).to(self.device)
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.enable_attention_slicing()
                if self.device == "cpu":
                    pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001 - optional memory optimisations
                pass
            self._pipe = pipe
            LOGGER.info("loaded %s on %s (%s)", model_id, self.device, dtype)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not load %s (%s); backend disabled", model_id, exc)
            self._failed = True
        return self._pipe

    def _ensure_ip_adapter(self, pipe) -> bool:
        """Load IP-Adapter lazily; only shots with a reference image need it."""
        if self._ip_adapter_loaded:
            return True
        try:
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHTS
            )
            self._ip_adapter_loaded = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("IP-Adapter unavailable (%s); characters will drift between shots", exc)
        return self._ip_adapter_loaded

    def _load_motion(self):
        if self._motion_pipe is not None or not self._probe:
            return self._motion_pipe
        torch, _ = self._probe
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            from diffusers.schedulers import DDIMScheduler

            base = str(self.config.preset_data["model"]) if self.config else "runwayml/stable-diffusion-v1-5"
            adapter = MotionAdapter.from_pretrained(MOTION_MODELS["animatediff"])
            pipe = AnimateDiffPipeline.from_pretrained(
                base,
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config, beta_schedule="linear", clip_sample=False
            )
            pipe.set_progress_bar_config(disable=True)
            self._motion_pipe = pipe
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("motion model unavailable (%s); falling back to camera motion", exc)
            self.enable_motion = False
        return self._motion_pipe

    # -- rendering --------------------------------------------------------- #

    @staticmethod
    def _reference_image(request: ShotRequest) -> Optional[Image.Image]:
        """The speaking character's reference art, if any has one."""
        ordered = request.speaking or request.characters
        for character in ordered:
            if character.reference_image and Path(character.reference_image).exists():
                try:
                    with Image.open(character.reference_image) as handle:
                        return handle.convert("RGB").resize((224, 224))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("could not read reference for %s: %s", character.name, exc)
        return None

    def render(self, request: ShotRequest) -> ShotVisual:
        torch, _ = self._probe  # type: ignore[misc]
        pipe = self._load()
        if pipe is None:
            raise RuntimeError("diffusion pipeline failed to load")

        # A per-call generator is reproducible and thread-safe, unlike seeding
        # the global RNG.
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        kwargs = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "width": request.image_size,
            "height": request.image_size,
            "generator": generator,
        }
        # Turbo checkpoints run with CFG disabled, where a negative prompt is not
        # merely useless: the pipeline rejects it.
        if request.guidance > 1.0 and request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt

        used = {STILL}
        reference = self._reference_image(request)
        if reference is not None and self._ensure_ip_adapter(pipe):
            pipe.set_ip_adapter_scale(0.62)
            kwargs["ip_adapter_image"] = reference
            used.add(IDENTITY)

        if request.want_motion and MOTION in self.capabilities():
            motion = self._load_motion()
            if motion is not None:
                try:
                    frames = motion(
                        prompt=request.prompt,
                        num_frames=16,
                        num_inference_steps=max(request.steps, 8),
                        guidance_scale=max(request.guidance, 7.0),
                        generator=generator,
                    ).frames[0]
                    used.add(MOTION)
                    return ShotVisual(
                        backend=self.name, frames=list(frames), frame_rate=8.0, used=used
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("motion render failed (%s); using a still", exc)

        image = pipe(**kwargs).images[0]
        return ShotVisual(backend=self.name, image=image, used=used)

    # -- training ---------------------------------------------------------- #

    def train_character(self, name: str, images: List[Path], output_dir: Path) -> Optional[Path]:
        """Train a character LoRA.

        Not implemented: doing this properly means a real training loop with
        gradient checkpointing and a prior-preservation dataset, and it cannot be
        written blind against hardware that has never run it. The seam exists so
        the CLI and character model already carry a `lora` path.
        """
        raise NotImplementedError(
            "LoRA character training is not implemented yet; it needs a CUDA machine "
            "to develop and verify against. See roadmap.md."
        )
