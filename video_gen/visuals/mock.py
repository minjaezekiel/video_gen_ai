"""A deterministic fake backend.

This exists because no neural inference can run on the development machine: it
claims every capability, including MOTION and IDENTITY, so the orchestration
around those paths -- frame-sequence handling, identity plumbing, capability
degradation -- is genuinely exercised by the test suite even though no model is
present. It renders flat, obviously synthetic frames on purpose; it must never
be mistaken for real output.
"""

from __future__ import annotations

from typing import List, Optional, Set

from PIL import Image, ImageDraw

from .base import IDENTITY, MOTION, STILL, TRAIN, ShotRequest, ShotVisual


class MockBackend:
    """Claims everything, renders identifiable test frames, never fails."""

    name = "mock"
    priority = 5
    parallel_safe = True

    def __init__(
        self,
        config=None,
        capabilities: Optional[Set[str]] = None,
        is_available: bool = True,
    ) -> None:
        self.config = config
        self._capabilities = set(capabilities or {STILL, IDENTITY, MOTION, TRAIN})
        self._available = is_available
        self.calls: List[ShotRequest] = []  # inspected by tests

    def available(self) -> bool:
        return self._available

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    def describe(self) -> str:
        return f"mock backend for testing (capabilities: {sorted(self._capabilities)})"

    def _frame(self, request: ShotRequest, index: int, total: int) -> Image.Image:
        size = (request.image_size, request.image_size)
        # Colour is a pure function of the seed, so tests can assert determinism.
        base = (
            (request.seed * 37 + index * 11) % 200 + 30,
            (request.seed * 17 + index * 23) % 200 + 30,
            (request.seed * 7 + index * 5) % 200 + 30,
        )
        image = Image.new("RGB", size, base)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(255, 255, 255), width=3)
        label = [
            "MOCK",
            request.shot_type,
            f"{index + 1}/{total}",
            ",".join(c.name for c in request.characters)[:24],
        ]
        draw.multiline_text((10, 10), "\n".join(label), fill=(255, 255, 255))
        return image

    def render(self, request: ShotRequest) -> ShotVisual:
        self.calls.append(request)
        used = {STILL}
        if request.needs_identity and IDENTITY in self._capabilities:
            used.add(IDENTITY)

        if request.want_motion and MOTION in self._capabilities:
            count = max(2, int(request.duration * request.fps))
            frames = [self._frame(request, i, count) for i in range(count)]
            used.add(MOTION)
            return ShotVisual(
                backend=self.name, frames=frames, frame_rate=float(request.fps), used=used
            )
        return ShotVisual(backend=self.name, image=self._frame(request, 0, 1), used=used)
