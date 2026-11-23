"""A music backend that claims everything, so the hosted path is testable.

The hosted tier cannot be exercised in the test suite without a paid key and a
network, which would make the suite neither offline nor free. This stands in
for it: same Protocol, same capabilities, deterministic samples.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

from .base import ALL_CAPABILITIES, ScoreRequest, ScoreResult


class MockMusic:
    """Deterministic tone at a fixed rate, with every capability advertised."""

    name = "mock"
    priority = 99

    def __init__(
        self,
        config=None,
        sample_rate: int = 16000,
        fail: bool = False,
        empty: bool = False,
        duration: Optional[float] = None,
    ) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.fail = fail
        self.empty = empty
        #: Deliberately unrelated to the request, so `fitted` is exercised.
        self.duration = duration
        self.calls = 0
        self.requests: list = []

    def available(self) -> bool:
        return True

    def capabilities(self) -> Set[str]:
        return set(ALL_CAPABILITIES)

    def render(self, request: ScoreRequest) -> ScoreResult:
        self.calls += 1
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("mock music backend was told to fail")
        if self.empty:
            return ScoreResult(
                backend=self.name,
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self.sample_rate,
            )
        seconds = self.duration if self.duration is not None else max(0.1, request.total_duration)
        count = max(1, int(seconds * self.sample_rate))
        t = np.arange(count, dtype=np.float32) / self.sample_rate
        samples = 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        return ScoreResult(
            backend=self.name,
            samples=samples,
            sample_rate=self.sample_rate,
            detail=f"mock {seconds:.1f}s at {self.sample_rate}Hz",
            used=set(ALL_CAPABILITIES),
        )

    def describe(self) -> str:
        return "mock music backend (tests only)"
