"""Concrete `Objective` implementations for the car game. Car-specific (reads
`CarEnvironment`'s `info` keys directly), unlike the game-agnostic `Objective` Protocol
(Phase 0) — lives in `neuroarena.sim`, not `neuroarena.interfaces`, matching where
`CarEnvironment` itself lives. See
`../../../docs/phases/phase-6-persistence.md`'s "Minimal concrete Objective" section for
why this exists: `neuroarena-train` needs a real `Objective` to build a `Trainer`, and
nothing else in the codebase had built one yet."""

from __future__ import annotations

from typing import Any

import numpy as np


class ProgressObjective:
    """Satisfies `Objective`. Fitness/reward come from `CarEnvironment`'s `progress` info
    field (signed, continuous distance along the track centerline); an episode is flagged
    done via `should_stop()` once `lap_progress` (progress / one lap) reaches 1.0 — this
    is the "one-lap completion" example the Phase 5 doc's pluggable-success-criteria bullet
    already named as composing for free from `lap_progress`, with no new env mechanism."""

    def __init__(self) -> None:
        self._last_progress = 0.0
        self._last_delta = 0.0
        self._cumulative = 0.0
        self._lap_progress = 0.0

    def reset(self) -> None:
        self._last_progress = 0.0
        self._last_delta = 0.0
        self._cumulative = 0.0
        self._lap_progress = 0.0

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        progress = float(info["progress"])
        self._last_delta = progress - self._last_progress
        self._last_progress = progress
        self._cumulative += self._last_delta
        self._lap_progress = float(info["lap_progress"])

    def should_stop(self) -> bool:
        return self._lap_progress >= 1.0

    def step_reward(self) -> float:
        return self._last_delta

    def fitness(self) -> float:
        return self._cumulative
