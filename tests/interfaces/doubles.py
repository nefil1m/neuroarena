from __future__ import annotations

from typing import Any

import numpy as np

from neuroarena.interfaces.spaces import Box, Space


class DummyEnvironment:
    """Minimal Environment double: 3-float obs, 2-float action, truncates after 5 steps."""

    observation_space: Space = Box(-1.0, 1.0, (3,))
    action_space: Space = Box(-1.0, 1.0, (2,))

    def __init__(self) -> None:
        self._t = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        self._t = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        self._t += 1
        return np.zeros(3, dtype=np.float32), False, self._t >= 5, {"t": self._t}


class DummyModel:
    """Minimal Model double: always emits zeros of the action shape."""

    def __init__(self, observation_space: Space, action_space: Space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        shape = self.action_space.shape  # type: ignore[union-attr]
        return np.zeros(shape, dtype=np.float32)


class DummyObjective:
    """Minimal Objective double: fitness = number of `update()` calls (steps survived).
    Never signals should_stop() — episodes only end via the env's own terminated/truncated."""

    def __init__(self) -> None:
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._steps += 1

    def should_stop(self) -> bool:
        return False

    def step_reward(self) -> float:
        return 1.0

    def fitness(self) -> float:
        return float(self._steps)
