"""Quick, runnable demonstration of the Phase 0 interface layer.

This isn't the game (that's Phase 1+) — Phase 0 is a pure-declaration
interface layer with no game and no trainer. This script is a sanity-check
that the pieces Phase 0 built actually compose: a hand-written Model and
Environment satisfy the platform's structural Protocols, `check_compatibility`
accepts a matching pair and rejects a mismatched one, and a `RunConfig`
round-trips through the versioned JSON codec.

Run: `uv run python examples/quickstart.py`
"""

from __future__ import annotations

import numpy as np

from neuroarena.config import RunConfig, dumps, loads
from neuroarena.interfaces import (
    Box,
    Environment,
    IncompatibleDescriptorsError,
    Model,
    Space,
    check_compatibility,
)


class ExampleEnvironment:
    """A 3-observation, 2-action environment that truncates after 5 steps."""

    observation_space: Space = Box(-1.0, 1.0, (3,))
    action_space: Space = Box(-1.0, 1.0, (2,))

    def __init__(self) -> None:
        self._t = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        self._t = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict]:
        self._t += 1
        return np.zeros(3, dtype=np.float32), False, self._t >= 5, {"t": self._t}


class ExampleModel:
    """A policy that always emits zeros, shaped to whatever action_space it's given."""

    def __init__(self, observation_space: Space, action_space: Space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=np.float32)  # type: ignore[union-attr]


def main() -> None:
    env = ExampleEnvironment()
    print(f"Environment satisfies the Environment protocol: {isinstance(env, Environment)}")

    matching_model = ExampleModel(env.observation_space, env.action_space)
    print(f"Model satisfies the Model protocol: {isinstance(matching_model, Model)}")

    check_compatibility(matching_model, env)
    print("check_compatibility(matching model, env): OK, no error raised")

    mismatched_model = ExampleModel(Box(-1.0, 1.0, (7,)), env.action_space)
    try:
        check_compatibility(mismatched_model, env)
    except IncompatibleDescriptorsError as exc:
        print(f"check_compatibility(mismatched model, env): correctly rejected:\n  {exc}")

    config = RunConfig(master_seed=42)
    text = dumps(config)
    restored = loads(text)
    print(f"RunConfig round-trip through JSON: {config == restored} -> {restored!r}")


if __name__ == "__main__":
    main()
