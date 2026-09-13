"""Wraps one NEAT genome as a `neuroarena.interfaces.protocols.Model`: a feed-forward
network whose output layer is `tanh`-activated (per `build_neat_config`'s template), so
`act()`'s output already lies in `[-1, 1]` with no separate clipping step (Phase 4 doc)."""

from __future__ import annotations

from typing import Any

import neat
import numpy as np

from neuroarena.interfaces.spaces import Space


class GenomeModel:
    """Satisfies `Model`. No hidden state: `reset()` is a no-op since `neat-python`
    feed-forward networks carry none across steps (Phase 0's `Model.reset()` docstring:
    "no-op for feedforward NEAT nets")."""

    def __init__(
        self,
        genome: Any,
        neat_config: neat.Config,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self._network = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        raw = self._network.activate(observation.tolist())
        return np.array(raw, dtype=np.float32)
