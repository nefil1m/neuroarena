from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from neuroarena.interfaces.spaces import Space

if TYPE_CHECKING:
    from neuroarena.config import RunConfig

Observation = np.ndarray
Action = np.ndarray


@runtime_checkable
class Environment(Protocol):
    """A game episode. Mirrors Gymnasium's Env contract, minus `reward`."""

    observation_space: Space
    action_space: Space

    def reset(self, *, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, bool, bool, dict[str, Any]]: ...


@runtime_checkable
class Visualizable(Protocol):
    """Optional capability, deliberately separate from `Environment` so no existing
    implementer has to change: an environment that can describe its current state for a
    viewer — a small mapping of named numbers (for the car game: `x`, `y`, `heading`).
    A trainer may read it from another thread while stepping, so an implementation must
    build the mapping from a single consistent read of its own state."""

    def visual_state(self) -> Mapping[str, float]: ...


@runtime_checkable
class Model(Protocol):
    """A policy: observation -> action, plus a per-episode reset hook."""

    observation_space: Space
    action_space: Space

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...


@runtime_checkable
class Objective(Protocol):
    """Defines what the agent is trying to do and how well it did.

    One config-selected component, driven by the trainer. `fitness()` feeds
    evolutionary backends; `step_reward()` feeds RL backends.
    """

    def reset(self) -> None: ...

    def update(
        self,
        observation: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None: ...

    def should_stop(self) -> bool: ...

    def step_reward(self) -> float: ...

    def fitness(self) -> float: ...


@dataclass(frozen=True)
class TrainingUpdate:
    """One unit of training progress. Fields may be added later (versioned)."""

    progress_index: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    population_size: int
    champion_metrics: dict[str, float]
    sim_time: float
    wall_time: float
    schema_version: int = 1


@runtime_checkable
class Trainer(Protocol):
    """A learning backend. Consumes Environment + Model + Objective + RunConfig only."""

    def run(self) -> Iterator[TrainingUpdate]: ...

    def update_config(self, partial: Mapping[str, Any]) -> None: ...

    def save_checkpoint(self, path: Path) -> None: ...

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
    ) -> Trainer: ...
