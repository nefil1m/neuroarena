"""Runs one genome's episode in isolation on a shared `Environment`, under a
per-generation step budget that force-truncates it the instant the budget runs out —
see the Phase 4 doc's "Generations always terminate" and "force-truncated ... scored on
partial progress" requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuroarena.interfaces.protocols import Environment, Model, Objective


@dataclass
class StepBudget:
    """Mutable counter shared across every genome evaluated within one generation."""

    remaining: int


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    final_info: dict[str, Any] = field(default_factory=dict)


def evaluate_genome(
    model: Model,
    env: Environment,
    objective: Objective,
    step_budget: StepBudget,
    seed: int,
) -> EvaluationResult:
    """One episode for `model` on `env`, scored by `objective` (reset first, per Phase 0's
    per-episode `Objective.reset()`/`Model.reset()` hooks). Consumes one unit of
    `step_budget` per `env.step()` call; if the budget is already at zero, the genome is
    scored on its freshly-reset state with zero steps taken — this is what lets a whole
    generation's total step count never exceed the ceiling regardless of population size."""
    observation = env.reset(seed=seed)
    model.reset()
    objective.reset()
    info: dict[str, Any] = {}
    while True:
        if step_budget.remaining <= 0:
            break
        action = model.act(observation)
        observation, terminated, truncated, info = env.step(action)
        step_budget.remaining -= 1
        objective.update(observation, action, terminated, truncated, info)
        if terminated or truncated or objective.should_stop():
            break
    return EvaluationResult(fitness=objective.fitness(), final_info=info)
