"""Runs a generation's whole batch of genomes concurrently, round-robin lockstep — one
`env.step()` per still-active genome per round — under a per-generation step budget shared
across the batch. See the Phase 4 doc's "Generations always terminate" requirement: a batch
stops the instant either the step budget runs out, or any genome's episode ends via
`Objective.should_stop()` (task success), at which point every other still-active genome is
force-truncated immediately and scored on partial progress, exactly like a normally-ended
episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuroarena.interfaces.protocols import Environment, Model, Objective


@dataclass
class StepBudget:
    """Mutable counter shared across every genome evaluated within one generation's batch."""

    remaining: int


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    final_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchEntry:
    """One genome's inputs for `evaluate_batch`. Concurrent evaluation needs independent
    per-genome state — its own `Environment` instance and its own `Objective` instance — not
    the single shared instances sequential per-genome evaluation could get away with reusing."""

    genome_id: int
    model: Model
    env: Environment
    objective: Objective
    seed: int


@dataclass
class _ActiveGenome:
    genome_id: int
    model: Model
    env: Environment
    objective: Objective
    observation: Any
    last_info: dict[str, Any]


def evaluate_batch(
    entries: list[BatchEntry], step_budget: StepBudget
) -> dict[int, EvaluationResult]:
    """Steps every entry's episode in round-robin lockstep. Each genome resets first (per
    Phase 0's per-episode `Objective.reset()`/`Model.reset()` hooks), then the batch advances
    one round at a time: every still-active genome takes exactly one `env.step()`, consuming
    one unit of `step_budget` each. A genome that ends on its own (`terminated` or
    `truncated`) simply drops out of the round-robin — it does not end the batch for anyone
    else. The whole batch stops, immediately, the instant either (a) `step_budget` is
    exhausted, or (b) any genome's `Objective.should_stop()` fires (task success) — whichever
    happens first. On trigger (b), a genome processed earlier in `entries` within the same
    round keeps the step it already took that round; one processed later does not get a
    chance to move that round. Every genome still active when the batch stops is
    force-truncated and scored on partial progress via `Objective.fitness()`, exactly like a
    normally-ended episode, with no separate penalty term."""
    active: list[_ActiveGenome] = []
    for entry in entries:
        observation = entry.env.reset(seed=entry.seed)
        entry.model.reset()
        entry.objective.reset()
        active.append(
            _ActiveGenome(entry.genome_id, entry.model, entry.env, entry.objective, observation, {})
        )

    results: dict[int, EvaluationResult] = {}
    while active and step_budget.remaining > 0:
        still_active: list[_ActiveGenome] = []
        stop_batch = False
        for i, genome in enumerate(active):
            if step_budget.remaining <= 0:
                still_active.extend(active[i:])
                break
            action = genome.model.act(genome.observation)
            observation, terminated, truncated, info = genome.env.step(action)
            step_budget.remaining -= 1
            genome.objective.update(observation, action, terminated, truncated, info)
            genome.observation = observation
            genome.last_info = info
            if terminated or truncated:
                results[genome.genome_id] = EvaluationResult(genome.objective.fitness(), info)
                continue
            if genome.objective.should_stop():
                results[genome.genome_id] = EvaluationResult(genome.objective.fitness(), info)
                stop_batch = True
                still_active.extend(active[i + 1 :])
                break
            still_active.append(genome)
        active = still_active
        if stop_batch:
            break

    for genome in active:
        results[genome.genome_id] = EvaluationResult(genome.objective.fitness(), genome.last_info)

    return results


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
    generation's total step count never exceed the ceiling regardless of population size.

    DEPRECATED: Use `evaluate_batch` instead. This function is kept for backward compatibility
    during the Task 1/Task 2 transition."""
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
