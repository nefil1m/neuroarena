"""Runs a generation's whole batch of genomes concurrently, round-robin lockstep — one
`env.step()` per still-active genome per round — under a per-generation step budget shared
across the batch. See the Phase 4 doc's "Generations always terminate" requirement: a batch
stops the instant either the step budget runs out, or any genome's episode ends via
`Objective.should_stop()` (task success), at which point every other still-active genome is
force-truncated immediately and scored on partial progress, exactly like a normally-ended
episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from neuroarena.interfaces.protocols import Environment, Model, Objective


@dataclass
class StepBudget:
    """Mutable counter shared across every genome evaluated within one generation's batch."""

    remaining: int


@dataclass(frozen=True)
class LiveGenome:
    """One still-active genome, as a viewer thread needs it: which genome, its own env
    (for `Visualizable.visual_state`) and its own objective (for live `fitness()`)."""

    genome_id: int
    env: Environment
    objective: Objective


@dataclass(init=False)
class BatchProgress:
    """Mutable, written by `evaluate_batch` at each round boundary — not just once at the
    end — so a caller on a different thread (Phase 7's dashboard poller, via
    `NeatTrainer.progress_snapshot`) can read live in-progress state while a generation's
    batch is still running. Plain attribute writes/reads are safe here under CPython's GIL:
    each field is a single atomic assignment, and brief cross-field inconsistency is fine
    for a display value polled independently on its own timer. `live` is replaced (never
    mutated) whenever the active set changes, so a reader on another thread always sees a
    consistent tuple."""

    active_count: int
    best_fitness_so_far: float | None = None
    live: tuple[LiveGenome, ...] = ()

    def __init__(self, active_count: int, best_fitness_so_far: float | None = None) -> None:
        # Use object.__setattr__ to bypass polymorphic dispatch: construction is not an
        # observable write (the object doesn't exist for pollers yet). This prevents test spies
        # from capturing spurious initialization writes, allowing them to intercept only actual
        # progress updates in evaluate_batch.
        object.__setattr__(self, "active_count", active_count)
        object.__setattr__(self, "best_fitness_so_far", best_fitness_so_far)
        object.__setattr__(self, "live", ())


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
class _ActiveEpisode:
    genome_id: int
    model: Model
    env: Environment
    objective: Objective
    observation: Any
    last_info: dict[str, Any]


def evaluate_batch(
    entries: list[BatchEntry],
    step_budget: StepBudget,
    *,
    progress: BatchProgress | None = None,
    pace: Callable[[], None] | None = None,
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
    normally-ended episode, with no separate penalty term. If a genome's episode ends via
    `terminated`/`truncated` on the same tick its `Objective.should_stop()` would also have
    fired, the episode-end takes priority: it is scored as a normal ending and does NOT
    trigger the collective stop for the rest of the batch. This ordering is inherited
    unchanged from the sequential evaluation this function replaced. If `progress` is given,
    it is updated at each round boundary and reflects the current active count / best finished
    fitness so far. If `pace` is given, it is called once at the end of every round that
    leaves at least one genome active and did not end the batch — the hook a caller uses to
    slow the batch to a chosen speed; it must not raise."""
    active: list[_ActiveEpisode] = []
    for entry in entries:
        observation = entry.env.reset(seed=entry.seed)
        entry.model.reset()
        entry.objective.reset()
        active.append(
            _ActiveEpisode(
                entry.genome_id, entry.model, entry.env, entry.objective, observation, {}
            )
        )
    if progress is not None:
        progress.active_count = len(active)
        progress.live = _live_genomes(active)

    results: dict[int, EvaluationResult] = {}
    while active and step_budget.remaining > 0:
        still_active: list[_ActiveEpisode] = []
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
        if progress is not None:
            progress.active_count = len(active)
            if len(active) != len(progress.live):
                progress.live = _live_genomes(active)
            _update_best_fitness(progress, results)
        if stop_batch:
            break
        if pace is not None and active:
            pace()

    for genome in active:
        results[genome.genome_id] = EvaluationResult(genome.objective.fitness(), genome.last_info)
    if progress is not None:
        progress.active_count = 0
        progress.live = ()
        _update_best_fitness(progress, results)

    return results


def _update_best_fitness(progress: BatchProgress, results: dict[int, EvaluationResult]) -> None:
    if not results:
        return
    best = max(r.fitness for r in results.values())
    if progress.best_fitness_so_far is None or best > progress.best_fitness_so_far:
        progress.best_fitness_so_far = best


def _live_genomes(active: list[_ActiveEpisode]) -> tuple[LiveGenome, ...]:
    return tuple(LiveGenome(g.genome_id, g.env, g.objective) for g in active)
