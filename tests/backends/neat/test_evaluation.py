from typing import Any

import numpy as np

from neuroarena.backends.neat.evaluation import (
    BatchEntry,
    BatchProgress,
    StepBudget,
    evaluate_batch,
)
from tests.interfaces.doubles import DummyEnvironment, DummyModel, DummyObjective


def _entry(genome_id: int, objective: DummyObjective | None = None) -> BatchEntry:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    return BatchEntry(
        genome_id=genome_id, model=model, env=env, objective=objective or DummyObjective(), seed=0
    )


def test_evaluate_batch_runs_a_single_genome_until_env_truncates() -> None:
    budget = StepBudget(remaining=100)
    results = evaluate_batch([_entry(0)], budget)
    # DummyEnvironment truncates after 5 steps (see tests/interfaces/doubles.py).
    assert results[0].fitness == 5.0
    assert results[0].final_info == {"t": 5}
    assert budget.remaining == 95


def test_evaluate_batch_force_truncates_a_single_genome_when_budget_runs_out_mid_episode() -> None:
    budget = StepBudget(remaining=3)
    results = evaluate_batch([_entry(0)], budget)
    assert results[0].fitness == 3.0
    assert budget.remaining == 0


def test_evaluate_batch_scores_zero_when_budget_is_already_exhausted() -> None:
    budget = StepBudget(remaining=0)
    results = evaluate_batch([_entry(0)], budget)
    assert results[0].fitness == 0.0
    assert results[0].final_info == {}


def test_evaluate_batch_stops_a_single_genome_early_when_objective_says_stop() -> None:
    class StopsImmediately(DummyObjective):
        def should_stop(self) -> bool:
            return True

    budget = StepBudget(remaining=100)
    results = evaluate_batch([_entry(0, StopsImmediately())], budget)
    assert results[0].fitness == 1.0
    assert budget.remaining == 99


def test_evaluate_batch_force_truncates_every_other_genome_the_instant_one_finishes() -> None:
    class SucceedsAfterTwoSteps(DummyObjective):
        def should_stop(self) -> bool:
            return self._steps >= 2

    # Iteration order matters: `early` is processed before the finisher within a round and so
    # keeps the round-2 step it already took; `late` is processed after and does not get one.
    entries = [
        _entry(genome_id=10),  # "early" — never stops on its own
        _entry(genome_id=20, objective=SucceedsAfterTwoSteps()),  # "finisher"
        _entry(genome_id=30),  # "late" — never stops on its own
    ]
    budget = StepBudget(remaining=100)
    results = evaluate_batch(entries, budget)

    assert results[10].fitness == 2.0  # processed before the finisher this round: kept its step
    assert results[20].fitness == 2.0  # the finisher itself
    assert results[30].fitness == 1.0  # processed after the finisher this round: cut off first
    assert budget.remaining == 95  # 2 + 2 + 1 = 5 steps consumed


def test_evaluate_batch_consumes_the_shared_budget_evenly_round_by_round() -> None:
    # None of these three genomes reach their own natural end (DummyEnvironment truncates at
    # 5 steps each) before the tight budget below runs out. Round-robin means the budget is
    # spent one step per genome per round, not front-loaded onto whichever genome runs first.
    entries = [_entry(genome_id=0), _entry(genome_id=1), _entry(genome_id=2)]
    budget = StepBudget(remaining=7)
    results = evaluate_batch(entries, budget)

    # Round 1 (budget 7->4): all three step once. Round 2 (budget 4->1): all three step again.
    # Round 3 (budget 1->0): only genome 0 gets its turn before the budget runs out.
    assert results[0].fitness == 3.0
    assert results[1].fitness == 2.0
    assert results[2].fitness == 2.0
    assert budget.remaining == 0


def test_evaluate_batch_lets_each_genome_run_to_its_own_natural_end() -> None:
    class TruncatesAfter(DummyEnvironment):
        def __init__(self, steps: int) -> None:
            super().__init__()
            self._limit = steps

        def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
            self._t += 1
            return np.zeros(3, dtype=np.float32), False, self._t >= self._limit, {"t": self._t}

    def entry(genome_id: int, steps: int) -> BatchEntry:
        env = TruncatesAfter(steps)
        model = DummyModel(env.observation_space, env.action_space)
        return BatchEntry(
            genome_id=genome_id, model=model, env=env, objective=DummyObjective(), seed=0
        )

    # DummyObjective never calls should_stop(), so nothing ever triggers the collective stop —
    # each genome must run all the way to its own natural truncation length, completely
    # unaffected by its shorter- or longer-lived siblings dropping out of the round-robin.
    entries = [entry(0, 2), entry(1, 9), entry(2, 4)]
    budget = StepBudget(remaining=100)
    results = evaluate_batch(entries, budget)

    assert results[0].fitness == 2.0
    assert results[1].fitness == 9.0
    assert results[2].fitness == 4.0
    assert budget.remaining == 100 - (2 + 9 + 4)


def test_evaluate_batch_writes_progress_at_every_round_boundary() -> None:
    """The dashboard polls this from a separate thread while evaluate_batch is still
    running, so progress must be written incrementally — once per round — not only once
    after evaluate_batch returns."""
    writes: list[int] = []

    class _SpyProgress(BatchProgress):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "active_count":
                writes.append(value)
            super().__setattr__(name, value)

    entries = [_entry(0), _entry(1)]
    budget = StepBudget(remaining=100)
    evaluate_batch(entries, budget, progress=_SpyProgress(active_count=0))
    # Both genomes truncate together at DummyEnvironment's 5-step limit: 2 active -> 0,
    # written once per round (5 rounds), not just once at the very end.
    assert writes[0] == 2
    assert writes[-1] == 0
    assert len(writes) >= 2


def test_evaluate_batch_progress_tracks_best_fitness_seen_so_far() -> None:
    class SucceedsAfterOneStep(DummyObjective):
        def should_stop(self) -> bool:
            return self._steps >= 1

    entries = [_entry(0, SucceedsAfterOneStep()), _entry(1)]
    budget = StepBudget(remaining=100)
    progress = BatchProgress(active_count=0)
    evaluate_batch(entries, budget, progress=progress)
    # Genome 0 finishes on step 1 with fitness 1.0; genome 1 is force-truncated by the
    # collective stop with fitness 1.0 too (one step taken before the stop fires) —
    # either way, best_fitness_so_far must reflect a finished genome by the time we return.
    assert progress.best_fitness_so_far == 1.0


def test_evaluate_batch_without_a_progress_argument_is_unaffected() -> None:
    results = evaluate_batch([_entry(0)], StepBudget(remaining=100))
    assert results[0].fitness == 5.0
