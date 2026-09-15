# Phase 4 — Concurrent Batch Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `NeatTrainer` evaluate a generation's whole batch of genomes concurrently (round-robin lockstep) instead of one full episode at a time, and add a second generation-termination trigger: any genome finishing (`Objective.should_stop()`) immediately force-truncates every other still-running genome in the same batch.

**Architecture:** Replace `evaluation.py`'s single-genome `evaluate_genome(model, env, objective, step_budget, seed)` with a batch-level `evaluate_batch(entries, step_budget)` that steps every entry's episode round-robin — one `env.step()` per still-active genome per round — and stops the whole batch the instant the shared `step_budget` runs out or any entry's `Objective.should_stop()` fires. `trainer.py`'s `fitness_function` builds one `BatchEntry` per genome (its own fresh `Environment` from `make_env()`, its own `Objective` via `copy.deepcopy`) and calls `evaluate_batch` once per generation instead of calling `evaluate_genome` in a loop.

**Tech Stack:** Python 3.12+, `neat-python`, `pytest`. No new dependencies.

**Spec:** [`../../phases/phase-4-learning-backend-neat.md`](../../phases/phase-4-learning-backend-neat.md) — see the Requirements section ("Each genome is evaluated in isolation", "Generations always terminate") and the 2026-09-15 Revision history entry for full rationale, the round-robin-vs-multiprocessing decision (with measured perf numbers), and the ripple-effect check against Phase 5/6 (no changes needed there).

## Global Constraints

- Genome-level isolation means *no shared physics/state between genomes*, not sequential execution order — genomes in a batch run concurrently (round-robin), each on its own `Environment` and `Objective` instance.
- A batch stops on whichever of two triggers fires first: (1) the shared per-generation step budget is exhausted, or (2) any genome's `Objective.should_stop()` fires. Every genome still active when the batch stops is force-truncated and scored via `Objective.fitness()` on whatever it reached — no separate penalty term, same rule as today's step-ceiling truncation.
- `generation_stats` (Phase 6) and `ProgressObjective`'s fitness math (Phase 5/6) need no changes — do not touch `src/neuroarena/persistence/generation_stats_repo.py` or `src/neuroarena/sim/objectives.py` in this plan.
- `Trainer.update_config` (the Phase 0 addition) is explicitly **out of scope** for this plan — do not add it to `NeatTrainer` here.

---

## File Structure

- **Modify `src/neuroarena/backends/neat/evaluation.py`** — replace `evaluate_genome` with `BatchEntry` (a small dataclass bundling one genome's per-batch inputs) and `evaluate_batch` (the round-robin lockstep loop). `StepBudget` and `EvaluationResult` are unchanged in shape.
- **Modify `tests/backends/neat/test_evaluation.py`** — the four existing `evaluate_genome` tests become batch-of-one `evaluate_batch` tests (same assertions, adapted call shape); two new tests cover the collective early-stop trigger and the round-robin fairness claim.
- **Modify `src/neuroarena/backends/neat/trainer.py`** — `_run_one_generation`'s `fitness_function` builds a `BatchEntry` per genome (fresh `Environment` via `self._make_env()`, fresh `Objective` via `copy.deepcopy(self._objective)`) and calls `evaluate_batch` once instead of looping `evaluate_genome`. Update the module and class docstrings, which currently describe sequential per-genome evaluation on one shared `Environment`.
- **Modify `tests/backends/neat/test_trainer.py`** — existing tests are expected to keep passing unmodified (see Task 2's note on why); two new regression tests confirm each genome gets its own `Environment` and `Objective` instance.
- **Modify `docs/phases/phase-4-learning-backend-neat.md`** — Task 3 links this plan from the "Implementation plan" section and flips status `REVISED` → `FINALIZED` once the code lands.

---

### Task 1: `evaluate_batch` replaces `evaluate_genome`

**Files:**
- Modify: `src/neuroarena/backends/neat/evaluation.py`
- Test: `tests/backends/neat/test_evaluation.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces.protocols.{Environment, Model, Objective}` (unchanged).
- Produces: `StepBudget(remaining: int)` (unchanged shape), `EvaluationResult(fitness: float, final_info: dict[str, Any])` (unchanged shape), `BatchEntry(genome_id: int, model: Model, env: Environment, objective: Objective, seed: int)`, `evaluate_batch(entries: list[BatchEntry], step_budget: StepBudget) -> dict[int, EvaluationResult]`. Consumed by Task 2's `NeatTrainer`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/backends/neat/test_evaluation.py` with:

```python
from neuroarena.backends.neat.evaluation import BatchEntry, StepBudget, evaluate_batch
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: FAIL — `ImportError: cannot import name 'BatchEntry'` (or `evaluate_batch`) from `neuroarena.backends.neat.evaluation`, since neither exists yet.

- [ ] **Step 3: Replace `evaluate_genome` with `evaluate_batch`**

Replace the entire contents of `src/neuroarena/backends/neat/evaluation.py` with:

```python
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
            _ActiveGenome(
                entry.genome_id, entry.model, entry.env, entry.objective, observation, {}
            )
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/neuroarena/backends/neat/evaluation.py && uv run ruff check src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py && uv run ruff format --check src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py`
Expected: no errors. Fix any and re-run before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py
git commit -m "feat(phase-4): evaluate_batch replaces evaluate_genome — concurrent round-robin batch evaluation with collective finish-early truncation"
```

---

### Task 2: Wire `NeatTrainer` to `evaluate_batch`

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Test: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `BatchEntry`, `evaluate_batch` (Task 1).
- Produces: no new public interface — `NeatTrainer` still satisfies `Trainer` exactly as before; this task only changes *how* `_run_one_generation` computes fitness internally.

- [ ] **Step 1: Confirm Task 1 broke `trainer.py`'s import (expected, not yet fixed)**

Run: `uv run pytest tests/backends/neat/ -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_genome' from 'neuroarena.backends.neat.evaluation'`. Task 1 removed `evaluate_genome`; `trainer.py` still imports it. This is expected at this point in the plan, not a regression — Step 4 below fixes the import.

- [ ] **Step 2: Write the two new failing regression tests**

Add to the end of `tests/backends/neat/test_trainer.py`:

```python
def test_each_genome_gets_its_own_environment_instance() -> None:
    constructed: list[DummyEnvironment] = []

    def make_env() -> DummyEnvironment:
        env = DummyEnvironment()
        constructed.append(env)
        return env

    trainer = NeatTrainer(make_env, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    # One instance for `self._env` (descriptors/checkpoint metadata) + one per genome — never
    # one instance reused across all five, which concurrent round-robin evaluation requires.
    assert len(constructed) == 6
    assert len(set(map(id, constructed))) == 6


def test_each_genome_gets_its_own_objective_instance() -> None:
    seen_ids: list[int] = []

    class TrackingObjective(DummyObjective):
        def reset(self) -> None:
            super().reset()
            seen_ids.append(id(self))

    trainer = NeatTrainer(DummyEnvironment, TrackingObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    # The `Objective` passed to `NeatTrainer.__init__` is never reset/used directly during a
    # generation — only per-genome deep copies of it are, so concurrent genomes never share
    # one mutable Objective instance the way sequential evaluation could get away with.
    assert len(seen_ids) == 5
    assert len(set(seen_ids)) == 5
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/backends/neat/test_trainer.py -k "own_environment_instance or own_objective_instance" -v`
Expected: FAIL — `ImportError` (trainer.py still imports the now-removed `evaluate_genome`), or an `AssertionError` once the import is fixed but before this task's rewiring lands.

- [ ] **Step 4: Update `trainer.py`'s docstrings and import**

In `src/neuroarena/backends/neat/trainer.py`, replace the module docstring (lines 1–5):

```python
"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating a
generation's whole batch of genomes concurrently (round-robin lockstep, see `evaluation.py`)
under a per-generation step budget shared across the batch. See
`../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this implements,
in particular why track switches only take effect at a generation boundary, why a
force-truncated genome is scored on partial progress rather than penalised, and why any
genome finishing ends the whole batch immediately for the rest."""
```

Replace the import line:

```python
from neuroarena.backends.neat.evaluation import BatchEntry, EvaluationResult, StepBudget, evaluate_batch
```

Replace the `NeatTrainer` class docstring:

```python
class NeatTrainer:
    """Satisfies `Trainer`. `self._env` is rebuilt once per generation (used for observation/
    action-space descriptors and checkpoint metadata) — but each genome in that generation's
    batch gets its OWN fresh `Environment` instance from `make_env`, since concurrent
    round-robin evaluation (see `evaluation.py`) runs every genome's episode at once, not one
    at a time on a shared instance. `make_env`'s closure still only changes at a generation
    boundary, so every genome within one generation is still compared on the same track (a
    live track switch, Phase 5, takes effect for the next generation's `make_env()` calls)."""
```

- [ ] **Step 5: Rewrite `_run_one_generation`**

Replace the whole `_run_one_generation` method in `src/neuroarena/backends/neat/trainer.py`:

```python
    def _run_one_generation(self) -> TrainingUpdate:
        self._env = self._make_env()
        step_budget = StepBudget(remaining=self._config.max_generation_steps)
        champion = _ChampionTracker()
        results: dict[int, EvaluationResult] = {}

        def fitness_function(genomes: list[tuple[int, Any]], neat_config: neat.Config) -> None:
            entries = [
                BatchEntry(
                    genome_id=genome_id,
                    model=GenomeModel(
                        genome, neat_config, self._env.observation_space, self._env.action_space
                    ),
                    env=self._make_env(),
                    # Concurrent batch evaluation needs one Objective instance per genome, not
                    # the single shared `self._objective` sequential evaluation could reuse — a
                    # deep copy keeps every genome's state independent for the round-robin.
                    objective=copy.deepcopy(self._objective),
                    seed=self._config.master_seed + genome_id,
                )
                for genome_id, genome in genomes
            ]
            batch_results = evaluate_batch(entries, step_budget)
            results.update(batch_results)
            genome_by_id = dict(genomes)
            for genome_id, result in batch_results.items():
                genome = genome_by_id[genome_id]
                genome.fitness = result.fitness
                champion.consider(genome, result)

        self._population.run(fitness_function, 1)

        steps_used = self._config.max_generation_steps - step_budget.remaining
        self._total_sim_steps += steps_used

        if self._champion_dir is not None and champion.genome is not None:
            path = self._champion_dir / f"gen_{self._generation:05d}.pkl"
            path.write_bytes(pickle.dumps(champion.genome))

        values = [result.fitness for result in results.values()]
        update = TrainingUpdate(
            progress_index=self._generation,
            best_fitness=max(values),
            mean_fitness=sum(values) / len(values),
            worst_fitness=min(values),
            population_size=len(values),
            champion_metrics=champion.metrics(),
            sim_time=self._total_sim_steps,
            wall_time=time.perf_counter() - self._wall_start,
        )
        self._generation += 1
        return update
```

- [ ] **Step 6: Run the full NEAT backend test suite**

Run: `uv run pytest tests/backends/neat/ -v`
Expected: PASS, all tests including the two new ones from Step 2. The pre-existing tests are expected to keep passing unmodified:
- Tests using `RunConfig()`'s default `max_generation_steps=300_000` (nearly everything) never hit budget starvation either way — round-robin vs. sequential produces the same per-genome outcomes when no genome is ever cut short.
- `test_generation_ceiling_bounds_total_steps_per_generation` (`max_generation_steps=7`, 5 genomes) only asserts `worst_fitness <= 5.0` and `sim_time <= 7` — both still hold under round-robin's more even step distribution, just with different (higher) exact per-genome step counts than the old sequential front-loading produced.

If anything unexpected fails, read the assertion, decide whether the new round-robin semantics genuinely changed the expected value (fix the test, with a comment explaining why), or whether it's a real bug in Step 5 (fix the code) — do not weaken an assertion just to make it pass.

- [ ] **Step 7: Run the whole project test suite**

Run: `uv run pytest -v`
Expected: PASS. This catches any other module importing `evaluate_genome` that a repo-wide grep might have missed.

- [ ] **Step 8: Type-check and lint**

Run: `uv run mypy --strict src/neuroarena/backends/neat/trainer.py && uv run ruff check src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py && uv run ruff format --check src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py`
Expected: no errors. Fix any and re-run before continuing.

- [ ] **Step 9: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-4): NeatTrainer evaluates each generation's batch via evaluate_batch — per-genome Environment/Objective instances, round-robin concurrency"
```

---

### Task 3: Record completion in the Phase 4 doc

**Files:**
- Modify: `docs/phases/phase-4-learning-backend-neat.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Update the "Implementation plan" section**

In `docs/phases/phase-4-learning-backend-neat.md`, replace the paragraph that currently reads:

```
**Needs a follow-up pass:** the 2026-09-15 revision above (concurrent round-robin batch evaluation + the collective finish-early trigger) changes `backends/neat/evaluation.py` and `backends/neat/trainer.py` behavior beyond what the plan below covers — it was written and executed before that revision existed. Not yet planned or built.
```

with:

```
The 2026-09-15 revision above (concurrent round-robin batch evaluation + the collective finish-early trigger) is implemented via a separate, later plan: [`../superpowers/plans/2026-09-15-phase-4-concurrent-batch-evaluation.md`](../superpowers/plans/2026-09-15-phase-4-concurrent-batch-evaluation.md), since the plan below was written and executed before that revision existed.
```

- [ ] **Step 2: Flip status back to FINALIZED and add a Revision history entry**

Change the Status line at the top of the doc from:

```
Status: **REVISED** — requirements settled as of 2026-09-13, then deliberately revised on 2026-09-15 (concurrent batch evaluation + collective finish-early trigger). The 2026-09-15 revision is not yet implemented — see "Implementation plan" below. Changing the requirements further needs its own deliberate revision (see `../WORKFLOW.md`).
```

to:

```
Status: **FINALIZED** — requirements settled as of 2026-09-13, revised 2026-09-15 (concurrent batch evaluation + collective finish-early trigger), implementation caught up the same day. Changing them needs a deliberate revision (see `../WORKFLOW.md`).
```

Add a new entry at the top of the Revision history section (immediately after `## Revision history`, before the existing 2026-09-15 entry):

```
- 2026-09-15 — Implemented the same-day revision above via `evaluate_batch` (`backends/neat/evaluation.py`) and a rewired `NeatTrainer._run_one_generation` (`backends/neat/trainer.py`) — round-robin lockstep batch evaluation, force-truncation the instant any genome's `Objective.should_stop()` fires. Status REVISED -> FINALIZED. See [`../superpowers/plans/2026-09-15-phase-4-concurrent-batch-evaluation.md`](../superpowers/plans/2026-09-15-phase-4-concurrent-batch-evaluation.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/phases/phase-4-learning-backend-neat.md
git commit -m "docs(phase-4): record concurrent-batch-evaluation implementation completion"
```
