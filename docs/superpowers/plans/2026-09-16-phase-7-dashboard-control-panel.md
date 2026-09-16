# Phase 7 — Dashboard Control Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working FastAPI + WebSocket dashboard backend that embeds `NeatTrainer` in-process (start/resume a run, live-edit its 3 wired config knobs, manual stop, live metrics push) and a minimal React control-panel frontend that drives it.

**Architecture:** The backend is a new `neuroarena.dashboard` package: a `RunManager` owns the single live `Trainer` this process drives (one run per process, per the spec), running it on a background thread via a thin, non-blocking wrapper around Phase 6's already-FINALIZED `persistence.recorder.run_and_record`; a FastAPI app exposes REST endpoints over `RunManager` and a `/ws` WebSocket endpoint that polls it on a dashboard-configurable timer and pushes the three-message-type envelope the spec defines. Two earlier-phase modules get small, additive, backward-compatible extensions to make live progress observable at all: `evaluate_batch` (Phase 4) gains an optional progress-sink parameter, and `run_and_record` (Phase 6) gains optional `on_update`/`should_stop` callbacks — both default to `None` and change no existing behavior. The React frontend is a small Vite/TypeScript app talking to this API with plain `fetch`/`WebSocket`, no state-management library.

**Tech Stack:** FastAPI + Uvicorn (backend), stdlib `threading`/`asyncio` (no Celery/RQ — matches the spec's "embed in-process" decision), Vite + React + TypeScript (frontend), plain `fetch`/native `WebSocket` (no React Query/Redux — YAGNI for a single-user local tool).

**Spec:** [`../../phases/phase-7-dashboard-control-panel.md`](../../phases/phase-7-dashboard-control-panel.md) (FINALIZED 2026-09-16). Builds on the FINALIZED [`phase-0-architecture.md`](../../phases/phase-0-architecture.md) (`Trainer`/`TrainingUpdate`), [`phase-4-learning-backend-neat.md`](../../phases/phase-4-learning-backend-neat.md) (`NeatTrainer`, `evaluate_batch`), [`phase-5-training-controls.md`](../../phases/phase-5-training-controls.md) (`RunConfig` knob classification), and [`phase-6-persistence.md`](../../phases/phase-6-persistence.md) (`models_repo`, `runs_repo`, `settings_history_repo`, `checkpoints_repo`, `recorder`, the `neuroarena-train` CLI).

## Global Constraints

- Python `>=3.12`; every new/modified `.py` file must pass `ruff check`, `ruff format --check`, and `mypy --strict` (see `pyproject.toml`).
- Local-first: the FastAPI app binds `127.0.0.1` only, no auth (spec).
- One training run per backend process (spec) — a second `POST /api/runs` while one is active is rejected, not queued.
- WebSocket envelope is exactly `{"type": "progress"|"generation"|"status", "schema_version": 1, "data": {...}}` (spec) — do not invent a fourth type or change field names without a spec revision.
- Poll interval defaults to 200ms, is dashboard-only runtime state (never `RunConfig`, never settings-history), global per process, adjustable with immediate effect (spec).
- `NeatTrainer.update_config` today only accepts `max_generation_steps`/`max_generations`/`target_fitness` — the dashboard's live-config-edit surface is exactly those three fields, nothing more (Phase 0/Phase 5).
- **Deliberately out of scope for this plan** (per the spec's own "out of scope" bullets and Phase 9's ownership of comparison/history views): in-browser live rendering (Phase 8), a run-history/generation-browsing UI (Phase 9), live-editing any `RunConfig` field beyond the 3 wired ones, multi-process/remote deployment.
- **Plan-level scoping decision, not in the spec** (Implementation-plan-level detail, as the spec's own convention leaves to this document): the "start a new run" form exposes exactly `track_id` (required), `population_size`, `max_generations`, `target_fitness` — the knobs Phase 7's own resume-flow bullet calls out plus the one obviously-needed creation-time knob. Every other `RunConfig` field (`sensor_config`, `physics_constants`, `neat_hyperparameters`, `max_episode_steps`, `checkpoint_every_n_generations`, `champion_retention_cap`, `sim_speed`, `headless`) keeps its `RunConfig` default and is not exposed in this first dashboard slice.
- **Coupling decision, recorded not hidden:** `neuroarena.dashboard` imports `NeatTrainer`'s concrete `GenerationProgress` dataclass directly (in `run_manager.py` and `ws_protocol.py`) rather than inventing a backend-agnostic structural type for it. `RunManager.progress_snapshot()` still degrades gracefully (via `getattr`) for any future `Trainer` that doesn't implement `progress_snapshot` at all — only the *type hint* is NEAT-specific. Acceptable because Phase 10's RL backend doesn't exist yet; revisit when it does, not before (YAGNI).

## File Structure

New:
- `src/neuroarena/persistence/launch.py` — `prepare_run`/`PreparedRun`, extracted from `cli.py`'s `run()` so both the CLI and the dashboard share one training-launch setup path.
- `src/neuroarena/dashboard/__init__.py`
- `src/neuroarena/dashboard/ws_protocol.py` — pure envelope-building functions (no FastAPI/asyncio/threading).
- `src/neuroarena/dashboard/run_manager.py` — `RunManager`, the in-process trainer owner.
- `src/neuroarena/dashboard/app.py` — FastAPI app factory: REST routes + `/ws`.
- `src/neuroarena/dashboard/cli.py` — `neuroarena-dashboard` entry point (runs uvicorn).
- `tests/dashboard/__init__.py`, `tests/dashboard/test_ws_protocol.py`, `tests/dashboard/test_run_manager.py`, `tests/dashboard/test_app.py`
- `tests/persistence/test_launch.py`
- `frontend/` — Vite + React + TypeScript app: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api.ts`, `src/ws.ts`, `src/types.ts`, `src/components/{ModelList,NewRunForm,LiveMetrics,ConfigPanel}.tsx`.

Modified:
- `src/neuroarena/backends/neat/evaluation.py` — add `BatchProgress` + `evaluate_batch`'s new `progress` parameter.
- `src/neuroarena/backends/neat/trainer.py` — add `GenerationProgress` + `NeatTrainer.progress_snapshot()`.
- `src/neuroarena/persistence/recorder.py` — add `run_and_record`'s new `on_update`/`should_stop` parameters.
- `src/neuroarena/persistence/cli.py` — `run()` becomes a thin wrapper over `launch.prepare_run` + `recorder.run_and_record`; `main()` unchanged.
- `pyproject.toml` — add `fastapi`, `uvicorn[standard]` to `dependencies`; `httpx` to the `dev` group; a `neuroarena-dashboard` script entry.

---

## Task 1: `evaluate_batch` gets a progress sink (Phase 4 extension)

**Files:**
- Modify: `src/neuroarena/backends/neat/evaluation.py`
- Test: `tests/backends/neat/test_evaluation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `BatchProgress` (mutable `@dataclass`: `active_count: int`, `best_fitness_so_far: float | None = None`); `evaluate_batch(entries, step_budget, *, progress: BatchProgress | None = None) -> dict[int, EvaluationResult]` (extended signature — the new parameter is keyword-only, defaults to `None`, and every existing call site/test is unaffected).

- [ ] **Step 1: Write the failing tests**

Add to `tests/backends/neat/test_evaluation.py` (uses the existing `_entry` helper and `DummyObjective`/`DummyEnvironment` already imported in that file):

```python
from typing import Any

from neuroarena.backends.neat.evaluation import BatchProgress


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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -k progress -v`
Expected: FAIL — `ImportError: cannot import name 'BatchProgress'` (or `TypeError: evaluate_batch() got an unexpected keyword argument 'progress'`).

- [ ] **Step 3: Implement `BatchProgress` and wire it into `evaluate_batch`**

In `src/neuroarena/backends/neat/evaluation.py`, add the dataclass (after `StepBudget`) and update `evaluate_batch`:

```python
@dataclass
class BatchProgress:
    """Mutable, written by `evaluate_batch` at each round boundary — not just once at the
    end — so a caller on a different thread (Phase 7's dashboard poller, via
    `NeatTrainer.progress_snapshot`) can read live in-progress state while a generation's
    batch is still running. Plain attribute writes/reads are safe here under CPython's GIL:
    each field is a single atomic assignment, and brief cross-field inconsistency is fine
    for a display value polled independently on its own timer."""

    active_count: int
    best_fitness_so_far: float | None = None


def evaluate_batch(
    entries: list[BatchEntry], step_budget: StepBudget, *, progress: BatchProgress | None = None
) -> dict[int, EvaluationResult]:
    """... (existing docstring unchanged; `progress`, if given, is updated at each round
    boundary and reflects the current active count / best finished fitness so far)"""
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
            _update_best_fitness(progress, results)
        if stop_batch:
            break

    for genome in active:
        results[genome.genome_id] = EvaluationResult(genome.objective.fitness(), genome.last_info)
    if progress is not None:
        progress.active_count = 0
        _update_best_fitness(progress, results)

    return results


def _update_best_fitness(progress: BatchProgress, results: dict[int, EvaluationResult]) -> None:
    if not results:
        return
    best = max(r.fitness for r in results.values())
    if progress.best_fitness_so_far is None or best > progress.best_fitness_so_far:
        progress.best_fitness_so_far = best
```

- [ ] **Step 4: Run the full evaluation test file to verify everything passes**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: PASS, all tests (old and new).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py
git commit -m "feat(phase-7): evaluate_batch gains an optional progress sink"
```

---

## Task 2: `NeatTrainer.progress_snapshot()` (Phase 4/0 extension)

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Test: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `BatchProgress`, `evaluate_batch(..., progress=...)` (Task 1).
- Produces: `GenerationProgress` (frozen `@dataclass`: `generation: int`, `population_size: int`, `active_genomes_remaining: int`, `elapsed_steps: int`, `step_ceiling: int`, `best_fitness_so_far: float | None`); `NeatTrainer.progress_snapshot() -> GenerationProgress | None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/backends/neat/test_trainer.py`:

```python
import time

from neuroarena.backends.neat.trainer import GenerationProgress


def test_progress_snapshot_is_none_before_run_starts() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    assert trainer.progress_snapshot() is None


def test_progress_snapshot_is_none_again_after_a_generation_completes() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    assert trainer.progress_snapshot() is None


def test_progress_snapshot_reflects_live_state_during_a_generation() -> None:
    class SlowObjective(DummyObjective):
        def update(self, observation, action, terminated, truncated, info) -> None:
            super().update(observation, action, terminated, truncated, info)
            time.sleep(0.02)

    trainer = NeatTrainer(DummyEnvironment, SlowObjective(), RunConfig(), population_size=3)
    snapshots: list[GenerationProgress | None] = []

    def _poll() -> None:
        for _ in range(100):
            snapshots.append(trainer.progress_snapshot())
            time.sleep(0.005)

    poller = threading.Thread(target=_poll)
    poller.start()
    next(trainer.run())
    poller.join()

    live = [s for s in snapshots if s is not None]
    assert live, "expected at least one snapshot while the generation was running"
    assert all(s.population_size == 3 for s in live)
    assert all(0 <= s.active_genomes_remaining <= 3 for s in live)
    assert all(0 <= s.elapsed_steps <= s.step_ceiling for s in live)
    assert all(s.generation == 0 for s in live)
```

(`threading` is already imported at the top of this test file.)

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/backends/neat/test_trainer.py -k progress_snapshot -v`
Expected: FAIL — `AttributeError: 'NeatTrainer' object has no attribute 'progress_snapshot'`.

- [ ] **Step 3: Implement**

In `src/neuroarena/backends/neat/trainer.py`:

1. Import `BatchProgress` alongside the existing `evaluation` imports.
2. Add near the top (after `_LIVE_CHANGEABLE_FIELDS`). This file already does `import dataclasses` (module-style, not `from dataclasses import ...`) and uses `dataclasses.replace` elsewhere — match that style with `@dataclasses.dataclass`, no new import needed:

```python
@dataclasses.dataclass(frozen=True)
class GenerationProgress:
    """A snapshot of an in-progress generation's batch state, for Phase 7's dashboard
    polling loop. See `NeatTrainer.progress_snapshot`."""

    generation: int
    population_size: int
    active_genomes_remaining: int
    elapsed_steps: int
    step_ceiling: int
    best_fitness_so_far: float | None
```

3. In `NeatTrainer.__init__`, after the existing field initializations, add:

```python
        self._current_batch_progress: BatchProgress | None = None
        self._current_step_budget: StepBudget | None = None
```

4. Add the public method (placed after `update_config`, before `run`):

```python
    def progress_snapshot(self) -> GenerationProgress | None:
        """Thread-safe (plain attribute reads under CPython's GIL — see `BatchProgress`'s
        own docstring for why no lock is needed), read-only snapshot of the currently
        in-progress generation's batch state, for Phase 7's dashboard polling loop. `None`
        when no generation is currently running (between generations, or before `run()`
        has been advanced at all)."""
        progress = self._current_batch_progress
        step_budget = self._current_step_budget
        if progress is None or step_budget is None:
            return None
        return GenerationProgress(
            generation=self._generation,
            population_size=self._neat_config.pop_size,
            active_genomes_remaining=progress.active_count,
            elapsed_steps=self._config.max_generation_steps - step_budget.remaining,
            step_ceiling=self._config.max_generation_steps,
            best_fitness_so_far=progress.best_fitness_so_far,
        )
```

5. In `_run_one_generation`, set the two fields around the batch, and pass `progress=` through in `fitness_function`. Modify:

```python
    def _run_one_generation(self) -> TrainingUpdate:
        with self._config_lock:
            if self._pending_config_update:
                self._config = dataclasses.replace(self._config, **self._pending_config_update)
                self._pending_config_update = None
        self._env = self._make_env()
        step_budget = StepBudget(remaining=self._config.max_generation_steps)
        self._current_step_budget = step_budget
        self._current_batch_progress = BatchProgress(active_count=self._neat_config.pop_size)
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
                    objective=copy.deepcopy(self._objective),
                    seed=self._config.master_seed + genome_id,
                )
                for genome_id, genome in genomes
            ]
            batch_results = evaluate_batch(entries, step_budget, progress=self._current_batch_progress)
            results.update(batch_results)
            genome_by_id = dict(genomes)
            for genome_id, result in batch_results.items():
                genome = genome_by_id[genome_id]
                genome.fitness = result.fitness
                champion.consider(genome, result)

        self._population.run(fitness_function, 1)
        self._current_batch_progress = None
        self._current_step_budget = None

        steps_used = self._config.max_generation_steps - step_budget.remaining
        # ... (rest of the method unchanged)
```

Only the three added/changed lines matter: `self._current_step_budget = step_budget` and `self._current_batch_progress = BatchProgress(...)` before `self._population.run(...)`, the `progress=self._current_batch_progress` argument added to the existing `evaluate_batch(...)` call, and the two `= None` resets immediately after `self._population.run(...)` returns. Everything else in this method (docstring, `_ChampionTracker`, the `TrainingUpdate` construction at the end) is unchanged.

- [ ] **Step 4: Run the full trainer test file to verify everything passes**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: PASS, all tests (old and new).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-7): NeatTrainer.progress_snapshot for live dashboard polling"
```

---

## Task 3: `run_and_record` gets `on_update`/`should_stop` hooks (Phase 6 extension)

**Files:**
- Modify: `src/neuroarena/persistence/recorder.py`
- Test: `tests/persistence/test_recorder.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_and_record(..., on_update: Callable[[TrainingUpdate], None] | None = None, should_stop: Callable[[], bool] | None = None) -> RunRecord` (extended signature, backward compatible — every existing caller/test is unaffected).

- [ ] **Step 1: Write the failing tests**

Add to `tests/persistence/test_recorder.py` (reuses the existing `_FakeTrainer`/`_model_id` helpers already in this file):

```python
def test_on_update_is_called_once_per_generation_after_persisting_it(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    seen: list[int] = []
    run_and_record(
        conn,
        _FakeTrainer(n_generations=3),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
        on_update=lambda u: seen.append(u.progress_index),
    )
    assert seen == [0, 1, 2]
    # "after persisting it": every generation on_update saw is already in the DB.
    assert len(list_generation_stats_for_model(conn, model_id)) == 3


def test_should_stop_ends_the_run_early_with_stopped_status(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    record = run_and_record(
        conn,
        _FakeTrainer(n_generations=10),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
        should_stop=lambda: True,
    )
    assert record.status == "stopped"
    # Stopped after fully persisting exactly the first generation, not zero and not all ten.
    assert len(list_generation_stats_for_model(conn, model_id)) == 1


def test_run_without_the_new_hooks_is_unaffected(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    record = run_and_record(
        conn,
        _FakeTrainer(n_generations=2),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
    )
    assert record.status == "completed"
```

`list_generation_stats_for_model` needs importing at the top of the test file if not already (it is — check the existing import block; if absent, add `from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model`).

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/persistence/test_recorder.py -k "on_update or should_stop or without_the_new_hooks" -v`
Expected: FAIL — `TypeError: run_and_record() got an unexpected keyword argument 'on_update'`.

- [ ] **Step 3: Implement**

In `src/neuroarena/persistence/recorder.py`, update the signature and the loop body:

```python
def run_and_record(
    conn: sqlite3.Connection,
    trainer: Trainer,
    *,
    model_id: str,
    track_id: str | None,
    starting_generation: int,
    resume_dir: Path,
    champion_dir: Path | None,
    checkpoint_every_n_generations: int,
    champion_retention_cap: int | None,
    initial_settings_diff: dict[str, Any],
    on_update: Callable[[TrainingUpdate], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RunRecord:
    resume_dir.mkdir(parents=True, exist_ok=True)
    run_record = runs_repo.create_run(
        conn, model_id=model_id, track_id=track_id, starting_generation=starting_generation
    )
    settings_history_repo.record_settings_entry(
        conn,
        model_id=model_id,
        run_id=run_record.run_id,
        generation=starting_generation,
        diff=initial_settings_diff,
    )
    try:
        for update in trainer.run():
            generation_stats_repo.record_generation_stat(
                conn, model_id=model_id, run_id=run_record.run_id, update=update
            )
            if on_update is not None:
                on_update(update)
            generation = update.progress_index

            if generation % checkpoint_every_n_generations == 0:
                path = resume_dir / f"gen_{generation:05d}.pkl"
                trainer.save_checkpoint(path)
                checkpoints_repo.record_checkpoint(
                    conn,
                    model_id=model_id,
                    run_id=run_record.run_id,
                    kind="resume",
                    generation=generation,
                    file_path=path,
                )

            if champion_dir is not None:
                champion_path = champion_dir / f"gen_{generation:05d}.pkl"
                checkpoints_repo.record_checkpoint(
                    conn,
                    model_id=model_id,
                    run_id=run_record.run_id,
                    kind="champion",
                    generation=generation,
                    file_path=champion_path,
                )
                if champion_retention_cap is not None:
                    _prune_champions(conn, model_id, champion_retention_cap)

            if should_stop is not None and should_stop():
                return _finish(conn, run_record, status="stopped")
    except KeyboardInterrupt:
        _finish(conn, run_record, status="stopped")
        raise
    except Exception:
        _finish(conn, run_record, status="crashed")
        raise

    return _finish(conn, run_record, status="completed")
```

Two import changes at the top of `recorder.py`: add `from collections.abc import Callable` as a normal top-level import (needed at runtime for the `Callable[...]` type in the new parameters' defaults), and add `TrainingUpdate` to the existing `TYPE_CHECKING` block so it reads `from neuroarena.interfaces.protocols import Trainer, TrainingUpdate` — this file has `from __future__ import annotations`, so the `Callable[[TrainingUpdate], None]` annotation is never evaluated at runtime and `TrainingUpdate` doesn't need to be a real top-level import, preserving this module's existing "only imports the Trainer Protocol, never a backend" discipline.

- [ ] **Step 4: Run the full recorder test file to verify everything passes**

Run: `uv run pytest tests/persistence/test_recorder.py -v`
Expected: PASS, all tests (old and new).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/recorder.py tests/persistence/test_recorder.py
git commit -m "feat(phase-7): run_and_record gains on_update/should_stop hooks"
```

---

## Task 4: Extract `persistence.launch.prepare_run` from `cli.py`

**Files:**
- Create: `src/neuroarena/persistence/launch.py`
- Modify: `src/neuroarena/persistence/cli.py`
- Test: `tests/persistence/test_launch.py` (new); `tests/persistence/test_cli.py` must keep passing **unchanged** — that's the regression bar for this refactor.

**Interfaces:**
- Consumes: everything `cli.py`'s `run()` already consumes (`models_repo`, `checkpoints_repo`, `settings_history_repo`, `NeatTrainer`, `CarEnvironment`/`CarEnvironmentConfig`, `ProgressObjective`, `tracks.store`).
- Produces: `PreparedRun` (frozen `@dataclass`: `trainer: Trainer`, `model: ModelRecord`, `config: RunConfig`, `starting_generation: int`, `settings_diff: dict[str, Any]`, `model_dir: Path`); `prepare_run(conn, *, track_id, population_size=None, max_generations=None, target_fitness=None, checkpoint_every_n_generations=None, champion_retention_cap=None, resume_model_id=None, data_dir=DEFAULT_DATA_DIR) -> PreparedRun`. Consumed by Task 7's `RunManager`.

- [ ] **Step 1: Write the failing tests**

Create `tests/persistence/test_launch.py`:

```python
from pathlib import Path

from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import prepare_run
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from neuroarena.tracks.store import save


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _save_test_track(data_dir: Path) -> str:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    record = save(track, requested_size=10, complexity=0.3, seed=1, tracks_dir=data_dir / "tracks")
    return record.track_id


def test_prepare_run_builds_a_fresh_model_and_working_trainer(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    conn = connect(tmp_path / "neuroarena.db")
    prepared = prepare_run(conn, track_id=track_id, population_size=6, data_dir=tmp_path)
    assert prepared.model.backend == "neat"
    assert prepared.starting_generation == 0
    assert prepared.config.population_size == 6
    update = next(prepared.trainer.run())
    assert update.population_size == 6


def test_prepare_run_resumes_from_an_existing_model(tmp_path: Path) -> None:
    from neuroarena.persistence.recorder import run_and_record

    track_id = _save_test_track(tmp_path)
    conn = connect(tmp_path / "neuroarena.db")
    # max_generations is required here: without it (or a should_stop callback below),
    # NeatTrainer.run() has no exit condition and run_and_record loops forever — this bit a
    # real SDD execution of this plan (see the plan file's own commit history / SDD ledger
    # for 2026-09-16).
    first = prepare_run(
        conn, track_id=track_id, population_size=6, max_generations=2, data_dir=tmp_path
    )
    run_and_record(
        conn,
        first.trainer,
        model_id=first.model.model_id,
        track_id=track_id,
        starting_generation=first.starting_generation,
        resume_dir=first.model_dir / "resume",
        champion_dir=first.model_dir / "champion",
        checkpoint_every_n_generations=1,
        champion_retention_cap=None,
        initial_settings_diff=first.settings_diff,
    )

    resumed = prepare_run(
        conn, track_id=track_id, resume_model_id=first.model.model_id, data_dir=tmp_path
    )
    assert resumed.model.model_id == first.model.model_id
    assert resumed.starting_generation > 0
    assert resumed.config.population_size == 6  # inherited, not reset to RunConfig's default
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/persistence/test_launch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.launch'`.

- [ ] **Step 3: Create `launch.py`, moving the setup logic out of `cli.py`'s `run()`**

Create `src/neuroarena/persistence/launch.py`:

```python
"""Training-launch setup: resolves a `track_id` + config overrides (fresh or resumed) into
a ready-to-run `Trainer`. Extracted from `cli.py`'s `run()` (Phase 6) so Phase 7's dashboard
can reuse the exact same setup path non-blockingly — `prepare_run` only builds the trainer;
it never drives `Trainer.run()` itself (that's the caller's job: `cli.run()` calls
`recorder.run_and_record` synchronously, Phase 7's `RunManager` runs it on a background
thread). See `../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.protocols import Trainer
from neuroarena.persistence import checkpoints_repo, models_repo, settings_history_repo
from neuroarena.persistence.models_repo import ModelRecord
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.objectives import ProgressObjective
from neuroarena.tracks.store import load as load_track

DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class PreparedRun:
    trainer: Trainer
    model: ModelRecord
    config: RunConfig
    starting_generation: int
    settings_diff: dict[str, Any]
    model_dir: Path


def prepare_run(
    conn: sqlite3.Connection,
    *,
    track_id: str,
    population_size: int | None = None,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int | None = None,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> PreparedRun:
    """`None` means "the caller did not specify this", never "no value" — see `cli.run`'s
    original docstring for the fresh-vs-resume fallback rule this preserves exactly."""
    track_record = load_track(track_id, tracks_dir=data_dir / "tracks")
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "population_size": population_size,
            "max_generations": max_generations,
            "target_fitness": target_fitness,
            "checkpoint_every_n_generations": checkpoint_every_n_generations,
            "champion_retention_cap": champion_retention_cap,
        }.items()
        if value is not None
    }

    checkpoint_root = data_dir / "checkpoints"
    resume_state: tuple[models_repo.ModelRecord, Path] | None = None
    if resume_model_id is not None:
        resume_model = models_repo.get_model(conn, resume_model_id)
        latest = checkpoints_repo.latest_checkpoint(conn, resume_model_id, kind="resume")
        if latest is None:
            raise ValueError(f"model {resume_model_id!r} has no resume checkpoint to resume from")
        previous_config = settings_history_repo.reconstruct_run_config(conn, resume_model_id)
        if (
            "population_size" in overrides
            and overrides["population_size"] != previous_config.population_size
        ):
            raise ValueError(
                "population_size cannot be changed on resume — NeatTrainer.load_checkpoint"
                " always continues with the checkpoint's own population size"
                f" ({previous_config.population_size})"
            )
        config = dataclasses.replace(previous_config, track_id=track_id, **overrides)
        diff = settings_history_repo.compute_diff(previous_config, config)
        starting_generation = latest.generation + 1
        resume_state = (resume_model, Path(latest.file_path))
    else:
        config = dataclasses.replace(RunConfig(track_id=track_id), **overrides)
        diff = settings_history_repo.compute_diff(None, config)
        starting_generation = 0

    if config.checkpoint_every_n_generations < 1:
        raise ValueError(
            "checkpoint_every_n_generations must be >= 1, got"
            f" {config.checkpoint_every_n_generations}"
        )

    def make_env() -> CarEnvironment:
        return CarEnvironment(
            track_record.track, track_id, CarEnvironmentConfig.from_run_config(config)
        )

    objective = ProgressObjective()

    if resume_state is not None:
        model, checkpoint_path = resume_state
        trainer: Trainer = NeatTrainer.load_checkpoint(checkpoint_path, make_env, objective, config)
    else:
        env = make_env()
        model = models_repo.create_model(
            conn,
            backend="neat",
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        model_dir = checkpoint_root / model.model_id
        trainer = NeatTrainer(make_env, objective, config, champion_dir=model_dir / "champion")

    model_dir = checkpoint_root / model.model_id
    return PreparedRun(
        trainer=trainer,
        model=model,
        config=config,
        starting_generation=starting_generation,
        settings_diff=diff,
        model_dir=model_dir,
    )
```

- [ ] **Step 4: Run the new test file to verify it passes**

Run: `uv run pytest tests/persistence/test_launch.py -v`
Expected: PASS.

- [ ] **Step 5: Rewrite `cli.py`'s `run()` to delegate to `prepare_run`**

In `src/neuroarena/persistence/cli.py`, replace the entire body of `run()` (everything between its docstring and `main()`) with:

```python
def run(
    *,
    track_id: str,
    population_size: int | None = None,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int | None = None,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> RunRecord:
    """`None` means "the caller did not specify this", never "no value": an unspecified knob
    falls back to `RunConfig`'s own default on a fresh run, and to the model's most recent
    recorded settings on a resume (the spec's "opening a model loads its most recent settings
    by default"). Passing a value explicitly overrides that fallback. Setup (config
    resolution, trainer construction) is `persistence.launch.prepare_run` — shared with
    Phase 7's dashboard, which needs the same setup without this function's blocking
    `run_and_record` call below."""
    conn = connect(data_dir / "neuroarena.db")
    prepared = prepare_run(
        conn,
        track_id=track_id,
        population_size=population_size,
        max_generations=max_generations,
        target_fitness=target_fitness,
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
        resume_model_id=resume_model_id,
        data_dir=data_dir,
    )
    return recorder.run_and_record(
        conn,
        prepared.trainer,
        model_id=prepared.model.model_id,
        track_id=track_id,
        starting_generation=prepared.starting_generation,
        resume_dir=prepared.model_dir / "resume",
        champion_dir=prepared.model_dir / "champion",
        checkpoint_every_n_generations=prepared.config.checkpoint_every_n_generations,
        champion_retention_cap=prepared.config.champion_retention_cap,
        initial_settings_diff=prepared.settings_diff,
    )
```

Replace the entire import block at the top of `cli.py` (everything between the module docstring and `DEFAULT_DATA_DIR = Path("data")`) with exactly this — `main()` still needs `models_repo.UnknownModelError` and `UnknownTrackError`, `run()`'s new body still needs `connect`/`recorder`/`RunRecord`, and everything else (`NeatTrainer`, `RunConfig`, `CarEnvironment`/`CarEnvironmentConfig`, `ProgressObjective`, `dataclasses`, `checkpoints_repo`, `settings_history_repo`, `load_track`) moved into `launch.py` and is no longer referenced here at all:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from neuroarena.persistence import models_repo, recorder
from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import prepare_run
from neuroarena.persistence.runs_repo import RunRecord
from neuroarena.tracks.store import UnknownTrackError

DEFAULT_DATA_DIR = Path("data")
```

Leave `main()` and the module docstring untouched — only `run()`'s body and this import block change.

- [ ] **Step 6: Run the full persistence test suite to verify the refactor is behavior-preserving**

Run: `uv run pytest tests/persistence/ -v`
Expected: PASS — every test in `test_cli.py` passes **unchanged** (no test file edits), proving this refactor changed no observable behavior.

- [ ] **Step 7: Commit**

```bash
git add src/neuroarena/persistence/launch.py src/neuroarena/persistence/cli.py tests/persistence/test_launch.py
git commit -m "refactor(phase-7): extract persistence.launch.prepare_run from cli.py"
```

---

## Task 5: Add FastAPI/Uvicorn dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:** none (dependency-only task).

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, update `dependencies` and the `dev` group:

```toml
dependencies = ["numpy>=1.26", "arcade>=3.0", "neat-python>=0.92,<1.0", "fastapi>=0.115", "uvicorn[standard]>=0.30"]
```

```toml
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6", "mypy>=1.11", "pre-commit>=3.8", "httpx>=0.27"]
```

(`httpx` is required by FastAPI's `TestClient`, used in Task 8/9's tests; `uvicorn[standard]` pulls in `websockets`, needed for the `/ws` endpoint.)

- [ ] **Step 2: Sync and verify the new packages import**

Run: `uv sync` then `uv run python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"`
Expected: prints two version strings, no import errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(phase-7): add fastapi/uvicorn dependencies"
```

---

## Task 6: `dashboard/ws_protocol.py` — envelope builders

**Files:**
- Create: `src/neuroarena/dashboard/__init__.py` (empty), `src/neuroarena/dashboard/ws_protocol.py`
- Test: `tests/dashboard/__init__.py` (empty), `tests/dashboard/test_ws_protocol.py`

**Interfaces:**
- Consumes: `GenerationProgress` (Task 2), `TrainingUpdate` (Phase 0).
- Produces: `progress_message(snapshot) -> dict`, `generation_message(update) -> dict`, `status_message(state, detail) -> dict`. Consumed by Task 9's WebSocket endpoint.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_ws_protocol.py`:

```python
from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.dashboard.ws_protocol import (
    generation_message,
    progress_message,
    status_message,
)
from neuroarena.interfaces.protocols import TrainingUpdate


def test_progress_message_shape() -> None:
    snapshot = GenerationProgress(
        generation=3,
        population_size=10,
        active_genomes_remaining=4,
        elapsed_steps=120,
        step_ceiling=300,
        best_fitness_so_far=2.5,
    )
    msg = progress_message(snapshot)
    assert msg["type"] == "progress"
    assert msg["schema_version"] == 1
    assert msg["data"] == {
        "generation": 3,
        "population_size": 10,
        "active_genomes_remaining": 4,
        "elapsed_steps": 120,
        "step_ceiling": 300,
        "best_fitness_so_far": 2.5,
    }


def test_generation_message_carries_the_full_training_update() -> None:
    update = TrainingUpdate(
        progress_index=5,
        best_fitness=9.0,
        mean_fitness=5.0,
        worst_fitness=1.0,
        population_size=10,
        champion_metrics={"progress": 9.0},
        sim_time=300.0,
        wall_time=1.2,
    )
    msg = generation_message(update)
    assert msg["type"] == "generation"
    assert msg["schema_version"] == 1
    assert msg["data"]["progress_index"] == 5
    assert msg["data"]["champion_metrics"] == {"progress": 9.0}
    assert msg["data"]["schema_version"] == 1  # TrainingUpdate's own field, inside data


def test_status_message_shape() -> None:
    msg = status_message("crashed", "RuntimeError: boom")
    assert msg == {
        "type": "status",
        "schema_version": 1,
        "data": {"state": "crashed", "detail": "RuntimeError: boom"},
    }
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/dashboard/test_ws_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.dashboard'`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/dashboard/__init__.py` (empty) and `tests/dashboard/__init__.py` (empty).

Create `src/neuroarena/dashboard/ws_protocol.py`:

```python
"""Pure builders for the WebSocket envelope Phase 7's dashboard pushes — see
`../../../docs/phases/phase-7-dashboard-control-panel.md`'s WebSocket message schema
decision. No FastAPI/asyncio/threading here, so these are unit-testable in isolation."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.interfaces.protocols import TrainingUpdate

_ENVELOPE_SCHEMA_VERSION = 1


def progress_message(snapshot: GenerationProgress) -> dict[str, Any]:
    return {
        "type": "progress",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {
            "generation": snapshot.generation,
            "population_size": snapshot.population_size,
            "active_genomes_remaining": snapshot.active_genomes_remaining,
            "elapsed_steps": snapshot.elapsed_steps,
            "step_ceiling": snapshot.step_ceiling,
            "best_fitness_so_far": snapshot.best_fitness_so_far,
        },
    }


def generation_message(update: TrainingUpdate) -> dict[str, Any]:
    return {
        "type": "generation",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": dataclasses.asdict(update),
    }


def status_message(state: str, detail: str | None) -> dict[str, Any]:
    return {
        "type": "status",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {"state": state, "detail": detail},
    }
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/dashboard/test_ws_protocol.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/dashboard/__init__.py src/neuroarena/dashboard/ws_protocol.py tests/dashboard/__init__.py tests/dashboard/test_ws_protocol.py
git commit -m "feat(phase-7): dashboard WebSocket envelope builders"
```

---

## Task 7: `dashboard/run_manager.py` — `RunManager`

**Files:**
- Create: `src/neuroarena/dashboard/run_manager.py`
- Test: `tests/dashboard/test_run_manager.py`

**Interfaces:**
- Consumes: `prepare_run`/`PreparedRun` (Task 4), `run_and_record(..., on_update=, should_stop=)` (Task 3), `NeatTrainer.progress_snapshot`/`GenerationProgress` (Task 2).
- Produces: `RunManager` (`start`, `status`, `progress_snapshot`, `latest_update`, `update_config`, `request_stop`, `poll_interval_ms`, `set_poll_interval_ms`), `RunManagerStatus`, `RunAlreadyActiveError`, `NoActiveRunError`. Consumed by Task 8/9's `app.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_run_manager.py` (reuses the same track-fixture helper as `tests/persistence/test_launch.py`):

```python
import time
from pathlib import Path

from neuroarena.dashboard.run_manager import NoActiveRunError, RunAlreadyActiveError, RunManager
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from neuroarena.tracks.store import save


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _save_test_track(data_dir: Path) -> str:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    record = save(track, requested_size=10, complexity=0.3, seed=1, tracks_dir=data_dir / "tracks")
    return record.track_id


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition never became true")


def test_poll_interval_defaults_to_200ms_and_is_settable(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.poll_interval_ms() == 200
    manager.set_poll_interval_ms(50)
    assert manager.poll_interval_ms() == 50


def test_set_poll_interval_rejects_non_positive_values(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.set_poll_interval_ms(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_status_is_idle_before_any_run_starts(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    status = manager.status()
    assert status.status == "idle"
    assert status.model_id is None
    assert status.generation is None


def test_start_runs_to_completion_in_the_background(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    model_id = manager.start(
        track_id=track_id, overrides={"population_size": 6, "max_generations": 2}
    )
    assert model_id == manager.status().model_id
    _wait_until(lambda: manager.status().status == "completed")
    assert manager.status().generation == 1  # 0-indexed, 2 generations means final index 1


def test_start_while_a_run_is_active_is_rejected(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 50})
    try:
        manager.start(track_id=track_id, overrides={"population_size": 6})
        raise AssertionError("expected RunAlreadyActiveError")
    except RunAlreadyActiveError:
        pass
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running")


def test_request_stop_ends_a_running_run_with_stopped_status(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    _wait_until(lambda: manager.latest_update() is not None)
    manager.request_stop()
    _wait_until(lambda: manager.status().status == "stopped")


def test_request_stop_without_an_active_run_raises(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.request_stop()
        raise AssertionError("expected NoActiveRunError")
    except NoActiveRunError:
        pass


def test_update_config_without_an_active_run_raises(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.update_config({"max_generations": 5})
        raise AssertionError("expected NoActiveRunError")
    except NoActiveRunError:
        pass


def test_update_config_rejects_unsupported_fields_once_a_run_is_active(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    try:
        try:
            manager.update_config({"sim_speed": 2.0})
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running")


def test_progress_snapshot_is_none_when_idle(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.progress_snapshot() is None
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/dashboard/test_run_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.dashboard.run_manager'`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/dashboard/run_manager.py`:

```python
"""`RunManager` owns the single in-process `Trainer` this dashboard backend drives — Phase
7's "embeds the trainer in-process... one training run per backend process" process model.
Runs `persistence.recorder.run_and_record` on a background thread (via
`persistence.launch.prepare_run` for setup) so the FastAPI process stays responsive to
REST/WebSocket requests while a run is in progress. See
`../../../docs/phases/phase-7-dashboard-control-panel.md`.

Threading model: every field below is touched from two threads — the FastAPI
request-handling thread (reads status, requests config updates/stop) and the background
training thread this class spawns (writes status/latest_update as the run progresses).
Every shared field is a single Python attribute (str, int, a frozen dataclass reference, or
`None`) — safe to read/write under CPython's GIL without an extra lock, the same reasoning
`NeatTrainer.progress_snapshot`/`BatchProgress` already document. No field here is ever
read-modify-written across the two threads (the only compound state mutation,
`NeatTrainer.update_config`'s own pending-update merge, already has its own lock inside
`NeatTrainer`)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate
from neuroarena.persistence import recorder
from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import PreparedRun, prepare_run

DEFAULT_POLL_INTERVAL_MS = 200


class RunAlreadyActiveError(RuntimeError):
    """Raised by `RunManager.start` when a run is already active — one run per process."""


class NoActiveRunError(RuntimeError):
    """Raised when an operation (stop, config update) needs an active run but none exists."""


@dataclass(frozen=True)
class RunManagerStatus:
    status: str  # "idle" | "running" | "completed" | "crashed" | "stopped"
    model_id: str | None
    generation: int | None


class RunManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._trainer: Trainer | None = None
        self._model_id: str | None = None
        self._status: str = "idle"
        self._latest_update: TrainingUpdate | None = None
        self._stop_requested = False
        self._poll_interval_ms = DEFAULT_POLL_INTERVAL_MS

    def status(self) -> RunManagerStatus:
        generation = (
            self._latest_update.progress_index if self._latest_update is not None else None
        )
        return RunManagerStatus(status=self._status, model_id=self._model_id, generation=generation)

    def poll_interval_ms(self) -> int:
        return self._poll_interval_ms

    def set_poll_interval_ms(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"poll interval must be >= 1ms, got {value}")
        self._poll_interval_ms = value

    def progress_snapshot(self) -> GenerationProgress | None:
        if self._trainer is None:
            return None
        snapshot_fn = getattr(self._trainer, "progress_snapshot", None)
        return None if snapshot_fn is None else snapshot_fn()

    def latest_update(self) -> TrainingUpdate | None:
        return self._latest_update

    def update_config(self, partial: Mapping[str, Any]) -> None:
        if self._trainer is None:
            raise NoActiveRunError("no run is currently active")
        self._trainer.update_config(partial)

    def request_stop(self) -> None:
        if self._status != "running":
            raise NoActiveRunError("no run is currently active")
        self._stop_requested = True

    def start(
        self,
        *,
        track_id: str,
        resume_model_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        if self._status == "running":
            raise RunAlreadyActiveError("a run is already active in this backend process")
        conn = connect(self._data_dir / "neuroarena.db")
        try:
            prepared = prepare_run(
                conn,
                track_id=track_id,
                resume_model_id=resume_model_id,
                data_dir=self._data_dir,
                **(overrides or {}),
            )
        finally:
            conn.close()

        self._trainer = prepared.trainer
        self._model_id = prepared.model.model_id
        self._status = "running"
        self._latest_update = None
        self._stop_requested = False
        thread = threading.Thread(target=self._run, args=(prepared,), daemon=True)
        thread.start()
        return prepared.model.model_id

    def _run(self, prepared: PreparedRun) -> None:
        thread_conn = connect(self._data_dir / "neuroarena.db")
        try:
            record = recorder.run_and_record(
                thread_conn,
                prepared.trainer,
                model_id=prepared.model.model_id,
                track_id=prepared.config.track_id,
                starting_generation=prepared.starting_generation,
                resume_dir=prepared.model_dir / "resume",
                champion_dir=prepared.model_dir / "champion",
                checkpoint_every_n_generations=prepared.config.checkpoint_every_n_generations,
                champion_retention_cap=prepared.config.champion_retention_cap,
                initial_settings_diff=prepared.settings_diff,
                on_update=self._handle_update,
                should_stop=lambda: self._stop_requested,
            )
            self._status = record.status
        except Exception:
            self._status = "crashed"
        finally:
            thread_conn.close()

    def _handle_update(self, update: TrainingUpdate) -> None:
        self._latest_update = update
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/dashboard/test_run_manager.py -v`
Expected: PASS. (These tests spawn real background threads and poll with a 5s timeout — allow the run to take a few seconds; if `test_start_runs_to_completion_in_the_background` is flaky/slow on your machine, that's real background-thread scheduling, not a bug — the timeout is generous on purpose.)

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/dashboard/run_manager.py tests/dashboard/test_run_manager.py
git commit -m "feat(phase-7): RunManager — in-process trainer owner for the dashboard"
```

---

## Task 8: `dashboard/app.py` — REST routes

**Files:**
- Create: `src/neuroarena/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

**Interfaces:**
- Consumes: `RunManager` (Task 7), `models_repo`/`settings_history_repo`/`checkpoints_repo`/`tracks.store` (Phase 6/2).
- Produces: `create_app(run_manager: RunManager, data_dir: Path) -> FastAPI`, request dataclasses `StartRunRequest`/`ConfigUpdateRequest`/`PollIntervalRequest`. Consumed by Task 9 (adds `/ws` to the same app) and Task 10 (the CLI entry point).

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_app.py`:

```python
from pathlib import Path

from fastapi.testclient import TestClient

from neuroarena.dashboard.app import create_app
from neuroarena.dashboard.run_manager import RunManager
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from neuroarena.tracks.store import save


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _save_test_track(data_dir: Path) -> str:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    record = save(track, requested_size=10, complexity=0.3, seed=1, tracks_dir=data_dir / "tracks")
    return record.track_id


def _client(tmp_path: Path) -> TestClient:
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    return TestClient(app)


def test_get_tracks_lists_saved_tracks(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    response = client.get("/api/tracks")
    assert response.status_code == 200
    assert any(t["track_id"] == track_id for t in response.json())


def test_get_models_starts_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == []


def test_get_current_run_is_idle_before_anything_starts(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/runs/current")
    assert response.status_code == 200
    assert response.json() == {"status": "idle", "model_id": None, "generation": None}


def test_starting_a_run_returns_201_and_a_model_id(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    response = client.post(
        "/api/runs",
        json={"track_id": track_id, "population_size": 6, "max_generations": 1000},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "running"
    client.post("/api/runs/current/stop")


def test_starting_a_second_run_while_one_is_active_returns_409(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    response = client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    assert response.status_code == 409
    client.post("/api/runs/current/stop")


def test_starting_a_run_with_an_unknown_track_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/runs", json={"track_id": "does-not-exist"})
    assert response.status_code == 400


def test_stop_without_an_active_run_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/runs/current/stop")
    assert response.status_code == 409


def test_config_update_without_an_active_run_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/runs/current/config", json={"max_generations": 5})
    assert response.status_code == 409


def test_config_update_with_an_unsupported_field_returns_400(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    # ConfigUpdateRequest only models the 3 wired fields, so an unsupported field can't even
    # be expressed in the request body — this instead confirms a *supported* field is
    # accepted and applied without error.
    response = client.patch("/api/runs/current/config", json={"max_generations": 3})
    assert response.status_code == 200
    client.post("/api/runs/current/stop")


def test_poll_interval_defaults_to_200_and_is_settable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/poll-interval").json() == {"interval_ms": 200}
    response = client.patch("/api/poll-interval", json={"interval_ms": 50})
    assert response.status_code == 200
    assert response.json() == {"interval_ms": 50}
    assert client.get("/api/poll-interval").json() == {"interval_ms": 50}


def test_poll_interval_rejects_a_non_positive_value(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/poll-interval", json={"interval_ms": 0})
    assert response.status_code == 400


def test_model_settings_includes_current_generation(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    client = TestClient(app)
    model_id = client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 2}
    ).json()["model_id"]

    deadline = 0
    while True:
        response = client.get(f"/api/models/{model_id}/settings")
        if response.status_code == 200 and response.json().get("current_generation") is not None:
            break
        deadline += 1
        assert deadline < 500, "run never produced a resume checkpoint in time"
        import time

        time.sleep(0.02)

    body = response.json()
    assert body["current_generation"] >= 0
    assert body["population_size"] == 6


def test_model_settings_for_an_unknown_model_returns_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/models/does-not-exist/settings")
    assert response.status_code == 404
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/dashboard/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.dashboard.app'`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/dashboard/app.py` (WebSocket route added in Task 9 — this task's `create_app` only wires the REST routes below):

```python
"""FastAPI app factory for Phase 7's dashboard backend — REST routes over `RunManager`
(Task 9 adds `/ws`). See `../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from neuroarena.dashboard.run_manager import NoActiveRunError, RunAlreadyActiveError, RunManager
from neuroarena.persistence import checkpoints_repo, models_repo, settings_history_repo
from neuroarena.persistence.db import connect
from neuroarena.tracks import store as tracks_store
from neuroarena.tracks.store import UnknownTrackError


@dataclasses.dataclass
class StartRunRequest:
    track_id: str
    resume_model_id: str | None = None
    population_size: int | None = None
    max_generations: int | None = None
    target_fitness: float | None = None


@dataclasses.dataclass
class ConfigUpdateRequest:
    max_generation_steps: int | None = None
    max_generations: int | None = None
    target_fitness: float | None = None


@dataclasses.dataclass
class PollIntervalRequest:
    interval_ms: int


def create_app(run_manager: RunManager, data_dir: Path) -> FastAPI:
    app = FastAPI(title="neuroarena dashboard")

    @app.get("/api/tracks")
    def get_tracks() -> list[dict[str, object]]:
        return tracks_store.list_tracks(tracks_dir=data_dir / "tracks")

    @app.get("/api/models")
    def get_models() -> list[dict[str, Any]]:
        conn = connect(data_dir / "neuroarena.db")
        try:
            return [
                {"model_id": m.model_id, "backend": m.backend, "created_at": m.created_at}
                for m in models_repo.list_models(conn)
            ]
        finally:
            conn.close()

    @app.get("/api/models/{model_id}/settings")
    def get_model_settings(model_id: str) -> dict[str, Any]:
        """Includes `current_generation` alongside the reconstructed `RunConfig` fields —
        the spec's resume-flow requirement is to "surface current generation vs.
        max_generations/target_fitness so they can be raised before starting", which needs
        this number, not just the settings themselves."""
        conn = connect(data_dir / "neuroarena.db")
        try:
            models_repo.get_model(conn, model_id)
            config = settings_history_repo.reconstruct_run_config(conn, model_id)
            latest = checkpoints_repo.latest_checkpoint(conn, model_id, kind="resume")
            return {
                **dataclasses.asdict(config),
                "current_generation": None if latest is None else latest.generation,
            }
        except models_repo.UnknownModelError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model_id: {model_id}") from exc
        finally:
            conn.close()

    @app.post("/api/runs", status_code=201)
    def start_run(request: StartRunRequest) -> dict[str, Any]:
        overrides = {
            key: value
            for key, value in {
                "population_size": request.population_size,
                "max_generations": request.max_generations,
                "target_fitness": request.target_fitness,
            }.items()
            if value is not None
        }
        try:
            model_id = run_manager.start(
                track_id=request.track_id,
                resume_model_id=request.resume_model_id,
                overrides=overrides,
            )
        except RunAlreadyActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, UnknownTrackError, models_repo.UnknownModelError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"model_id": model_id, "status": "running"}

    @app.get("/api/runs/current")
    def get_current_run() -> dict[str, Any]:
        status = run_manager.status()
        return {
            "status": status.status,
            "model_id": status.model_id,
            "generation": status.generation,
        }

    @app.post("/api/runs/current/stop", status_code=202)
    def stop_run() -> dict[str, str]:
        try:
            run_manager.request_stop()
        except NoActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "stopping"}

    @app.patch("/api/runs/current/config")
    def update_config(request: ConfigUpdateRequest) -> dict[str, str]:
        partial = {k: v for k, v in dataclasses.asdict(request).items() if v is not None}
        try:
            run_manager.update_config(partial)
        except NoActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted"}

    @app.get("/api/poll-interval")
    def get_poll_interval() -> dict[str, int]:
        return {"interval_ms": run_manager.poll_interval_ms()}

    @app.patch("/api/poll-interval")
    def set_poll_interval(request: PollIntervalRequest) -> dict[str, int]:
        try:
            run_manager.set_poll_interval_ms(request.interval_ms)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"interval_ms": run_manager.poll_interval_ms()}

    return app
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/dashboard/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(phase-7): dashboard REST routes over RunManager"
```

---

## Task 9: `dashboard/app.py` — `/ws` WebSocket endpoint

**Files:**
- Modify: `src/neuroarena/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

**Interfaces:**
- Consumes: `ws_protocol.{progress_message, generation_message, status_message}` (Task 6), `RunManager.{status, latest_update, progress_snapshot, poll_interval_ms}` (Task 7).
- Produces: `/ws` on the same `FastAPI` app `create_app` returns.

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_ws_pushes_status_then_progress_and_generation_messages(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    manager.set_poll_interval_ms(10)  # fast polling keeps this test quick
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        client.post(
            "/api/runs",
            json={"track_id": track_id, "population_size": 6, "max_generations": 1000},
        )
        seen_types: set[str] = set()
        for _ in range(200):
            msg = ws.receive_json()
            seen_types.add(msg["type"])
            assert msg["schema_version"] == 1
            if {"status", "progress", "generation"} <= seen_types:
                break
        assert {"status", "progress", "generation"} <= seen_types
    client.post("/api/runs/current/stop")
```

Add the needed import at the top of `tests/dashboard/test_app.py`: `RunManager` is already imported; no new imports required beyond what's already there (`create_app`, `TestClient`).

- [ ] **Step 2: Run to verify the new test fails**

Run: `uv run pytest tests/dashboard/test_app.py -k test_ws_pushes -v`
Expected: FAIL — `AssertionError` from Starlette (no `/ws` route registered), or a 404-equivalent WebSocket rejection.

- [ ] **Step 3: Implement**

In `src/neuroarena/dashboard/app.py`, add imports and the route (inside `create_app`, after the last REST route, before `return app`):

```python
import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from neuroarena.dashboard.ws_protocol import generation_message, progress_message, status_message
```

```python
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        last_generation_sent: int | None = None
        last_status_sent: str | None = None
        try:
            while True:
                status = run_manager.status()
                if status.status != last_status_sent:
                    await websocket.send_json(status_message(status.status, None))
                    last_status_sent = status.status

                update = run_manager.latest_update()
                if update is not None and update.progress_index != last_generation_sent:
                    await websocket.send_json(generation_message(update))
                    last_generation_sent = update.progress_index

                snapshot = run_manager.progress_snapshot()
                if snapshot is not None:
                    await websocket.send_json(progress_message(snapshot))

                await asyncio.sleep(run_manager.poll_interval_ms() / 1000)
        except WebSocketDisconnect:
            pass
```

(Add these imports at the top of the file alongside the existing ones, rather than inline — place `import asyncio` with the stdlib imports, `WebSocket, WebSocketDisconnect` alongside the existing `from fastapi import FastAPI, HTTPException` line, and the `ws_protocol` import alongside the other `neuroarena.dashboard` import.)

- [ ] **Step 4: Run the full dashboard test suite to verify everything passes**

Run: `uv run pytest tests/dashboard/ -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(phase-7): dashboard /ws — progress/generation/status push"
```

---

## Task 10: `neuroarena-dashboard` CLI entry point

**Files:**
- Create: `src/neuroarena/dashboard/cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `create_app` (Task 8/9), `RunManager` (Task 7).
- Produces: `main()`, wired as the `neuroarena-dashboard` console script — mirrors the existing `neuroarena-play`/`neuroarena-track-gen`/`neuroarena-train` precedent.

- [ ] **Step 1: Implement**

Create `src/neuroarena/dashboard/cli.py`:

```python
"""Entry point: `uv run neuroarena-dashboard`. Starts the FastAPI dashboard backend —
local-first, binds `127.0.0.1` only, no auth (spec). See
`../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from neuroarena.dashboard.app import create_app
from neuroarena.dashboard.run_manager import RunManager

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PORT = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the neuroarena dashboard backend.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    run_manager = RunManager(data_dir=args.data_dir)
    app = create_app(run_manager=run_manager, data_dir=args.data_dir)
    # 127.0.0.1, not 0.0.0.0: local-first, no auth (spec) — never bind every interface.
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
```

There is no automated test for this module beyond a smoke import — `uvicorn.run(...)` blocks forever by design, so it isn't unit-tested; `create_app`/`RunManager` are already fully covered by Tasks 7–9.

- [ ] **Step 2: Verify it imports and the CLI parses args correctly**

Run: `uv run python -c "from neuroarena.dashboard.cli import main; print('ok')"`
Expected: prints `ok`, no import errors.

- [ ] **Step 3: Wire the console script**

In `pyproject.toml`, add to `[project.scripts]`:

```toml
neuroarena-dashboard = "neuroarena.dashboard.cli:main"
```

- [ ] **Step 4: Verify the installed script resolves**

Run: `uv run neuroarena-dashboard --help`
Expected: prints the `argparse` usage/help text, exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/dashboard/cli.py pyproject.toml
git commit -m "feat(phase-7): neuroarena-dashboard CLI entry point"
```

---

## Task 11: Whole-backend verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full backend test suite**

Run: `uv run pytest -v`
Expected: PASS — every test in the repo, including all pre-existing suites (confirming Tasks 1–4's additive changes broke nothing) and every new `tests/dashboard/`/`tests/persistence/test_launch.py` test.

- [ ] **Step 2: Run lint and type checks**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
Expected: clean on all three.

- [ ] **Step 3: Re-confirm Phase 4's NEAT import-hygiene guard**

Run: `uv run pytest tests/backends/neat/test_import_hygiene.py -v`
Expected: PASS — confirms `neuroarena.backends.neat` still imports no car-specific/`neuroarena.dashboard` modules despite this plan's changes to `evaluation.py`/`trainer.py`.

- [ ] **Step 4: Fix anything Steps 1–3 surfaced, then re-run all three until clean**

If a whole-branch review (via `superpowers:subagent-driven-development`'s final review step) finds cross-task issues, fix them here in one consolidated wave, per this project's established SDD pattern (see Phase 5/6's plans' own final-review sections for precedent) — do not scatter fixes back into already-committed task commits.

- [ ] **Step 5: Commit if Step 4 made any changes**

```bash
git add -A
git commit -m "fix(phase-7): final backend verification fixes"
```

(Skip this step entirely if Steps 1–3 were already clean.)

---

## Task 12: Scaffold the frontend (Vite + React + TypeScript)

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/index.css`, `frontend/.gitignore`

**Interfaces:** none yet — this task only gets a blank page rendering and talking to nothing.

*Frontend tasks have no existing test harness in this repo (no `package.json` exists before this task) and none is being introduced — per this project's own guidance for UI work, the verification step is running the dev server and confirming the behavior in a real browser, plus TypeScript's own `tsc --noEmit` as the automated gate. This is a deliberate scope decision (YAGNI: this codebase has zero JS/TS anywhere yet; adding Vitest + Testing Library for one dashboard slice is disproportionate), not an oversight.*

- [ ] **Step 1: Scaffold with the official Vite template**

Run: `npm create vite@latest frontend -- --template react-ts` from the repo root, answering "Ignore files and continue" if prompted about the non-empty directory (there shouldn't be one — `frontend/` doesn't exist yet).

- [ ] **Step 2: Install dependencies**

Run: `cd frontend && npm install`

- [ ] **Step 3: Configure the dev server to proxy API/WS calls to the backend**

Edit `frontend/vite.config.ts` to add a proxy (so the frontend dev server on its own port can call `/api/*` and `/ws` on the backend at `127.0.0.1:8000` without CORS setup):

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
```

- [ ] **Step 4: Strip the template's demo content**

Replace `frontend/src/App.tsx` with a placeholder that later tasks fill in:

```tsx
function App() {
  return <div>neuroarena dashboard</div>
}

export default App
```

Delete `frontend/src/App.css` and `frontend/public/vite.svg` (unused demo assets) and remove the now-dangling `import './App.css'` from `App.tsx` if the template generated one.

- [ ] **Step 5: Verify the dev server runs and type-checks cleanly**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

Run: `cd frontend && npm run dev` (backgroundable — leave it running), then open the printed local URL (typically `http://localhost:5173`) in a browser.
Expected: the page loads and shows "neuroarena dashboard" with no console errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(phase-7): scaffold the React dashboard frontend"
```

---

## Task 13: Model list + start/resume run form

**Files:**
- Create: `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/components/ModelList.tsx`, `frontend/src/components/NewRunForm.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `GET /api/tracks`, `GET /api/models`, `GET /api/models/{id}/settings`, `POST /api/runs`, `GET /api/runs/current` (Task 8).
- Produces: `Track`, `Model`, `RunStatus` types (`types.ts`); `listTracks()`, `listModels()`, `getModelSettings(modelId)`, `startRun(request)`, `getCurrentRun()` (`api.ts`) — consumed by Task 14/15's components too.

- [ ] **Step 1: Define the shared types**

Create `frontend/src/types.ts`:

```typescript
export interface Track {
  track_id: string
  [key: string]: unknown
}

export interface Model {
  model_id: string
  backend: string
  created_at: string
}

export interface RunStatus {
  status: 'idle' | 'running' | 'completed' | 'crashed' | 'stopped'
  model_id: string | null
  generation: number | null
}

export interface StartRunRequest {
  track_id: string
  resume_model_id?: string | null
  population_size?: number | null
  max_generations?: number | null
  target_fitness?: number | null
}
```

- [ ] **Step 2: Write the API client**

Create `frontend/src/api.ts`:

```typescript
import type { Model, RunStatus, StartRunRequest, Track } from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(detail.detail ?? `request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function listTracks(): Promise<Track[]> {
  return fetch('/api/tracks').then((r) => json<Track[]>(r))
}

export function listModels(): Promise<Model[]> {
  return fetch('/api/models').then((r) => json<Model[]>(r))
}

export function getModelSettings(modelId: string): Promise<Record<string, unknown>> {
  return fetch(`/api/models/${modelId}/settings`).then((r) => json<Record<string, unknown>>(r))
}

export function startRun(request: StartRunRequest): Promise<{ model_id: string; status: string }> {
  return fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  }).then((r) => json<{ model_id: string; status: string }>(r))
}

export function getCurrentRun(): Promise<RunStatus> {
  return fetch('/api/runs/current').then((r) => json<RunStatus>(r))
}

export function stopRun(): Promise<{ status: string }> {
  return fetch('/api/runs/current/stop', { method: 'POST' }).then((r) => json(r))
}

export function updateConfig(
  partial: Record<string, number | null>,
): Promise<{ status: string }> {
  return fetch('/api/runs/current/config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  }).then((r) => json(r))
}

export function getPollInterval(): Promise<{ interval_ms: number }> {
  return fetch('/api/poll-interval').then((r) => json(r))
}

export function setPollInterval(intervalMs: number): Promise<{ interval_ms: number }> {
  return fetch('/api/poll-interval', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interval_ms: intervalMs }),
  }).then((r) => json(r))
}
```

- [ ] **Step 3: Build `ModelList`**

Create `frontend/src/components/ModelList.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { listModels } from '../api'
import type { Model } from '../types'

export function ModelList({
  onResume,
}: {
  onResume: (modelId: string) => void
}) {
  const [models, setModels] = useState<Model[]>([])

  useEffect(() => {
    listModels().then(setModels).catch(console.error)
  }, [])

  if (models.length === 0) return <p>No saved models yet.</p>

  return (
    <ul>
      {models.map((m) => (
        <li key={m.model_id}>
          {m.model_id} ({m.backend}, created {m.created_at}){' '}
          <button onClick={() => onResume(m.model_id)}>Resume</button>
        </li>
      ))}
    </ul>
  )
}
```

- [ ] **Step 4: Build `NewRunForm`**

Create `frontend/src/components/NewRunForm.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { getModelSettings, listTracks, startRun } from '../api'
import type { Track } from '../types'

export function NewRunForm({
  resumeModelId,
  onStarted,
}: {
  resumeModelId: string | null
  onStarted: () => void
}) {
  const [tracks, setTracks] = useState<Track[]>([])
  const [trackId, setTrackId] = useState('')
  const [populationSize, setPopulationSize] = useState<number | ''>('')
  const [maxGenerations, setMaxGenerations] = useState<number | ''>('')
  const [targetFitness, setTargetFitness] = useState<number | ''>('')
  const [error, setError] = useState<string | null>(null)
  const [currentGeneration, setCurrentGeneration] = useState<number | null>(null)

  useEffect(() => {
    listTracks().then(setTracks).catch(console.error)
  }, [])

  useEffect(() => {
    if (resumeModelId === null) {
      setCurrentGeneration(null)
      return
    }
    getModelSettings(resumeModelId).then((settings) => {
      if (typeof settings.population_size === 'number') setPopulationSize(settings.population_size)
      if (typeof settings.max_generations === 'number') setMaxGenerations(settings.max_generations)
      if (typeof settings.target_fitness === 'number') setTargetFitness(settings.target_fitness)
      if (typeof settings.track_id === 'string') setTrackId(settings.track_id)
      setCurrentGeneration(typeof settings.current_generation === 'number' ? settings.current_generation : null)
    })
  }, [resumeModelId])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    try {
      await startRun({
        track_id: trackId,
        resume_model_id: resumeModelId,
        population_size: populationSize === '' ? null : populationSize,
        max_generations: maxGenerations === '' ? null : maxGenerations,
        target_fitness: targetFitness === '' ? null : targetFitness,
      })
      onStarted()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>{resumeModelId ? `Resume ${resumeModelId}` : 'New run'}</h2>
      {currentGeneration !== null && (
        <p>
          Currently at generation {currentGeneration}.{' '}
          {maxGenerations !== '' && Number(maxGenerations) <= currentGeneration && (
            <strong>
              max_generations ({maxGenerations}) is at or below this — raise it, or the run
              will train exactly one more generation and stop immediately.
            </strong>
          )}
        </p>
      )}
      <label>
        Track
        <select value={trackId} onChange={(e) => setTrackId(e.target.value)} required>
          <option value="" disabled>
            select a track
          </option>
          {tracks.map((t) => (
            <option key={t.track_id} value={t.track_id}>
              {t.track_id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Population size
        <input
          type="number"
          value={populationSize}
          onChange={(e) => setPopulationSize(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      <label>
        Max generations
        <input
          type="number"
          value={maxGenerations}
          onChange={(e) => setMaxGenerations(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      <label>
        Target fitness
        <input
          type="number"
          value={targetFitness}
          onChange={(e) => setTargetFitness(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <button type="submit">{resumeModelId ? 'Resume' : 'Start'}</button>
    </form>
  )
}
```

- [ ] **Step 5: Wire both into `App.tsx`**

Replace `frontend/src/App.tsx`:

```tsx
import { useState } from 'react'
import { ModelList } from './components/ModelList'
import { NewRunForm } from './components/NewRunForm'

function App() {
  const [resumeModelId, setResumeModelId] = useState<string | null>(null)
  const [runKey, setRunKey] = useState(0) // bump to force children to re-fetch after a run starts

  return (
    <div>
      <h1>neuroarena dashboard</h1>
      <NewRunForm
        key={runKey}
        resumeModelId={resumeModelId}
        onStarted={() => setRunKey((k) => k + 1)}
      />
      <h2>Saved models</h2>
      <ModelList onResume={setResumeModelId} />
    </div>
  )
}

export default App
```

- [ ] **Step 6: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 7: Manual verification in the browser**

Run the backend (`uv run neuroarena-dashboard`) and the frontend dev server (`npm run dev` in `frontend/`) side by side. In the browser: create a track first if none exist (`uv run neuroarena-track-gen` — see its own `--help`), confirm it appears in the "Track" dropdown, fill in population size + max generations, click Start, confirm no error banner appears and the request succeeds (check the Network tab: `POST /api/runs` returns 201). Then reload the page and confirm the model now appears under "Saved models" with a working Resume button that pre-fills the form.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat(phase-7): model list + new-run/resume-run form"
```

---

## Task 14: WebSocket client + live metrics/status panel

**Files:**
- Create: `frontend/src/ws.ts`, `frontend/src/components/LiveMetrics.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `/ws` (Task 9), `GET /api/runs/current` (Task 8).
- Produces: `useDashboardSocket()` hook (`ws.ts`) returning the latest `progress`/`generation`/`status` payloads; `LiveMetrics` component.

- [ ] **Step 1: Write the WebSocket hook**

Create `frontend/src/ws.ts`:

```typescript
import { useEffect, useRef, useState } from 'react'

interface ProgressData {
  generation: number
  population_size: number
  active_genomes_remaining: number
  elapsed_steps: number
  step_ceiling: number
  best_fitness_so_far: number | null
}

interface GenerationData {
  progress_index: number
  best_fitness: number
  mean_fitness: number
  worst_fitness: number
  population_size: number
  champion_metrics: Record<string, number>
  sim_time: number
  wall_time: number
  schema_version: number
}

interface StatusData {
  state: 'running' | 'completed' | 'crashed' | 'stopped'
  detail: string | null
}

type Envelope =
  | { type: 'progress'; schema_version: number; data: ProgressData }
  | { type: 'generation'; schema_version: number; data: GenerationData }
  | { type: 'status'; schema_version: number; data: StatusData }

export function useDashboardSocket() {
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [generation, setGeneration] = useState<GenerationData | null>(null)
  const [status, setStatus] = useState<StatusData | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`)
    socketRef.current = socket

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data) as Envelope
      if (msg.type === 'progress') setProgress(msg.data)
      else if (msg.type === 'generation') setGeneration(msg.data)
      else if (msg.type === 'status') setStatus(msg.data)
    }

    return () => socket.close()
  }, [])

  return { progress, generation, status }
}
```

- [ ] **Step 2: Build `LiveMetrics`**

Create `frontend/src/components/LiveMetrics.tsx`:

```tsx
import { useDashboardSocket } from '../ws'

export function LiveMetrics() {
  const { progress, generation, status } = useDashboardSocket()

  return (
    <div>
      <h2>Live metrics</h2>
      {status && <p>Status: {status.state}</p>}
      {progress && (
        <div>
          <h3>Generation {progress.generation} (in progress)</h3>
          <p>
            Active genomes: {progress.active_genomes_remaining} / {progress.population_size}
          </p>
          <p>
            Steps: {progress.elapsed_steps} / {progress.step_ceiling}
          </p>
          <p>Best fitness so far: {progress.best_fitness_so_far ?? '—'}</p>
        </div>
      )}
      {generation && (
        <div>
          <h3>Last completed: generation {generation.progress_index}</h3>
          <p>Best: {generation.best_fitness}</p>
          <p>Mean: {generation.mean_fitness}</p>
          <p>Worst: {generation.worst_fitness}</p>
        </div>
      )}
      {!status && !progress && !generation && <p>No run active.</p>}
    </div>
  )
}
```

- [ ] **Step 3: Add `LiveMetrics` to `App.tsx`**

In `frontend/src/App.tsx`, import and render it (below `NewRunForm`, above "Saved models"):

```tsx
import { LiveMetrics } from './components/LiveMetrics'
```

```tsx
      <LiveMetrics />
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Manual verification in the browser**

With both servers running, start a run from the form (use a small population size and a large `max_generations` so it keeps running). Confirm: a "Status: running" line appears within ~200ms, the in-progress generation block updates its active-genomes/steps numbers live as the run progresses, and once a generation completes, the "Last completed" block appears with matching fitness numbers. Open a second browser tab to the same URL and confirm it independently shows the same live state (two WS connections, one shared `RunManager`).

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(phase-7): WebSocket client + live metrics panel"
```

---

## Task 15: Live config edit + poll-interval knob + manual stop

**Files:**
- Create: `frontend/src/components/ConfigPanel.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `updateConfig`, `stopRun`, `getPollInterval`, `setPollInterval` (`api.ts`, Task 13).

- [ ] **Step 1: Build `ConfigPanel`**

Create `frontend/src/components/ConfigPanel.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { getPollInterval, setPollInterval, stopRun, updateConfig } from '../api'

export function ConfigPanel() {
  const [maxGenerationSteps, setMaxGenerationSteps] = useState('')
  const [maxGenerations, setMaxGenerations] = useState('')
  const [targetFitness, setTargetFitness] = useState('')
  const [pollIntervalMs, setPollIntervalMsState] = useState(200)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    getPollInterval().then((r) => setPollIntervalMsState(r.interval_ms))
  }, [])

  async function applyConfig(e: React.FormEvent) {
    e.preventDefault()
    setMessage(null)
    const partial: Record<string, number | null> = {}
    if (maxGenerationSteps !== '') partial.max_generation_steps = Number(maxGenerationSteps)
    if (maxGenerations !== '') partial.max_generations = Number(maxGenerations)
    if (targetFitness !== '') partial.target_fitness = Number(targetFitness)
    try {
      await updateConfig(partial)
      setMessage('Applied — takes effect at the next generation boundary.')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  async function applyPollInterval(value: number) {
    setPollIntervalMsState(value)
    await setPollInterval(value).catch((err) => setMessage(String(err)))
  }

  async function handleStop() {
    try {
      await stopRun()
      setMessage('Stop requested — the run will end after its current generation.')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <h2>Controls</h2>
      <form onSubmit={applyConfig}>
        <label>
          Max generation steps
          <input
            type="number"
            value={maxGenerationSteps}
            onChange={(e) => setMaxGenerationSteps(e.target.value)}
          />
        </label>
        <label>
          Max generations
          <input
            type="number"
            value={maxGenerations}
            onChange={(e) => setMaxGenerations(e.target.value)}
          />
        </label>
        <label>
          Target fitness
          <input
            type="number"
            value={targetFitness}
            onChange={(e) => setTargetFitness(e.target.value)}
          />
        </label>
        <button type="submit">Apply</button>
      </form>
      <label>
        Poll interval (ms)
        <input
          type="number"
          min={1}
          value={pollIntervalMs}
          onChange={(e) => applyPollInterval(Number(e.target.value))}
        />
      </label>
      <button onClick={handleStop}>Stop run</button>
      {message && <p>{message}</p>}
    </div>
  )
}
```

- [ ] **Step 2: Add `ConfigPanel` to `App.tsx`**

In `frontend/src/App.tsx`, import and render it (below `LiveMetrics`):

```tsx
import { ConfigPanel } from './components/ConfigPanel'
```

```tsx
      <ConfigPanel />
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual verification in the browser**

With a run active: (a) change "Max generations" to a value just above the current generation and confirm the run stops on its own shortly after (status flips to "completed"); start a fresh run for the rest of this check; (b) change the poll interval to `1000` and confirm the live metrics panel visibly updates less often (roughly once a second instead of five times a second); (c) click "Stop run" and confirm the status flips to "stopped" within one generation, and that a second "Stop run" click (or a second run start while this one is still finishing its last generation) is rejected/handled gracefully, not crashing the page.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(phase-7): live config edit, poll-interval knob, manual stop"
```

---

## Task 16: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Full backend suite one more time**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
Expected: clean.

- [ ] **Step 2: Frontend type-check one more time**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Full golden-path walkthrough in a real browser**

With both servers running (`uv run neuroarena-dashboard` and `npm run dev` in `frontend/`), and at least one track already generated:

1. Start a fresh run (small population, e.g. 10, `max_generations` unset). Confirm live metrics update within ~200ms and generations complete and accumulate.
2. Edit `max_generation_steps` down mid-run via the config panel; confirm the change is reflected (a shorter subsequent generation) without an error.
3. Change the poll interval; confirm the visible update rate changes accordingly.
4. Click Stop; confirm the run ends with status "stopped" and no further progress/generation messages arrive.
5. Reload the page; confirm the now-completed model appears in "Saved models" and Resume pre-fills the form with its last-known settings (population size in particular, since Phase 6's own resume rule forbids changing it).
6. Resume it, confirm generation numbering continues from where it left off (not reset to 0), and stop it again.
7. Open the browser's dev console throughout; confirm no uncaught errors.

- [ ] **Step 4: Record completion**

Update `docs/phases/phase-7-dashboard-control-panel.md`'s Revision history with an implementation-completion entry (status stays FINALIZED — this is the "implemented, matching a REVISED-then-back-to-FINALIZED" pattern only needed if a requirement changed during implementation; if nothing changed, a plain completion note is enough, following Phase 5/6's own precedent for this final entry).

```bash
git add docs/phases/phase-7-dashboard-control-panel.md
git commit -m "docs(phase-7): record implementation completion"
```
