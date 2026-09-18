# Phase 8 — Game Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native `arcade` viewer window, opened and controlled from the dashboard panel, that shows a training run's whole concurrent batch live (ghost cars, best highlighted) with three camera modes, zoom, an overlay, and a speed-preset control that paces the trainer.

**Architecture:** The NEAT trainer gains two small additive hooks (a thread-safe per-car visual snapshot, and a per-round pacing callback). The dashboard backend owns the speed preset (runtime state), one viewer child process, and a new `/ws/viewer` endpoint that pushes 60 Hz car frames plus run/view messages. The viewer is its own process: a background thread reads `/ws/viewer` into a latest-state slot, and the `arcade` window draws it (track/asset loading by `track_id`, camera math in a pure, unit-tested module). The React panel gets a speed selector and a "Game window" box.

**Tech Stack:** Python 3.12, `arcade` 3.x, FastAPI/uvicorn, `websockets` (sync client), NEAT (`neat-python`), React + TypeScript (Vite).

**Spec:** `docs/phases/phase-8-dashboard-game-viewer.md` (FINALIZED 2026-09-18 — read its Requirements and Design sections first).

## Global Constraints

- Python `>=3.12`; every new/modified `.py` file must pass `uv run ruff check`, `uv run ruff format --check` and `uv run mypy --strict` (the repo's pre-commit hook runs all three over the **whole repo** on every commit — so no task may leave a broken intermediate import or type error; each task's commit must leave the repo green).
- `neuroarena.backends.neat` must not import `neuroarena.sim`, `neuroarena.render`, `arcade`, etc. (`tests/backends/neat/test_import_hygiene.py` enforces it). The trainer therefore reads car poses only through the new optional `Visualizable` protocol, never by importing car code.
- `neuroarena.sim` must not import `arcade` (`tests/sim/test_import_hygiene.py`).
- The panel's existing `/ws` endpoint keeps **exactly** its three message types (`progress`/`generation`/`status`) — all viewer traffic goes on the **new** `/ws/viewer` endpoint. Envelope shape everywhere: `{"type": ..., "schema_version": 1, "data": {...}}`.
- Speed is **dashboard-only runtime state**: never a `RunConfig` field, never written to the settings history, global per backend process, defaults to `"max"` (unpaced) every time the backend starts. Presets, exactly: `0.25x`, `0.5x`, `1x`, `2x`, `4x`, `8x`, `max`. `1x` = each simulation round takes the game's fixed timestep `TICK_DT = 1/60 s`.
- The backend samples and pushes viewer frames at a fixed **60 Hz**; no adjustable rate, no client-side interpolation.
- Exactly one viewer at a time, owned by the backend (child process); the backend terminates it on shutdown. Localhost only (`127.0.0.1`), no auth.
- The viewer loads the track and sprite assets itself (track from `<data_dir>/tracks` by `track_id`, assets via the existing `AssetManifest`); there are no track/asset API endpoints.
- Rank = `Objective.fitness()` (live, per car), descending; ties broken by lower `genome_id`. Camera modes exactly: `fit`, `follow_best`, `follow_rank`. Followed-rank choice is held until that car drops out, then falls back to the best remaining car. Control is panel-only (no keyboard shortcuts in the viewer window).
- Pure logic (view settings, camera/follow math, overlay text, message handling) lives in modules that import neither `arcade` nor FastAPI, so it is unit-testable; the `arcade` window only draws.
- Line length is 100 (`ruff` E501). Code in this plan is written to that limit, but if `ruff format --check` or `ruff check` complains about code copied from it, run `uv run ruff format <files>` and hand-wrap any over-long comment or docstring line — the behavior must not change.
- Tests that touch threads/sockets must be bounded (wall-clock deadlines / `join(timeout)`); never write a loop or wait with no termination condition. Use fake clocks for pacing tests.

## File Structure

New:
- `src/neuroarena/dashboard/pacing.py` — `SPEED_PRESETS`, `DEFAULT_SPEED_PRESET`, `Pacer` (per-round sleep to hit a speed).
- `src/neuroarena/dashboard/viewer_manager.py` — `ViewerManager`: spawn/terminate the one viewer child process + hold the current `ViewSettings`.
- `src/neuroarena/render/view_settings.py` — pure: `CAMERA_MODES`, zoom bounds, `ViewSettings` (shared by dashboard and viewer).
- `src/neuroarena/render/view_model.py` — pure: `CarView`, `Camera`, `rank_cars`, `track_bounds`, `ViewModel` (follow/camera logic), `overlay_lines`.
- `src/neuroarena/render/viewer_client.py` — pure-ish: `ViewerState`, `ViewerClient` (thread + `websockets` sync client, message handling).
- `src/neuroarena/render/scene.py` — `TrackScene` (the static track drawing extracted from `PlayWindow`) + `heading_to_sprite_angle`.
- `src/neuroarena/render/viewer.py` — `ViewerWindow` (`arcade.Window`) + `main()`; run as `python -m neuroarena.render.viewer`.
- `frontend/src/components/SpeedControl.tsx`, `frontend/src/components/ViewerPanel.tsx`.
- Tests: `tests/interfaces/test_visualizable.py`, `tests/dashboard/test_pacing.py`, `tests/dashboard/test_viewer_manager.py`, `tests/render/test_view_settings.py`, `tests/render/test_view_model.py`, `tests/render/test_viewer_client.py`, `tests/render/test_viewer_import.py`.

Modified:
- `src/neuroarena/interfaces/protocols.py` — add `Visualizable`.
- `src/neuroarena/sim/car_env.py` — `CarEnvironment.visual_state()`.
- `src/neuroarena/backends/neat/evaluation.py` — `LiveGenome`, `BatchProgress.live`, `evaluate_batch(pace=...)`.
- `src/neuroarena/backends/neat/trainer.py` — `GenomeVisual`, `BatchVisuals`, `NeatTrainer.visual_snapshot()`, `NeatTrainer.set_pace()`.
- `src/neuroarena/dashboard/run_manager.py` — speed state, pacer wiring, `visual_snapshot()`, `current_track_id()`.
- `src/neuroarena/dashboard/ws_protocol.py` — viewer message builders.
- `src/neuroarena/dashboard/app.py` — `/api/speed`, `/api/viewer*`, `/ws/viewer`, lifespan closes the viewer, optional `viewer_manager` param.
- `src/neuroarena/dashboard/cli.py` — builds the `ViewerManager` with the right URL.
- `src/neuroarena/render/window.py`, `src/neuroarena/render/play.py` — use `TrackScene`.
- `src/neuroarena/config/run_config.py` — docstring note only (`sim_speed` is unused; speed lives in the dashboard).
- `pyproject.toml` (+ `uv.lock`) — explicit `websockets` dependency.
- `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/App.tsx`.
- `README.md`, `docs/phases/phase-8-dashboard-game-viewer.md` (Implementation plan section + revision entry).

---

### Task 1: `Visualizable` protocol + `CarEnvironment.visual_state()`

**Files:**
- Modify: `src/neuroarena/interfaces/protocols.py`
- Modify: `src/neuroarena/sim/car_env.py`
- Create: `tests/interfaces/test_visualizable.py`
- Modify (append): `tests/sim/test_car_env.py`

**Interfaces:**
- Produces: `Visualizable` (runtime-checkable Protocol with `visual_state(self) -> Mapping[str, float]`); `CarEnvironment.visual_state()` returning `{"x": float, "y": float, "heading": float}`. Task 3's trainer snapshot relies on both.

This is a **new, separate** Protocol — do not add the method to `Environment` (that would break every existing implementer, including the test doubles).

- [ ] **Step 1: Write the failing tests**

Create `tests/interfaces/test_visualizable.py`:

```python
from collections.abc import Mapping

from neuroarena.interfaces.protocols import Visualizable
from tests.interfaces.doubles import DummyEnvironment


def test_an_environment_without_visual_state_is_not_visualizable() -> None:
    assert not isinstance(DummyEnvironment(), Visualizable)


def test_an_object_with_visual_state_is_visualizable() -> None:
    class Drawable:
        def visual_state(self) -> Mapping[str, float]:
            return {"x": 1.0}

    assert isinstance(Drawable(), Visualizable)
```

Append to `tests/sim/test_car_env.py` (the file already imports `numpy as np`, `CarEnvironment`, and defines `_track()`; add `from neuroarena.interfaces.protocols import Environment, Visualizable` by extending the existing `Environment` import):

```python
def test_car_environment_is_visualizable_and_reports_the_cars_pose() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    assert isinstance(env, Visualizable)
    env.reset()
    before = env.visual_state()
    assert set(before) == {"x", "y", "heading"}
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    after = env.visual_state()
    assert (after["x"], after["y"]) != (before["x"], before["y"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/interfaces/test_visualizable.py tests/sim/test_car_env.py -v`
Expected: FAIL — `ImportError: cannot import name 'Visualizable'`.

- [ ] **Step 3: Implement**

In `src/neuroarena/interfaces/protocols.py`, add directly after the `Environment` class (`Mapping` is already imported at the top of the file):

```python
@runtime_checkable
class Visualizable(Protocol):
    """Optional capability, deliberately separate from `Environment` so no existing
    implementer has to change: an environment that can describe its current state for a
    viewer — a small mapping of named numbers (for the car game: `x`, `y`, `heading`).
    A trainer may read it from another thread while stepping, so an implementation must
    build the mapping from a single consistent read of its own state."""

    def visual_state(self) -> Mapping[str, float]: ...
```

In `src/neuroarena/sim/car_env.py`, add `from collections.abc import Mapping` to the imports (next to the existing `from dataclasses import dataclass`), and add this method to `CarEnvironment` directly after `reset`:

```python
    def visual_state(self) -> Mapping[str, float]:
        """Satisfies `Visualizable`. One read of `self._game.car` (a frozen state replaced
        atomically each tick), so a viewer thread never sees a torn pose."""
        car = self._game.car
        return {"x": car.x, "y": car.y, "heading": car.heading}
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/interfaces tests/sim -q`
Expected: PASS (including `tests/sim/test_import_hygiene.py`).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/interfaces/protocols.py src/neuroarena/sim/car_env.py tests/interfaces/test_visualizable.py tests/sim/test_car_env.py
git commit -m "feat(phase-8): Visualizable protocol and CarEnvironment.visual_state"
```

---

### Task 2: `evaluate_batch` — live genomes + per-round pacing hook

**Files:**
- Modify: `src/neuroarena/backends/neat/evaluation.py`
- Modify (append): `tests/backends/neat/test_evaluation.py`

**Interfaces:**
- Produces: `LiveGenome(genome_id: int, env: Environment, objective: Objective)` (frozen dataclass); `BatchProgress.live: tuple[LiveGenome, ...]` (replaced — never mutated in place — whenever the set of still-active genomes changes, and cleared to `()` when the batch ends); `evaluate_batch(entries, step_budget, *, progress=None, pace: Callable[[], None] | None = None)` where `pace` is called once at the end of every round that leaves at least one genome active and did not end the batch. Task 3 relies on all of this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/neat/test_evaluation.py` (it already imports `Any`, `np`, `BatchEntry`, `BatchProgress`, `StepBudget`, `evaluate_batch`, and the doubles):

```python
class _TruncatesAfterTwoSteps(DummyEnvironment):
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        observation, terminated, _truncated, info = super().step(action)
        return observation, terminated, self._t >= 2, info


def test_evaluate_batch_publishes_live_genomes_and_clears_them_at_the_end() -> None:
    seen: list[tuple[int, ...]] = []

    class _SpyProgress(BatchProgress):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "live":
                seen.append(tuple(g.genome_id for g in value))
            super().__setattr__(name, value)

    short_env = _TruncatesAfterTwoSteps()
    short_entry = BatchEntry(
        genome_id=0,
        model=DummyModel(short_env.observation_space, short_env.action_space),
        env=short_env,
        objective=DummyObjective(),
        seed=0,
    )
    entries = [short_entry, _entry(1)]
    evaluate_batch(entries, StepBudget(remaining=100), progress=_SpyProgress(active_count=0))
    assert seen[0] == (0, 1)  # both alive at the start
    assert (1,) in seen  # genome 0 dropped out after its episode ended
    assert seen[-1] == ()  # cleared when the batch is over


def test_evaluate_batch_live_genomes_expose_each_genomes_env_and_objective() -> None:
    entries = [_entry(0)]
    progress = BatchProgress(active_count=0)
    captured: list[Any] = []

    def pace() -> None:
        captured.append(progress.live)

    evaluate_batch(entries, StepBudget(remaining=100), progress=progress, pace=pace)
    live = captured[0][0]
    assert live.genome_id == 0
    assert live.env is entries[0].env
    assert live.objective is entries[0].objective


def test_evaluate_batch_calls_pace_once_per_round_while_genomes_remain() -> None:
    calls: list[int] = []
    evaluate_batch([_entry(0)], StepBudget(remaining=100), pace=lambda: calls.append(1))
    # DummyEnvironment truncates after 5 steps = 5 rounds; the last round leaves no active
    # genome, so there is nothing left to pace for.
    assert len(calls) == 4


def test_evaluate_batch_does_not_pace_after_a_collective_stop() -> None:
    class SucceedsAfterOneStep(DummyObjective):
        def should_stop(self) -> bool:
            return self._steps >= 1

    calls: list[int] = []
    evaluate_batch(
        [_entry(0, SucceedsAfterOneStep()), _entry(1)],
        StepBudget(remaining=100),
        pace=lambda: calls.append(1),
    )
    assert calls == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: FAIL (`BatchProgress` has no `live`; `evaluate_batch` got unexpected keyword `pace`).

- [ ] **Step 3: Implement**

In `src/neuroarena/backends/neat/evaluation.py`:

1. Add `from collections.abc import Callable` to the imports (with the other stdlib imports).

2. Add `LiveGenome` right after `StepBudget`:

```python
@dataclass(frozen=True)
class LiveGenome:
    """One still-active genome, as a viewer thread needs it: which genome, its own env
    (for `Visualizable.visual_state`) and its own objective (for live `fitness()`)."""

    genome_id: int
    env: Environment
    objective: Objective
```

3. In `BatchProgress`, add the field and initialise it. Replace the field block and `__init__` with:

```python
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
```

Also extend `BatchProgress`'s docstring with one sentence: "`live` is replaced (never mutated) whenever the active set changes, so a reader on another thread always sees a consistent tuple."

4. Change the `evaluate_batch` signature and body. New signature:

```python
def evaluate_batch(
    entries: list[BatchEntry],
    step_budget: StepBudget,
    *,
    progress: BatchProgress | None = None,
    pace: Callable[[], None] | None = None,
) -> dict[int, EvaluationResult]:
```

Add to the docstring's last sentence: "If `pace` is given, it is called once at the end of every round that leaves at least one genome active and did not end the batch — the hook a caller uses to slow the batch to a chosen speed; it must not raise."

Replace the progress handling. After the reset loop:

```python
    if progress is not None:
        progress.active_count = len(active)
        progress.live = _live_genomes(active)
```

Inside the `while` loop, replace the block after `active = still_active` with:

```python
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
```

At the end of the function, replace the final progress block with:

```python
    if progress is not None:
        progress.active_count = 0
        progress.live = ()
        _update_best_fitness(progress, results)
```

Add the helper at the bottom of the file:

```python
def _live_genomes(active: list[_ActiveEpisode]) -> tuple[LiveGenome, ...]:
    return tuple(LiveGenome(g.genome_id, g.env, g.objective) for g in active)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/backends/neat -q`
Expected: PASS (all existing evaluation tests included — the `active_count` spy test still sees only `active_count` writes).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py
git commit -m "feat(phase-8): evaluate_batch publishes live genomes and takes a pace hook"
```

---

### Task 3: `NeatTrainer.visual_snapshot()` and `set_pace()`

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Modify (append): `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `Visualizable` (Task 1), `BatchProgress.live`, `evaluate_batch(pace=...)` (Task 2).
- Produces: `GenomeVisual(genome_id: int, state: Mapping[str, float], fitness: float)`, `BatchVisuals(generation: int, population_size: int, genomes: tuple[GenomeVisual, ...])` (both frozen dataclasses in `trainer.py`); `NeatTrainer.visual_snapshot() -> BatchVisuals | None` (`None` when no generation is running; genomes whose env is not `Visualizable` are omitted); `NeatTrainer.set_pace(pace: Callable[[], None] | None) -> None`. Tasks 5, 8, 9 rely on these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/neat/test_trainer.py` (it already imports `threading`, `time`, `typing`, `np`, `NeatTrainer`, `RunConfig`, and the doubles; add `from collections.abc import Mapping` and extend the trainer import to include `BatchVisuals` if not present):

```python
class _VisibleEnvironment(DummyEnvironment):
    def visual_state(self) -> Mapping[str, float]:
        return {"x": float(self._t), "y": 0.0, "heading": 0.0}


class _SlowObjective(DummyObjective):
    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, typing.Any],
    ) -> None:
        super().update(observation, action, terminated, truncated, info)
        time.sleep(0.02)


def _poll_visual_snapshots(trainer: NeatTrainer) -> list[BatchVisuals | None]:
    snapshots: list[BatchVisuals | None] = []

    def _poll() -> None:
        for _ in range(100):
            snapshots.append(trainer.visual_snapshot())
            time.sleep(0.005)

    poller = threading.Thread(target=_poll)
    poller.start()
    next(trainer.run())
    poller.join(timeout=30)
    return snapshots


def test_visual_snapshot_is_none_before_run_and_after_a_generation() -> None:
    trainer = NeatTrainer(_VisibleEnvironment, DummyObjective(), RunConfig(), population_size=4)
    assert trainer.visual_snapshot() is None
    next(trainer.run())
    assert trainer.visual_snapshot() is None


def test_visual_snapshot_reports_each_live_genomes_state_and_fitness() -> None:
    trainer = NeatTrainer(_VisibleEnvironment, _SlowObjective(), RunConfig(), population_size=3)
    live = [s for s in _poll_visual_snapshots(trainer) if s is not None]
    assert live, "expected at least one snapshot while the generation was running"
    populated = [s for s in live if s.genomes]
    assert populated, "expected at least one snapshot with live genomes"
    for snapshot in populated:
        assert snapshot.generation == 0
        assert snapshot.population_size == 3
        assert len(snapshot.genomes) <= 3
        for genome in snapshot.genomes:
            assert set(genome.state) == {"x", "y", "heading"}
            assert genome.fitness >= 0.0


def test_visual_snapshot_omits_genomes_whose_env_is_not_visualizable() -> None:
    trainer = NeatTrainer(DummyEnvironment, _SlowObjective(), RunConfig(), population_size=3)
    live = [s for s in _poll_visual_snapshots(trainer) if s is not None]
    assert live
    assert all(s.genomes == () for s in live)


def test_set_pace_is_called_once_per_round_during_a_generation() -> None:
    calls: list[int] = []
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=3)
    trainer.set_pace(lambda: calls.append(1))
    next(trainer.run())
    # DummyEnvironment truncates after 5 steps: 5 rounds, and the last leaves nobody to pace.
    assert len(calls) == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backends/neat/test_trainer.py -k "visual_snapshot or set_pace" -v`
Expected: FAIL — `NeatTrainer` has no `visual_snapshot` / `set_pace` (and `BatchVisuals` import fails).

- [ ] **Step 3: Implement**

In `src/neuroarena/backends/neat/trainer.py`:

1. Extend the protocols import: `from neuroarena.interfaces.protocols import TrainingUpdate, Visualizable`.

2. Add after the `GenerationProgress` dataclass:

```python
@dataclasses.dataclass(frozen=True)
class GenomeVisual:
    """One still-active genome as a viewer sees it: the environment's own visual state
    (see `Visualizable`) plus the genome's live `Objective.fitness()`."""

    genome_id: int
    state: Mapping[str, float]
    fitness: float


@dataclasses.dataclass(frozen=True)
class BatchVisuals:
    """A snapshot of every still-active genome of the in-progress generation, for Phase 8's
    viewer. See `NeatTrainer.visual_snapshot`."""

    generation: int
    population_size: int
    genomes: tuple[GenomeVisual, ...]
```

3. In `NeatTrainer.__init__`, next to `self._current_step_budget: StepBudget | None = None`, add:

```python
        # Phase 8: optional per-round pacing callback (see `set_pace`). Deliberately not part
        # of the `Trainer` protocol — a caller sets it with `getattr`, like `progress_snapshot`.
        self._pace: Callable[[], None] | None = None
```

4. Add these two methods directly after `progress_snapshot`:

```python
    def set_pace(self, pace: Callable[[], None] | None) -> None:
        """Installs (or clears) a callback invoked once per simulation round while a
        generation is running — the hook a caller uses to slow the batch down to a chosen
        speed. It runs on the training thread and must not raise. Takes effect from the next
        generation's `evaluate_batch` call at the latest; setting it before `run()` is the
        normal use."""
        self._pace = pace

    def visual_snapshot(self) -> BatchVisuals | None:
        """Thread-safe (single reads of immutable/atomically-replaced values — see
        `BatchProgress.live`), read-only snapshot of every still-active genome of the
        in-progress generation: its environment's `Visualizable.visual_state()` and its live
        `Objective.fitness()`, for Phase 8's viewer. Genomes whose environment is not
        `Visualizable` are omitted. `None` when no generation is running."""
        progress = self._current_batch_progress
        if progress is None:
            return None
        genomes = tuple(
            GenomeVisual(
                genome_id=live.genome_id,
                state=dict(live.env.visual_state()),
                fitness=live.objective.fitness(),
            )
            for live in progress.live
            if isinstance(live.env, Visualizable)
        )
        return BatchVisuals(
            generation=self._generation,
            population_size=self._neat_config.pop_size,
            genomes=genomes,
        )
```

5. In `_run_one_generation`'s `fitness_function`, pass the hook:

```python
            batch_results = evaluate_batch(
                entries, step_budget, progress=self._current_batch_progress, pace=self._pace
            )
```

- [ ] **Step 4: Run to verify they pass**

Run: `timeout 300 uv run pytest tests/backends/neat -q`
Expected: PASS, including `test_import_hygiene.py` (the trainer imports only `neuroarena.interfaces.protocols`).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-8): NeatTrainer.visual_snapshot and set_pace"
```

---

### Task 4: `dashboard/pacing.py` — speed presets and `Pacer`

**Files:**
- Create: `src/neuroarena/dashboard/pacing.py`
- Create: `tests/dashboard/test_pacing.py`

**Interfaces:**
- Consumes: `TICK_DT` from `neuroarena.sim.game`.
- Produces: `SPEED_PRESETS: dict[str, float | None]` (exactly `"0.25x"`, `"0.5x"`, `"1x"`, `"2x"`, `"4x"`, `"8x"`, `"max"` → `None`, in that order); `DEFAULT_SPEED_PRESET = "max"`; `Pacer(speed: Callable[[], float | None], *, tick_dt=TICK_DT, clock=time.perf_counter, sleep=time.sleep, max_lag_s=0.25)` whose `__call__()` sleeps just enough that successive calls are `tick_dt / speed` apart, and never sleeps when `speed()` is `None`. Task 5 relies on these.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_pacing.py`:

```python
import pytest

from neuroarena.dashboard.pacing import DEFAULT_SPEED_PRESET, SPEED_PRESETS, Pacer
from neuroarena.sim.game import TICK_DT


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _pacer(speed: float | None, fake: _FakeTime) -> Pacer:
    return Pacer(lambda: speed, clock=fake.clock, sleep=fake.sleep)


def test_presets_are_exactly_the_agreed_set_in_order() -> None:
    assert list(SPEED_PRESETS) == ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "max"]
    assert SPEED_PRESETS["1x"] == 1.0
    assert SPEED_PRESETS["max"] is None
    assert DEFAULT_SPEED_PRESET == "max"


def test_max_speed_never_sleeps() -> None:
    fake = _FakeTime()
    pacer = _pacer(None, fake)
    for _ in range(5):
        pacer()
    assert fake.sleeps == []


def test_real_time_sleeps_one_tick_per_call() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    for _ in range(3):
        pacer()
    assert fake.sleeps == pytest.approx([TICK_DT] * 3)


def test_double_speed_sleeps_half_a_tick() -> None:
    fake = _FakeTime()
    pacer = _pacer(2.0, fake)
    pacer()
    assert fake.sleeps == pytest.approx([TICK_DT / 2])


def test_work_between_calls_is_subtracted_from_the_sleep() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    pacer()  # sleeps a full tick
    fake.now += 0.005  # 5 ms of "simulation work"
    pacer()
    assert fake.sleeps[1] == pytest.approx(TICK_DT - 0.005)


def test_a_speed_change_takes_effect_immediately() -> None:
    fake = _FakeTime()
    speed: list[float | None] = [1.0]
    pacer = Pacer(lambda: speed[0], clock=fake.clock, sleep=fake.sleep)
    pacer()
    speed[0] = 4.0
    pacer()
    assert fake.sleeps == pytest.approx([TICK_DT, TICK_DT / 4])


def test_falling_far_behind_does_not_cause_a_catch_up_burst() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    pacer()  # sleeps one tick
    fake.now += 1.0  # a very slow round
    pacer()  # behind by far more than the allowed lag: no sleep, deadline resets
    pacer()  # back to normal pacing, not a burst of zero-sleep calls
    assert len(fake.sleeps) == 2
    assert fake.sleeps[1] == pytest.approx(TICK_DT)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/dashboard/test_pacing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.dashboard.pacing'`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/dashboard/pacing.py`:

```python
"""Speed presets and the `Pacer` that slows a training batch to a chosen speed — Phase 8's
"speed control paces the trainer" requirement. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`. Speed is dashboard-only runtime
state (not a `RunConfig` field), so nothing here is persisted."""

from __future__ import annotations

import time
from collections.abc import Callable

from neuroarena.sim.game import TICK_DT

SPEED_PRESETS: dict[str, float | None] = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
    "8x": 8.0,
    "max": None,  # unpaced: as fast as the CPU allows
}
DEFAULT_SPEED_PRESET = "max"


class Pacer:
    """Callable installed as `NeatTrainer.set_pace`'s hook: called once per simulation round
    (one `TICK_DT` of simulated time for every active car), it sleeps so that successive
    rounds are `TICK_DT / speed` of wall-clock time apart. It can only slow a run down — if a
    round already takes longer than its target, it does not sleep. `speed()` is read on every
    call, so a change of speed takes effect on the very next round. Runs on the training
    thread."""

    def __init__(
        self,
        speed: Callable[[], float | None],
        *,
        tick_dt: float = TICK_DT,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        max_lag_s: float = 0.25,
    ) -> None:
        self._speed = speed
        self._tick_dt = tick_dt
        self._clock = clock
        self._sleep = sleep
        self._max_lag_s = max_lag_s
        self._deadline: float | None = None
        self._last_speed: float | None = None

    def __call__(self) -> None:
        speed = self._speed()
        if speed is None:
            self._deadline = None
            self._last_speed = None
            return
        now = self._clock()
        if self._deadline is None or speed != self._last_speed:
            self._deadline = now
        self._last_speed = speed
        self._deadline += self._tick_dt / speed
        delay = self._deadline - now
        if delay > 0:
            self._sleep(delay)
        elif delay < -self._max_lag_s:
            # Fell too far behind (a slow round, a debugger pause): resync to "now" rather
            # than racing through the following rounds with no sleep to catch up.
            self._deadline = now
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/dashboard/test_pacing.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/dashboard/pacing.py tests/dashboard/test_pacing.py
git commit -m "feat(phase-8): speed presets and Pacer"
```

---

### Task 5: `RunManager` speed state + visual snapshot, and `/api/speed`

**Files:**
- Modify: `src/neuroarena/dashboard/run_manager.py`
- Modify: `src/neuroarena/dashboard/app.py`
- Modify (append): `tests/dashboard/test_run_manager.py`, `tests/dashboard/test_app.py`

**Interfaces:**
- Consumes: `Pacer`, `SPEED_PRESETS`, `DEFAULT_SPEED_PRESET` (Task 4); `NeatTrainer.set_pace`, `visual_snapshot`, `BatchVisuals` (Task 3).
- Produces: `RunManager.speed_preset() -> str`, `RunManager.set_speed_preset(preset: str) -> None` (raises `ValueError` for an unknown preset), `RunManager.speed_multiplier() -> float | None`, `RunManager.visual_snapshot() -> BatchVisuals | None` (`None` unless a run is `"running"`), `RunManager.current_track_id() -> str | None`; REST `GET /api/speed` → `{"preset": str, "presets": list[str]}`, `PATCH /api/speed` body `{"preset": str}` → `{"preset": str}` (400 for an unknown preset). Tasks 9 and 12 rely on these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_run_manager.py` (it already has `_save_test_track`, `_wait_until`, `RunManager`, `Path`):

```python
def test_speed_preset_defaults_to_max_and_is_settable(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.speed_preset() == "max"
    assert manager.speed_multiplier() is None
    manager.set_speed_preset("2x")
    assert manager.speed_preset() == "2x"
    assert manager.speed_multiplier() == 2.0


def test_set_speed_preset_rejects_an_unknown_preset(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    with pytest.raises(ValueError):
        manager.set_speed_preset("3x")
    assert manager.speed_preset() == "max"


def test_visual_snapshot_is_none_when_idle(tmp_path: Path) -> None:
    assert RunManager(data_dir=tmp_path).visual_snapshot() is None
    assert RunManager(data_dir=tmp_path).current_track_id() is None


def test_a_running_run_exposes_track_id_pacer_and_car_poses(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    try:
        assert manager.current_track_id() == track_id
        # the trainer got this manager's pacer installed
        assert getattr(manager._trainer, "_pace", None) is manager._pacer

        def _has_cars() -> bool:
            snapshot = manager.visual_snapshot()
            return snapshot is not None and len(snapshot.genomes) > 0

        _wait_until(_has_cars, timeout=30.0)
        snapshot = manager.visual_snapshot()
        assert snapshot is not None
        assert snapshot.population_size == 6
        assert set(snapshot.genomes[0].state) == {"x", "y", "heading"}
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running", timeout=60.0)
```

Append to `tests/dashboard/test_app.py` (it already has `_client`):

```python
def test_get_speed_defaults_to_max_and_lists_the_presets(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/speed")
    assert response.status_code == 200
    assert response.json() == {
        "preset": "max",
        "presets": ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "max"],
    }


def test_patch_speed_sets_the_preset(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/speed", json={"preset": "2x"})
    assert response.status_code == 200
    assert response.json() == {"preset": "2x"}
    assert client.get("/api/speed").json()["preset"] == "2x"


def test_patch_speed_rejects_an_unknown_preset_with_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/speed", json={"preset": "3x"})
    assert response.status_code == 400
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/dashboard/test_run_manager.py tests/dashboard/test_app.py -k "speed or visual_snapshot or track_id_pacer" -v`
Expected: FAIL (`RunManager` has no `speed_preset` etc.; `/api/speed` is a 404).

- [ ] **Step 3: Implement `RunManager` changes**

In `src/neuroarena/dashboard/run_manager.py`:

1. Imports: change to `from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress` and add `from neuroarena.dashboard.pacing import DEFAULT_SPEED_PRESET, SPEED_PRESETS, Pacer`.

2. In `__init__`, after `self._poll_interval_ms = DEFAULT_POLL_INTERVAL_MS`:

```python
        self._speed_preset = DEFAULT_SPEED_PRESET
        self._pacer = Pacer(self.speed_multiplier)
        self._track_id: str | None = None
```

3. Add these methods after `set_poll_interval_ms`:

```python
    def speed_preset(self) -> str:
        return self._speed_preset

    def set_speed_preset(self, preset: str) -> None:
        if preset not in SPEED_PRESETS:
            raise ValueError(f"unknown speed preset {preset!r}; choose from {list(SPEED_PRESETS)}")
        self._speed_preset = preset

    def speed_multiplier(self) -> float | None:
        """The current preset as a multiplier of real time; `None` means unpaced ("max")."""
        return SPEED_PRESETS[self._speed_preset]

    def current_track_id(self) -> str | None:
        return self._track_id

    def visual_snapshot(self) -> BatchVisuals | None:
        """Every still-active car of the in-progress generation (see
        `NeatTrainer.visual_snapshot`), or `None` unless a run is running — a finished run's
        stale trainer must not keep producing frames."""
        if self._trainer is None or self._status != "running":
            return None
        snapshot_fn = getattr(self._trainer, "visual_snapshot", None)
        return None if snapshot_fn is None else snapshot_fn()
```

4. In `start()`, right after `self._trainer = prepared.trainer` add:

```python
            self._pacer = Pacer(self.speed_multiplier)  # fresh deadline for each run
            set_pace = getattr(prepared.trainer, "set_pace", None)
            if set_pace is not None:
                set_pace(self._pacer)
            self._track_id = track_id
```

Also update the class docstring's "Threading model" paragraph with one sentence: "The speed preset is a single `str` attribute read by the `Pacer` on the training thread each round."

- [ ] **Step 4: Implement the REST routes**

In `src/neuroarena/dashboard/app.py`: add the import `from neuroarena.dashboard.pacing import SPEED_PRESETS`, add a request dataclass next to `PollIntervalRequest`:

```python
@dataclasses.dataclass
class SpeedRequest:
    preset: str
```

and add these routes after the poll-interval routes (before the `/ws` endpoint):

```python
    @app.get("/api/speed")
    def get_speed() -> dict[str, Any]:
        return {"preset": run_manager.speed_preset(), "presets": list(SPEED_PRESETS)}

    @app.patch("/api/speed")
    def set_speed(request: SpeedRequest) -> dict[str, str]:
        try:
            run_manager.set_speed_preset(request.preset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"preset": run_manager.speed_preset()}
```

- [ ] **Step 5: Run to verify they pass**

Run: `timeout 400 uv run pytest tests/dashboard -q`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/dashboard/run_manager.py src/neuroarena/dashboard/app.py tests/dashboard/test_run_manager.py tests/dashboard/test_app.py
git commit -m "feat(phase-8): speed preset state, pacer wiring and /api/speed"
```

---

### Task 6: `render/view_settings.py` — pure `ViewSettings`

**Files:**
- Create: `src/neuroarena/render/view_settings.py`
- Create: `tests/render/test_view_settings.py`

**Interfaces:**
- Produces: `CAMERA_MODES = ("fit", "follow_best", "follow_rank")`, `ZOOM_MIN = 0.25`, `ZOOM_MAX = 4.0`; frozen dataclass `ViewSettings(zoom: float = 1.0, camera_mode: str = "fit", follow_rank: int = 1, follow_seq: int = 0)` that validates in `__post_init__` (`ValueError`), with `updated(partial: Mapping[str, Any]) -> ViewSettings` (accepts only `zoom`/`camera_mode`/`follow_rank`; bumps `follow_seq` whenever `follow_rank` is present, even if unchanged), `to_dict() -> dict[str, Any]`, and classmethod `from_dict(data: Mapping[str, Any])`. Used by Tasks 7, 8, 9, 10 (this module imports neither `arcade` nor FastAPI).

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_view_settings.py`:

```python
import pytest

from neuroarena.render.view_settings import CAMERA_MODES, ZOOM_MAX, ZOOM_MIN, ViewSettings


def test_defaults() -> None:
    settings = ViewSettings()
    assert settings.zoom == 1.0
    assert settings.camera_mode == "fit"
    assert settings.follow_rank == 1
    assert settings.follow_seq == 0
    assert CAMERA_MODES == ("fit", "follow_best", "follow_rank")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_mode": "orbit"},
        {"zoom": ZOOM_MIN - 0.01},
        {"zoom": ZOOM_MAX + 0.01},
        {"follow_rank": 0},
    ],
)
def test_invalid_settings_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ViewSettings(**kwargs)  # type: ignore[arg-type]


def test_updated_merges_a_partial_and_leaves_the_original_alone() -> None:
    original = ViewSettings()
    changed = original.updated({"zoom": 2.0, "camera_mode": "follow_best"})
    assert changed == ViewSettings(zoom=2.0, camera_mode="follow_best")
    assert original == ViewSettings()


def test_setting_follow_rank_bumps_follow_seq_even_when_unchanged() -> None:
    first = ViewSettings().updated({"follow_rank": 3})
    assert (first.follow_rank, first.follow_seq) == (3, 1)
    again = first.updated({"follow_rank": 3})
    assert (again.follow_rank, again.follow_seq) == (3, 2)
    assert first.updated({"zoom": 2.0}).follow_seq == 1  # other keys do not bump it


def test_updated_rejects_unknown_keys_and_invalid_values() -> None:
    with pytest.raises(ValueError):
        ViewSettings().updated({"bogus": 1})
    with pytest.raises(ValueError):
        ViewSettings().updated({"camera_mode": "orbit"})


def test_round_trips_through_a_dict() -> None:
    settings = ViewSettings(zoom=1.5, camera_mode="follow_rank", follow_rank=4, follow_seq=7)
    assert ViewSettings.from_dict(settings.to_dict()) == settings
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/render/test_view_settings.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/render/view_settings.py`:

```python
"""View settings shared by the dashboard backend and the viewer process: the camera modes,
zoom bounds and the validated `ViewSettings` record. Pure — imports neither `arcade` nor
FastAPI — so both sides (and their tests) can use it. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CAMERA_MODES = ("fit", "follow_best", "follow_rank")
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0
_PARTIAL_KEYS = frozenset({"zoom", "camera_mode", "follow_rank"})


@dataclass(frozen=True)
class ViewSettings:
    """`follow_rank` is 1-based and only matters in `follow_rank` mode. `follow_seq` counts
    how many times a follow-rank choice has been made, so choosing the same rank twice is
    still a new choice the viewer re-resolves (the choice is held on a *car* until it drops
    out — see `view_model.ViewModel`)."""

    zoom: float = 1.0
    camera_mode: str = "fit"
    follow_rank: int = 1
    follow_seq: int = 0

    def __post_init__(self) -> None:
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {self.camera_mode!r}")
        if not ZOOM_MIN <= self.zoom <= ZOOM_MAX:
            raise ValueError(f"zoom must be within [{ZOOM_MIN}, {ZOOM_MAX}], got {self.zoom}")
        if self.follow_rank < 1:
            raise ValueError(f"follow_rank must be >= 1, got {self.follow_rank}")

    def updated(self, partial: Mapping[str, Any]) -> ViewSettings:
        unknown = set(partial) - _PARTIAL_KEYS
        if unknown:
            raise ValueError(f"unsupported view setting(s): {sorted(unknown)}")
        follow_seq = self.follow_seq + (1 if "follow_rank" in partial else 0)
        return ViewSettings(
            zoom=float(partial.get("zoom", self.zoom)),
            camera_mode=str(partial.get("camera_mode", self.camera_mode)),
            follow_rank=int(partial.get("follow_rank", self.follow_rank)),
            follow_seq=follow_seq,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewSettings:
        return cls(
            zoom=float(data["zoom"]),
            camera_mode=str(data["camera_mode"]),
            follow_rank=int(data["follow_rank"]),
            follow_seq=int(data["follow_seq"]),
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/render -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/render/view_settings.py tests/render/test_view_settings.py
git commit -m "feat(phase-8): ViewSettings shared by dashboard and viewer"
```

---

### Task 7: `render/view_model.py` — ranking, follow logic, camera, overlay (pure)

**Files:**
- Create: `src/neuroarena/render/view_model.py`
- Create: `tests/render/test_view_model.py`

**Interfaces:**
- Consumes: `ViewSettings` (Task 6); `Track` from `neuroarena.sim.track` (existing: `Track.cells`, `Track.cell_size`, `Track.cell_center(cell)`).
- Produces (all importable without `arcade`): `CarView(genome_id: int, x: float, y: float, heading: float, fitness: float)`; `Camera(center_x, center_y, scale)`; `rank_cars(cars) -> list[CarView]` (fitness descending, ties by lower `genome_id`); `rank_of(cars, car) -> int | None` (1-based); `track_bounds(track) -> tuple[min_x, min_y, max_x, max_y]`; `ViewModel(bounds, window_size)` with `followed_car(cars, settings) -> CarView | None` and `camera(followed, settings) -> Camera`; `overlay_lines(...) -> list[str]`; constants `FOLLOW_BASE_SCALE = 0.5`, `FIT_MARGIN = 1.15`. Tasks 10 and 11 rely on these names.

Semantics to implement exactly: `fit` → no followed car; `follow_best` → the top-ranked car every frame; `follow_rank` → when the mode was just entered or `follow_seq` changed, lock onto the car currently at `min(follow_rank, len(cars))`; keep following **that car** while it is alive; once it is gone, fall back to the best remaining car and keep following that one. The camera's `scale` is arcade's zoom factor (world units → pixels): follow modes use `FOLLOW_BASE_SCALE × settings.zoom`; fit mode uses the scale that fits the whole track in the window divided by `FIT_MARGIN`, times `settings.zoom`, centred on the track bounds; with no followed car the camera is the fit camera.

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_view_model.py`:

```python
import pytest

from neuroarena.render.view_model import (
    FIT_MARGIN,
    FOLLOW_BASE_SCALE,
    CarView,
    ViewModel,
    overlay_lines,
    rank_cars,
    rank_of,
    track_bounds,
)
from neuroarena.render.view_settings import ViewSettings
from neuroarena.sim.track import Facing, GridCell, TileKind, Track


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


def _track() -> Track:
    return Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)


def _car(genome_id: int, fitness: float, x: float = 0.0, y: float = 0.0) -> CarView:
    return CarView(genome_id=genome_id, x=x, y=y, heading=0.0, fitness=fitness)


WINDOW = (1280, 720)


def _model() -> ViewModel:
    return ViewModel(track_bounds(_track()), WINDOW)


def test_rank_cars_orders_by_fitness_then_lower_genome_id() -> None:
    cars = [_car(5, 1.0), _car(2, 3.0), _car(9, 3.0), _car(1, 2.0)]
    assert [c.genome_id for c in rank_cars(cars)] == [2, 9, 1, 5]


def test_rank_of_is_one_based_and_none_for_an_unknown_car() -> None:
    cars = [_car(1, 1.0), _car(2, 5.0)]
    assert rank_of(cars, cars[1]) == 1
    assert rank_of(cars, cars[0]) == 2
    assert rank_of(cars, _car(99, 0.0)) is None


def test_track_bounds_cover_every_cell_edge() -> None:
    track = _track()
    half = track.cell_size / 2
    assert track_bounds(track) == (
        -half,
        -half,
        3 * track.cell_size + half,
        2 * track.cell_size + half,
    )


def test_fit_mode_follows_nobody() -> None:
    assert _model().followed_car([_car(1, 1.0)], ViewSettings()) is None


def test_follow_best_tracks_the_leader_as_it_changes() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_best")
    a, b = _car(1, 1.0), _car(2, 2.0)
    assert model.followed_car([a, b], settings) == b
    b_behind = _car(2, 0.5)
    assert model.followed_car([a, b_behind], settings) == a


def test_follow_rank_locks_onto_a_car_and_holds_it_when_ranks_change() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=2, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0), _car(3, 1.0)]
    assert model.followed_car(cars, settings) is cars[1]
    # car 2 is overtaken by car 3, but the choice is held on the car, not the rank
    overtaken = [_car(1, 3.0), _car(2, 0.5), _car(3, 1.0)]
    assert model.followed_car(overtaken, settings) == overtaken[1]


def test_follow_rank_falls_back_to_the_best_remaining_car_when_the_followed_one_drops_out() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=2, follow_seq=1)
    model.followed_car([_car(1, 3.0), _car(2, 2.0), _car(3, 1.0)], settings)
    remaining = [_car(1, 3.0), _car(3, 1.0)]  # car 2 crashed
    assert model.followed_car(remaining, settings).genome_id == 1  # type: ignore[union-attr]
    # and it stays on that car afterwards, even if car 3 later out-scores it
    later = [_car(1, 3.0), _car(3, 9.0)]
    assert model.followed_car(later, settings).genome_id == 1  # type: ignore[union-attr]


def test_a_new_follow_choice_re_resolves_even_for_the_same_rank() -> None:
    model = _model()
    first = ViewSettings(camera_mode="follow_rank", follow_rank=1, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0)]
    assert model.followed_car(cars, first).genome_id == 1  # type: ignore[union-attr]
    swapped = [_car(1, 1.0), _car(2, 2.0)]  # car 2 now leads, car 1 is still followed
    assert model.followed_car(swapped, first).genome_id == 1  # type: ignore[union-attr]
    again = ViewSettings(camera_mode="follow_rank", follow_rank=1, follow_seq=2)
    assert model.followed_car(swapped, again).genome_id == 2  # type: ignore[union-attr]


def test_follow_rank_larger_than_the_field_picks_the_last_car() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=50, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0)]
    assert model.followed_car(cars, settings).genome_id == 2  # type: ignore[union-attr]


def test_no_cars_means_nobody_to_follow() -> None:
    settings = ViewSettings(camera_mode="follow_best")
    assert _model().followed_car([], settings) is None


def test_fit_camera_centres_on_the_track_and_scales_to_the_window() -> None:
    track = _track()
    min_x, min_y, max_x, max_y = track_bounds(track)
    model = _model()
    camera = model.camera(None, ViewSettings())
    expected = min(WINDOW[0] / (max_x - min_x), WINDOW[1] / (max_y - min_y)) / FIT_MARGIN
    assert camera.center_x == pytest.approx((min_x + max_x) / 2)
    assert camera.center_y == pytest.approx((min_y + max_y) / 2)
    assert camera.scale == pytest.approx(expected)
    assert model.camera(None, ViewSettings(zoom=2.0)).scale == pytest.approx(expected * 2)


def test_follow_camera_centres_on_the_car_at_the_base_scale() -> None:
    car = _car(1, 1.0, x=123.0, y=-45.0)
    camera = _model().camera(car, ViewSettings(zoom=2.0))
    assert (camera.center_x, camera.center_y) == (123.0, -45.0)
    assert camera.scale == pytest.approx(FOLLOW_BASE_SCALE * 2.0)


def _lines(**overrides: object) -> list[str]:
    base: dict[str, object] = {
        "connected": True,
        "run_state": "running",
        "generation": 12,
        "alive": 87,
        "population_size": 150,
        "speed": "1x",
        "settings": ViewSettings(),
        "followed_rank": None,
    }
    base.update(overrides)
    return overlay_lines(**base)  # type: ignore[arg-type]


def test_overlay_shows_generation_cars_speed_and_camera() -> None:
    assert _lines() == [
        "Generation 12",
        "Cars: 87 / 150",
        "Speed: 1x",
        "Camera: whole track",
    ]


def test_overlay_describes_each_camera_mode() -> None:
    assert "Camera: following the best car" in _lines(
        settings=ViewSettings(camera_mode="follow_best")
    )
    following = ViewSettings(camera_mode="follow_rank", follow_rank=3)
    assert "Camera: following #3" in _lines(settings=following, followed_rank=3)
    assert "Camera: following #?" in _lines(settings=following, followed_rank=None)


def test_overlay_when_disconnected_idle_or_starting() -> None:
    assert _lines(connected=False) == ["Disconnected from the dashboard backend - retrying..."]
    assert _lines(run_state="idle", generation=None, alive=0) == ["Waiting for a run..."]
    assert _lines(run_state="running", generation=None, alive=0) == ["Run starting..."]


def test_overlay_marks_a_finished_run() -> None:
    assert _lines(run_state="stopped")[-1] == "Run stopped"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/render/test_view_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.render.view_model'`.

- [ ] **Step 3: Implement**

Create `src/neuroarena/render/view_model.py`:

```python
"""Pure logic for the viewer window: ranking, which car the camera follows, the camera
transform, and the overlay text. Imports neither `arcade` nor FastAPI, so it is unit-testable
without a display — the window only draws what this module decides. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from neuroarena.render.view_settings import ViewSettings
from neuroarena.sim.track import Track

FOLLOW_BASE_SCALE = 0.5
"""World units -> pixels when following a car at zoom 1.0 (a 512-unit cell is 256 px)."""
FIT_MARGIN = 1.15
"""The whole-track view leaves ~15% breathing room around the track."""

Bounds = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


@dataclass(frozen=True)
class CarView:
    genome_id: int
    x: float
    y: float
    heading: float
    fitness: float


@dataclass(frozen=True)
class Camera:
    center_x: float
    center_y: float
    scale: float  # arcade's zoom factor: world units -> pixels


def rank_cars(cars: Sequence[CarView]) -> list[CarView]:
    """Best first: live fitness descending, ties broken by the lower genome id."""
    return sorted(cars, key=lambda car: (-car.fitness, car.genome_id))


def rank_of(cars: Sequence[CarView], car: CarView) -> int | None:
    for index, ranked in enumerate(rank_cars(cars), start=1):
        if ranked.genome_id == car.genome_id:
            return index
    return None


def track_bounds(track: Track) -> Bounds:
    half = track.cell_size / 2
    centers = [track.cell_center(cell) for cell in track.cells]
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    return (min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half)


class ViewModel:
    """Holds the little state the follow logic needs (which car a follow-rank choice locked
    onto) and computes the camera. One instance per loaded track."""

    def __init__(self, bounds: Bounds, window_size: tuple[int, int]) -> None:
        self._bounds = bounds
        self._window_size = window_size
        self._followed_id: int | None = None
        self._last_mode: str | None = None
        self._applied_seq: int | None = None

    def followed_car(self, cars: Sequence[CarView], settings: ViewSettings) -> CarView | None:
        ranked = rank_cars(cars)
        mode = settings.camera_mode
        if not ranked or mode == "fit":
            self._followed_id = None
            self._last_mode = mode
            return None
        if mode == "follow_best":
            self._followed_id = ranked[0].genome_id
            self._last_mode = mode
            return ranked[0]
        # follow_rank: (re)resolve the rank to a car when the mode was just entered or a new
        # choice was made; otherwise keep following that same car.
        if mode != self._last_mode or settings.follow_seq != self._applied_seq:
            index = min(settings.follow_rank, len(ranked)) - 1
            self._followed_id = ranked[index].genome_id
            self._applied_seq = settings.follow_seq
        self._last_mode = mode
        followed = next((car for car in ranked if car.genome_id == self._followed_id), None)
        if followed is None:
            # The followed car crashed or finished: fall back to the best remaining car and
            # keep following that one, so the camera never sits on an empty spot.
            followed = ranked[0]
            self._followed_id = followed.genome_id
        return followed

    def camera(self, followed: CarView | None, settings: ViewSettings) -> Camera:
        if followed is not None:
            return Camera(followed.x, followed.y, FOLLOW_BASE_SCALE * settings.zoom)
        min_x, min_y, max_x, max_y = self._bounds
        width_px, height_px = self._window_size
        fit = min(width_px / (max_x - min_x), height_px / (max_y - min_y)) / FIT_MARGIN
        return Camera((min_x + max_x) / 2, (min_y + max_y) / 2, fit * settings.zoom)


def overlay_lines(
    *,
    connected: bool,
    run_state: str,
    generation: int | None,
    alive: int,
    population_size: int,
    speed: str,
    settings: ViewSettings,
    followed_rank: int | None,
) -> list[str]:
    if not connected:
        return ["Disconnected from the dashboard backend - retrying..."]
    if run_state == "idle" and alive == 0 and generation is None:
        return ["Waiting for a run..."]
    if run_state == "running" and generation is None:
        return ["Run starting..."]
    lines: list[str] = []
    if generation is not None:
        lines.append(f"Generation {generation}")
    lines.append(f"Cars: {alive} / {population_size}")
    lines.append(f"Speed: {speed}")
    if settings.camera_mode == "fit":
        camera = "whole track"
    elif settings.camera_mode == "follow_best":
        camera = "following the best car"
    else:
        camera = "following #?" if followed_rank is None else f"following #{followed_rank}"
    lines.append(f"Camera: {camera}")
    if run_state in ("completed", "stopped", "crashed"):
        lines.append(f"Run {run_state}")
    return lines
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/render -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/render/view_model.py tests/render/test_view_model.py
git commit -m "feat(phase-8): view model - ranking, follow logic, camera, overlay"
```

---

### Task 8: Viewer protocol messages + `ViewerManager`

**Files:**
- Modify: `src/neuroarena/dashboard/ws_protocol.py`
- Create: `src/neuroarena/dashboard/viewer_manager.py`
- Create: `tests/dashboard/fakes.py`
- Create: `tests/dashboard/test_viewer_manager.py`
- Modify (append): `tests/dashboard/test_ws_protocol.py`

**Interfaces:**
- Consumes: `BatchVisuals` (Task 3), `ViewSettings` (Task 6).
- Produces: `viewer_frame_message(snapshot, speed) -> dict`, `viewer_run_message(state, track_id, model_id) -> dict`, `viewer_view_message(settings) -> dict` (envelope `{"type": "frame"|"run"|"view", "schema_version": 1, "data": {...}}`); `ViewerManager(*, url: str, data_dir: Path, spawn=..., terminate_timeout_s: float = 5.0)` with `open() -> bool` (spawns; `False` if already open), `close() -> bool` (terminates; `False` if not open), `is_open() -> bool`, `shutdown() -> None` (same as close), `settings() -> ViewSettings`, `update_settings(partial) -> ViewSettings` (raises `ValueError`, leaves settings unchanged). The child command line is exactly `[sys.executable, "-m", "neuroarena.render.viewer", "--url", <url>, "--data-dir", <data_dir>]`. Test helpers `FakeProcess`/`FakeSpawner` in `tests/dashboard/fakes.py` are reused by Task 9.

Wire format (exact):
- `frame`: `data = {"generation": int, "population_size": int, "speed": str, "cars": [{"id": int, "x": float, "y": float, "heading": float, "fitness": float}, ...]}`; a genome whose state lacks any of `x`/`y`/`heading` is skipped.
- `run`: `data = {"state": str, "track_id": str | null, "model_id": str | null}`.
- `view`: `data = ViewSettings.to_dict()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/fakes.py`:

```python
"""Test doubles for the viewer child process — no real process is ever spawned in tests."""

from __future__ import annotations

import subprocess


class FakeProcess:
    def __init__(self, args: list[str], *, stubborn: bool = False) -> None:
        self.args = args
        self.terminated = False
        self.killed = False
        self._stubborn = stubborn
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self._stubborn:
            self._returncode = -15

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout or 0.0)
        return self._returncode

    def die(self) -> None:
        """Simulates the window being closed by hand / the process crashing."""
        self._returncode = 0


class FakeSpawner:
    def __init__(self, *, stubborn: bool = False) -> None:
        self.processes: list[FakeProcess] = []
        self._stubborn = stubborn

    def __call__(self, args: list[str]) -> FakeProcess:
        process = FakeProcess(args, stubborn=self._stubborn)
        self.processes.append(process)
        return process
```

Create `tests/dashboard/test_viewer_manager.py`:

```python
import sys
from pathlib import Path

import pytest

from neuroarena.dashboard.viewer_manager import ViewerManager
from neuroarena.render.view_settings import ViewSettings
from tests.dashboard.fakes import FakeSpawner


def _manager(tmp_path: Path, spawner: FakeSpawner) -> ViewerManager:
    return ViewerManager(
        url="ws://127.0.0.1:8000/ws/viewer",
        data_dir=tmp_path,
        spawn=spawner,
        terminate_timeout_s=0.1,
    )


def test_open_spawns_the_viewer_with_the_agreed_command_line(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    assert manager.open() is True
    assert spawner.processes[0].args == [
        sys.executable,
        "-m",
        "neuroarena.render.viewer",
        "--url",
        "ws://127.0.0.1:8000/ws/viewer",
        "--data-dir",
        str(tmp_path),
    ]
    assert manager.is_open() is True


def test_a_second_open_while_one_is_running_does_nothing(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.open() is False
    assert len(spawner.processes) == 1


def test_is_open_turns_false_when_the_window_is_closed_by_hand(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    spawner.processes[0].die()
    assert manager.is_open() is False
    assert manager.open() is True  # and it can be opened again
    assert len(spawner.processes) == 2


def test_close_terminates_the_process_and_is_idempotent(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.close() is True
    assert spawner.processes[0].terminated is True
    assert manager.is_open() is False
    assert manager.close() is False


def test_close_kills_a_process_that_ignores_terminate(tmp_path: Path) -> None:
    spawner = FakeSpawner(stubborn=True)
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.close() is True
    assert spawner.processes[0].terminated is True
    assert spawner.processes[0].killed is True


def test_shutdown_closes_the_viewer(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    manager.shutdown()
    assert spawner.processes[0].terminated is True


def test_settings_default_and_update_merge_a_partial(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeSpawner())
    assert manager.settings() == ViewSettings()
    updated = manager.update_settings({"zoom": 2.0, "camera_mode": "follow_rank", "follow_rank": 3})
    assert updated == ViewSettings(zoom=2.0, camera_mode="follow_rank", follow_rank=3, follow_seq=1)
    assert manager.settings() == updated


def test_an_invalid_settings_update_raises_and_changes_nothing(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeSpawner())
    with pytest.raises(ValueError):
        manager.update_settings({"camera_mode": "orbit"})
    assert manager.settings() == ViewSettings()
```

Append to `tests/dashboard/test_ws_protocol.py` (add these imports at the top of the file: `from neuroarena.backends.neat.trainer import BatchVisuals, GenomeVisual`, `from neuroarena.dashboard.ws_protocol import viewer_frame_message, viewer_run_message, viewer_view_message`, `from neuroarena.render.view_settings import ViewSettings`):

```python
def test_viewer_frame_message_shape_and_skips_genomes_without_a_full_pose() -> None:
    snapshot = BatchVisuals(
        generation=4,
        population_size=10,
        genomes=(
            GenomeVisual(7, {"x": 1.0, "y": 2.0, "heading": 0.5}, 3.5),
            GenomeVisual(8, {"x": 1.0}, 1.0),  # incomplete: skipped
        ),
    )
    assert viewer_frame_message(snapshot, "2x") == {
        "type": "frame",
        "schema_version": 1,
        "data": {
            "generation": 4,
            "population_size": 10,
            "speed": "2x",
            "cars": [{"id": 7, "x": 1.0, "y": 2.0, "heading": 0.5, "fitness": 3.5}],
        },
    }


def test_viewer_run_message_shape() -> None:
    assert viewer_run_message("running", "track1", "model1") == {
        "type": "run",
        "schema_version": 1,
        "data": {"state": "running", "track_id": "track1", "model_id": "model1"},
    }


def test_viewer_view_message_shape() -> None:
    message = viewer_view_message(ViewSettings(zoom=2.0))
    assert message["type"] == "view"
    assert message["schema_version"] == 1
    assert message["data"] == ViewSettings(zoom=2.0).to_dict()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/dashboard/test_viewer_manager.py tests/dashboard/test_ws_protocol.py -v`
Expected: FAIL — missing `viewer_manager` module and viewer message builders.

- [ ] **Step 3: Implement the message builders**

In `src/neuroarena/dashboard/ws_protocol.py`: extend the imports (`from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress` and `from neuroarena.render.view_settings import ViewSettings`), add one sentence to the module docstring — "The `frame`/`run`/`view` builders below are used only by the viewer's own `/ws/viewer` endpoint; the panel's `/ws` keeps exactly `progress`/`generation`/`status`." — and append:

```python
_POSE_KEYS = frozenset({"x", "y", "heading"})


def viewer_frame_message(snapshot: BatchVisuals, speed: str) -> dict[str, Any]:
    cars = [
        {
            "id": genome.genome_id,
            "x": genome.state["x"],
            "y": genome.state["y"],
            "heading": genome.state["heading"],
            "fitness": genome.fitness,
        }
        for genome in snapshot.genomes
        if _POSE_KEYS <= genome.state.keys()
    ]
    return {
        "type": "frame",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {
            "generation": snapshot.generation,
            "population_size": snapshot.population_size,
            "speed": speed,
            "cars": cars,
        },
    }


def viewer_run_message(state: str, track_id: str | None, model_id: str | None) -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {"state": state, "track_id": track_id, "model_id": model_id},
    }


def viewer_view_message(settings: ViewSettings) -> dict[str, Any]:
    return {
        "type": "view",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": settings.to_dict(),
    }
```

- [ ] **Step 4: Implement `ViewerManager`**

Create `src/neuroarena/dashboard/viewer_manager.py`:

```python
"""`ViewerManager` owns the one game-viewer child process (Phase 8) and the current view
settings. The viewer is its own process because `arcade` runs its event loop on the process's
main thread, which uvicorn already owns. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from neuroarena.render.view_settings import ViewSettings


class _Process(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def _spawn(args: list[str]) -> _Process:
    return subprocess.Popen(args)


class ViewerManager:
    def __init__(
        self,
        *,
        url: str,
        data_dir: Path,
        spawn: Callable[[list[str]], _Process] = _spawn,
        terminate_timeout_s: float = 5.0,
    ) -> None:
        self._url = url
        self._data_dir = data_dir
        self._spawn = spawn
        self._terminate_timeout_s = terminate_timeout_s
        self._lock = threading.Lock()
        self._process: _Process | None = None
        self._settings = ViewSettings()

    def open(self) -> bool:
        """Spawns the viewer; `False` (and does nothing) if one is already running."""
        with self._lock:
            if self._running_locked():
                return False
            self._process = self._spawn(
                [
                    sys.executable,
                    "-m",
                    "neuroarena.render.viewer",
                    "--url",
                    self._url,
                    "--data-dir",
                    str(self._data_dir),
                ]
            )
            return True

    def close(self) -> bool:
        """Terminates the viewer (kills it if it ignores `terminate`); `False` if none was
        running."""
        with self._lock:
            process = self._process
            self._process = None
            if process is None or process.poll() is not None:
                return False
            process.terminate()
            try:
                process.wait(timeout=self._terminate_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return True

    def is_open(self) -> bool:
        with self._lock:
            return self._running_locked()

    def shutdown(self) -> None:
        self.close()

    def settings(self) -> ViewSettings:
        with self._lock:
            return self._settings

    def update_settings(self, partial: Mapping[str, Any]) -> ViewSettings:
        with self._lock:
            self._settings = self._settings.updated(partial)  # raises ValueError, unchanged
            return self._settings

    def _running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/dashboard/test_viewer_manager.py tests/dashboard/test_ws_protocol.py -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/dashboard/ws_protocol.py src/neuroarena/dashboard/viewer_manager.py tests/dashboard/fakes.py tests/dashboard/test_viewer_manager.py tests/dashboard/test_ws_protocol.py
git commit -m "feat(phase-8): viewer protocol messages and ViewerManager"
```

---

### Task 9: Viewer REST routes, `/ws/viewer`, shutdown, CLI wiring

**Files:**
- Modify: `src/neuroarena/dashboard/app.py`
- Modify: `src/neuroarena/dashboard/cli.py`
- Modify (append): `tests/dashboard/test_app.py`

**Interfaces:**
- Consumes: `ViewerManager` (Task 8), `viewer_*_message` (Task 8), `RunManager.visual_snapshot/current_track_id/speed_preset/status` (Task 5), `ViewSettings` (Task 6).
- Produces: `create_app(run_manager, data_dir, viewer_manager=None)`; REST `GET /api/viewer` → `{"open": bool, "settings": {...ViewSettings dict}}`, `POST /api/viewer/open`, `POST /api/viewer/close` (both return the same body as `GET`), `PATCH /api/viewer/settings` body `{zoom?, camera_mode?, follow_rank?}` → the settings dict (400 on an invalid value); WebSocket `/ws/viewer` (sends a `run` message on connect and whenever `(status, track_id, model_id)` changes, a `view` message on connect and whenever the settings change, and a `frame` message every ~1/60 s while a snapshot exists). App shutdown closes the viewer **before** stopping the run. Task 12's frontend and Task 11's viewer consume these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app.py` (add imports at the top: `from neuroarena.dashboard.viewer_manager import ViewerManager` and `from tests.dashboard.fakes import FakeSpawner`):

```python
def _viewer_client(tmp_path: Path, spawner: FakeSpawner) -> TestClient:
    manager = RunManager(data_dir=tmp_path)
    viewer = ViewerManager(
        url="ws://127.0.0.1:8000/ws/viewer",
        data_dir=tmp_path,
        spawn=spawner,
        terminate_timeout_s=0.1,
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    return TestClient(app)


def test_viewer_starts_closed_with_default_settings(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    response = client.get("/api/viewer")
    assert response.status_code == 200
    assert response.json() == {
        "open": False,
        "settings": {"zoom": 1.0, "camera_mode": "fit", "follow_rank": 1, "follow_seq": 0},
    }


def test_open_and_close_the_viewer_are_idempotent(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    client = _viewer_client(tmp_path, spawner)
    assert client.post("/api/viewer/open").json()["open"] is True
    assert client.post("/api/viewer/open").json()["open"] is True
    assert len(spawner.processes) == 1
    assert client.post("/api/viewer/close").json()["open"] is False
    assert client.post("/api/viewer/close").json()["open"] is False


def test_patch_viewer_settings_updates_and_validates(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    ok = client.patch(
        "/api/viewer/settings",
        json={"camera_mode": "follow_rank", "follow_rank": 3, "zoom": 2.0},
    )
    assert ok.status_code == 200
    assert ok.json() == {"zoom": 2.0, "camera_mode": "follow_rank", "follow_rank": 3, "follow_seq": 1}
    assert client.patch("/api/viewer/settings", json={"camera_mode": "orbit"}).status_code == 400
    assert client.patch("/api/viewer/settings", json={"zoom": 99}).status_code == 400
    assert client.get("/api/viewer").json()["settings"]["zoom"] == 2.0  # unchanged by the bad ones


def test_app_shutdown_closes_the_viewer(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = RunManager(data_dir=tmp_path)
    viewer = ViewerManager(
        url="ws://x/ws/viewer", data_dir=tmp_path, spawn=spawner, terminate_timeout_s=0.1
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    with TestClient(app) as client:
        client.post("/api/viewer/open")
        assert spawner.processes[0].terminated is False
    assert spawner.processes[0].terminated is True


def test_ws_viewer_sends_run_and_view_messages_on_connect_when_idle(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    with client.websocket_connect("/ws/viewer") as ws:
        first = ws.receive_json()
        second = ws.receive_json()
    assert {first["type"], second["type"]} == {"run", "view"}
    run = first if first["type"] == "run" else second
    assert run["data"] == {"state": "idle", "track_id": None, "model_id": None}


def test_ws_viewer_streams_frames_with_car_poses_for_a_running_run(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _viewer_client(tmp_path, FakeSpawner())
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    run_messages: list[dict[str, object]] = []
    frame: dict[str, object] | None = None
    try:
        with client.websocket_connect("/ws/viewer") as ws:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline and frame is None:
                message = ws.receive_json()
                if message["type"] == "run":
                    run_messages.append(message["data"])
                elif message["type"] == "frame" and message["data"]["cars"]:
                    frame = message["data"]
    finally:
        client.post("/api/runs/current/stop")
    assert run_messages and run_messages[-1]["track_id"] == track_id
    assert frame is not None
    assert frame["population_size"] == 6
    assert frame["speed"] == "max"
    car = frame["cars"][0]  # type: ignore[index]
    assert set(car) == {"id", "x", "y", "heading", "fitness"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/dashboard/test_app.py -k "viewer" -v`
Expected: FAIL — `create_app() got an unexpected keyword argument 'viewer_manager'`.

- [ ] **Step 3: Implement**

In `src/neuroarena/dashboard/app.py`:

1. Imports: add `from neuroarena.dashboard.viewer_manager import ViewerManager`, extend the ws_protocol import to include `viewer_frame_message, viewer_run_message, viewer_view_message`, and add `from neuroarena.render.view_settings import ViewSettings`.

2. Add module constants after the imports:

```python
DEFAULT_VIEWER_URL = "ws://127.0.0.1:8000/ws/viewer"
VIEWER_FRAME_INTERVAL_S = 1 / 60
```

3. Add a request dataclass after `SpeedRequest`:

```python
@dataclasses.dataclass
class ViewSettingsRequest:
    zoom: float | None = None
    camera_mode: str | None = None
    follow_rank: int | None = None
```

4. Change the signature and lifespan of `create_app`:

```python
def create_app(
    run_manager: RunManager, data_dir: Path, viewer_manager: ViewerManager | None = None
) -> FastAPI:
    viewer = (
        viewer_manager
        if viewer_manager is not None
        else ViewerManager(url=DEFAULT_VIEWER_URL, data_dir=data_dir)
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        # Blocking (bounded) work — run off the event loop so it can't stall other handlers.
        # The viewer goes first so it does not sit "disconnected" while the run winds down.
        await asyncio.to_thread(viewer.shutdown)
        await asyncio.to_thread(run_manager.shutdown)
```

5. Add these routes after the speed routes and before `/ws`:

```python
    def _viewer_state() -> dict[str, Any]:
        return {"open": viewer.is_open(), "settings": viewer.settings().to_dict()}

    @app.get("/api/viewer")
    def get_viewer() -> dict[str, Any]:
        return _viewer_state()

    @app.post("/api/viewer/open")
    def open_viewer() -> dict[str, Any]:
        try:
            viewer.open()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not start the viewer: {exc}") from exc
        return _viewer_state()

    @app.post("/api/viewer/close")
    def close_viewer() -> dict[str, Any]:
        viewer.close()
        return _viewer_state()

    @app.patch("/api/viewer/settings")
    def update_viewer_settings(request: ViewSettingsRequest) -> dict[str, Any]:
        partial = {k: v for k, v in dataclasses.asdict(request).items() if v is not None}
        try:
            settings = viewer.update_settings(partial)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return settings.to_dict()
```

6. Add the viewer WebSocket after the existing `/ws` endpoint (before `return app`):

```python
    @app.websocket("/ws/viewer")
    async def ws_viewer(websocket: WebSocket) -> None:
        await websocket.accept()
        last_run: tuple[str, str | None, str | None] | None = None
        last_settings: ViewSettings | None = None
        try:
            while True:
                status = run_manager.status()
                run_key = (status.status, run_manager.current_track_id(), status.model_id)
                if run_key != last_run:
                    await websocket.send_json(viewer_run_message(*run_key))
                    last_run = run_key

                settings = viewer.settings()
                if settings != last_settings:
                    await websocket.send_json(viewer_view_message(settings))
                    last_settings = settings

                snapshot = run_manager.visual_snapshot()
                if snapshot is not None:
                    await websocket.send_json(
                        viewer_frame_message(snapshot, run_manager.speed_preset())
                    )

                await asyncio.sleep(VIEWER_FRAME_INTERVAL_S)
        except WebSocketDisconnect:
            pass
```

Update the module docstring's first sentence to also mention the speed/viewer routes and `/ws/viewer`.

In `src/neuroarena/dashboard/cli.py`, add `from neuroarena.dashboard.viewer_manager import ViewerManager` and build the manager with the real port:

```python
    run_manager = RunManager(data_dir=args.data_dir)
    viewer_manager = ViewerManager(
        url=f"ws://127.0.0.1:{args.port}/ws/viewer", data_dir=args.data_dir
    )
    app = create_app(run_manager=run_manager, data_dir=args.data_dir, viewer_manager=viewer_manager)
```

- [ ] **Step 4: Run to verify they pass**

Run: `timeout 400 uv run pytest tests/dashboard -q`
Expected: PASS (all earlier dashboard tests included — `create_app(run_manager=..., data_dir=...)` still works without `viewer_manager`).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add src/neuroarena/dashboard/app.py src/neuroarena/dashboard/cli.py tests/dashboard/test_app.py
git commit -m "feat(phase-8): viewer REST routes, /ws/viewer, shutdown and CLI wiring"
```

---

### Task 10: `render/viewer_client.py` — the viewer's connection to the backend

**Files:**
- Create: `src/neuroarena/render/viewer_client.py`
- Create: `tests/render/test_viewer_client.py`
- Modify: `pyproject.toml` (+ `uv.lock`)

**Interfaces:**
- Consumes: `CarView` (Task 7), `ViewSettings` (Task 6); the wire format defined in Task 8.
- Produces: `ViewerState` (frozen dataclass: `run_state: str = "idle"`, `track_id: str | None = None`, `generation: int | None = None`, `population_size: int = 0`, `speed: str = "max"`, `cars: tuple[CarView, ...] = ()`, `settings: ViewSettings`, `connected: bool = False`); `ViewerClient(url, *, connect=..., retry_delay_s=1.0, give_up_after_s=15.0, clock=time.monotonic, sleep=time.sleep)` with `start()` (daemon thread, reconnects forever), `stop()`, `state() -> ViewerState` (thread-safe copy), `apply_message(message)`, `run_once()` (one connect-and-consume lifecycle; never raises on connection failure), `should_exit() -> bool` (`True` once the client has been disconnected for more than `give_up_after_s`, counting from construction). Task 11's window consumes `state()` and `should_exit()`.

Message handling (`apply_message`): `run` sets `run_state`/`track_id` and **clears `cars` when a run newly enters `"running"`**; `view` replaces `settings`; `frame` replaces `generation`/`population_size`/`speed`/`cars`; any other `type` is ignored (forward compatibility). A malformed message (missing keys, wrong types) is logged and skipped by `run_once`, never fatal.

- [ ] **Step 1: Add the explicit dependency**

`websockets` is currently only a transitive dependency (via `uvicorn[standard]`); the viewer imports it directly. In `pyproject.toml`, add `"websockets>=12"` to `dependencies`, then run `uv sync` (this updates `uv.lock`).

- [ ] **Step 2: Write the failing tests**

Create `tests/render/test_viewer_client.py`:

```python
import json
from collections.abc import Iterable
from typing import Any

from neuroarena.render.view_model import CarView
from neuroarena.render.view_settings import ViewSettings
from neuroarena.render.viewer_client import ViewerClient, ViewerState


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeConnection:
    def __init__(self, messages: Iterable[str | bytes]) -> None:
        self._messages = list(messages)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(self._messages)


def _client(connect: Any = None, fake: _FakeTime | None = None) -> ViewerClient:
    fake = fake or _FakeTime()
    return ViewerClient(
        "ws://test/ws/viewer",
        connect=connect or (lambda url: _FakeConnection([])),
        clock=fake.clock,
        sleep=fake.sleep,
        give_up_after_s=15.0,
    )


def _frame(cars: list[dict[str, float]], generation: int = 3) -> dict[str, Any]:
    return {
        "type": "frame",
        "schema_version": 1,
        "data": {"generation": generation, "population_size": 10, "speed": "2x", "cars": cars},
    }


def _run(state: str, track_id: str | None = "t1") -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": 1,
        "data": {"state": state, "track_id": track_id, "model_id": "m1"},
    }


def test_initial_state_is_idle_and_disconnected() -> None:
    state = _client().state()
    assert state == ViewerState()
    assert state.connected is False


def test_a_frame_message_replaces_the_cars_and_metadata() -> None:
    client = _client()
    client.apply_message(
        _frame([{"id": 4, "x": 1.0, "y": 2.0, "heading": 0.5, "fitness": 9.0}], generation=7)
    )
    state = client.state()
    assert state.generation == 7
    assert state.population_size == 10
    assert state.speed == "2x"
    assert state.cars == (CarView(genome_id=4, x=1.0, y=2.0, heading=0.5, fitness=9.0),)


def test_a_view_message_replaces_the_settings() -> None:
    client = _client()
    settings = ViewSettings(zoom=2.0, camera_mode="follow_best")
    client.apply_message({"type": "view", "schema_version": 1, "data": settings.to_dict()})
    assert client.state().settings == settings


def test_a_run_message_sets_state_and_track_and_clears_cars_when_a_run_starts() -> None:
    client = _client()
    car = {"id": 1, "x": 0.0, "y": 0.0, "heading": 0.0, "fitness": 1.0}
    client.apply_message(_frame([car]))
    client.apply_message(_run("stopped"))  # a finished run keeps its last frame
    assert len(client.state().cars) == 1
    assert client.state().run_state == "stopped"
    client.apply_message(_run("running", track_id="t2"))  # a new run clears the field
    state = client.state()
    assert (state.run_state, state.track_id, state.cars) == ("running", "t2", ())


def test_an_unknown_message_type_is_ignored() -> None:
    client = _client()
    client.apply_message({"type": "future-thing", "schema_version": 9, "data": {}})
    assert client.state() == ViewerState()


def test_run_once_applies_every_message_then_marks_disconnected() -> None:
    messages = [json.dumps(_run("running")), json.dumps(_frame([])).encode()]
    client = _client(connect=lambda url: _FakeConnection(messages))
    client.run_once()
    state = client.state()
    assert state.run_state == "running"
    assert state.generation == 3
    assert state.connected is False  # the fake connection ended


def test_run_once_skips_malformed_messages_and_keeps_going() -> None:
    messages = ["not json", json.dumps({"type": "frame"}), json.dumps(_run("running"))]
    client = _client(connect=lambda url: _FakeConnection(messages))
    client.run_once()
    assert client.state().run_state == "running"


def test_run_once_does_not_raise_when_the_backend_is_unreachable() -> None:
    def refuse(url: str) -> Any:
        raise ConnectionRefusedError("nobody home")

    client = _client(connect=refuse)
    client.run_once()  # must not raise
    assert client.state().connected is False


def test_should_exit_only_after_being_disconnected_past_the_grace_period() -> None:
    fake = _FakeTime()
    client = _client(fake=fake)
    assert client.should_exit() is False
    fake.now = 14.0
    assert client.should_exit() is False
    fake.now = 16.0
    assert client.should_exit() is True


def test_repeated_failed_attempts_do_not_reset_the_grace_period() -> None:
    fake = _FakeTime()

    def refuse(url: str) -> Any:
        raise ConnectionRefusedError

    client = _client(connect=refuse, fake=fake)
    fake.now = 10.0
    client.run_once()
    fake.now = 16.0
    client.run_once()
    assert client.should_exit() is True


def test_connecting_resets_the_grace_period() -> None:
    fake = _FakeTime()
    connected_states: list[bool] = []

    class _Watching(_FakeConnection):
        def __iter__(self) -> Any:
            connected_states.append(client.state().connected)
            return iter([])

    client = _client(connect=lambda url: _Watching([]), fake=fake)
    fake.now = 100.0
    client.run_once()
    assert connected_states == [True]  # was marked connected while the connection was open
    assert client.should_exit() is False  # a fresh disconnect just now: the clock restarted
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/render/test_viewer_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.render.viewer_client'`.

- [ ] **Step 4: Implement**

Create `src/neuroarena/render/viewer_client.py`:

```python
"""The viewer process's connection to the dashboard backend: a background thread reads
`/ws/viewer` into a latest-state slot that the (main-thread) `arcade` window reads each frame.
Pure of `arcade`, so the message handling is unit-testable. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as _ws_connect

from neuroarena.render.view_model import CarView
from neuroarena.render.view_settings import ViewSettings

_log = logging.getLogger(__name__)

ConnectFn = Callable[[str], AbstractContextManager[Iterable[str | bytes]]]


def _default_connect(url: str) -> AbstractContextManager[Iterable[str | bytes]]:
    return _ws_connect(url)


@dataclass(frozen=True)
class ViewerState:
    run_state: str = "idle"
    track_id: str | None = None
    generation: int | None = None
    population_size: int = 0
    speed: str = "max"
    cars: tuple[CarView, ...] = ()
    settings: ViewSettings = field(default_factory=ViewSettings)
    connected: bool = False


class ViewerClient:
    def __init__(
        self,
        url: str,
        *,
        connect: ConnectFn = _default_connect,
        retry_delay_s: float = 1.0,
        give_up_after_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._connect = connect
        self._retry_delay_s = retry_delay_s
        self._give_up_after_s = give_up_after_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._state = ViewerState()
        self._stop = threading.Event()
        self._disconnected_since: float | None = clock()

    def state(self) -> ViewerState:
        with self._lock:
            return self._state

    def should_exit(self) -> bool:
        """True once disconnected for longer than the grace period (counted from
        construction until the first successful connection)."""
        with self._lock:
            since = self._disconnected_since
        return since is not None and self._clock() - since > self._give_up_after_s

    def start(self) -> None:
        threading.Thread(target=self._loop, name="viewer-client", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> None:
        """One connection lifecycle: connect, then consume messages until it ends. Never
        raises on a refused/lost connection — the loop simply tries again."""
        try:
            with self._connect(self._url) as connection:
                self._set_connected(True)
                for raw in connection:
                    if self._stop.is_set():
                        return
                    self._handle_raw(raw)
        except (OSError, WebSocketException):
            _log.info("viewer connection to %s ended", self._url)
        finally:
            self._set_connected(False)

    def apply_message(self, message: Mapping[str, Any]) -> None:
        kind = message["type"]
        data = message["data"]
        with self._lock:
            state = self._state
            if kind == "run":
                cars = state.cars
                if data["state"] == "running" and state.run_state != "running":
                    cars = ()  # a new run starts on an empty field
                self._state = replace(
                    state, run_state=data["state"], track_id=data["track_id"], cars=cars
                )
            elif kind == "view":
                self._state = replace(state, settings=ViewSettings.from_dict(data))
            elif kind == "frame":
                self._state = replace(
                    state,
                    generation=int(data["generation"]),
                    population_size=int(data["population_size"]),
                    speed=str(data["speed"]),
                    cars=tuple(
                        CarView(
                            genome_id=int(car["id"]),
                            x=float(car["x"]),
                            y=float(car["y"]),
                            heading=float(car["heading"]),
                            fitness=float(car["fitness"]),
                        )
                        for car in data["cars"]
                    ),
                )
            # any other type: ignored, so a newer backend can add messages

    def _handle_raw(self, raw: str | bytes) -> None:
        try:
            self.apply_message(json.loads(raw))
        except (KeyError, TypeError, ValueError):
            _log.warning("ignoring malformed viewer message: %.200r", raw)

    def _set_connected(self, connected: bool) -> None:
        with self._lock:
            self._state = replace(self._state, connected=connected)
            if connected:
                self._disconnected_since = None
            elif self._disconnected_since is None:
                self._disconnected_since = self._clock()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            if self._stop.is_set():
                return
            self._sleep(self._retry_delay_s)
```

If mypy rejects the `_ws_connect(url)` return against `AbstractContextManager[Iterable[str | bytes]]`, fix it with a narrow, commented `cast` (the real `ClientConnection` is both a context manager and an iterable of `str | bytes`) rather than loosening the module's types.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/render/test_viewer_client.py -v`
Expected: PASS (12 tests).

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
```bash
git add pyproject.toml uv.lock src/neuroarena/render/viewer_client.py tests/render/test_viewer_client.py
git commit -m "feat(phase-8): ViewerClient - the viewer's backend connection"
```

---

### Task 11: `TrackScene` extraction + the viewer window

**Files:**
- Create: `src/neuroarena/render/scene.py`
- Create: `src/neuroarena/render/viewer.py`
- Modify: `src/neuroarena/render/window.py`
- Modify: `src/neuroarena/render/play.py`
- Create: `tests/render/test_viewer_import.py`

**Interfaces:**
- Consumes: `ViewerClient`/`ViewerState` (Task 10), `ViewModel`/`rank_cars`/`rank_of`/`track_bounds`/`overlay_lines`/`CarView` (Task 7), `AssetManifest` (existing), `tracks.store.load` (existing: `load(track_id, tracks_dir) -> TrackRecord` with `.track`; raises `UnknownTrackError`).
- Produces: `neuroarena.render.scene.TrackScene(track, manifest)` (`.draw()` draws grass, asphalt, kerbs and the start/finish decal; must be created after an `arcade.Window` exists), `heading_to_sprite_angle(heading)`, `DEFAULT_MANIFEST_PATH`; `neuroarena.render.viewer.ViewerWindow(client, data_dir)` and `main()`. The viewer is launched by `ViewerManager` (Task 8) as `python -m neuroarena.render.viewer --url URL --data-dir DIR`.

This task has two parts: a **behavior-preserving refactor** of `PlayWindow` (its drawing moves into `TrackScene`; the play window must look and drive exactly as before), and the new **viewer window**. The window classes cannot be unit-tested (they need a GL context); verification is an import smoke test, the repo gates, an automated no-crash smoke run, and a manual visual check by the user (Task 13).

- [ ] **Step 1: Write the import smoke test**

Create `tests/render/test_viewer_import.py`:

```python
import pytest


def test_the_viewer_and_scene_modules_import_without_opening_a_window() -> None:
    pytest.importorskip("arcade")
    import neuroarena.render.scene as scene
    import neuroarena.render.viewer as viewer

    assert callable(viewer.main)
    assert scene.DEFAULT_MANIFEST_PATH.name == "manifest.json"
```

Run: `uv run pytest tests/render/test_viewer_import.py -v` — Expected: FAIL (`ModuleNotFoundError: neuroarena.render.scene`).

- [ ] **Step 2: Create `scene.py` (extracted from `PlayWindow`)**

Create `src/neuroarena/render/scene.py`:

```python
"""The static part of the game's picture — grass background, asphalt tiles, kerbs and the
start/finish decal — shared by the play window and the viewer window. Imports `neuroarena.sim`
and `arcade`; never the reverse. A `TrackScene` allocates GPU `SpriteList`s, so it must be
created after an `arcade.Window` exists."""

from __future__ import annotations

import math
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.sim.track import DRIVABLE_WIDTH, Segment, Track, boundary_segments

DEFAULT_MANIFEST_PATH = Path(__file__).parent / "assets/road_01/manifest.json"
BACKGROUND_MARGIN_CELLS = 2
KERB_COLOR = arcade.color.WHITE_SMOKE
KERB_LINE_WIDTH = 10.0
KERB_ARC_STEPS = 24  # smoother than the collision resolver's default — this is visual only


def heading_to_sprite_angle(heading: float) -> float:
    """`arcade.Sprite.angle` rotates clockwise from the texture's native orientation.
    The car art's native (angle=0) nose points world-heading +90° (up) — confirmed by
    actually driving it: the first cut had this 180° off and drove the car tail-first."""
    return 90.0 - math.degrees(heading)


class TrackScene:
    def __init__(self, track: Track, manifest: AssetManifest) -> None:
        self.track = track
        self.manifest = manifest
        self.background_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.tile_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.decor_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.kerb_segments: list[Segment] = boundary_segments(track, arc_steps=KERB_ARC_STEPS)
        self._build_background()
        self._build_track()
        self._build_decor()

    def _build_background(self) -> None:
        cell_size = self.track.cell_size
        xs = [cell[0] for cell in self.track.cells]
        ys = [cell[1] for cell in self.track.cells]
        margin = BACKGROUND_MARGIN_CELLS
        for gx in range(min(xs) - margin, max(xs) + margin + 1):
            for gy in range(min(ys) - margin, max(ys) + margin + 1):
                sprite = arcade.Sprite(str(self.manifest.background["grass"]))
                sprite.width = cell_size
                sprite.height = cell_size
                sprite.center_x, sprite.center_y = gx * cell_size, gy * cell_size
                self.background_sprites.append(sprite)

    def _build_track(self) -> None:
        # One undirected, edge-to-edge asphalt fill per cell — no per-TileKind art or
        # rotation needed, so this can't reintroduce either the seam or the wrong-corner
        # bugs a rotated bitmap kerb had. Kerbs are drawn separately in `draw` from the
        # same boundary geometry the collision resolver uses (see `kerb_segments`).
        cell_size = self.track.cell_size
        for cell in self.track.cells:
            cx, cy = self.track.cell_center(cell)
            sprite = arcade.Sprite(str(self.manifest.surface))
            sprite.width = cell_size
            sprite.height = cell_size
            sprite.center_x, sprite.center_y = cx, cy
            self.tile_sprites.append(sprite)

    def _build_decor(self) -> None:
        cx, cy = self.track.cell_center(self.track.start_cell)
        sprite = arcade.Sprite(str(self.manifest.decor["start_finish"]))
        # Native art is a wide banner (long axis = the checkered cross-lane line, short axis
        # = along-travel thickness) sized to the drivable width, not the full kerb-to-kerb cell.
        native_aspect = sprite.height / sprite.width
        sprite.width = DRIVABLE_WIDTH
        sprite.height = DRIVABLE_WIDTH * native_aspect
        sprite.center_x, sprite.center_y = cx, cy
        # Native (angle=0) long axis is horizontal, which already crosses a vertical (N/S)
        # road correctly; a horizontal (E/W) road needs the band rotated 90° to still cross
        # it perpendicular to travel.
        sprite.angle = 90.0 - math.degrees(self.track.start_facing.heading_radians)
        self.decor_sprites.append(sprite)

    def draw(self) -> None:
        """Draws the whole static scene; the caller has already activated its camera."""
        self.background_sprites.draw()
        self.tile_sprites.draw()
        for (x1, y1), (x2, y2) in self.kerb_segments:
            arcade.draw_line(x1, y1, x2, y2, KERB_COLOR, KERB_LINE_WIDTH)
        self.decor_sprites.draw()
```

- [ ] **Step 3: Refactor `window.py` and `play.py` onto `TrackScene`**

Replace `src/neuroarena/render/window.py` with (behavior identical to before; the docstring is unchanged except it now says the static scene lives in `scene.py`):

```python
"""arcade window: draws the track/decor/car and runs a fixed-step accumulator loop
decoupled from the display framerate. The static track drawing lives in `scene.py`.
Imports `neuroarena.sim`; never the reverse."""

from __future__ import annotations

import arcade

from neuroarena.render.input import KeyboardInput
from neuroarena.render.manifest import AssetManifest
from neuroarena.render.scene import TrackScene, heading_to_sprite_angle
from neuroarena.sim.game import TICK_DT, Game

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class PlayWindow(arcade.Window):
    def __init__(self, game: Game, manifest: AssetManifest) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "neuroarena — Phase 1")
        self.game = game
        self.manifest = manifest
        self.input = KeyboardInput()
        self._accumulator = 0.0

        self.background_color = arcade.color.DARK_SPRING_GREEN
        self.scene = TrackScene(game.track, manifest)
        self.car_sprite = arcade.Sprite(str(manifest.car))
        self.car_sprite.width = game.constants.car_width
        self.car_sprite.height = game.constants.car_length

        self.camera = arcade.Camera2D()
        self._sync_car_sprite()

    def _sync_car_sprite(self) -> None:
        self.car_sprite.center_x = self.game.car.x
        self.car_sprite.center_y = self.game.car.y
        self.car_sprite.angle = heading_to_sprite_angle(self.game.car.heading)
        self.camera.position = (self.car_sprite.center_x, self.car_sprite.center_y)

    def on_update(self, delta_time: float) -> None:
        steering, throttle = self.input.poll()
        self._accumulator += delta_time
        while self._accumulator >= TICK_DT:
            self.game.tick(steering, throttle)
            self._accumulator -= TICK_DT
        self._sync_car_sprite()

    def on_draw(self) -> None:
        self.clear()
        with self.camera.activate():
            self.scene.draw()
            arcade.draw_sprite(self.car_sprite)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_press(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_release(symbol)
```

(The window title keeps its existing text; do not change it.)

In `src/neuroarena/render/play.py`: replace the `MANIFEST_PATH = ...` line and its use with the shared constant — import `from neuroarena.render.scene import DEFAULT_MANIFEST_PATH`, delete the `MANIFEST_PATH` definition, and call `AssetManifest.load(DEFAULT_MANIFEST_PATH)`. (`tracks/cli.py` has its own `MANIFEST_PATH` and only constructs `PlayWindow(game, manifest)` — the constructor signature is unchanged, so leave it alone.)

- [ ] **Step 4: Create the viewer window**

Create `src/neuroarena/render/viewer.py`:

```python
"""The Phase 8 game viewer: an `arcade` window, run as its own process (opened by the
dashboard backend), that draws every still-active car of a training run's current generation
live on the real track art. It only draws — what to follow, the camera, and the overlay text
are decided by `view_model.py`; the backend connection is `viewer_client.py`. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`. Run manually with
`uv run python -m neuroarena.render.viewer --url ws://127.0.0.1:8000/ws/viewer`."""

from __future__ import annotations

import argparse
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.render.scene import DEFAULT_MANIFEST_PATH, TrackScene, heading_to_sprite_angle
from neuroarena.render.view_model import (
    CarView,
    ViewModel,
    overlay_lines,
    rank_cars,
    rank_of,
    track_bounds,
)
from neuroarena.render.viewer_client import ViewerClient, ViewerState
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.tracks import store as tracks_store

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DEFAULT_URL = "ws://127.0.0.1:8000/ws/viewer"
GHOST_ALPHA = 150
RING_RADIUS = 90.0
RING_WIDTH = 6
BEST_RING_COLOR = arcade.color.YELLOW
FOLLOWED_RING_COLOR = arcade.color.WHITE
OVERLAY_ROWS = 6


class ViewerWindow(arcade.Window):
    def __init__(self, client: ViewerClient, data_dir: Path) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "neuroarena — game viewer")
        self._client = client
        self._tracks_dir = data_dir / "tracks"
        self._manifest = AssetManifest.load(DEFAULT_MANIFEST_PATH)
        # The viewer draws the sprite at the default car size; a run configured with custom
        # physics constants would draw slightly off-size ghosts (accepted, see the plan notes).
        self._constants = PhysicsConstants()
        self.background_color = arcade.color.DARK_SPRING_GREEN
        self._camera = arcade.Camera2D()
        self._scene: TrackScene | None = None
        self._view: ViewModel | None = None
        self._loaded_track_id: str | None = None
        self._track_error: str | None = None
        self._sprites: list[arcade.Sprite] = []
        self._sprite_list: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self._overlay = [
            arcade.Text("", 14, WINDOW_HEIGHT - 26 - row * 22, arcade.color.WHITE, 15)
            for row in range(OVERLAY_ROWS)
        ]

    def _ensure_scene(self, track_id: str | None) -> None:
        if track_id is None or track_id == self._loaded_track_id:
            return
        self._loaded_track_id = track_id
        self._scene = None
        self._view = None
        self._track_error = None
        try:
            record = tracks_store.load(track_id, tracks_dir=self._tracks_dir)
            self._scene = TrackScene(record.track, self._manifest)
            self._view = ViewModel(track_bounds(record.track), (self.width, self.height))
        except Exception as exc:  # any load failure shows an overlay; it must not kill the viewer
            self._track_error = f"Could not load track {track_id}: {exc}"

    def _ensure_sprites(self, count: int) -> None:
        while len(self._sprites) < count:
            sprite = arcade.Sprite(str(self._manifest.car))
            sprite.width = self._constants.car_width
            sprite.height = self._constants.car_length
            sprite.alpha = 0
            self._sprites.append(sprite)
            self._sprite_list.append(sprite)

    def _place_cars(self, ranked: list[CarView]) -> None:
        """Sprite `k` shows the car at rank `len - 1 - k`, so the best car is the last sprite
        in the list and is drawn on top. Unused sprites are made fully transparent."""
        self._ensure_sprites(len(ranked))
        for index, sprite in enumerate(self._sprites):
            if index >= len(ranked):
                sprite.alpha = 0
                continue
            rank_index = len(ranked) - 1 - index
            car = ranked[rank_index]
            sprite.center_x, sprite.center_y = car.x, car.y
            sprite.angle = heading_to_sprite_angle(car.heading)
            sprite.alpha = 255 if rank_index == 0 else GHOST_ALPHA

    def _draw_rings(self, ranked: list[CarView], followed: CarView | None) -> None:
        if not ranked:
            return
        best = ranked[0]
        arcade.draw_circle_outline(best.x, best.y, RING_RADIUS, BEST_RING_COLOR, RING_WIDTH)
        if followed is not None and followed.genome_id != best.genome_id:
            arcade.draw_circle_outline(
                followed.x, followed.y, RING_RADIUS, FOLLOWED_RING_COLOR, RING_WIDTH
            )

    def _draw_overlay(self, state: ViewerState, followed_rank: int | None) -> None:
        lines = overlay_lines(
            connected=state.connected,
            run_state=state.run_state,
            generation=state.generation,
            alive=len(state.cars),
            population_size=state.population_size,
            speed=state.speed,
            settings=state.settings,
            followed_rank=followed_rank,
        )
        if self._track_error is not None:
            lines.append(self._track_error)
        for row, text in enumerate(self._overlay):
            text.text = lines[row] if row < len(lines) else ""
            text.draw()

    def on_draw(self) -> None:
        state = self._client.state()
        self._ensure_scene(state.track_id)
        self.clear()
        followed_rank: int | None = None
        if self._scene is not None and self._view is not None:
            ranked = rank_cars(state.cars)
            followed = self._view.followed_car(state.cars, state.settings)
            camera = self._view.camera(followed, state.settings)
            self._camera.position = (camera.center_x, camera.center_y)
            self._camera.zoom = camera.scale
            self._place_cars(ranked)
            with self._camera.activate():
                self._scene.draw()
                self._sprite_list.draw()
                self._draw_rings(ranked, followed)
            followed_rank = None if followed is None else rank_of(state.cars, followed)
        self._draw_overlay(state, followed_rank)
        if self._client.should_exit():
            arcade.exit()


def main() -> None:
    parser = argparse.ArgumentParser(description="neuroarena game viewer (opened by the dashboard).")
    parser.add_argument("--url", default=DEFAULT_URL, help="the dashboard's /ws/viewer URL")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    client = ViewerClient(args.url)
    client.start()
    ViewerWindow(client, args.data_dir)
    try:
        arcade.run()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
```

If `arcade` 3.x's API differs from what is written above (for example `Camera2D.zoom`, `arcade.Text`, `draw_circle_outline`, `arcade.exit`), check the installed version's docs/source (`uv run python -c "import arcade; print(arcade.__version__)"`) and adapt the call — keep the behavior — and note the change in your report.

- [ ] **Step 5: Run the gates**

Run: `uv run pytest tests -q && uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src`
Expected: PASS / clean (including `tests/sim/test_import_hygiene.py` and `tests/backends/neat/test_import_hygiene.py`).

- [ ] **Step 6: Automated no-crash smoke run**

A display is available on this machine (WSLg). This checks that the play window still opens after the refactor and that the viewer starts, connects and survives — it cannot judge how it *looks* (the user does that in Task 13). Use a scratch directory outside the repo and clean up every process you start (record PIDs with `$!`; do **not** use `pkill -f`).

```bash
SCRATCH=$(mktemp -d)
uv run neuroarena-track-gen --size 30 --seed 1 --tracks-dir "$SCRATCH/data/tracks"
uv run neuroarena-dashboard --data-dir "$SCRATCH/data" --port 8765 > "$SCRATCH/backend.log" 2>&1 &
BACKEND=$!
sleep 4
TRACK=$(ls "$SCRATCH/data/tracks" | grep -v manifest | head -1 | sed 's/.json//')
curl -s -X POST localhost:8765/api/runs -H 'content-type: application/json' \
  -d "{\"track_id\": \"$TRACK\", \"population_size\": 20}"
curl -s -X PATCH localhost:8765/api/speed -H 'content-type: application/json' -d '{"preset": "1x"}'
curl -s -X POST localhost:8765/api/viewer/open
sleep 10
curl -s localhost:8765/api/viewer          # expect "open": true (the viewer process is still alive)
curl -s -X POST localhost:8765/api/viewer/close
curl -s -X POST localhost:8765/api/runs/current/stop
sleep 3
kill $BACKEND
cat "$SCRATCH/backend.log" | tail -20      # expect no tracebacks from the viewer child
```
Report the `GET /api/viewer` output and whether the backend log shows a viewer traceback. Separately confirm the play window still opens: `timeout 8 uv run neuroarena-play` (expect it to run until the timeout kills it, with no traceback).

- [ ] **Step 7: Commit**

```bash
git add src/neuroarena/render/scene.py src/neuroarena/render/viewer.py src/neuroarena/render/window.py src/neuroarena/render/play.py tests/render/test_viewer_import.py
git commit -m "feat(phase-8): TrackScene extraction and the game viewer window"
```

---

### Task 12: Frontend — speed selector and "Game window" panel

**Files:**
- Modify: `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/App.tsx`
- Create: `frontend/src/components/SpeedControl.tsx`, `frontend/src/components/ViewerPanel.tsx`

**Interfaces:**
- Consumes: the REST routes from Tasks 5 and 9 (`GET/PATCH /api/speed`, `GET /api/viewer`, `POST /api/viewer/open|close`, `PATCH /api/viewer/settings`).
- Produces: the two components, rendered in `App.tsx` between `ConfigPanel` and "Saved models".

Node is not on the default PATH here: in every shell command that uses node/npm run `export PATH=$HOME/.nvm/versions/node/v24.18.0/bin:$PATH` first. There is no JS test harness (a deliberate project decision); the gates are `npx tsc -b`, `npm run build` and `npm run lint` (one known pre-existing lint warning in `NewRunForm.tsx` is expected — no *new* warnings), plus a curl-level check of the backend routes the components call. Follow the conventions already used in `NewRunForm.tsx` / `ConfigPanel.tsx` (cancel guards on effects, `err instanceof Error ? err.message : String(err)` for messages, controlled inputs).

- [ ] **Step 1: Types and API client**

Append to `frontend/src/types.ts`:

```ts
export interface SpeedState {
  preset: string
  presets: string[]
}

export type CameraMode = 'fit' | 'follow_best' | 'follow_rank'

export interface ViewSettings {
  zoom: number
  camera_mode: CameraMode
  follow_rank: number
  follow_seq: number
}

export interface ViewerState {
  open: boolean
  settings: ViewSettings
}
```

In `frontend/src/api.ts`, extend the type import to `import type { Model, RunStatus, SpeedState, StartRunRequest, Track, ViewSettings, ViewerState } from './types'` and append:

```ts
export function getSpeed(): Promise<SpeedState> {
  return fetch('/api/speed').then((r) => json<SpeedState>(r))
}

export function setSpeed(preset: string): Promise<{ preset: string }> {
  return fetch('/api/speed', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset }),
  }).then((r) => json<{ preset: string }>(r))
}

export function getViewer(): Promise<ViewerState> {
  return fetch('/api/viewer').then((r) => json<ViewerState>(r))
}

export function openViewer(): Promise<ViewerState> {
  return fetch('/api/viewer/open', { method: 'POST' }).then((r) => json<ViewerState>(r))
}

export function closeViewer(): Promise<ViewerState> {
  return fetch('/api/viewer/close', { method: 'POST' }).then((r) => json<ViewerState>(r))
}

export function updateViewerSettings(
  partial: Record<string, number | string>,
): Promise<ViewSettings> {
  return fetch('/api/viewer/settings', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  }).then((r) => json<ViewSettings>(r))
}
```

- [ ] **Step 2: `SpeedControl`**

Create `frontend/src/components/SpeedControl.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { getSpeed, setSpeed } from '../api'

export function SpeedControl() {
  const [presets, setPresets] = useState<string[]>([])
  const [preset, setPresetState] = useState('max')
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getSpeed()
      .then((speed) => {
        if (cancelled) return
        setPresets(speed.presets)
        setPresetState(speed.preset)
      })
      .catch((err) => {
        if (!cancelled) setMessage(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function choose(next: string) {
    const previous = preset
    setPresetState(next)
    setMessage(null)
    try {
      await setSpeed(next)
    } catch (err) {
      setPresetState(previous)
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <h2>Speed</h2>
      <label>
        Simulation speed
        <select value={preset} onChange={(e) => choose(e.target.value)}>
          {presets.map((p) => (
            <option key={p} value={p}>
              {p === 'max' ? 'Max (unpaced)' : p}
            </option>
          ))}
        </select>
      </label>
      <p>1x is real time. Slower speeds pace training so it is watchable, and make it take longer.</p>
      {message && <p role="alert">{message}</p>}
    </div>
  )
}
```

- [ ] **Step 3: `ViewerPanel`**

Create `frontend/src/components/ViewerPanel.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react'
import { closeViewer, getViewer, openViewer, updateViewerSettings } from '../api'
import type { CameraMode, ViewerState } from '../types'

const POLL_MS = 1000 // notice a window closed by hand

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function ViewerPanel() {
  const [viewer, setViewer] = useState<ViewerState | null>(null)
  const [rank, setRank] = useState('1')
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(() => {
    getViewer()
      .then(setViewer)
      .catch((err) => setMessage(describe(err)))
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  async function handleOpen() {
    setMessage(null)
    try {
      setViewer(await openViewer())
    } catch (err) {
      setMessage(describe(err))
    }
  }

  async function handleClose() {
    setMessage(null)
    try {
      setViewer(await closeViewer())
    } catch (err) {
      setMessage(describe(err))
    }
  }

  async function applySettings(partial: Record<string, number | string>) {
    setMessage(null)
    try {
      const settings = await updateViewerSettings(partial)
      setViewer((current) => (current === null ? current : { ...current, settings }))
    } catch (err) {
      setMessage(describe(err))
    }
  }

  const rankValue = Number(rank)
  const rankValid = Number.isInteger(rankValue) && rankValue >= 1

  return (
    <div>
      <h2>Game window</h2>
      <p>
        Opens the game on the machine running the backend, showing every car of the current
        generation. Status: {viewer === null ? '…' : viewer.open ? 'open' : 'closed'}
      </p>
      <button type="button" onClick={handleOpen} disabled={viewer === null || viewer.open}>
        Open game window
      </button>
      <button type="button" onClick={handleClose} disabled={viewer === null || !viewer.open}>
        Close game window
      </button>
      {viewer !== null && (
        <div>
          <label>
            Camera
            <select
              value={viewer.settings.camera_mode}
              onChange={(e) => applySettings({ camera_mode: e.target.value as CameraMode })}
            >
              <option value="fit">Whole track</option>
              <option value="follow_best">Follow the best car</option>
              <option value="follow_rank">Follow a chosen car</option>
            </select>
          </label>
          <label>
            Zoom ({viewer.settings.zoom}×)
            <input
              type="range"
              min={0.25}
              max={4}
              step={0.25}
              value={viewer.settings.zoom}
              onChange={(e) => applySettings({ zoom: Number(e.target.value) })}
            />
          </label>
          <label>
            Follow rank
            <input
              type="number"
              min={1}
              step={1}
              value={rank}
              onChange={(e) => setRank(e.target.value)}
            />
          </label>
          <button
            type="button"
            disabled={!rankValid}
            onClick={() => applySettings({ camera_mode: 'follow_rank', follow_rank: rankValue })}
          >
            Follow this rank
          </button>
        </div>
      )}
      {message && <p role="alert">{message}</p>}
    </div>
  )
}
```

- [ ] **Step 4: Wire into `App.tsx`**

In `frontend/src/App.tsx`, add the imports and render both components between `<ConfigPanel />` and the "Saved models" heading:

```tsx
import { SpeedControl } from './components/SpeedControl'
import { ViewerPanel } from './components/ViewerPanel'
```
```tsx
      <ConfigPanel />
      <SpeedControl />
      <ViewerPanel />
      <h2>Saved models</h2>
```

- [ ] **Step 5: Verify**

```bash
export PATH=$HOME/.nvm/versions/node/v24.18.0/bin:$PATH
cd frontend && npx tsc -b && npm run build && npm run lint
```
Expected: `tsc` clean, build OK, lint shows only the one known `NewRunForm.tsx` `set-state-in-effect` warning.

Backend contract check (proves the URLs/bodies the components send are the ones the backend accepts) — from the repo root, with a scratch data dir and `$!`-tracked PIDs as in Task 11:
```bash
SCRATCH=$(mktemp -d)
uv run neuroarena-dashboard --data-dir "$SCRATCH" --port 8766 > "$SCRATCH/backend.log" 2>&1 &
BACKEND=$!
sleep 4
curl -s localhost:8766/api/speed
curl -s -X PATCH localhost:8766/api/speed -H 'content-type: application/json' -d '{"preset": "2x"}'
curl -s localhost:8766/api/viewer
curl -s -X PATCH localhost:8766/api/viewer/settings -H 'content-type: application/json' \
  -d '{"camera_mode": "follow_rank", "follow_rank": 2}'
kill $BACKEND
```
Expected: the JSON shapes match the types added in Step 1.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/api.ts frontend/src/App.tsx frontend/src/components/SpeedControl.tsx frontend/src/components/ViewerPanel.tsx
git commit -m "feat(phase-8): speed selector and game-window panel"
```

---

### Task 13: Docs, `sim_speed` note, and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `docs/phases/phase-8-dashboard-game-viewer.md`
- Modify: `src/neuroarena/config/run_config.py` (docstring only)

**Interfaces:** none (documentation and verification only).

- [ ] **Step 1: `RunConfig` docstring note**

In `src/neuroarena/config/run_config.py`, add one paragraph at the end of the `RunConfig` class docstring, just before its closing `"""`, in the surrounding style:

```
    `sim_speed` stays an unused stub on purpose: simulation speed is dashboard-only runtime
    state (`neuroarena.dashboard.pacing`), deliberately not a `RunConfig` field and never
    persisted, so that a resumed run cannot inherit a real-time pacing speed."""
```
(Merge with the docstring's existing final line so the docstring still ends with a single `"""`.)

- [ ] **Step 2: README**

In `README.md`:
1. In "What it does", extend the **Web dashboard** bullet's end with: `open a game window that shows the whole generation live, and set the training speed.` (adjust punctuation), and replace the "Not there yet" paragraph with: `Not there yet: run-history charts and model inspection in the dashboard.`
2. In the `neuroarena-dashboard` section's "From the panel you can:" list, add two bullets:

```
- open a game window — a separate `arcade` viewer, on the machine running the backend — that
  shows every car of the current generation live on the real track art (the best car
  highlighted), with three camera modes (whole track, follow the best car, follow a chosen
  rank), zoom, and a small overlay (generation, cars alive, speed, camera);
- set the simulation speed: `0.25x`, `0.5x`, `1x` (real time), `2x`, `4x`, `8x`, or `Max`.
  Slower speeds pace training so the game window is watchable, and make training take longer;
  `Max` (the default every time the backend starts) runs as fast as the CPU allows.
```
3. After that list, add: `The game window needs a display on the same machine as the backend; the panel itself can be used remotely. To run the viewer by hand: \`uv run python -m neuroarena.render.viewer --url ws://127.0.0.1:8000/ws/viewer\`.`
4. In "Project layout", update the `render/` line to `arcade game window, viewer window, input, sprite assets`.
(No mention of any phase number anywhere in the README.)

- [ ] **Step 3: Phase doc — Implementation plan + revision entry**

In `docs/phases/phase-8-dashboard-game-viewer.md`, replace the `_Do not write this section until Requirements above is FINALIZED._` placeholder in "Implementation plan" with (matching the style of Phases 5–7):

```
Detailed, step-by-step plan: [`../superpowers/plans/2026-09-18-phase-8-game-viewer.md`](../superpowers/plans/2026-09-18-phase-8-game-viewer.md).

Thirteen tasks, each ending in an independently testable, committed deliverable:

1. `Visualizable` protocol + `CarEnvironment.visual_state()`.
2. `evaluate_batch`: live-genome publication and the per-round `pace` hook.
3. `NeatTrainer.visual_snapshot()` and `set_pace()`.
4. `dashboard/pacing.py`: speed presets and `Pacer`.
5. `RunManager` speed state, pacer wiring, visual snapshot; `/api/speed`.
6. `render/view_settings.py` (pure `ViewSettings`).
7. `render/view_model.py` (ranking, follow logic, camera, overlay — pure).
8. Viewer protocol messages and `ViewerManager`.
9. Viewer REST routes, `/ws/viewer`, shutdown and CLI wiring.
10. `render/viewer_client.py` (the viewer's backend connection).
11. `TrackScene` extraction and the viewer window.
12. Frontend: speed selector and game-window panel.
13. Docs, `sim_speed` note and end-to-end verification.
```

Append to the Revision history: `- 2026-09-18 — Implementation plan written and executed: [...]` — write this entry **after** Step 4 passes, stating that the phase is implemented, listing any plan-level deviations that occurred during implementation (or "none"), and adding an "Open items" line: the visual check of the game window by the user (the automated smoke run confirms only that it starts and stays up), and that a run configured with custom `physics_constants` draws default-sized car sprites.

- [ ] **Step 4: Full verification**

```bash
export PATH=$HOME/.nvm/versions/node/v24.18.0/bin:$PATH
timeout 600 uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src
(cd frontend && npx tsc -b && npm run build && npm run lint)
```
Expected: all green (lint: only the one known warning). Then repeat the Task 11 Step 6 smoke run once more against the finished tree (backend + run + `1x` speed + open viewer + 10 s + close), and additionally confirm pacing works end to end: with a run active, `curl PATCH /api/speed {"preset":"1x"}` then `{"preset":"max"}`, sampling `GET /api/runs/current` (or `/ws` progress) shows the generation rate drop and recover. Report the numbers.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/phases/phase-8-dashboard-game-viewer.md src/neuroarena/config/run_config.py
git commit -m "docs(phase-8): README, plan section, sim_speed note and completion entry"
```

---

## Self-Review

**Spec coverage:** view content (whole batch, ghosts, best highlighted, cars drop out) → Tasks 1–3 (snapshot of live genomes), 7 (rank/best), 11 (drawing); three camera modes + zoom + follow-by-rank with hold-and-fall-back → Tasks 6, 7; overlay (generation, cars alive/total, speed, camera + followed rank) → Task 7 (`overlay_lines`), 11; speed presets, pacing hook, runtime-state-not-`RunConfig`, default `max` → Tasks 2–5; `sim_speed` stub left and documented → Task 13; 60 Hz frame stream on a separate `/ws/viewer` with the panel's `/ws` unchanged → Tasks 8–9; separate viewer process, single instance, backend-owned, terminated on shutdown, crash/hand-close shown as closed → Tasks 8, 9, 12; viewer loads track/assets itself → Task 11; never-blank behaviour (cut at generation boundary, "waiting for a run", keep last frame with final status, doesn't self-close on run end) → Tasks 7 (`overlay_lines`), 10 (`run` handling keeps cars until a new run), 11; panel controls (open/close, camera mode, zoom, follow rank, speed) → Task 12; disconnected/retry/exit-after-grace and track-load-failure overlay → Tasks 10, 11; ranking by `Objective.fitness()` → Tasks 3, 7; docs/README → Task 13.

**Placeholders:** none — every code step has complete code; the only open-ended instruction is Task 11's "adapt to the installed `arcade` version if an API name differs", which names the calls at risk and requires a report.

**Type consistency:** `BatchVisuals`/`GenomeVisual` (Task 3) → `viewer_frame_message` (Task 8) reads `.genomes[*].genome_id/.state/.fitness` and `.generation/.population_size` ✓; wire `cars[*]` keys `id/x/y/heading/fitness` (Task 8) ↔ `ViewerClient.apply_message` (Task 10) ✓; `ViewSettings.to_dict/from_dict` keys (`zoom/camera_mode/follow_rank/follow_seq`) used identically in Tasks 6, 8, 9, 10, 12 ✓; `CarView` fields (Task 7) ↔ Task 10 ↔ Task 11 ✓; `RunManager.visual_snapshot/current_track_id/speed_preset` (Task 5) ↔ Task 9's `/ws/viewer` ✓; `ViewerManager` API (`open/close/is_open/shutdown/settings/update_settings`) (Task 8) ↔ Task 9 ✓; `FakeSpawner` (Task 8) used in Task 9 ✓; `heading_to_sprite_angle` / `DEFAULT_MANIFEST_PATH` (Task 11 `scene.py`) used by `window.py`, `play.py`, `viewer.py` ✓.

**Known limitations recorded, not built:** the viewer draws default-size car sprites (a run with custom `physics_constants` would look slightly off-size); pacing can only slow a run, never speed one past what the CPU allows (at `8x` a large population may already be CPU-bound); the window's drawing is verified by a no-crash smoke run plus the user's visual check, not by automated tests.
