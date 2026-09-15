# NeatTrainer.update_config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `NeatTrainer` satisfy the `Trainer.update_config` method Phase 0's doc already added on 2026-09-15, so a live-changeable `RunConfig` field can be pushed into a trainer that is already mid-`run()` — the mechanism Phase 7's dashboard needs for live config edits.

**Architecture:** `update_config` validates every key in its `partial` dict against a small, explicit allow-list of the three `RunConfig` fields `NeatTrainer` already re-reads fresh every generation, then stages the change under a `threading.Lock` rather than applying it in place. `_run_one_generation` applies any staged change atomically as its very first action — before it reads `self._config` for anything — so a generation already in progress when `update_config` is called is never affected; only the next one sees it.

**Tech Stack:** Python 3.12+ standard library only (`threading`, `dataclasses`). No new dependencies.

**Spec:** [`../../phases/phase-0-architecture.md`](../../phases/phase-0-architecture.md) — the `Trainer` interface section already documents `update_config`'s full contract (added 2026-09-15). [`../../phases/phase-5-training-controls.md`](../../phases/phase-5-training-controls.md)'s "Knob classification" bullet is the authority on which `RunConfig` fields are live-changeable at all, and this plan's scope decision (below) narrows that further to what `NeatTrainer` can safely support today.

## Global Constraints

- `update_config`'s signature must match Phase 0's doc exactly: `update_config(self, partial: Mapping[str, Any]) -> None`.
- **Scope, decided with the user:** only these three `RunConfig` fields are supported — `max_generation_steps`, `max_generations`, `target_fitness`. Every other key `partial` might contain — `neat_hyperparameters`, `physics_constants`, `max_episode_steps`, `track_id`, `sim_speed`, `population_size`, `headless`, `sensor_config`, anything else — must raise `ValueError` naming exactly what's unsupported and why. Do not silently accept and no-op on an unsupported key; do not attempt to wire any of the excluded fields in this plan.
- Thread-safety is a real requirement, not decoration: `update_config` must be safe to call from a thread other than the one driving `run()`, and a generation already in progress must never observe a config value that changed mid-generation.
- Do not touch `src/neuroarena/backends/neat/evaluation.py`, `src/neuroarena/sim/objectives.py`, or `src/neuroarena/persistence/generation_stats_repo.py` — out of scope.

---

## File Structure

- **Modify `src/neuroarena/backends/neat/trainer.py`** — add a module-level `_LIVE_CHANGEABLE_FIELDS` constant, two new `NeatTrainer.__init__` attributes (`_config_lock`, `_pending_config_update`), a new public `update_config` method, and three new lines at the very top of `_run_one_generation` that apply any staged update before anything else runs. Update the module and class docstrings to mention it.
- **Modify `tests/backends/neat/test_trainer.py`** — four new tests: rejecting an unsupported field, applying starting the next generation (not immediately), merging multiple staged updates, and a real thread-safety test proving a generation already in progress is unaffected by a concurrent `update_config` call.
- **Modify `docs/phases/phase-0-architecture.md`** — Task 2 links this plan, updates the `update_config` bullet to say it's now implemented (for the three scoped fields), and flips status `REVISED` → `FINALIZED`.

---

### Task 1: `NeatTrainer.update_config`

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Test: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: nothing new — `RunConfig` (`neuroarena.config`), already imported under `TYPE_CHECKING` in `trainer.py`, is used only via `dataclasses.replace(self._config, **partial)` (no new import of `RunConfig` itself needed at runtime, since it's accessed only through the existing `self._config` instance).
- Produces: `NeatTrainer.update_config(self, partial: Mapping[str, Any]) -> None` — matches `Trainer.update_config` exactly, satisfying the Protocol Phase 0 already declares.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/backends/neat/test_trainer.py`:

```python
def test_update_config_rejects_unsupported_fields() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    with pytest.raises(ValueError, match="population_size"):
        trainer.update_config({"population_size": 10})
    with pytest.raises(ValueError, match="neat_hyperparameters"):
        trainer.update_config({"neat_hyperparameters": {"compatibility_threshold": 1.0}})


def test_update_config_applies_starting_the_next_generation_not_immediately() -> None:
    config = RunConfig(max_generation_steps=300_000)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    run = trainer.run()
    gen0 = next(run)
    # DummyEnvironment truncates every genome at 5 steps, DummyObjective never should_stop()s,
    # so all 5 genomes run their full natural length under the original, generous budget.
    assert gen0.sim_time == 25.0

    trainer.update_config({"max_generation_steps": 7})
    gen1 = next(run)
    # Round-robin distributes a tight budget=7 across 5 genomes: round 1 (7->2, all 5 step),
    # round 2 (2->0, only 2 of the 5 get a second step before the budget runs out) = 7 total.
    assert gen1.sim_time - gen0.sim_time == 7.0


def test_update_config_merges_multiple_pending_updates_before_they_apply() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    trainer.update_config({"max_generations": 2})
    # Deliberately unreachable on its own (DummyObjective's fitness never exceeds 5.0) — if
    # this call silently clobbered the max_generations update above instead of merging with
    # it, the run would never stop and this test would hang until pytest's own timeout.
    trainer.update_config({"target_fitness": 1_000_000.0})
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1]


def test_update_config_does_not_affect_a_generation_already_in_progress() -> None:
    started = threading.Event()
    release = threading.Event()

    class PausesOnFirstUpdateCall(DummyObjective):
        _paused = False

        def update(
            self,
            observation: np.ndarray,
            action: np.ndarray,
            terminated: bool,
            truncated: bool,
            info: dict[str, typing.Any],
        ) -> None:
            super().update(observation, action, terminated, truncated, info)
            if not PausesOnFirstUpdateCall._paused:
                PausesOnFirstUpdateCall._paused = True
                started.set()
                assert release.wait(timeout=5), "test deadlocked waiting for release"

    config = RunConfig(max_generation_steps=300_000)
    trainer = NeatTrainer(
        DummyEnvironment, PausesOnFirstUpdateCall(), config, population_size=5
    )
    run = trainer.run()

    results: list[TrainingUpdate] = []

    def drive_generation_zero() -> None:
        results.append(next(run))

    thread = threading.Thread(target=drive_generation_zero)
    thread.start()
    assert started.wait(timeout=5), "generation 0 never reached its first genome step"

    # Called from this (main) thread while generation 0 is paused mid-step in the background
    # thread above — this is the concurrent-call scenario update_config must handle safely.
    trainer.update_config({"max_generation_steps": 7})
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "background generation never finished"

    gen0 = results[0]
    # Generation 0's step_budget was already captured from the ORIGINAL config before the
    # pause; the concurrent update must not retroactively shrink a generation in progress.
    assert gen0.sim_time == 25.0

    gen1 = next(run)
    # The staged update only takes effect starting the next generation boundary.
    assert gen1.sim_time - gen0.sim_time == 7.0
```

Add the two new imports these tests need at the top of `tests/backends/neat/test_trainer.py` — check the file's current import block first; `random`, `typing`, `warnings`, `Path`, `numpy as np`, `pytest`, and the various `neuroarena.*` names are already imported, so only `threading` is missing:

```python
import threading
```

(Add it alphabetically among the existing top-level `import` lines — `random`, `threading`, `typing`, `warnings`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backends/neat/test_trainer.py -k "update_config" -v`
Expected: FAIL — `AttributeError: 'NeatTrainer' object has no attribute 'update_config'`.

- [ ] **Step 3: Add the module-level constant**

In `src/neuroarena/backends/neat/trainer.py`, add this right after the existing `_CHECKPOINT_SCHEMA_VERSION = 2` line:

```python
# Phase 5's Knob classification names more live-changeable `RunConfig` fields than this
# trainer actually re-reads per generation; only these three are wired to `update_config`
# today — see `update_config`'s own docstring for why the rest are excluded.
_LIVE_CHANGEABLE_FIELDS = frozenset({"max_generation_steps", "max_generations", "target_fitness"})
```

- [ ] **Step 4: Update the imports**

At the top of `src/neuroarena/backends/neat/trainer.py`, replace:

```python
import copy
import itertools
import numbers
import pickle
import random
import time
from collections.abc import Callable, Iterator
```

with:

```python
import copy
import dataclasses
import itertools
import numbers
import pickle
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
```

- [ ] **Step 5: Update `NeatTrainer.__init__`**

In `src/neuroarena/backends/neat/trainer.py`, find this block inside `__init__`:

```python
        self._make_env = make_env
        self._objective = objective
        self._config = config
        self._champion_dir = champion_dir
```

Replace it with:

```python
        self._make_env = make_env
        self._objective = objective
        self._config = config
        # Guards `_pending_config_update` — `update_config` may be called from a different
        # thread than the one driving `run()` (Phase 0's `Trainer.update_config` contract).
        self._config_lock = threading.Lock()
        self._pending_config_update: dict[str, Any] | None = None
        self._champion_dir = champion_dir
```

- [ ] **Step 6: Add `update_config`**

In `src/neuroarena/backends/neat/trainer.py`, add this new method immediately after `__init__` ends (right before the `def run(self) -> Iterator[TrainingUpdate]:` method):

```python
    def update_config(self, partial: Mapping[str, Any]) -> None:
        """Satisfies `Trainer.update_config` (Phase 0). Thread-safe: safe to call from a
        different thread than the one driving `run()` — e.g. a dashboard request handler.
        The change is staged, not applied in place, and is applied atomically at the top of
        `_run_one_generation`'s next call, so a generation already in progress when this is
        called is never affected — only the *next* generation sees it.

        Only the three `RunConfig` fields `NeatTrainer` already re-reads fresh every
        generation are supported today — see Phase 5's Knob classification
        (`../../../docs/phases/phase-5-training-controls.md`) for the full live-changeable
        list. The rest of that list is deliberately not wired here yet: `neat_hyperparameters`
        needs careful handling of live `neat-python` internal state (id allocators, speciation
        bookkeeping) that Phase 5 itself left as an open question; `physics_constants`,
        `max_episode_steps`, `track_id`, and `sim_speed` only take effect through the
        caller-supplied `make_env` closure, which `NeatTrainer` does not control and cannot
        push a live update into."""
        unsupported = set(partial) - _LIVE_CHANGEABLE_FIELDS
        if unsupported:
            raise ValueError(
                f"NeatTrainer.update_config does not support {sorted(unsupported)} yet — "
                f"only {sorted(_LIVE_CHANGEABLE_FIELDS)} are wired. neat_hyperparameters "
                "needs careful handling of live neat-python internal state not yet designed; "
                "physics_constants/max_episode_steps/track_id/sim_speed only take effect "
                "through the caller-supplied make_env closure, which NeatTrainer does not "
                "control."
            )
        with self._config_lock:
            self._pending_config_update = {
                **(self._pending_config_update or {}),
                **dict(partial),
            }
```

- [ ] **Step 7: Apply the staged update at the top of `_run_one_generation`**

In `src/neuroarena/backends/neat/trainer.py`, find the start of `_run_one_generation`:

```python
    def _run_one_generation(self) -> TrainingUpdate:
        self._env = self._make_env()
```

Replace it with:

```python
    def _run_one_generation(self) -> TrainingUpdate:
        with self._config_lock:
            if self._pending_config_update is not None:
                self._config = dataclasses.replace(self._config, **self._pending_config_update)
                self._pending_config_update = None
        self._env = self._make_env()
```

- [ ] **Step 8: Update the module and class docstrings**

In `src/neuroarena/backends/neat/trainer.py`, the module docstring currently ends with `...genome finishing ends the whole batch immediately for the rest."""` — append one sentence before the closing `"""` so the docstring reads:

```python
"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating a
generation's whole batch of genomes concurrently (round-robin lockstep, see `evaluation.py`)
under a per-generation step budget shared across the batch. See
`../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this implements,
in particular why track switches only take effect at a generation boundary, why a
force-truncated genome is scored on partial progress rather than penalised, and why any
genome finishing ends the whole batch immediately for the rest. `update_config` (Phase 0)
lets a caller stage a live-changeable config field mid-`run()`, applied atomically at the
next generation boundary — see that method's own docstring for exactly which fields."""
```

Leave the `NeatTrainer` class docstring as-is — it already describes `self._env`/`make_env` semantics unrelated to this change.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/backends/neat/test_trainer.py -k "update_config" -v`
Expected: PASS (4 tests).

- [ ] **Step 10: Run the full NEAT backend test suite, then the whole project**

Run: `uv run pytest tests/backends/neat/ -v`
Expected: PASS, all tests including the four new ones — nothing else in this file should need changes, since every existing test either never calls `update_config` (unaffected) or exercises behavior this change doesn't touch.

Run: `uv run pytest -v`
Expected: PASS, all green.

- [ ] **Step 11: Type-check and lint**

Run: `uv run mypy --strict src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py`
Expected: no errors.

Run: `uv run ruff check src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py && uv run ruff format --check src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py`
Expected: no errors.

- [ ] **Step 12: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-0): NeatTrainer.update_config — thread-safe, staged, generation-boundary-effective config updates for max_generation_steps/max_generations/target_fitness"
```

---

### Task 2: Record completion in the Phase 0 doc

**Files:**
- Modify: `docs/phases/phase-0-architecture.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Update the `update_config` bullet**

In `docs/phases/phase-0-architecture.md`, find this bullet in the `Trainer` interface section:

```
- **`update_config`** (added 2026-09-15, see Revision history) lets a live-changeable config field be updated while `run()` is already executing, without tearing the trainer down — Phase 7's dashboard needs this for live config edits. It is safe to call from another thread than the one driving `run()`: the change is staged, not applied in place, and every backend applies it atomically at the top of its own next unit of progress (NEAT: next generation boundary) so no genome/rollout ever sees a config mutate mid-evaluation. **Not yet implemented by `NeatTrainer`** — the Protocol carries the method, the NEAT backend doesn't satisfy it yet (see Phase 4's Implementation plan note).
```

Replace the final sentence (`**Not yet implemented by...`) with:

```
Implemented by `NeatTrainer` for `max_generation_steps`/`max_generations`/`target_fitness` — see [`../superpowers/plans/2026-09-15-neat-trainer-update-config.md`](../superpowers/plans/2026-09-15-neat-trainer-update-config.md). The rest of Phase 5's live-changeable knob list (`neat_hyperparameters`, `physics_constants`, `max_episode_steps`, `track_id`, `sim_speed`) is explicitly rejected by `NeatTrainer.update_config` today, not silently ignored — see that method's own docstring for why each is excluded.
```

- [ ] **Step 2: Flip status back to FINALIZED and add a Revision history entry**

Change the Status line at the top of the doc from:

```
Status: **REVISED** — requirements settled as of 2026-09-10, then deliberately revised on 2026-09-15 to add `Trainer.update_config` (Phase 7 needs it). The NEAT backend (Phase 4) does not implement it yet — see the `Trainer` interface section below. Changing the requirements further needs its own deliberate revision (see `../WORKFLOW.md`). The two items under Open questions are deferred by design and do not block.
```

to:

```
Status: **FINALIZED** — requirements settled as of 2026-09-10, revised 2026-09-15 to add `Trainer.update_config` (Phase 7 needs it), implemented by `NeatTrainer` the same day (for the fields it can safely support — see the `Trainer` interface section below). Changing the requirements further needs its own deliberate revision (see `../WORKFLOW.md`). The two items under Open questions are deferred by design and do not block.
```

Add a new entry at the top of the Revision history section (immediately after `## Revision history`, before the existing 2026-09-15 "Deliberate revision, status → REVISED" entry):

```
- 2026-09-15 — Implemented `Trainer.update_config` on `NeatTrainer`, for the same day's revision above — scoped to `max_generation_steps`/`max_generations`/`target_fitness` (the fields `NeatTrainer` already re-reads fresh every generation); every other Phase-5-classified live-changeable field is explicitly rejected with an error naming why, not silently accepted. Status REVISED -> FINALIZED. See [`../superpowers/plans/2026-09-15-neat-trainer-update-config.md`](../superpowers/plans/2026-09-15-neat-trainer-update-config.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/phases/phase-0-architecture.md
git commit -m "docs(phase-0): record NeatTrainer.update_config implementation completion"
```
