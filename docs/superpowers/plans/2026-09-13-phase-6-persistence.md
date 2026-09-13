# Phase 6 — Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SQLite-backed persistence for run/model metadata and history (five tables: `models`, `runs`, `generation_stats`, `settings_history`, `checkpoints`), file-based checkpoints reusing Phase 4's unmodified NEAT pickle format, and a `neuroarena-train` CLI that actually launches, persists, and can resume a real training run — the first working end-to-end training entrypoint in this codebase.

**Architecture:** A new top-level package `neuroarena/persistence/` holds a migration-driven SQLite connection helper (`db.py`), one thin repository module per table (dataclasses in/out, each write self-committing — no cross-table transactions, matching this codebase's existing "plain dataclass + explicit function" style from `tracks/store.py`), a `recorder.py` that wraps any `Trainer.run()` iterator to persist `TrainingUpdate`s and manage checkpoint cadence/retention, and `cli.py` (`neuroarena-train`) that wires `tracks.store` + `ProgressObjective` (new, in `neuroarena/sim/objectives.py`) + `NeatTrainer` + the recorder together. `RunConfig` gains two new scalar knobs (`checkpoint_every_n_generations`, `champion_retention_cap`). `NeatTrainer`, `build_neat_config`, and the checkpoint pickle format are **not modified** — the recorder only calls the public `Trainer` protocol (`run()`, `save_checkpoint()`) and independently reconstructs the champion-file naming convention `NeatTrainer` already uses (`champion_dir/gen_<generation:05d>.pkl`) rather than reading any private attribute.

**Not built in this plan (explicitly out of scope):** anything Phase 7's dashboard owns — starting/stopping/pausing a *running* process, live mid-run config edits, or a UI over any of this. `--resume` re-specifies the full `RunConfig` from CLI flags/defaults each time, not a "change only these fields" flow — good enough for a dev CLI; a real partial-override resume UX is Phase 7's job. Objective *selection* among multiple objectives (this plan ships exactly one, `ProgressObjective`) — see the phase doc's Revision history on why one minimal objective was added here at all. Mirroring Phase 2's track manifest into SQLite — explicitly decided against in the phase doc.

**Tech Stack:** Python 3.12+, stdlib `sqlite3` (no ORM, no new dependency), `neat-python`, `numpy`, `pytest`, `mypy --strict`, `ruff`.

**Spec:** [`../../phases/phase-6-persistence.md`](../../phases/phase-6-persistence.md) (FINALIZED 2026-09-13). Builds on the FINALIZED [`../../phases/phase-0-architecture.md`](../../phases/phase-0-architecture.md) (`Trainer`/`Objective` protocols, `TrainingUpdate`), [`../../phases/phase-2-track-generation.md`](../../phases/phase-2-track-generation.md) (`tracks.store`), [`../../phases/phase-4-learning-backend-neat.md`](../../phases/phase-4-learning-backend-neat.md) (`NeatTrainer`, unmodified), and [`../../phases/phase-5-training-controls.md`](../../phases/phase-5-training-controls.md) (`RunConfig`, `CarEnvironmentConfig.from_run_config`).

## Global Constraints

- Python 3.12+; `uv` for deps; `pytest` for tests; `ruff` for lint+format; `mypy --strict` (see `pyproject.toml`'s `[tool.mypy]`). No new production dependencies — `sqlite3` is stdlib.
- New package `src/neuroarena/persistence/` (`db.py`, `migrations/0001_initial.sql`, `models_repo.py`, `runs_repo.py`, `generation_stats_repo.py`, `checkpoints_repo.py`, `settings_history_repo.py`, `recorder.py`, `cli.py`) plus one new file `src/neuroarena/sim/objectives.py`. Modifies `src/neuroarena/config/run_config.py` (two new fields) and `pyproject.toml` (`[project.scripts]` gains `neuroarena-train`).
- `src/neuroarena/backends/neat/{trainer,config,model,evaluation}.py` are **not modified** anywhere in this plan.
- Every repository write function commits immediately (`conn.commit()` inside the function) — no multi-statement transactions. Appropriate for a single-user local SQLite file; simplest thing that keeps a crash mid-run from leaving a torn transaction, at the cost of no atomic multi-table writes (not needed here — each row stands alone).
- `conn.row_factory = sqlite3.Row` is set once in `db.connect()`; every repo module reads rows by column name (`row["field"]`), never by index.
- `data/` is already gitignored (`.gitignore`: `data/`) — `data/neuroarena.db` and `data/checkpoints/` are runtime artifacts, never committed, matching `data/tracks/`'s existing treatment.
- Primary keys for `models`/`checkpoints` (and `run_id`) are `uuid.uuid4().hex` strings, matching `TrackRecord.track_id`'s existing convention (`tracks/store.py`) — not autoincrement ints. `generation_stats`/`settings_history` rows use an autoincrement `id` since nothing references them individually by id.
- Every test that touches SQLite or the filesystem uses `tmp_path` (pytest's built-in fixture) for the db file and any checkpoint/track directories — never `data/` — matching `tests/tracks/test_store.py`'s existing pattern.

---

## Task 1: Migration-driven SQLite connection (`db.py`) + initial schema

**Files:**
- Create: `src/neuroarena/persistence/__init__.py` (empty)
- Create: `src/neuroarena/persistence/db.py`
- Create: `src/neuroarena/persistence/migrations/0001_initial.sql`
- Test: `tests/persistence/__init__.py` (empty)
- Test: `tests/persistence/test_db.py`

**Interfaces:**
- Produces: `db.connect(db_path: Path) -> sqlite3.Connection` — applies any unapplied migration in `migrations/` (ordered by the leading number in the filename) via `PRAGMA user_version`, sets `PRAGMA foreign_keys = ON` and `conn.row_factory = sqlite3.Row`, and is idempotent (safe to call repeatedly against the same file). Consumed by every later task.

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/__init__.py` (empty file) and `tests/persistence/test_db.py`:

```python
from pathlib import Path

from neuroarena.persistence.db import connect


def test_connect_creates_all_five_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"models", "runs", "generation_stats", "settings_history", "checkpoints"} <= tables


def test_connect_sets_user_version(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_connect_enables_foreign_keys(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connect_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    connect(db_path)
    conn = connect(db_path)  # second connect on the same file must not error or re-apply
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_row_factory_supports_column_access(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO models (model_id, backend, observation_space, action_space, created_at) "
        "VALUES ('m1', 'neat', '{}', '{}', 'now')"
    )
    conn.commit()
    row = conn.execute("SELECT * FROM models WHERE model_id = 'm1'").fetchone()
    assert row["backend"] == "neat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence'`.

- [ ] **Step 3: Write the migration file and connection helper**

Create an empty `src/neuroarena/persistence/__init__.py` (this project's existing packages — `backends/__init__.py`, `config/__init__.py`, etc. — are all empty marker files; match that convention).

Create `src/neuroarena/persistence/migrations/0001_initial.sql`:

```sql
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    observation_space TEXT NOT NULL,
    action_space TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    track_id TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    starting_generation INTEGER NOT NULL
);

CREATE TABLE generation_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    best_fitness REAL NOT NULL,
    mean_fitness REAL NOT NULL,
    worst_fitness REAL NOT NULL,
    population_size INTEGER NOT NULL,
    champion_metrics TEXT NOT NULL,
    sim_time REAL NOT NULL,
    wall_time REAL NOT NULL
);

CREATE TABLE settings_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    diff TEXT NOT NULL
);

CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    kind TEXT NOT NULL,
    generation INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE INDEX idx_generation_stats_model ON generation_stats(model_id, generation);
CREATE INDEX idx_settings_history_model ON settings_history(model_id, generation);
CREATE INDEX idx_checkpoints_model_kind ON checkpoints(model_id, kind, generation);
```

Create `src/neuroarena/persistence/db.py`:

```python
"""Migration-driven SQLite connection helper. Schema versioning follows the same
versioned-and-migrated shape used elsewhere in this codebase (`RunConfig.schema_version`,
NEAT checkpoint `schema_version`, `TrackRecord.schema_version`) — see
`../../../docs/phases/phase-6-persistence.md`'s "Schema versioning" section — applied to
the database as a whole via `PRAGMA user_version` plus ordered `.sql` files here."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DEFAULT_DB_PATH = Path("data/neuroarena.db")


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for path in migrations:
        version = int(path.stem.split("_")[0])
        if version <= current_version:
            continue
        conn.executescript(path.read_text())
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_db.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/__init__.py src/neuroarena/persistence/db.py \
        src/neuroarena/persistence/migrations/0001_initial.sql \
        tests/persistence/__init__.py tests/persistence/test_db.py
git commit -m "feat(phase-6): SQLite migration runner + initial 5-table schema"
```

---

## Task 2: `models` repository + `Space` JSON codec

**Files:**
- Create: `src/neuroarena/persistence/models_repo.py`
- Test: `tests/persistence/test_models_repo.py`

**Interfaces:**
- Consumes: `db.connect` (Task 1), `neuroarena.interfaces.spaces.{Box,Discrete,Space}`.
- Produces: `ModelRecord` (frozen dataclass: `model_id: str`, `backend: str`, `observation_space: Space`, `action_space: Space`, `created_at: str`); `create_model(conn, *, backend: str, observation_space: Space, action_space: Space) -> ModelRecord`; `get_model(conn, model_id: str) -> ModelRecord` (raises `UnknownModelError` — a `KeyError` subclass, matching `tracks/store.py`'s `UnknownTrackError` precedent); `list_models(conn) -> list[ModelRecord]`. Consumed by Task 9 (recorder) and Task 10 (CLI).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_models_repo.py`:

```python
from pathlib import Path

import pytest

from neuroarena.interfaces.spaces import Box, Discrete
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import UnknownModelError, create_model, get_model, list_models


def test_create_model_round_trips_spaces(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    )
    loaded = get_model(conn, record.model_id)
    assert loaded == record
    assert loaded.observation_space == Box(-1.0, 1.0, (10,))


def test_create_model_round_trips_discrete_space(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_model(conn, backend="neat", observation_space=Box(-1.0, 1.0, (3,)), action_space=Discrete(4))
    loaded = get_model(conn, record.model_id)
    assert loaded.action_space == Discrete(4)


def test_get_model_unknown_id_raises(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    with pytest.raises(UnknownModelError):
        get_model(conn, "does-not-exist")


def test_list_models_returns_all_created(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    for _ in range(3):
        create_model(conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,)))
    assert len(list_models(conn)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_models_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.models_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/neuroarena/persistence/models_repo.py`:

```python
"""`models` table: the persistent trainable artifact's identity, backend, and the
observation/action descriptors it was trained against. Informational only — resume-safety
enforcement stays exactly where Phase 0 put it (`interfaces.compat.check_compatibility`,
and `NeatTrainer.load_checkpoint`'s own descriptor check), not here. See
`../../../docs/phases/phase-6-persistence.md`'s "Entity model" and "Schema" sections."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from neuroarena.interfaces.spaces import Box, Discrete, Space


class UnknownModelError(KeyError):
    """Raised when a model_id has no matching row."""


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    backend: str
    observation_space: Space
    action_space: Space
    created_at: str


def create_model(
    conn: sqlite3.Connection, *, backend: str, observation_space: Space, action_space: Space
) -> ModelRecord:
    record = ModelRecord(
        model_id=uuid.uuid4().hex,
        backend=backend,
        observation_space=observation_space,
        action_space=action_space,
        created_at=datetime.now(UTC).isoformat(),
    )
    conn.execute(
        "INSERT INTO models (model_id, backend, observation_space, action_space, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            record.model_id,
            record.backend,
            _space_to_json(record.observation_space),
            _space_to_json(record.action_space),
            record.created_at,
        ),
    )
    conn.commit()
    return record


def get_model(conn: sqlite3.Connection, model_id: str) -> ModelRecord:
    row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    if row is None:
        raise UnknownModelError(model_id)
    return _record_from_row(row)


def list_models(conn: sqlite3.Connection) -> list[ModelRecord]:
    rows = conn.execute("SELECT * FROM models ORDER BY created_at").fetchall()
    return [_record_from_row(row) for row in rows]


def _record_from_row(row: sqlite3.Row) -> ModelRecord:
    return ModelRecord(
        model_id=row["model_id"],
        backend=row["backend"],
        observation_space=_space_from_json(row["observation_space"]),
        action_space=_space_from_json(row["action_space"]),
        created_at=row["created_at"],
    )


def _space_to_json(space: Space) -> str:
    if isinstance(space, Box):
        return json.dumps(
            {"kind": "Box", "low": space.low, "high": space.high, "shape": list(space.shape), "dtype": space.dtype}
        )
    return json.dumps({"kind": "Discrete", "n": space.n})


def _space_from_json(text: str) -> Space:
    raw = json.loads(text)
    if raw["kind"] == "Box":
        return Box(low=raw["low"], high=raw["high"], shape=tuple(raw["shape"]), dtype=raw["dtype"])
    return Discrete(n=raw["n"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_models_repo.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/models_repo.py tests/persistence/test_models_repo.py
git commit -m "feat(phase-6): models repository + Space JSON codec"
```

---

## Task 3: `runs` repository

**Files:**
- Create: `src/neuroarena/persistence/runs_repo.py`
- Test: `tests/persistence/test_runs_repo.py`

**Interfaces:**
- Consumes: `models_repo.create_model` (Task 2, to satisfy the `model_id` foreign key in tests).
- Produces: `RunRecord` (frozen dataclass: `run_id: str`, `model_id: str`, `track_id: str | None`, `started_at: str`, `ended_at: str | None`, `status: str`, `starting_generation: int`); `create_run(conn, *, model_id: str, track_id: str | None, starting_generation: int) -> RunRecord` (status always starts `"running"`); `update_run_status(conn, run_id: str, *, status: str, ended_at: str | None = None) -> None`; `get_run(conn, run_id: str) -> RunRecord` (raises `UnknownRunError`); `list_runs_for_model(conn, model_id: str) -> list[RunRecord]`. Consumed by Task 9 (recorder).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_runs_repo.py`:

```python
from pathlib import Path

import pytest

from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.runs_repo import (
    UnknownRunError,
    create_run,
    get_run,
    list_runs_for_model,
    update_run_status,
)


def _model_id(conn) -> str:  # type: ignore[no-untyped-def]
    return create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    ).model_id


def test_create_run_starts_as_running(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_run(conn, model_id=_model_id(conn), track_id="t1", starting_generation=0)
    assert record.status == "running"
    assert record.ended_at is None
    assert record.starting_generation == 0


def test_update_run_status_persists(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_run(conn, model_id=_model_id(conn), track_id="t1", starting_generation=0)
    update_run_status(conn, record.run_id, status="completed", ended_at="2026-09-13T00:00:00")
    reloaded = get_run(conn, record.run_id)
    assert reloaded.status == "completed"
    assert reloaded.ended_at == "2026-09-13T00:00:00"


def test_get_run_unknown_id_raises(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    with pytest.raises(UnknownRunError):
        get_run(conn, "does-not-exist")


def test_list_runs_for_model_returns_only_that_models_runs(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_a, model_b = _model_id(conn), _model_id(conn)
    create_run(conn, model_id=model_a, track_id="t1", starting_generation=0)
    create_run(conn, model_id=model_a, track_id="t1", starting_generation=10)
    create_run(conn, model_id=model_b, track_id="t1", starting_generation=0)
    assert len(list_runs_for_model(conn, model_a)) == 2
    assert len(list_runs_for_model(conn, model_b)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_runs_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.runs_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/neuroarena/persistence/runs_repo.py`:

```python
"""`runs` table: one training session (fresh start or resume) against a model, producing
one contiguous segment of `generation_stats`. See
`../../../docs/phases/phase-6-persistence.md`'s "Entity model" section — a model has many
runs. `status` values this codebase produces: `"running"` (set by `create_run`),
`"completed"` (the recorder's normal exit) and `"crashed"` (an unhandled exception mid-run).
`"stopped"` is reserved for a future manual-stop control surface (Phase 7) — nothing in
this plan sets it."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


class UnknownRunError(KeyError):
    """Raised when a run_id has no matching row."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    model_id: str
    track_id: str | None
    started_at: str
    ended_at: str | None
    status: str
    starting_generation: int


def create_run(
    conn: sqlite3.Connection, *, model_id: str, track_id: str | None, starting_generation: int
) -> RunRecord:
    record = RunRecord(
        run_id=uuid.uuid4().hex,
        model_id=model_id,
        track_id=track_id,
        started_at=datetime.now(UTC).isoformat(),
        ended_at=None,
        status="running",
        starting_generation=starting_generation,
    )
    conn.execute(
        "INSERT INTO runs (run_id, model_id, track_id, started_at, ended_at, status, starting_generation) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record.run_id,
            record.model_id,
            record.track_id,
            record.started_at,
            record.ended_at,
            record.status,
            record.starting_generation,
        ),
    )
    conn.commit()
    return record


def update_run_status(
    conn: sqlite3.Connection, run_id: str, *, status: str, ended_at: str | None = None
) -> None:
    conn.execute("UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?", (status, ended_at, run_id))
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> RunRecord:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise UnknownRunError(run_id)
    return _record_from_row(row)


def list_runs_for_model(conn: sqlite3.Connection, model_id: str) -> list[RunRecord]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE model_id = ? ORDER BY started_at", (model_id,)
    ).fetchall()
    return [_record_from_row(row) for row in rows]


def _record_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        model_id=row["model_id"],
        track_id=row["track_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        status=row["status"],
        starting_generation=row["starting_generation"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_runs_repo.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/runs_repo.py tests/persistence/test_runs_repo.py
git commit -m "feat(phase-6): runs repository"
```

---

## Task 4: `generation_stats` repository

**Files:**
- Create: `src/neuroarena/persistence/generation_stats_repo.py`
- Test: `tests/persistence/test_generation_stats_repo.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces.protocols.TrainingUpdate`, `models_repo.create_model`, `runs_repo.create_run` (test setup only).
- Produces: `GenerationStatRecord` (frozen dataclass mirroring `TrainingUpdate`'s fields plus `model_id`/`run_id`); `record_generation_stat(conn, *, model_id: str, run_id: str, update: TrainingUpdate) -> None`; `list_generation_stats_for_model(conn, model_id: str) -> list[GenerationStatRecord]` (ordered by `generation`). Consumed by Task 9 (recorder).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_generation_stats_repo.py`:

```python
from pathlib import Path

from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import (
    list_generation_stats_for_model,
    record_generation_stat,
)
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.runs_repo import create_run


def _setup(conn) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    model = create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    )
    run = create_run(conn, model_id=model.model_id, track_id="t1", starting_generation=0)
    return model.model_id, run.run_id


def test_record_and_list_round_trips(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    update = TrainingUpdate(
        progress_index=0,
        best_fitness=10.0,
        mean_fitness=5.0,
        worst_fitness=1.0,
        population_size=6,
        champion_metrics={"progress": 10.0, "crashed": 0.0},
        sim_time=100.0,
        wall_time=1.5,
    )
    record_generation_stat(conn, model_id=model_id, run_id=run_id, update=update)

    records = list_generation_stats_for_model(conn, model_id)
    assert len(records) == 1
    assert records[0].generation == 0
    assert records[0].best_fitness == 10.0
    assert records[0].champion_metrics == {"progress": 10.0, "crashed": 0.0}


def test_list_is_ordered_by_generation(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    for generation in (2, 0, 1):
        update = TrainingUpdate(
            progress_index=generation,
            best_fitness=1.0,
            mean_fitness=1.0,
            worst_fitness=1.0,
            population_size=1,
            champion_metrics={},
            sim_time=0.0,
            wall_time=0.0,
        )
        record_generation_stat(conn, model_id=model_id, run_id=run_id, update=update)
    assert [r.generation for r in list_generation_stats_for_model(conn, model_id)] == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_generation_stats_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.generation_stats_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/neuroarena/persistence/generation_stats_repo.py`:

```python
"""`generation_stats` table: one row per `TrainingUpdate` a run's `Trainer.run()` yields —
this *is* the run history (Phase 0's "run history & per-generation records" requirement),
enough to redraw a fitness curve and compare generations without reloading a checkpoint."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from neuroarena.interfaces.protocols import TrainingUpdate


@dataclass(frozen=True)
class GenerationStatRecord:
    model_id: str
    run_id: str
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    population_size: int
    champion_metrics: dict[str, float]
    sim_time: float
    wall_time: float


def record_generation_stat(
    conn: sqlite3.Connection, *, model_id: str, run_id: str, update: TrainingUpdate
) -> None:
    conn.execute(
        "INSERT INTO generation_stats "
        "(model_id, run_id, generation, best_fitness, mean_fitness, worst_fitness, "
        " population_size, champion_metrics, sim_time, wall_time) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            model_id,
            run_id,
            update.progress_index,
            update.best_fitness,
            update.mean_fitness,
            update.worst_fitness,
            update.population_size,
            json.dumps(update.champion_metrics),
            update.sim_time,
            update.wall_time,
        ),
    )
    conn.commit()


def list_generation_stats_for_model(conn: sqlite3.Connection, model_id: str) -> list[GenerationStatRecord]:
    rows = conn.execute(
        "SELECT * FROM generation_stats WHERE model_id = ? ORDER BY generation", (model_id,)
    ).fetchall()
    return [
        GenerationStatRecord(
            model_id=row["model_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            best_fitness=row["best_fitness"],
            mean_fitness=row["mean_fitness"],
            worst_fitness=row["worst_fitness"],
            population_size=row["population_size"],
            champion_metrics=json.loads(row["champion_metrics"]),
            sim_time=row["sim_time"],
            wall_time=row["wall_time"],
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_generation_stats_repo.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/generation_stats_repo.py tests/persistence/test_generation_stats_repo.py
git commit -m "feat(phase-6): generation_stats repository"
```

---

## Task 5: `checkpoints` repository

**Files:**
- Create: `src/neuroarena/persistence/checkpoints_repo.py`
- Test: `tests/persistence/test_checkpoints_repo.py`

**Interfaces:**
- Consumes: `models_repo.create_model`, `runs_repo.create_run` (test setup only).
- Produces: `CheckpointRecord` (frozen dataclass: `checkpoint_id: str`, `model_id: str`, `run_id: str`, `kind: str`, `generation: int`, `file_path: str`, `created_at: str`, `schema_version: int`); `record_checkpoint(conn, *, model_id: str, run_id: str, kind: str, generation: int, file_path: Path) -> CheckpointRecord` (`kind` is `"resume"` or `"champion"`; `schema_version` is this row format's own version — a module constant, unrelated to whatever schema-versioning the checkpoint *file's own contents* use, e.g. `NeatTrainer`'s internal pickle `schema_version`); `list_checkpoints(conn, model_id: str, kind: str | None = None) -> list[CheckpointRecord]` (ordered by `generation`); `latest_checkpoint(conn, model_id: str, kind: str) -> CheckpointRecord | None`; `delete_checkpoint(conn, checkpoint_id: str) -> None`. Consumed by Task 9 (recorder) and Task 10 (CLI, for `--resume`).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_checkpoints_repo.py`:

```python
from pathlib import Path

from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.checkpoints_repo import (
    delete_checkpoint,
    latest_checkpoint,
    list_checkpoints,
    record_checkpoint,
)
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.runs_repo import create_run


def _setup(conn) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    model = create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    )
    run = create_run(conn, model_id=model.model_id, track_id="t1", starting_generation=0)
    return model.model_id, run.run_id


def test_record_and_list_checkpoints(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    record_checkpoint(
        conn, model_id=model_id, run_id=run_id, kind="resume", generation=0, file_path=tmp_path / "gen_0.pkl"
    )
    record_checkpoint(
        conn, model_id=model_id, run_id=run_id, kind="resume", generation=10, file_path=tmp_path / "gen_10.pkl"
    )
    resumes = list_checkpoints(conn, model_id, kind="resume")
    assert [r.generation for r in resumes] == [0, 10]


def test_list_checkpoints_filters_by_kind(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    record_checkpoint(
        conn, model_id=model_id, run_id=run_id, kind="resume", generation=0, file_path=tmp_path / "r.pkl"
    )
    record_checkpoint(
        conn, model_id=model_id, run_id=run_id, kind="champion", generation=0, file_path=tmp_path / "c.pkl"
    )
    assert len(list_checkpoints(conn, model_id, kind="champion")) == 1
    assert len(list_checkpoints(conn, model_id)) == 2


def test_latest_checkpoint_returns_highest_generation(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    for generation in (0, 20, 10):
        record_checkpoint(
            conn, model_id=model_id, run_id=run_id, kind="resume", generation=generation,
            file_path=tmp_path / f"gen_{generation}.pkl",
        )
    latest = latest_checkpoint(conn, model_id, kind="resume")
    assert latest is not None
    assert latest.generation == 20


def test_latest_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, _ = _setup(conn)
    assert latest_checkpoint(conn, model_id, kind="resume") is None


def test_delete_checkpoint_removes_the_row(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    record = record_checkpoint(
        conn, model_id=model_id, run_id=run_id, kind="champion", generation=0, file_path=tmp_path / "c.pkl"
    )
    delete_checkpoint(conn, record.checkpoint_id)
    assert list_checkpoints(conn, model_id, kind="champion") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_checkpoints_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.checkpoints_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/neuroarena/persistence/checkpoints_repo.py`:

```python
"""`checkpoints` table: metadata pointing at an opaque backend-owned checkpoint file
(Phase 4's unmodified NEAT pickle format) — never parses the file. Both checkpoint kinds
(`"resume"`: full trainer-state, all kept; `"champion"`: one genome per generation,
retention-capped) share this one table, distinguished by `kind`. See
`../../../docs/phases/phase-6-persistence.md`'s "Checkpoint format" and "Checkpoint
cadence and retention" sections."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_ROW_SCHEMA_VERSION = 1
"""This row format's own version — independent of whatever schema_version a checkpoint
FILE's own contents carry internally (e.g. NeatTrainer's pickle payload)."""


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    model_id: str
    run_id: str
    kind: str
    generation: int
    file_path: str
    created_at: str
    schema_version: int


def record_checkpoint(
    conn: sqlite3.Connection, *, model_id: str, run_id: str, kind: str, generation: int, file_path: Path
) -> CheckpointRecord:
    record = CheckpointRecord(
        checkpoint_id=uuid.uuid4().hex,
        model_id=model_id,
        run_id=run_id,
        kind=kind,
        generation=generation,
        file_path=str(file_path),
        created_at=datetime.now(UTC).isoformat(),
        schema_version=_ROW_SCHEMA_VERSION,
    )
    conn.execute(
        "INSERT INTO checkpoints "
        "(checkpoint_id, model_id, run_id, kind, generation, file_path, created_at, schema_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.checkpoint_id,
            record.model_id,
            record.run_id,
            record.kind,
            record.generation,
            record.file_path,
            record.created_at,
            record.schema_version,
        ),
    )
    conn.commit()
    return record


def list_checkpoints(conn: sqlite3.Connection, model_id: str, kind: str | None = None) -> list[CheckpointRecord]:
    if kind is None:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE model_id = ? ORDER BY generation", (model_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE model_id = ? AND kind = ? ORDER BY generation",
            (model_id, kind),
        ).fetchall()
    return [_record_from_row(row) for row in rows]


def latest_checkpoint(conn: sqlite3.Connection, model_id: str, kind: str) -> CheckpointRecord | None:
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE model_id = ? AND kind = ? ORDER BY generation DESC LIMIT 1",
        (model_id, kind),
    ).fetchone()
    return None if row is None else _record_from_row(row)


def delete_checkpoint(conn: sqlite3.Connection, checkpoint_id: str) -> None:
    conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
    conn.commit()


def _record_from_row(row: sqlite3.Row) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=row["checkpoint_id"],
        model_id=row["model_id"],
        run_id=row["run_id"],
        kind=row["kind"],
        generation=row["generation"],
        file_path=row["file_path"],
        created_at=row["created_at"],
        schema_version=row["schema_version"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_checkpoints_repo.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/checkpoints_repo.py tests/persistence/test_checkpoints_repo.py
git commit -m "feat(phase-6): checkpoints repository"
```

---

## Task 6: New `RunConfig` knobs (`checkpoint_every_n_generations`, `champion_retention_cap`)

**Files:**
- Modify: `src/neuroarena/config/run_config.py`
- Modify: `tests/config/test_run_config.py`

**Interfaces:**
- Produces: `RunConfig.checkpoint_every_n_generations: int` (default `10`), `RunConfig.champion_retention_cap: int | None` (default `None`). Both plain scalars — no `config/serialization.py` change needed (mirrors Task 1 of the Phase 5 plan: JSON already round-trips `int`/`None` generically). Consumed by Task 9 (recorder) and Task 10 (CLI).

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_run_config.py`:

```python
def test_checkpoint_every_n_generations_default() -> None:
    assert RunConfig().checkpoint_every_n_generations == 10


def test_champion_retention_cap_defaults_to_none() -> None:
    assert RunConfig().champion_retention_cap is None


def test_champion_retention_cap_is_settable() -> None:
    assert RunConfig(champion_retention_cap=20).champion_retention_cap == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_run_config.py -v`
Expected: FAIL — `TypeError: RunConfig.__init__() got an unexpected keyword argument 'champion_retention_cap'`.

- [ ] **Step 3: Add the fields**

In `src/neuroarena/config/run_config.py`, add two fields to the dataclass (after `target_fitness`, before `sensor_config`):

```python
    max_generations: int | None = None
    target_fitness: float | None = None
    checkpoint_every_n_generations: int = 10
    champion_retention_cap: int | None = None
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
```

Update the class docstring's final sentence to add:

```python
    `checkpoint_every_n_generations` and `champion_retention_cap` are Phase 6 knobs
    consumed by `neuroarena.persistence.recorder` — the first sets resume-checkpoint
    cadence (every save kept, never overwritten), the second optionally bounds how many
    champion checkpoints (Phase 4's per-generation capture) are retained."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_run_config.py tests/config/test_serialization.py -v`
Expected: PASS — serialization tests must still pass unchanged (both new fields are plain scalars JSON already round-trips).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/config/run_config.py tests/config/test_run_config.py
git commit -m "feat(phase-6): add checkpoint_every_n_generations and champion_retention_cap RunConfig knobs"
```

---

## Task 7: `settings_history` repository + diff/reconstruction

**Files:**
- Create: `src/neuroarena/persistence/settings_history_repo.py`
- Test: `tests/persistence/test_settings_history_repo.py`

**Interfaces:**
- Consumes: `neuroarena.config.RunConfig`, `neuroarena.config.serialization.loads`, `models_repo.create_model`/`runs_repo.create_run` (test setup only).
- Produces: `SettingsHistoryEntry` (frozen dataclass: `model_id: str`, `run_id: str`, `generation: int`, `changed_at: str`, `diff: dict[str, Any]`); `compute_diff(previous: RunConfig | None, new: RunConfig) -> dict[str, Any]` (pure — `previous=None` returns every field, i.e. the first entry for a model is always a full-field diff, per the phase doc's granularity decision); `record_settings_entry(conn, *, model_id: str, run_id: str, generation: int, diff: dict[str, Any]) -> SettingsHistoryEntry`; `list_settings_history_for_model(conn, model_id: str) -> list[SettingsHistoryEntry]` (ordered by `generation`); `reconstruct_run_config(conn, model_id: str) -> RunConfig` (replays every entry's diff forward, oldest first, merging into an accumulator; a model with zero entries reconstructs to `RunConfig()` defaults). Consumed by Task 10 (CLI, to compute the resume-time diff).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_settings_history_repo.py`:

```python
from pathlib import Path

from neuroarena.config import RunConfig
from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.runs_repo import create_run
from neuroarena.persistence.settings_history_repo import (
    compute_diff,
    list_settings_history_for_model,
    record_settings_entry,
    reconstruct_run_config,
)


def _setup(conn) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    model = create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    )
    run = create_run(conn, model_id=model.model_id, track_id="t1", starting_generation=0)
    return model.model_id, run.run_id


def test_compute_diff_with_no_previous_returns_every_field() -> None:
    diff = compute_diff(None, RunConfig(population_size=200))
    assert diff["population_size"] == 200
    assert "master_seed" in diff  # every field present, not just the one that "changed"


def test_compute_diff_with_previous_returns_only_changed_fields() -> None:
    previous = RunConfig(population_size=150, max_episode_steps=3000)
    new = RunConfig(population_size=200, max_episode_steps=3000)
    diff = compute_diff(previous, new)
    assert diff == {"population_size": 200}


def test_record_and_list_settings_history(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    diff = compute_diff(None, RunConfig(population_size=200))
    record_settings_entry(conn, model_id=model_id, run_id=run_id, generation=0, diff=diff)

    entries = list_settings_history_for_model(conn, model_id)
    assert len(entries) == 1
    assert entries[0].diff["population_size"] == 200


def test_reconstruct_run_config_with_no_history_returns_defaults(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, _ = _setup(conn)
    assert reconstruct_run_config(conn, model_id) == RunConfig()


def test_reconstruct_run_config_replays_diffs_forward(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    initial = RunConfig(population_size=150)
    record_settings_entry(conn, model_id=model_id, run_id=run_id, generation=0, diff=compute_diff(None, initial))

    changed = RunConfig(population_size=200)
    record_settings_entry(
        conn, model_id=model_id, run_id=run_id, generation=10, diff=compute_diff(initial, changed)
    )

    reconstructed = reconstruct_run_config(conn, model_id)
    assert reconstructed.population_size == 200
    assert reconstructed.max_episode_steps == RunConfig().max_episode_steps  # untouched field survives
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_settings_history_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.settings_history_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/neuroarena/persistence/settings_history_repo.py`:

```python
"""`settings_history` table: the ordered log of every `RunConfig` change over a model's
training lifetime (Phase 0/OVERVIEW requirement). Diff-only granularity — each entry
stores only the fields that changed relative to the model's previous effective settings,
except the first entry for a model, which has nothing to diff against and so stores every
field. `reconstruct_run_config` replays entries forward to answer "what were this model's
settings most recently" — see `../../../docs/phases/phase-6-persistence.md`'s
"Settings-history location and granularity" decision."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from neuroarena.config import RunConfig
from neuroarena.config.serialization import loads as config_loads


@dataclass(frozen=True)
class SettingsHistoryEntry:
    model_id: str
    run_id: str
    generation: int
    changed_at: str
    diff: dict[str, Any]


def compute_diff(previous: RunConfig | None, new: RunConfig) -> dict[str, Any]:
    new_fields = dataclasses.asdict(new)
    if previous is None:
        return new_fields
    previous_fields = dataclasses.asdict(previous)
    return {key: value for key, value in new_fields.items() if previous_fields.get(key) != value}


def record_settings_entry(
    conn: sqlite3.Connection, *, model_id: str, run_id: str, generation: int, diff: dict[str, Any]
) -> SettingsHistoryEntry:
    entry = SettingsHistoryEntry(
        model_id=model_id, run_id=run_id, generation=generation,
        changed_at=datetime.now(UTC).isoformat(), diff=diff,
    )
    conn.execute(
        "INSERT INTO settings_history (model_id, run_id, generation, changed_at, diff) VALUES (?, ?, ?, ?, ?)",
        (entry.model_id, entry.run_id, entry.generation, entry.changed_at, json.dumps(entry.diff)),
    )
    conn.commit()
    return entry


def list_settings_history_for_model(conn: sqlite3.Connection, model_id: str) -> list[SettingsHistoryEntry]:
    rows = conn.execute(
        "SELECT * FROM settings_history WHERE model_id = ? ORDER BY generation", (model_id,)
    ).fetchall()
    return [
        SettingsHistoryEntry(
            model_id=row["model_id"], run_id=row["run_id"], generation=row["generation"],
            changed_at=row["changed_at"], diff=json.loads(row["diff"]),
        )
        for row in rows
    ]


def reconstruct_run_config(conn: sqlite3.Connection, model_id: str) -> RunConfig:
    merged: dict[str, Any] = {}
    for entry in list_settings_history_for_model(conn, model_id):
        merged.update(entry.diff)
    if not merged:
        return RunConfig()
    return config_loads(json.dumps(merged))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_settings_history_repo.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/settings_history_repo.py tests/persistence/test_settings_history_repo.py
git commit -m "feat(phase-6): settings_history repository with diff/reconstruction"
```

---

## Task 8: `ProgressObjective`

**Files:**
- Create: `src/neuroarena/sim/objectives.py`
- Test: `tests/sim/test_objectives.py`

**Interfaces:**
- Consumes: nothing new — reads `info["progress"]`/`info["lap_progress"]`, both already emitted by `CarEnvironment.step()` (Phase 3, `sim/car_env.py`).
- Produces: `ProgressObjective`, satisfying `neuroarena.interfaces.protocols.Objective`: `fitness()` = cumulative `progress` seen across `update()` calls since the last `reset()`; `step_reward()` = the most recent step's `progress` delta; `should_stop()` = `lap_progress >= 1.0`. Consumed by Task 10 (CLI).

- [ ] **Step 1: Write the failing test**

Create `tests/sim/test_objectives.py`:

```python
import numpy as np

from neuroarena.sim.objectives import ProgressObjective


def _obs() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


def _act() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_fitness_accumulates_progress_deltas() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 5.0, "lap_progress": 0.1})
    objective.update(_obs(), _act(), False, False, {"progress": 8.0, "lap_progress": 0.16})
    assert objective.fitness() == 8.0  # cumulative = final progress, since it started at 0


def test_step_reward_is_the_latest_delta() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 5.0, "lap_progress": 0.1})
    objective.update(_obs(), _act(), False, False, {"progress": 8.0, "lap_progress": 0.16})
    assert objective.step_reward() == 3.0


def test_should_stop_true_once_lap_progress_reaches_one() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 100.0, "lap_progress": 0.99})
    assert objective.should_stop() is False
    objective.update(_obs(), _act(), False, False, {"progress": 102.0, "lap_progress": 1.0})
    assert objective.should_stop() is True


def test_reset_clears_accumulated_state() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 50.0, "lap_progress": 0.5})
    objective.reset()
    assert objective.fitness() == 0.0
    assert objective.should_stop() is False
    objective.update(_obs(), _act(), False, False, {"progress": 3.0, "lap_progress": 0.03})
    assert objective.step_reward() == 3.0  # delta from the post-reset baseline (0), not the old one
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sim/test_objectives.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.sim.objectives'`.

- [ ] **Step 3: Write the objective**

Create `src/neuroarena/sim/objectives.py`:

```python
"""Concrete `Objective` implementations for the car game. Car-specific (reads
`CarEnvironment`'s `info` keys directly), unlike the game-agnostic `Objective` Protocol
(Phase 0) — lives in `neuroarena.sim`, not `neuroarena.interfaces`, matching where
`CarEnvironment` itself lives. See
`../../../docs/phases/phase-6-persistence.md`'s "Minimal concrete Objective" section for
why this exists: `neuroarena-train` needs a real `Objective` to build a `Trainer`, and
nothing else in the codebase had built one yet."""

from __future__ import annotations

from typing import Any

import numpy as np


class ProgressObjective:
    """Satisfies `Objective`. Fitness/reward come from `CarEnvironment`'s `progress` info
    field (signed, continuous distance along the track centerline); an episode is flagged
    done via `should_stop()` once `lap_progress` (progress / one lap) reaches 1.0 — this
    is the "one-lap completion" example the Phase 5 doc's pluggable-success-criteria bullet
    already named as composing for free from `lap_progress`, with no new env mechanism."""

    def __init__(self) -> None:
        self._last_progress = 0.0
        self._last_delta = 0.0
        self._cumulative = 0.0
        self._lap_progress = 0.0

    def reset(self) -> None:
        self._last_progress = 0.0
        self._last_delta = 0.0
        self._cumulative = 0.0
        self._lap_progress = 0.0

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        progress = float(info["progress"])
        self._last_delta = progress - self._last_progress
        self._last_progress = progress
        self._cumulative += self._last_delta
        self._lap_progress = float(info["lap_progress"])

    def should_stop(self) -> bool:
        return self._lap_progress >= 1.0

    def step_reward(self) -> float:
        return self._last_delta

    def fitness(self) -> float:
        return self._cumulative
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sim/test_objectives.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/sim/objectives.py tests/sim/test_objectives.py
git commit -m "feat(phase-6): add ProgressObjective, the first concrete Objective in this codebase"
```

---

## Task 9: Persistence recorder

**Files:**
- Create: `src/neuroarena/persistence/recorder.py`
- Test: `tests/persistence/test_recorder.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces.protocols.{Trainer, TrainingUpdate}`, `generation_stats_repo.record_generation_stat`, `checkpoints_repo.{record_checkpoint, list_checkpoints, delete_checkpoint}`, `runs_repo.{create_run, update_run_status}`, `settings_history_repo.record_settings_entry` (writes the one up-front `settings_history` row described below — the diff itself is computed by the caller, Task 10, and passed in as `initial_settings_diff`).
- Produces: `run_and_record(conn, trainer: Trainer, *, model_id: str, track_id: str | None, starting_generation: int, resume_dir: Path, champion_dir: Path | None, checkpoint_every_n_generations: int, champion_retention_cap: int | None, initial_settings_diff: dict[str, Any]) -> RunRecord`. Creates the `runs` row itself (status starts `"running"`, moves to `"completed"` on normal exhaustion of `trainer.run()` or `"crashed"` if it raises), writes one `settings_history` entry up front, one `generation_stats` row per `TrainingUpdate`, a `"resume"`-kind checkpoint (via `trainer.save_checkpoint`) every `checkpoint_every_n_generations` generations (every save kept), and — if `champion_dir` is not `None` — registers that generation's champion file (`champion_dir/gen_<generation:05d>.pkl`, `NeatTrainer`'s existing unconditional per-generation write) as a `"champion"`-kind checkpoint row, then prunes the oldest champion rows/files beyond `champion_retention_cap` if it is set. Consumed by Task 10 (CLI).

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_recorder.py`. This uses a small in-repo fake `Trainer` (not `NeatTrainer` — the recorder is backend-agnostic, and a fake keeps the test fast and focused on the recorder's own bookkeeping) that yields a fixed sequence of `TrainingUpdate`s and writes its own champion files, mimicking exactly what `NeatTrainer` does:

```python
from pathlib import Path
from typing import Iterator

import pytest

from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.checkpoints_repo import list_checkpoints
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.recorder import run_and_record
from neuroarena.persistence.runs_repo import get_run


class _FakeTrainer:
    """Yields `n_generations` updates; if `champion_dir` was given, writes one champion
    file per generation, mirroring NeatTrainer's `champion_dir/gen_<N:05d>.pkl` convention."""

    def __init__(self, n_generations: int, champion_dir: Path | None = None) -> None:
        self._n = n_generations
        self._champion_dir = champion_dir
        self.save_checkpoint_calls: list[Path] = []

    def run(self) -> Iterator[TrainingUpdate]:
        for generation in range(self._n):
            if self._champion_dir is not None:
                self._champion_dir.mkdir(parents=True, exist_ok=True)
                (self._champion_dir / f"gen_{generation:05d}.pkl").write_bytes(b"champion")
            yield TrainingUpdate(
                progress_index=generation, best_fitness=float(generation), mean_fitness=float(generation),
                worst_fitness=float(generation), population_size=6, champion_metrics={}, sim_time=0.0,
                wall_time=0.0,
            )

    def save_checkpoint(self, path: Path) -> None:
        self.save_checkpoint_calls.append(path)
        path.write_bytes(b"resume")


class _CrashingTrainer:
    def run(self) -> Iterator[TrainingUpdate]:
        yield TrainingUpdate(
            progress_index=0, best_fitness=1.0, mean_fitness=1.0, worst_fitness=1.0,
            population_size=1, champion_metrics={}, sim_time=0.0, wall_time=0.0,
        )
        raise RuntimeError("boom")

    def save_checkpoint(self, path: Path) -> None:
        pass


def _model_id(conn) -> str:  # type: ignore[no-untyped-def]
    return create_model(
        conn, backend="neat", observation_space=Box(-1.0, 1.0, (10,)), action_space=Box(-1.0, 1.0, (2,))
    ).model_id


def test_records_one_generation_stat_row_per_update(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=5)
    run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=None, checkpoint_every_n_generations=100,
        champion_retention_cap=None, initial_settings_diff={"population_size": 6},
    )
    assert len(list_generation_stats_for_model(conn, model_id)) == 5


def test_marks_run_completed_on_normal_exit(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=2)
    record = run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=None, checkpoint_every_n_generations=100,
        champion_retention_cap=None, initial_settings_diff={},
    )
    assert get_run(conn, record.run_id).status == "completed"


def test_marks_run_crashed_on_exception(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _CrashingTrainer()
    with pytest.raises(RuntimeError):
        run_and_record(
            conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
            resume_dir=tmp_path / "resume", champion_dir=None, checkpoint_every_n_generations=100,
            champion_retention_cap=None, initial_settings_diff={},
        )
    runs = list_checkpoints(conn, model_id)  # any run row exists at all
    assert runs == []  # no checkpoints were due, just confirming no crash-time exception leaked here
    from neuroarena.persistence.runs_repo import list_runs_for_model

    run_rows = list_runs_for_model(conn, model_id)
    assert len(run_rows) == 1
    assert run_rows[0].status == "crashed"


def test_saves_a_resume_checkpoint_every_n_generations_and_keeps_all(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=7)
    run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=None, checkpoint_every_n_generations=3,
        champion_retention_cap=None, initial_settings_diff={},
    )
    resumes = list_checkpoints(conn, model_id, kind="resume")
    assert [r.generation for r in resumes] == [0, 3, 6]  # every Nth kept, none overwritten
    for record in resumes:
        assert Path(record.file_path).is_file()


def test_registers_and_retains_champion_checkpoints_by_default(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    champion_dir = tmp_path / "champion"
    trainer = _FakeTrainer(n_generations=5, champion_dir=champion_dir)
    run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=champion_dir, checkpoint_every_n_generations=100,
        champion_retention_cap=None, initial_settings_diff={},
    )
    champions = list_checkpoints(conn, model_id, kind="champion")
    assert len(champions) == 5  # unbounded by default


def test_prunes_champion_checkpoints_beyond_the_retention_cap(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    champion_dir = tmp_path / "champion"
    trainer = _FakeTrainer(n_generations=5, champion_dir=champion_dir)
    run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=champion_dir, checkpoint_every_n_generations=100,
        champion_retention_cap=2, initial_settings_diff={},
    )
    champions = list_checkpoints(conn, model_id, kind="champion")
    assert [r.generation for r in champions] == [3, 4]  # only the newest 2 survive
    assert not (champion_dir / "gen_00000.pkl").is_file()  # pruned files are actually deleted
    assert (champion_dir / "gen_00004.pkl").is_file()


def test_records_the_initial_settings_diff_at_starting_generation(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=1)
    record = run_and_record(
        conn, trainer, model_id=model_id, track_id="t1", starting_generation=0,
        resume_dir=tmp_path / "resume", champion_dir=None, checkpoint_every_n_generations=100,
        champion_retention_cap=None, initial_settings_diff={"population_size": 42},
    )
    from neuroarena.persistence.settings_history_repo import list_settings_history_for_model

    entries = list_settings_history_for_model(conn, model_id)
    assert len(entries) == 1
    assert entries[0].run_id == record.run_id
    assert entries[0].diff == {"population_size": 42}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_recorder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.recorder'`.

- [ ] **Step 3: Write the recorder**

Create `src/neuroarena/persistence/recorder.py`:

```python
"""Wraps any `Trainer.run()` iterator to persist its progress. Backend-agnostic — takes
only the `Trainer` Protocol (Phase 0), never imports `NeatTrainer` — and independently
reconstructs the champion-file naming convention `NeatTrainer` already uses
(`champion_dir/gen_<generation:05d>.pkl`) rather than reading any private attribute, so it
works unchanged for Phase 10's future deep-RL backend (with `champion_dir=None`, since
nothing has said deep RL captures per-step champions the same way). See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" and "Checkpoint
cadence and retention" sections."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuroarena.persistence import checkpoints_repo, generation_stats_repo, runs_repo, settings_history_repo
from neuroarena.persistence.runs_repo import RunRecord

if TYPE_CHECKING:
    from neuroarena.interfaces.protocols import Trainer


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
) -> RunRecord:
    resume_dir.mkdir(parents=True, exist_ok=True)
    run_record = runs_repo.create_run(
        conn, model_id=model_id, track_id=track_id, starting_generation=starting_generation
    )
    settings_history_repo.record_settings_entry(
        conn, model_id=model_id, run_id=run_record.run_id, generation=starting_generation,
        diff=initial_settings_diff,
    )
    try:
        for update in trainer.run():
            generation_stats_repo.record_generation_stat(
                conn, model_id=model_id, run_id=run_record.run_id, update=update
            )
            generation = update.progress_index

            if generation % checkpoint_every_n_generations == 0:
                path = resume_dir / f"gen_{generation:05d}.pkl"
                trainer.save_checkpoint(path)
                checkpoints_repo.record_checkpoint(
                    conn, model_id=model_id, run_id=run_record.run_id, kind="resume",
                    generation=generation, file_path=path,
                )

            if champion_dir is not None:
                champion_path = champion_dir / f"gen_{generation:05d}.pkl"
                checkpoints_repo.record_checkpoint(
                    conn, model_id=model_id, run_id=run_record.run_id, kind="champion",
                    generation=generation, file_path=champion_path,
                )
                if champion_retention_cap is not None:
                    _prune_champions(conn, model_id, champion_retention_cap)
    except Exception:
        runs_repo.update_run_status(
            conn, run_record.run_id, status="crashed", ended_at=datetime.now(UTC).isoformat()
        )
        raise

    runs_repo.update_run_status(
        conn, run_record.run_id, status="completed", ended_at=datetime.now(UTC).isoformat()
    )
    return run_record


def _prune_champions(conn: sqlite3.Connection, model_id: str, cap: int) -> None:
    records = checkpoints_repo.list_checkpoints(conn, model_id, kind="champion")
    excess_count = len(records) - cap
    if excess_count <= 0:
        return
    for record in records[:excess_count]:  # oldest first, since list_checkpoints orders by generation
        Path(record.file_path).unlink(missing_ok=True)
        checkpoints_repo.delete_checkpoint(conn, record.checkpoint_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_recorder.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/recorder.py tests/persistence/test_recorder.py
git commit -m "feat(phase-6): persistence recorder (generation stats, checkpoint cadence, champion retention)"
```

---

## Task 10: `neuroarena-train` CLI

**Files:**
- Create: `src/neuroarena/persistence/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/persistence/test_cli.py`

**Interfaces:**
- Consumes: `neuroarena.tracks.store.load`, `neuroarena.sim.car_env.{CarEnvironment, CarEnvironmentConfig}`, `neuroarena.sim.objectives.ProgressObjective`, `neuroarena.backends.neat.trainer.NeatTrainer`, `neuroarena.config.RunConfig`, every `persistence.*_repo` module, `persistence.recorder.run_and_record`.
- Produces: `run(*, track_id: str, population_size: int = 150, max_generations: int | None = None, target_fitness: float | None = None, checkpoint_every_n_generations: int = 10, champion_retention_cap: int | None = None, resume_model_id: str | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> RunRecord` and `main() -> None` (argparse wrapper), mirroring `tracks/cli.py`'s `run()`/`main()` split. `[project.scripts]` gains `neuroarena-train = "neuroarena.persistence.cli:main"`.

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_cli.py`. Uses a tiny hand-built track (same 10-cell rounded-rectangle fixture the Phase 5 end-to-end test uses) saved via `tracks.store.save` so `cli.run()` can resolve it by id, and small population/generation counts to keep the test fast:

```python
from pathlib import Path

from neuroarena.persistence.checkpoints_repo import list_checkpoints
from neuroarena.persistence.cli import run
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model
from neuroarena.persistence.models_repo import get_model
from neuroarena.persistence.runs_repo import get_run
from neuroarena.persistence.settings_history_repo import list_settings_history_for_model
from neuroarena.sim.track import Facing, GridCell, Track, TileKind
from neuroarena.tracks.store import save


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE, (1, 0): K.STRAIGHT_EW, (2, 0): K.STRAIGHT_EW, (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS, (3, 2): K.CURVE_SW, (2, 2): K.STRAIGHT_EW, (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE, (0, 1): K.STRAIGHT_NS,
    }


def _save_test_track(data_dir: Path) -> str:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    record = save(track, requested_size=10, complexity=0.3, seed=1, tracks_dir=data_dir / "tracks")
    return record.track_id


def test_fresh_run_creates_model_run_and_stats(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    result = run(
        track_id=track_id, population_size=6, max_generations=2, checkpoint_every_n_generations=1,
        data_dir=tmp_path,
    )

    conn = connect(tmp_path / "neuroarena.db")
    model = get_model(conn, result.model_id)
    assert model.backend == "neat"
    run_record = get_run(conn, result.run_id)
    assert run_record.status == "completed"
    assert len(list_generation_stats_for_model(conn, result.model_id)) == 2
    assert len(list_checkpoints(conn, result.model_id, kind="resume")) == 2
    assert len(list_checkpoints(conn, result.model_id, kind="champion")) == 2
    assert len(list_settings_history_for_model(conn, result.model_id)) == 1


def test_resume_continues_generation_numbering_and_records_a_diff(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    first = run(
        track_id=track_id, population_size=6, max_generations=2, checkpoint_every_n_generations=1,
        data_dir=tmp_path,
    )

    second = run(
        track_id=track_id, population_size=6, max_generations=4, target_fitness=1e9,
        checkpoint_every_n_generations=1, resume_model_id=first.model_id, data_dir=tmp_path,
    )

    assert second.model_id == first.model_id
    assert second.run_id != first.run_id

    conn = connect(tmp_path / "neuroarena.db")
    stats = list_generation_stats_for_model(conn, first.model_id)
    # generation numbering is cumulative per model, not reset per run
    assert [s.generation for s in stats] == [0, 1, 2, 3]

    entries = list_settings_history_for_model(conn, first.model_id)
    assert len(entries) == 2
    assert entries[1].diff == {"max_generations": 4, "target_fitness": 1e9}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/persistence/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.persistence.cli'`.

- [ ] **Step 3: Write the CLI**

Create `src/neuroarena/persistence/cli.py`:

```python
"""Entry point: `uv run neuroarena-train`. The first working training-launch entrypoint in
this codebase — closes the "no phase has named a training-launch entrypoint yet" gap
Phase 5's implementation plan explicitly left open, the same way `tracks/cli.py` closed
the equivalent gap for track generation (Phase 2). Resolves `track_id` into a `Track` +
`make_env`, builds a `ProgressObjective`/`NeatTrainer`, and drives `run()` through
`persistence.recorder.run_and_record`. See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" section."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.persistence import checkpoints_repo, models_repo, recorder, settings_history_repo
from neuroarena.persistence.db import connect
from neuroarena.persistence.runs_repo import RunRecord
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.objectives import ProgressObjective
from neuroarena.tracks.store import load as load_track

DEFAULT_DATA_DIR = Path("data")


def run(
    *,
    track_id: str,
    population_size: int = 150,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int = 10,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> RunRecord:
    conn = connect(data_dir / "neuroarena.db")
    track_record = load_track(track_id, tracks_dir=data_dir / "tracks")
    config = RunConfig(
        population_size=population_size,
        track_id=track_id,
        max_generations=max_generations,
        target_fitness=target_fitness,
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
    )

    def make_env() -> CarEnvironment:
        return CarEnvironment(track_record.track, track_id, CarEnvironmentConfig.from_run_config(config))

    objective = ProgressObjective()
    checkpoint_root = data_dir / "checkpoints"

    if resume_model_id is not None:
        model = models_repo.get_model(conn, resume_model_id)
        latest = checkpoints_repo.latest_checkpoint(conn, resume_model_id, kind="resume")
        if latest is None:
            raise ValueError(f"model {resume_model_id!r} has no resume checkpoint to resume from")
        trainer = NeatTrainer.load_checkpoint(Path(latest.file_path), make_env, objective, config)
        starting_generation = latest.generation
        previous_config = settings_history_repo.reconstruct_run_config(conn, resume_model_id)
        diff = settings_history_repo.compute_diff(previous_config, config)
    else:
        env = make_env()
        model = models_repo.create_model(
            conn, backend="neat", observation_space=env.observation_space, action_space=env.action_space
        )
        model_dir = checkpoint_root / model.model_id
        trainer = NeatTrainer(make_env, objective, config, champion_dir=model_dir / "champion")
        starting_generation = 0
        diff = settings_history_repo.compute_diff(None, config)

    model_dir = checkpoint_root / model.model_id
    return recorder.run_and_record(
        conn,
        trainer,
        model_id=model.model_id,
        track_id=track_id,
        starting_generation=starting_generation,
        resume_dir=model_dir / "resume",
        champion_dir=model_dir / "champion",
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
        initial_settings_diff=diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a NEAT training run against a saved track.")
    parser.add_argument("--track-id", required=True, help="a track_id from `neuroarena-track-gen`")
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--target-fitness", type=float, default=None)
    parser.add_argument("--checkpoint-every-n-generations", type=int, default=10)
    parser.add_argument("--champion-retention-cap", type=int, default=None)
    parser.add_argument("--resume", dest="resume_model_id", default=None, help="a model_id to resume")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    record = run(
        track_id=args.track_id,
        population_size=args.population_size,
        max_generations=args.max_generations,
        target_fitness=args.target_fitness,
        checkpoint_every_n_generations=args.checkpoint_every_n_generations,
        champion_retention_cap=args.champion_retention_cap,
        resume_model_id=args.resume_model_id,
        data_dir=args.data_dir,
    )
    print(f"model_id={record.model_id} run_id={record.run_id} status={record.status}")


if __name__ == "__main__":
    main()
```

In `pyproject.toml`, add to `[project.scripts]`:

```toml
[project.scripts]
neuroarena-play = "neuroarena.render.play:main"
neuroarena-track-gen = "neuroarena.tracks.cli:main"
neuroarena-train = "neuroarena.persistence.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/persistence/test_cli.py -v -s`
Expected: PASS (2 tests), completing in well under a minute (population 6, 2-4 generations, 10-cell track).

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/persistence/cli.py pyproject.toml tests/persistence/test_cli.py
git commit -m "feat(phase-6): neuroarena-train CLI (fresh start + resume)"
```

---

## Task 11: Whole-suite verification + import hygiene

**Files:**
- No new files. Verification only.

**Interfaces:** None — this task only runs the existing full suite plus the two guards every prior phase's final task has run.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS, every test from Phases 0-6.

- [ ] **Step 2: Lint and format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 3: Type-check**

Run: `uv run mypy src`
Expected: clean (`--strict`). `sqlite3` is a stdlib module with bundled type stubs — no `[[tool.mypy.overrides]]` entry should be needed; if one is, add it to `pyproject.toml` and note why in the commit message.

- [ ] **Step 4: Confirm the NEAT backend's import-hygiene guard still passes unmodified**

Run: `uv run pytest tests/backends/neat/test_import_hygiene.py -v`
Expected: PASS — this task touches nothing under `src/neuroarena/backends/`, so this guard (Phase 4) should need no changes at all; running it is a confirmation, not new work.

- [ ] **Step 5: Commit (only if any of the above required a fix)**

```bash
git add -A
git commit -m "fix(phase-6): whole-suite verification fixes"
```

If nothing needed fixing, skip this step — there is nothing to commit.

---

## Post-plan: update the phase doc

After all tasks pass, update `docs/phases/phase-6-persistence.md`:
1. Change the "Implementation plan" section from `_Not yet written._` to a link: `` Detailed, step-by-step plan: [`../superpowers/plans/2026-09-13-phase-6-persistence.md`](../superpowers/plans/2026-09-13-phase-6-persistence.md). `` plus a short summary of the 11 tasks (mirroring Phase 4/5's "N tasks, each ending in an independently testable, committed deliverable" list).
2. Add a Revision history entry recording: final test count, `ruff`/`mypy --strict` status, any implementation-level findings from task reviews (subagent-driven-development's per-task review, if that execution mode is chosen), and explicit confirmation that `NeatTrainer`/`build_neat_config`/the checkpoint pickle format were not modified anywhere in this plan.
3. Commit: `git commit -m "docs(phase-6): record implementation completion in the phase doc"`.

This mirrors the exact pattern Phase 4 and Phase 5 both followed (see their own Revision history's final entries) — do not skip it; a phase doc whose Implementation plan section still says "not yet written" after the code is merged is exactly the kind of drift `WORKFLOW.md`'s Session end checklist exists to prevent.
