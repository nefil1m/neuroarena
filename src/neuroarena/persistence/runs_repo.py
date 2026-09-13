"""`runs` table: one training session (fresh start or resume) against a model, producing
one contiguous segment of `generation_stats`. See
`../../../docs/phases/phase-6-persistence.md`'s "Entity model" section — a model has many
runs. `status` values this codebase produces: `"running"` (set by `create_run`),
`"completed"` (the recorder's normal exit), `"crashed"` (an unhandled exception mid-run) and
`"stopped"` (the recorder's `KeyboardInterrupt` handler — Ctrl-C on a headless CLI run; a
future manual-stop control surface in Phase 7 sets the same value)."""

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
        "INSERT INTO runs ("
        "run_id, model_id, track_id, started_at, ended_at, status, starting_generation"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    conn.execute(
        "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?", (status, ended_at, run_id)
    )
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
