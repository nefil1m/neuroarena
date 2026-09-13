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
    conn: sqlite3.Connection,
    *,
    model_id: str,
    run_id: str,
    kind: str,
    generation: int,
    file_path: Path,
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
        "INSERT INTO checkpoints\n"
        "(checkpoint_id, model_id, run_id, kind, generation, file_path, created_at,"
        " schema_version)\n"
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


def list_checkpoints(
    conn: sqlite3.Connection, model_id: str, kind: str | None = None
) -> list[CheckpointRecord]:
    # `created_at` breaks generation ties deterministically (this table's PK is an
    # unordered UUID, so there is no autoincrement `id` to fall back on): a resume from an
    # older-than-latest checkpoint replays generations that already have rows.
    if kind is None:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE model_id = ? ORDER BY generation, created_at",
            (model_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE model_id = ? AND kind = ?"
            " ORDER BY generation, created_at",
            (model_id, kind),
        ).fetchall()
    return [_record_from_row(row) for row in rows]


def latest_checkpoint(
    conn: sqlite3.Connection, model_id: str, kind: str
) -> CheckpointRecord | None:
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE model_id = ? AND kind = ?"
        " ORDER BY generation DESC, created_at DESC LIMIT 1",
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
