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
        model_id=model_id,
        run_id=run_id,
        generation=generation,
        changed_at=datetime.now(UTC).isoformat(),
        diff=diff,
    )
    conn.execute(
        "INSERT INTO settings_history"
        " (model_id, run_id, generation, changed_at, diff) VALUES (?, ?, ?, ?, ?)",
        (entry.model_id, entry.run_id, entry.generation, entry.changed_at, json.dumps(entry.diff)),
    )
    conn.commit()
    return entry


def list_settings_history_for_model(
    conn: sqlite3.Connection, model_id: str
) -> list[SettingsHistoryEntry]:
    rows = conn.execute(
        "SELECT * FROM settings_history WHERE model_id = ? ORDER BY generation", (model_id,)
    ).fetchall()
    return [
        SettingsHistoryEntry(
            model_id=row["model_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            changed_at=row["changed_at"],
            diff=json.loads(row["diff"]),
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
