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
            {
                "kind": "Box",
                "low": space.low,
                "high": space.high,
                "shape": list(space.shape),
                "dtype": space.dtype,
            }
        )
    return json.dumps({"kind": "Discrete", "n": space.n})


def _space_from_json(text: str) -> Space:
    raw = json.loads(text)
    if raw["kind"] == "Box":
        return Box(low=raw["low"], high=raw["high"], shape=tuple(raw["shape"]), dtype=raw["dtype"])
    return Discrete(n=raw["n"])
