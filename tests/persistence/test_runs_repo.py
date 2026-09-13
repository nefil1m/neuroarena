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
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
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
