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
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
    )
    run = create_run(conn, model_id=model.model_id, track_id="t1", starting_generation=0)
    return model.model_id, run.run_id


def test_record_and_list_checkpoints(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    record_checkpoint(
        conn,
        model_id=model_id,
        run_id=run_id,
        kind="resume",
        generation=0,
        file_path=tmp_path / "gen_0.pkl",
    )
    record_checkpoint(
        conn,
        model_id=model_id,
        run_id=run_id,
        kind="resume",
        generation=10,
        file_path=tmp_path / "gen_10.pkl",
    )
    resumes = list_checkpoints(conn, model_id, kind="resume")
    assert [r.generation for r in resumes] == [0, 10]


def test_list_checkpoints_filters_by_kind(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    record_checkpoint(
        conn,
        model_id=model_id,
        run_id=run_id,
        kind="resume",
        generation=0,
        file_path=tmp_path / "r.pkl",
    )
    record_checkpoint(
        conn,
        model_id=model_id,
        run_id=run_id,
        kind="champion",
        generation=0,
        file_path=tmp_path / "c.pkl",
    )
    assert len(list_checkpoints(conn, model_id, kind="champion")) == 1
    assert len(list_checkpoints(conn, model_id)) == 2


def test_latest_checkpoint_returns_highest_generation(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id, run_id = _setup(conn)
    for generation in (0, 20, 10):
        record_checkpoint(
            conn,
            model_id=model_id,
            run_id=run_id,
            kind="resume",
            generation=generation,
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
        conn,
        model_id=model_id,
        run_id=run_id,
        kind="champion",
        generation=0,
        file_path=tmp_path / "c.pkl",
    )
    delete_checkpoint(conn, record.checkpoint_id)
    assert list_checkpoints(conn, model_id, kind="champion") == []
