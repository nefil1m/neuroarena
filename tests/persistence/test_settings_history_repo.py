from pathlib import Path

from neuroarena.config import RunConfig
from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.runs_repo import create_run
from neuroarena.persistence.settings_history_repo import (
    compute_diff,
    list_settings_history_for_model,
    reconstruct_run_config,
    record_settings_entry,
)


def _setup(conn) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    model = create_model(
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
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
    record_settings_entry(
        conn, model_id=model_id, run_id=run_id, generation=0, diff=compute_diff(None, initial)
    )

    changed = RunConfig(population_size=200)
    record_settings_entry(
        conn, model_id=model_id, run_id=run_id, generation=10, diff=compute_diff(initial, changed)
    )

    reconstructed = reconstruct_run_config(conn, model_id)
    assert reconstructed.population_size == 200
    assert (
        reconstructed.max_episode_steps == RunConfig().max_episode_steps
    )  # untouched field survives
