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
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
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
