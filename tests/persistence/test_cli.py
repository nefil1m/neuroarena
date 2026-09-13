from pathlib import Path

from neuroarena.persistence.checkpoints_repo import list_checkpoints
from neuroarena.persistence.cli import run
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model
from neuroarena.persistence.models_repo import get_model
from neuroarena.persistence.runs_repo import get_run
from neuroarena.persistence.settings_history_repo import list_settings_history_for_model
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from neuroarena.tracks.store import save


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _save_test_track(data_dir: Path) -> str:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    record = save(track, requested_size=10, complexity=0.3, seed=1, tracks_dir=data_dir / "tracks")
    return record.track_id


def test_fresh_run_creates_model_run_and_stats(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    result = run(
        track_id=track_id,
        population_size=6,
        max_generations=2,
        checkpoint_every_n_generations=1,
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
        track_id=track_id,
        population_size=6,
        max_generations=2,
        checkpoint_every_n_generations=1,
        data_dir=tmp_path,
    )

    second = run(
        track_id=track_id,
        population_size=6,
        max_generations=4,
        target_fitness=1e9,
        checkpoint_every_n_generations=1,
        resume_model_id=first.model_id,
        data_dir=tmp_path,
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
