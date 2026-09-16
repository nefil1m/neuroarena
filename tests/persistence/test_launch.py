from pathlib import Path

from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import prepare_run
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


def test_prepare_run_builds_a_fresh_model_and_working_trainer(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    conn = connect(tmp_path / "neuroarena.db")
    prepared = prepare_run(conn, track_id=track_id, population_size=6, data_dir=tmp_path)
    assert prepared.model.backend == "neat"
    assert prepared.starting_generation == 0
    assert prepared.config.population_size == 6
    update = next(prepared.trainer.run())
    assert update.population_size == 6


def test_prepare_run_resumes_from_an_existing_model(tmp_path: Path) -> None:
    from neuroarena.persistence.recorder import run_and_record

    track_id = _save_test_track(tmp_path)
    conn = connect(tmp_path / "neuroarena.db")
    first = prepare_run(
        conn, track_id=track_id, population_size=6, max_generations=2, data_dir=tmp_path
    )
    run_and_record(
        conn,
        first.trainer,
        model_id=first.model.model_id,
        track_id=track_id,
        starting_generation=first.starting_generation,
        resume_dir=first.model_dir / "resume",
        champion_dir=first.model_dir / "champion",
        checkpoint_every_n_generations=1,
        champion_retention_cap=None,
        initial_settings_diff=first.settings_diff,
    )

    resumed = prepare_run(
        conn, track_id=track_id, resume_model_id=first.model.model_id, data_dir=tmp_path
    )
    assert resumed.model.model_id == first.model.model_id
    assert resumed.starting_generation > 0
    assert resumed.config.population_size == 6  # inherited, not reset to RunConfig's default
