import sys
from pathlib import Path

import pytest

from neuroarena.persistence.checkpoints_repo import list_checkpoints
from neuroarena.persistence.cli import main, run
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model
from neuroarena.persistence.models_repo import get_model, list_models
from neuroarena.persistence.runs_repo import get_run, list_runs_for_model
from neuroarena.persistence.settings_history_repo import (
    list_settings_history_for_model,
    reconstruct_run_config,
)
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

    # `starting_generation` means the same thing on both paths: the first generation this
    # run's own segment actually recorded (a fresh run's 0 equals its first stat's 0).
    second_run = get_run(conn, second.run_id)
    second_segment = [s.generation for s in stats if s.run_id == second.run_id]
    assert second_run.starting_generation == second_segment[0]


def test_resume_without_flags_inherits_the_models_last_known_settings(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    first = run(
        track_id=track_id,
        population_size=6,
        max_generations=2,
        checkpoint_every_n_generations=1,
        data_dir=tmp_path,
    )

    # No population_size / checkpoint cadence re-specified: both must come from the model's
    # settings history, not from `run()`'s own defaults (150 / 10).
    second = run(
        track_id=track_id,
        max_generations=4,
        resume_model_id=first.model_id,
        data_dir=tmp_path,
    )

    conn = connect(tmp_path / "neuroarena.db")
    entries = list_settings_history_for_model(conn, first.model_id)
    assert entries[1].diff == {"max_generations": 4}  # population_size did NOT "change"
    assert reconstruct_run_config(conn, first.model_id).population_size == 6
    assert reconstruct_run_config(conn, first.model_id).checkpoint_every_n_generations == 1

    stats = list_generation_stats_for_model(conn, first.model_id)
    second_segment = [s for s in stats if s.run_id == second.run_id]
    assert [s.generation for s in second_segment] == [2, 3]
    assert {s.population_size for s in second_segment} == {6}  # still training 6 genomes
    # the inherited cadence of 1 (not the default 10) kept checkpointing every generation
    resumes = list_checkpoints(conn, first.model_id, kind="resume")
    assert [c.generation for c in resumes] == [0, 1, 2, 3]


def test_resume_rejects_a_changed_population_size(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    first = run(
        track_id=track_id,
        population_size=6,
        max_generations=2,
        checkpoint_every_n_generations=1,
        data_dir=tmp_path,
    )
    conn = connect(tmp_path / "neuroarena.db")
    runs_before = list_runs_for_model(conn, first.model_id)
    entries_before = list_settings_history_for_model(conn, first.model_id)

    with pytest.raises(ValueError, match="population_size cannot be changed on resume"):
        run(
            track_id=track_id,
            population_size=8,  # NeatTrainer.load_checkpoint cannot honour this
            max_generations=4,
            resume_model_id=first.model_id,
            data_dir=tmp_path,
        )

    # the rejected attempt wrote nothing — no half-started run, no false settings diff
    assert len(list_runs_for_model(conn, first.model_id)) == len(runs_before)
    assert len(list_settings_history_for_model(conn, first.model_id)) == len(entries_before)


def test_resume_accepts_a_repeated_population_size(tmp_path: Path) -> None:
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
        population_size=6,  # explicitly re-stating the current value is not a change
        max_generations=3,
        resume_model_id=first.model_id,
        data_dir=tmp_path,
    )
    assert second.status == "completed"


def test_rejects_a_checkpoint_cadence_below_one_before_writing_anything(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_every_n_generations must be >= 1"):
        run(
            track_id=track_id,
            population_size=6,
            max_generations=1,
            checkpoint_every_n_generations=0,  # would be a ZeroDivisionError mid-run
            data_dir=tmp_path,
        )
    conn = connect(tmp_path / "neuroarena.db")
    assert list_models(conn) == []  # no orphan model/run rows left behind


def test_main_reports_bad_input_as_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["neuroarena-train", "--track-id", "nope", "--data-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2  # argparse's usage-error exit, not a raw traceback
    assert "unknown track_id" in capsys.readouterr().err
