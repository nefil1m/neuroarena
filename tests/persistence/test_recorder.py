from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.interfaces.spaces import Box
from neuroarena.persistence.checkpoints_repo import list_checkpoints, record_checkpoint
from neuroarena.persistence.db import connect
from neuroarena.persistence.generation_stats_repo import list_generation_stats_for_model
from neuroarena.persistence.models_repo import create_model
from neuroarena.persistence.recorder import run_and_record
from neuroarena.persistence.runs_repo import create_run, get_run


class _FakeTrainer:
    """Yields `n_generations` updates; if `champion_dir` was given, writes one champion
    file per generation, mirroring NeatTrainer's `champion_dir/gen_<N:05d>.pkl` convention."""

    def __init__(self, n_generations: int, champion_dir: Path | None = None) -> None:
        self._n = n_generations
        self._champion_dir = champion_dir
        self.save_checkpoint_calls: list[Path] = []

    def run(self) -> Iterator[TrainingUpdate]:
        for generation in range(self._n):
            if self._champion_dir is not None:
                self._champion_dir.mkdir(parents=True, exist_ok=True)
                (self._champion_dir / f"gen_{generation:05d}.pkl").write_bytes(b"champion")
            yield TrainingUpdate(
                progress_index=generation,
                best_fitness=float(generation),
                mean_fitness=float(generation),
                worst_fitness=float(generation),
                population_size=6,
                champion_metrics={},
                sim_time=0.0,
                wall_time=0.0,
            )

    def save_checkpoint(self, path: Path) -> None:
        self.save_checkpoint_calls.append(path)
        path.write_bytes(b"resume")

    def update_config(self, partial: Mapping[str, Any]) -> None:
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance

    @classmethod
    def load_checkpoint(cls, path, make_env, objective, config):
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance


class _CrashingTrainer:
    def run(self) -> Iterator[TrainingUpdate]:
        yield TrainingUpdate(
            progress_index=0,
            best_fitness=1.0,
            mean_fitness=1.0,
            worst_fitness=1.0,
            population_size=1,
            champion_metrics={},
            sim_time=0.0,
            wall_time=0.0,
        )
        raise RuntimeError("boom")

    def save_checkpoint(self, path: Path) -> None:
        pass

    def update_config(self, partial: Mapping[str, Any]) -> None:
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance

    @classmethod
    def load_checkpoint(cls, path, make_env, objective, config):
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance


class _InterruptedTrainer:
    """Stands in for a human hitting Ctrl-C partway through a headless run."""

    def run(self) -> Iterator[TrainingUpdate]:
        yield TrainingUpdate(
            progress_index=0,
            best_fitness=1.0,
            mean_fitness=1.0,
            worst_fitness=1.0,
            population_size=1,
            champion_metrics={},
            sim_time=0.0,
            wall_time=0.0,
        )
        raise KeyboardInterrupt

    def save_checkpoint(self, path: Path) -> None:
        pass

    def update_config(self, partial: Mapping[str, Any]) -> None:
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance

    @classmethod
    def load_checkpoint(cls, path, make_env, objective, config):
        raise NotImplementedError  # unused by the recorder; present only for Trainer conformance


def _model_id(conn) -> str:  # type: ignore[no-untyped-def]
    return create_model(
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
    ).model_id


def test_records_one_generation_stat_row_per_update(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=5)
    run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={"population_size": 6},
    )
    assert len(list_generation_stats_for_model(conn, model_id)) == 5


def test_marks_run_completed_on_normal_exit(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=2)
    record = run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
    )
    assert get_run(conn, record.run_id).status == "completed"
    # the *returned* record reflects the final state too — `update_run_status` writes the DB
    # but cannot mutate the frozen dataclass `create_run` handed back at the start
    assert record.status == "completed"
    assert record.ended_at is not None


def test_marks_run_stopped_on_keyboard_interrupt(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _InterruptedTrainer()
    with pytest.raises(KeyboardInterrupt):  # Ctrl-C still propagates to the caller
        run_and_record(
            conn,
            trainer,
            model_id=model_id,
            track_id="t1",
            starting_generation=0,
            resume_dir=tmp_path / "resume",
            champion_dir=None,
            checkpoint_every_n_generations=100,
            champion_retention_cap=None,
            initial_settings_diff={},
        )
    from neuroarena.persistence.runs_repo import list_runs_for_model

    run_rows = list_runs_for_model(conn, model_id)
    assert len(run_rows) == 1
    assert run_rows[0].status == "stopped"  # not left at "running" forever
    assert run_rows[0].ended_at is not None


def test_marks_run_crashed_on_exception(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _CrashingTrainer()
    with pytest.raises(RuntimeError):
        run_and_record(
            conn,
            trainer,
            model_id=model_id,
            track_id="t1",
            starting_generation=0,
            resume_dir=tmp_path / "resume",
            champion_dir=None,
            checkpoint_every_n_generations=100,
            champion_retention_cap=None,
            initial_settings_diff={},
        )
    # checkpoint_every_n_generations=100 means no *subsequent* checkpoint is due, but the
    # generation-0 checkpoint always fires (0 % N == 0 for any N — see the [0, 3, 6] cadence
    # test below), and it does so before the crash on the next generation is ever reached.
    # (_CrashingTrainer.save_checkpoint is a no-op, so unlike _FakeTrainer it never actually
    # writes gen_00000.pkl to disk — only the DB row reflects the pre-crash checkpoint call.)
    resumes = list_checkpoints(conn, model_id, kind="resume")
    assert [r.generation for r in resumes] == [0]

    from neuroarena.persistence.runs_repo import list_runs_for_model

    run_rows = list_runs_for_model(conn, model_id)
    assert len(run_rows) == 1
    assert run_rows[0].status == "crashed"


def test_saves_a_resume_checkpoint_every_n_generations_and_keeps_all(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=7)
    run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=3,
        champion_retention_cap=None,
        initial_settings_diff={},
    )
    resumes = list_checkpoints(conn, model_id, kind="resume")
    assert [r.generation for r in resumes] == [0, 3, 6]  # every Nth kept, none overwritten
    for record in resumes:
        assert Path(record.file_path).is_file()


def test_registers_and_retains_champion_checkpoints_by_default(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    champion_dir = tmp_path / "champion"
    trainer = _FakeTrainer(n_generations=5, champion_dir=champion_dir)
    run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=champion_dir,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
    )
    champions = list_checkpoints(conn, model_id, kind="champion")
    assert len(champions) == 5  # unbounded by default


def test_prunes_champion_checkpoints_beyond_the_retention_cap(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    champion_dir = tmp_path / "champion"
    trainer = _FakeTrainer(n_generations=5, champion_dir=champion_dir)
    run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=champion_dir,
        checkpoint_every_n_generations=100,
        champion_retention_cap=2,
        initial_settings_diff={},
    )
    champions = list_checkpoints(conn, model_id, kind="champion")
    assert [r.generation for r in champions] == [3, 4]  # only the newest 2 survive
    assert not (champion_dir / "gen_00000.pkl").is_file()  # pruned files are actually deleted
    assert (champion_dir / "gen_00004.pkl").is_file()


def test_pruning_keeps_a_file_a_surviving_row_still_references(tmp_path: Path) -> None:
    # Resuming from an older-than-latest checkpoint replays generations, so a second
    # champion row can point at the *same* file as an earlier row. Pruning the earlier row
    # must not unlink a file the surviving duplicate still references.
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    shared_path = champion_dir / "gen_00000.pkl"
    shared_path.write_bytes(b"champion")

    previous_run = create_run(conn, model_id=model_id, track_id="t1", starting_generation=0)
    stale = record_checkpoint(
        conn,
        model_id=model_id,
        run_id=previous_run.run_id,
        kind="champion",
        generation=0,
        file_path=shared_path,
    )

    trainer = _FakeTrainer(n_generations=2, champion_dir=champion_dir)  # replays gens 0 and 1
    run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=champion_dir,
        checkpoint_every_n_generations=100,
        champion_retention_cap=2,
        initial_settings_diff={},
    )

    champions = list_checkpoints(conn, model_id, kind="champion")
    assert stale.checkpoint_id not in {c.checkpoint_id for c in champions}  # the old row is gone
    assert [c.generation for c in champions] == [0, 1]  # the replayed duplicate survived
    assert shared_path.is_file()  # ...and its file was NOT deleted out from under it


def test_records_the_initial_settings_diff_at_starting_generation(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    trainer = _FakeTrainer(n_generations=1)
    record = run_and_record(
        conn,
        trainer,
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={"population_size": 42},
    )
    from neuroarena.persistence.settings_history_repo import list_settings_history_for_model

    entries = list_settings_history_for_model(conn, model_id)
    assert len(entries) == 1
    assert entries[0].run_id == record.run_id
    assert entries[0].diff == {"population_size": 42}


def test_on_update_is_called_once_per_generation_after_persisting_it(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    seen: list[int] = []
    run_and_record(
        conn,
        _FakeTrainer(n_generations=3),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
        on_update=lambda u: seen.append(u.progress_index),
    )
    assert seen == [0, 1, 2]
    # "after persisting it": every generation on_update saw is already in the DB.
    assert len(list_generation_stats_for_model(conn, model_id)) == 3


def test_should_stop_ends_the_run_early_with_stopped_status(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    record = run_and_record(
        conn,
        _FakeTrainer(n_generations=10),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
        should_stop=lambda: True,
    )
    assert record.status == "stopped"
    # Stopped after fully persisting exactly the first generation, not zero and not all ten.
    assert len(list_generation_stats_for_model(conn, model_id)) == 1


def test_run_without_the_new_hooks_is_unaffected(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    model_id = _model_id(conn)
    record = run_and_record(
        conn,
        _FakeTrainer(n_generations=2),
        model_id=model_id,
        track_id="t1",
        starting_generation=0,
        resume_dir=tmp_path / "resume",
        champion_dir=None,
        checkpoint_every_n_generations=100,
        champion_retention_cap=None,
        initial_settings_diff={},
    )
    assert record.status == "completed"
