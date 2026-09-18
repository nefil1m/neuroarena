import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from neuroarena.dashboard.run_manager import NoActiveRunError, RunAlreadyActiveError, RunManager
from neuroarena.persistence.db import connect as persistence_connect
from neuroarena.persistence.launch import prepare_run as persistence_prepare_run
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


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition never became true")


def test_poll_interval_defaults_to_200ms_and_is_settable(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.poll_interval_ms() == 200
    manager.set_poll_interval_ms(50)
    assert manager.poll_interval_ms() == 50


def test_set_poll_interval_rejects_non_positive_values(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.set_poll_interval_ms(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_status_is_idle_before_any_run_starts(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    status = manager.status()
    assert status.status == "idle"
    assert status.model_id is None
    assert status.generation is None


def test_start_runs_to_completion_in_the_background(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    model_id = manager.start(
        track_id=track_id, overrides={"population_size": 6, "max_generations": 2}
    )
    assert model_id == manager.status().model_id
    _wait_until(lambda: manager.status().status == "completed")
    assert manager.status().generation == 1  # 0-indexed, 2 generations means final index 1


def test_start_while_a_run_is_active_is_rejected(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 50})
    try:
        manager.start(track_id=track_id, overrides={"population_size": 6})
        raise AssertionError("expected RunAlreadyActiveError")
    except RunAlreadyActiveError:
        pass
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running")


def test_request_stop_ends_a_running_run_with_stopped_status(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    _wait_until(lambda: manager.latest_update() is not None)
    manager.request_stop()
    _wait_until(lambda: manager.status().status == "stopped")


def test_request_stop_without_an_active_run_raises(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.request_stop()
        raise AssertionError("expected NoActiveRunError")
    except NoActiveRunError:
        pass


def test_update_config_without_an_active_run_raises(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    try:
        manager.update_config({"max_generations": 5})
        raise AssertionError("expected NoActiveRunError")
    except NoActiveRunError:
        pass


def test_update_config_rejects_unsupported_fields_once_a_run_is_active(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    try:
        try:
            manager.update_config({"sim_speed": 2.0})
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running")


def test_progress_snapshot_is_none_when_idle(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.progress_snapshot() is None


def test_worker_connect_failure_ends_crashed_and_allows_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    real_connect = persistence_connect
    main = threading.main_thread()

    def flaky_connect(path: Path) -> sqlite3.Connection:
        if threading.current_thread() is not main:
            raise sqlite3.OperationalError("boom")
        return real_connect(path)

    monkeypatch.setattr("neuroarena.dashboard.run_manager.connect", flaky_connect)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 2})
    _wait_until(lambda: manager.status().status == "crashed", timeout=20)
    monkeypatch.setattr("neuroarena.dashboard.run_manager.connect", real_connect)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 2})
    _wait_until(lambda: manager.status().status == "completed", timeout=20)


def test_prepare_run_failure_restores_previous_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(data_dir=tmp_path)

    def failing_prepare(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad track")

    monkeypatch.setattr("neuroarena.dashboard.run_manager.prepare_run", failing_prepare)
    with pytest.raises(ValueError):
        manager.start(track_id="nope")
    assert manager.status().status == "idle"


def test_concurrent_starts_yield_exactly_one_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    real_prepare = persistence_prepare_run

    def slow_prepare(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)  # widen the check/transition window
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr("neuroarena.dashboard.run_manager.prepare_run", slow_prepare)
    n = 6
    barrier = threading.Barrier(n, timeout=5)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            manager.start(
                track_id=track_id, overrides={"population_size": 6, "max_generations": 50}
            )
            result = "ok"
        except RunAlreadyActiveError:
            result = "rejected"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    try:
        assert sorted(outcomes) == ["ok"] + ["rejected"] * (n - 1)
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running")


def _model_runs(data_dir: Path, model_id: str) -> list[Any]:
    from neuroarena.persistence import runs_repo

    conn = persistence_connect(data_dir / "neuroarena.db")
    try:
        return runs_repo.list_runs_for_model(conn, model_id)
    finally:
        conn.close()


def test_update_config_after_the_run_has_ended_raises(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 2})
    _wait_until(lambda: manager.status().status == "completed", timeout=20)
    with pytest.raises(NoActiveRunError):
        manager.update_config({"max_generations": 5})


def test_shutdown_with_an_active_run_leaves_a_stopped_row(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    model_id = manager.start(
        track_id=track_id, overrides={"population_size": 6, "max_generations": 1000}
    )
    _wait_until(lambda: manager.latest_update() is not None)
    manager.shutdown(timeout=20)
    (run,) = _model_runs(tmp_path, model_id)
    assert run.status == "stopped"
    assert run.ended_at is not None


def test_shutdown_finalizes_the_row_itself_when_the_worker_outlives_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    release = threading.Event()
    real_run = manager._run

    def stuck_run(prepared: Any) -> None:
        release.wait(timeout=20)
        real_run(prepared)

    monkeypatch.setattr(manager, "_run", stuck_run)
    # Create the row the way the worker would, but keep the worker from ever starting it.
    from neuroarena.persistence import runs_repo

    model_id = manager.start(
        track_id=track_id, overrides={"population_size": 6, "max_generations": 1000}
    )
    conn = persistence_connect(tmp_path / "neuroarena.db")
    try:
        runs_repo.create_run(conn, model_id=model_id, track_id=track_id, starting_generation=0)
    finally:
        conn.close()
    try:
        manager.shutdown(timeout=0.1)
        (run,) = _model_runs(tmp_path, model_id)
        assert run.status == "stopped"
        assert run.ended_at is not None
    finally:
        release.set()
        manager._stop_requested = True
        if manager._thread is not None:
            manager._thread.join(timeout=20)


def test_shutdown_when_idle_is_a_noop(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    manager.shutdown(timeout=0.1)
    assert manager.status().status == "idle"


def test_speed_preset_defaults_to_max_and_is_settable(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    assert manager.speed_preset() == "max"
    assert manager.speed_multiplier() is None
    manager.set_speed_preset("2x")
    assert manager.speed_preset() == "2x"
    assert manager.speed_multiplier() == 2.0


def test_set_speed_preset_rejects_an_unknown_preset(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    with pytest.raises(ValueError):
        manager.set_speed_preset("3x")
    assert manager.speed_preset() == "max"


def test_visual_snapshot_is_none_when_idle(tmp_path: Path) -> None:
    assert RunManager(data_dir=tmp_path).visual_snapshot() is None
    assert RunManager(data_dir=tmp_path).current_track_id() is None


def test_a_running_run_exposes_track_id_pacer_and_car_poses(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    manager.start(track_id=track_id, overrides={"population_size": 6, "max_generations": 1000})
    try:
        assert manager.current_track_id() == track_id
        # the trainer got this manager's pacer installed
        assert getattr(manager._trainer, "_pace", None) is manager._pacer

        def _has_cars() -> bool:
            snapshot = manager.visual_snapshot()
            return snapshot is not None and len(snapshot.genomes) > 0

        _wait_until(_has_cars, timeout=30.0)
        snapshot = manager.visual_snapshot()
        assert snapshot is not None
        assert snapshot.population_size == 6
        assert set(snapshot.genomes[0].state) == {"x", "y", "heading"}
    finally:
        manager.request_stop()
        _wait_until(lambda: manager.status().status != "running", timeout=60.0)


def test_a_stop_request_unpaces_the_run_but_not_the_reported_preset(tmp_path: Path) -> None:
    manager = RunManager(data_dir=tmp_path)
    manager.set_speed_preset("2x")
    manager._status = "running"  # pure state test: no real run needed
    assert manager.speed_multiplier() == 2.0
    manager.request_stop()
    assert manager.speed_multiplier() is None
    assert manager.speed_preset() == "2x"
