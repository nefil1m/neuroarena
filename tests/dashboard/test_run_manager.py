import time
from collections.abc import Callable
from pathlib import Path

from neuroarena.dashboard.run_manager import NoActiveRunError, RunAlreadyActiveError, RunManager
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
