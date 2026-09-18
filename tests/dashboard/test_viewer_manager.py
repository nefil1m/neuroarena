import sys
import threading
import time
from pathlib import Path

import pytest

from neuroarena.dashboard.viewer_manager import ViewerManager
from neuroarena.render.view_settings import ViewSettings
from tests.dashboard.fakes import BlockingWaitProcess, FakeSpawner, NeverExitsProcess


def _manager(tmp_path: Path, spawner: FakeSpawner) -> ViewerManager:
    return ViewerManager(
        url="ws://127.0.0.1:8000/ws/viewer",
        data_dir=tmp_path,
        spawn=spawner,
        terminate_timeout_s=0.1,
    )


def test_open_spawns_the_viewer_with_the_agreed_command_line(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    assert manager.open() is True
    assert spawner.processes[0].args == [
        sys.executable,
        "-m",
        "neuroarena.render.viewer",
        "--url",
        "ws://127.0.0.1:8000/ws/viewer",
        "--data-dir",
        str(tmp_path),
    ]
    assert manager.is_open() is True


def test_a_second_open_while_one_is_running_does_nothing(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.open() is False
    assert len(spawner.processes) == 1


def test_is_open_turns_false_when_the_window_is_closed_by_hand(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    spawner.processes[0].die()
    assert manager.is_open() is False
    assert manager.open() is True  # and it can be opened again
    assert len(spawner.processes) == 2


def test_close_terminates_the_process_and_is_idempotent(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.close() is True
    assert spawner.processes[0].terminated is True
    assert manager.is_open() is False
    assert manager.close() is False


def test_close_kills_a_process_that_ignores_terminate(tmp_path: Path) -> None:
    spawner = FakeSpawner(stubborn=True)
    manager = _manager(tmp_path, spawner)
    manager.open()
    assert manager.close() is True
    assert spawner.processes[0].terminated is True
    assert spawner.processes[0].killed is True


def test_shutdown_closes_the_viewer(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = _manager(tmp_path, spawner)
    manager.open()
    manager.shutdown()
    assert spawner.processes[0].terminated is True


def test_settings_default_and_update_merge_a_partial(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeSpawner())
    assert manager.settings() == ViewSettings()
    updated = manager.update_settings({"zoom": 2.0, "camera_mode": "follow_rank", "follow_rank": 3})
    assert updated == ViewSettings(zoom=2.0, camera_mode="follow_rank", follow_rank=3, follow_seq=1)
    assert manager.settings() == updated


def test_an_invalid_settings_update_raises_and_changes_nothing(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeSpawner())
    with pytest.raises(ValueError):
        manager.update_settings({"camera_mode": "orbit"})
    assert manager.settings() == ViewSettings()


def test_settings_do_not_wait_for_a_close_stuck_in_a_process_wait(tmp_path: Path) -> None:
    process = BlockingWaitProcess([])
    manager = ViewerManager(
        url="ws://x", data_dir=tmp_path, spawn=lambda args: process, terminate_timeout_s=0.1
    )
    manager.open()
    closer = threading.Thread(target=manager.close, daemon=True)
    closer.start()
    try:
        assert process.in_wait.wait(timeout=5.0)
        start = time.monotonic()
        assert manager.settings() == ViewSettings()
        manager.update_settings({"zoom": 2.0})
        assert time.monotonic() - start < 1.0
    finally:
        process.release.set()
        closer.join(timeout=10.0)
    assert not closer.is_alive()


def test_close_returns_even_if_the_process_never_exits_after_kill(tmp_path: Path) -> None:
    process = NeverExitsProcess([])
    manager = ViewerManager(
        url="ws://x", data_dir=tmp_path, spawn=lambda args: process, terminate_timeout_s=0.05
    )
    manager.open()
    assert manager.close() is True
    assert process.killed is True
