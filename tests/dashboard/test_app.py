import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from neuroarena.dashboard.app import create_app
from neuroarena.dashboard.run_manager import RunManager
from neuroarena.dashboard.viewer_manager import ViewerManager
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from neuroarena.tracks.store import save
from tests.dashboard.fakes import FakeSpawner


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


def _client(tmp_path: Path) -> TestClient:
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    return TestClient(app)


def test_get_tracks_lists_saved_tracks(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    response = client.get("/api/tracks")
    assert response.status_code == 200
    assert any(t["track_id"] == track_id for t in response.json())


def test_get_models_starts_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == []


def test_get_current_run_is_idle_before_anything_starts(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/runs/current")
    assert response.status_code == 200
    assert response.json() == {"status": "idle", "model_id": None, "generation": None}


def test_starting_a_run_returns_201_and_a_model_id(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    response = client.post(
        "/api/runs",
        json={"track_id": track_id, "population_size": 6, "max_generations": 1000},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "running"
    client.post("/api/runs/current/stop")


def test_starting_a_second_run_while_one_is_active_returns_409(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    response = client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    assert response.status_code == 409
    client.post("/api/runs/current/stop")


def test_starting_a_run_with_an_unknown_track_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/runs", json={"track_id": "does-not-exist"})
    assert response.status_code == 400


def test_stop_without_an_active_run_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/runs/current/stop")
    assert response.status_code == 409


def test_config_update_without_an_active_run_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/runs/current/config", json={"max_generations": 5})
    assert response.status_code == 409


def test_config_update_with_a_supported_field_returns_200(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
    )
    # ConfigUpdateRequest only models the 3 wired fields; this confirms a supported field is
    # accepted while a run is active (no 400 path exists for unsupported fields here).
    response = client.patch("/api/runs/current/config", json={"max_generations": 3})
    assert response.status_code == 200
    client.post("/api/runs/current/stop")


def test_poll_interval_defaults_to_200_and_is_settable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/poll-interval").json() == {"interval_ms": 200}
    response = client.patch("/api/poll-interval", json={"interval_ms": 50})
    assert response.status_code == 200
    assert response.json() == {"interval_ms": 50}
    assert client.get("/api/poll-interval").json() == {"interval_ms": 50}


def test_poll_interval_rejects_a_non_positive_value(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/poll-interval", json={"interval_ms": 0})
    assert response.status_code == 400


def test_model_settings_includes_current_generation(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    client = TestClient(app)
    model_id = client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 2}
    ).json()["model_id"]

    deadline = 0
    while True:
        response = client.get(f"/api/models/{model_id}/settings")
        if response.status_code == 200 and response.json().get("current_generation") is not None:
            break
        deadline += 1
        assert deadline < 500, "run never produced a resume checkpoint in time"
        import time

        time.sleep(0.02)

    body = response.json()
    assert body["current_generation"] >= 0
    assert body["population_size"] == 6


def test_model_settings_for_an_unknown_model_returns_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/models/does-not-exist/settings")
    assert response.status_code == 404


def test_ws_pushes_status_then_progress_and_generation_messages(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    manager.set_poll_interval_ms(10)  # fast polling keeps this test quick
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        client.post(
            "/api/runs",
            json={"track_id": track_id, "population_size": 6, "max_generations": 1000},
        )
        seen_types: set[str] = set()
        deadline = time.monotonic() + 30  # wall-clock bound; progress msgs arrive every tick
        while time.monotonic() < deadline:
            msg = ws.receive_json()
            seen_types.add(msg["type"])
            assert msg["schema_version"] == 1
            if {"status", "progress", "generation"} <= seen_types:
                break
        assert {"status", "progress", "generation"} <= seen_types
    client.post("/api/runs/current/stop")


def test_config_update_after_the_run_has_ended_returns_409(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    client = _client(tmp_path)
    client.post(
        "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 2}
    )
    deadline = time.monotonic() + 20
    while client.get("/api/runs/current").json()["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    response = client.patch("/api/runs/current/config", json={"max_generations": 5})
    assert response.status_code == 409


def test_app_shutdown_stops_the_active_run_and_finalizes_its_row(tmp_path: Path) -> None:
    from neuroarena.persistence import runs_repo
    from neuroarena.persistence.db import connect

    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    app = create_app(run_manager=manager, data_dir=tmp_path)
    with TestClient(app) as client:
        model_id = client.post(
            "/api/runs",
            json={"track_id": track_id, "population_size": 6, "max_generations": 1000},
        ).json()["model_id"]
        deadline = time.monotonic() + 20
        while manager.latest_update() is None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
    conn = connect(tmp_path / "neuroarena.db")
    try:
        (run,) = runs_repo.list_runs_for_model(conn, model_id)
    finally:
        conn.close()
    assert run.status == "stopped"
    assert run.ended_at is not None


def test_get_speed_defaults_to_max_and_lists_the_presets(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/speed")
    assert response.status_code == 200
    assert response.json() == {
        "preset": "max",
        "presets": ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "max"],
    }


def test_patch_speed_sets_the_preset(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/speed", json={"preset": "2x"})
    assert response.status_code == 200
    assert response.json() == {"preset": "2x"}
    assert client.get("/api/speed").json()["preset"] == "2x"


def test_patch_speed_rejects_an_unknown_preset_with_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.patch("/api/speed", json={"preset": "3x"})
    assert response.status_code == 400


def _viewer_client(tmp_path: Path, spawner: FakeSpawner) -> TestClient:
    manager = RunManager(data_dir=tmp_path)
    viewer = ViewerManager(
        url="ws://127.0.0.1:8000/ws/viewer",
        data_dir=tmp_path,
        spawn=spawner,
        terminate_timeout_s=0.1,
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    return TestClient(app)


def test_viewer_starts_closed_with_default_settings(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    response = client.get("/api/viewer")
    assert response.status_code == 200
    assert response.json() == {
        "open": False,
        "settings": {"zoom": 1.0, "camera_mode": "fit", "follow_rank": 1, "follow_seq": 0},
    }


def test_open_and_close_the_viewer_are_idempotent(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    client = _viewer_client(tmp_path, spawner)
    assert client.post("/api/viewer/open").json()["open"] is True
    assert client.post("/api/viewer/open").json()["open"] is True
    assert len(spawner.processes) == 1
    assert client.post("/api/viewer/close").json()["open"] is False
    assert client.post("/api/viewer/close").json()["open"] is False


def test_patch_viewer_settings_updates_and_validates(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    ok = client.patch(
        "/api/viewer/settings",
        json={"camera_mode": "follow_rank", "follow_rank": 3, "zoom": 2.0},
    )
    assert ok.status_code == 200
    assert ok.json() == {
        "zoom": 2.0,
        "camera_mode": "follow_rank",
        "follow_rank": 3,
        "follow_seq": 1,
    }
    assert client.patch("/api/viewer/settings", json={"camera_mode": "orbit"}).status_code == 400
    assert client.patch("/api/viewer/settings", json={"zoom": 99}).status_code == 400
    assert client.get("/api/viewer").json()["settings"]["zoom"] == 2.0  # unchanged by the bad ones


def test_app_shutdown_closes_the_viewer(tmp_path: Path) -> None:
    spawner = FakeSpawner()
    manager = RunManager(data_dir=tmp_path)
    viewer = ViewerManager(
        url="ws://x/ws/viewer", data_dir=tmp_path, spawn=spawner, terminate_timeout_s=0.1
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    with TestClient(app) as client:
        client.post("/api/viewer/open")
        assert spawner.processes[0].terminated is False
    assert spawner.processes[0].terminated is True


def test_ws_viewer_sends_run_and_view_messages_on_connect_when_idle(tmp_path: Path) -> None:
    client = _viewer_client(tmp_path, FakeSpawner())
    with client.websocket_connect("/ws/viewer") as ws:
        first = ws.receive_json()
        second = ws.receive_json()
    assert {first["type"], second["type"]} == {"run", "view"}
    run = first if first["type"] == "run" else second
    assert run["data"] == {"state": "idle", "track_id": None, "model_id": None}


def test_ws_viewer_closes_the_socket_when_the_loop_body_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(data_dir=tmp_path)

    def boom() -> None:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(manager, "visual_snapshot", boom)
    viewer = ViewerManager(
        url="ws://x/ws/viewer", data_dir=tmp_path, spawn=FakeSpawner(), terminate_timeout_s=0.1
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    outcome: list[str] = []

    def drive() -> None:
        with TestClient(app) as client, client.websocket_connect("/ws/viewer") as ws:
            first = ws.receive_json()
            second = ws.receive_json()
            assert {first["type"], second["type"]} == {"run", "view"}
            try:
                ws.receive_json()
            except WebSocketDisconnect:
                outcome.append("closed")

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    thread.join(timeout=20.0)
    assert not thread.is_alive(), "/ws/viewer did not end after the loop body failed"
    assert outcome == ["closed"]


def test_ws_viewer_streams_frames_with_car_poses_for_a_running_run(tmp_path: Path) -> None:
    track_id = _save_test_track(tmp_path)
    manager = RunManager(data_dir=tmp_path)
    viewer = ViewerManager(
        url="ws://127.0.0.1:8000/ws/viewer",
        data_dir=tmp_path,
        spawn=FakeSpawner(),
        terminate_timeout_s=0.1,
    )
    app = create_app(run_manager=manager, data_dir=tmp_path, viewer_manager=viewer)
    run_messages: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []

    with TestClient(app) as client:
        client.post(
            "/api/runs", json={"track_id": track_id, "population_size": 6, "max_generations": 1000}
        )

        def read_until_a_frame() -> None:
            with client.websocket_connect("/ws/viewer") as ws:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline and not frames:
                    message = ws.receive_json()
                    if message["type"] == "run":
                        run_messages.append(message["data"])
                    elif message["type"] == "frame" and message["data"]["cars"]:
                        frames.append(message["data"])

        reader = threading.Thread(target=read_until_a_frame, daemon=True)
        reader.start()
        reader.join(timeout=30.0)
        timed_out = reader.is_alive()
        client.post("/api/runs/current/stop")
        end = time.monotonic() + 60.0
        while client.get("/api/runs/current").json()["status"] == "running":
            assert time.monotonic() < end, "the run did not stop within 60 s"
            time.sleep(0.05)
        reader.join(timeout=10.0)
        assert not timed_out, "no /ws/viewer frame with cars arrived within 30 s"

    assert run_messages and run_messages[-1]["track_id"] == track_id
    assert frames
    frame = frames[0]
    assert frame["population_size"] == 6
    assert frame["speed"] == "max"
    car = frame["cars"][0]  # type: ignore[index]
    assert set(car) == {"id", "x", "y", "heading", "fitness"}
