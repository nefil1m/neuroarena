import time
from pathlib import Path

from fastapi.testclient import TestClient

from neuroarena.dashboard.app import create_app
from neuroarena.dashboard.run_manager import RunManager
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
