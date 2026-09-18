import json
from collections.abc import Iterable
from typing import Any

from neuroarena.render.view_model import CarView
from neuroarena.render.view_settings import ViewSettings
from neuroarena.render.viewer_client import ViewerClient, ViewerState


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeConnection:
    def __init__(self, messages: Iterable[str | bytes]) -> None:
        self._messages = list(messages)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(self._messages)


def _client(connect: Any = None, fake: _FakeTime | None = None) -> ViewerClient:
    fake = fake or _FakeTime()
    return ViewerClient(
        "ws://test/ws/viewer",
        connect=connect or (lambda url: _FakeConnection([])),
        clock=fake.clock,
        sleep=fake.sleep,
        give_up_after_s=15.0,
    )


def _frame(cars: list[dict[str, float]], generation: int = 3) -> dict[str, Any]:
    return {
        "type": "frame",
        "schema_version": 1,
        "data": {"generation": generation, "population_size": 10, "speed": "2x", "cars": cars},
    }


def _run(state: str, track_id: str | None = "t1") -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": 1,
        "data": {"state": state, "track_id": track_id, "model_id": "m1"},
    }


def test_initial_state_is_idle_and_disconnected() -> None:
    state = _client().state()
    assert state == ViewerState()
    assert state.connected is False


def test_a_frame_message_replaces_the_cars_and_metadata() -> None:
    client = _client()
    client.apply_message(
        _frame([{"id": 4, "x": 1.0, "y": 2.0, "heading": 0.5, "fitness": 9.0}], generation=7)
    )
    state = client.state()
    assert state.generation == 7
    assert state.population_size == 10
    assert state.speed == "2x"
    assert state.cars == (CarView(genome_id=4, x=1.0, y=2.0, heading=0.5, fitness=9.0),)


def test_a_view_message_replaces_the_settings() -> None:
    client = _client()
    settings = ViewSettings(zoom=2.0, camera_mode="follow_best")
    client.apply_message({"type": "view", "schema_version": 1, "data": settings.to_dict()})
    assert client.state().settings == settings


def test_a_run_message_sets_state_and_track_and_clears_cars_when_a_run_starts() -> None:
    client = _client()
    car = {"id": 1, "x": 0.0, "y": 0.0, "heading": 0.0, "fitness": 1.0}
    client.apply_message(_frame([car]))
    client.apply_message(_run("stopped"))  # a finished run keeps its last frame
    assert len(client.state().cars) == 1
    assert client.state().run_state == "stopped"
    client.apply_message(_run("running", track_id="t2"))  # a new run clears the field
    state = client.state()
    assert (state.run_state, state.track_id, state.cars) == ("running", "t2", ())


def test_an_unknown_message_type_is_ignored() -> None:
    client = _client()
    client.apply_message({"type": "future-thing", "schema_version": 9, "data": {}})
    assert client.state() == ViewerState()


def test_run_once_applies_every_message_then_marks_disconnected() -> None:
    messages: list[str | bytes] = [json.dumps(_run("running")), json.dumps(_frame([])).encode()]
    client = _client(connect=lambda url: _FakeConnection(messages))
    client.run_once()
    state = client.state()
    assert state.run_state == "running"
    assert state.generation == 3
    assert state.connected is False  # the fake connection ended


def test_run_once_skips_malformed_messages_and_keeps_going() -> None:
    messages: list[str | bytes] = [
        "not json",
        json.dumps({"type": "frame"}),
        json.dumps(_run("running")),
    ]
    client = _client(connect=lambda url: _FakeConnection(messages))
    client.run_once()
    assert client.state().run_state == "running"


def test_run_once_does_not_raise_when_the_backend_is_unreachable() -> None:
    def refuse(url: str) -> Any:
        raise ConnectionRefusedError("nobody home")

    client = _client(connect=refuse)
    client.run_once()  # must not raise
    assert client.state().connected is False


def test_should_exit_only_after_being_disconnected_past_the_grace_period() -> None:
    fake = _FakeTime()
    client = _client(fake=fake)
    assert client.should_exit() is False
    fake.now = 14.0
    assert client.should_exit() is False
    fake.now = 16.0
    assert client.should_exit() is True


def test_repeated_failed_attempts_do_not_reset_the_grace_period() -> None:
    fake = _FakeTime()

    def refuse(url: str) -> Any:
        raise ConnectionRefusedError

    client = _client(connect=refuse, fake=fake)
    fake.now = 10.0
    client.run_once()
    fake.now = 16.0
    client.run_once()
    assert client.should_exit() is True


def test_connecting_resets_the_grace_period() -> None:
    fake = _FakeTime()
    connected_states: list[bool] = []

    class _Watching(_FakeConnection):
        def __iter__(self) -> Any:
            connected_states.append(client.state().connected)
            return iter([])

    client = _client(connect=lambda url: _Watching([]), fake=fake)
    fake.now = 100.0
    client.run_once()
    assert connected_states == [True]  # was marked connected while the connection was open
    assert client.should_exit() is False  # a fresh disconnect just now: the clock restarted
