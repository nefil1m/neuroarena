"""The viewer process's connection to the dashboard backend: a background thread reads
`/ws/viewer` into a latest-state slot that the (main-thread) `arcade` window reads each frame.
Pure of `arcade`, so the message handling is unit-testable. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as _ws_connect

from neuroarena.render.view_model import CarView
from neuroarena.render.view_settings import ViewSettings

_log = logging.getLogger(__name__)

ConnectFn = Callable[[str], AbstractContextManager[Iterable[str | bytes]]]


def _default_connect(url: str) -> AbstractContextManager[Iterable[str | bytes]]:
    return _ws_connect(url)


@dataclass(frozen=True)
class ViewerState:
    run_state: str = "idle"
    track_id: str | None = None
    generation: int | None = None
    population_size: int = 0
    speed: str = "max"
    cars: tuple[CarView, ...] = ()
    settings: ViewSettings = field(default_factory=ViewSettings)
    connected: bool = False


class ViewerClient:
    def __init__(
        self,
        url: str,
        *,
        connect: ConnectFn = _default_connect,
        retry_delay_s: float = 1.0,
        give_up_after_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._connect = connect
        self._retry_delay_s = retry_delay_s
        self._give_up_after_s = give_up_after_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._state = ViewerState()
        self._stop = threading.Event()
        self._disconnected_since: float | None = clock()

    def state(self) -> ViewerState:
        with self._lock:
            return self._state

    def should_exit(self) -> bool:
        """True once disconnected for longer than the grace period (counted from
        construction until the first successful connection)."""
        with self._lock:
            since = self._disconnected_since
        return since is not None and self._clock() - since > self._give_up_after_s

    def start(self) -> None:
        threading.Thread(target=self._loop, name="viewer-client", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> None:
        """One connection lifecycle: connect, then consume messages until it ends. Never
        raises on a refused/lost connection — the loop simply tries again."""
        try:
            with self._connect(self._url) as connection:
                self._set_connected(True)
                for raw in connection:
                    if self._stop.is_set():
                        return
                    self._handle_raw(raw)
        except (OSError, WebSocketException):
            _log.info("viewer connection to %s ended", self._url)
        finally:
            self._set_connected(False)

    def apply_message(self, message: Mapping[str, Any]) -> None:
        kind = message["type"]
        data = message["data"]
        with self._lock:
            state = self._state
            if kind == "run":
                cars = state.cars
                if data["state"] == "running" and state.run_state != "running":
                    cars = ()  # a new run starts on an empty field
                self._state = replace(
                    state, run_state=data["state"], track_id=data["track_id"], cars=cars
                )
            elif kind == "view":
                self._state = replace(state, settings=ViewSettings.from_dict(data))
            elif kind == "frame":
                self._state = replace(
                    state,
                    generation=int(data["generation"]),
                    population_size=int(data["population_size"]),
                    speed=str(data["speed"]),
                    cars=tuple(
                        CarView(
                            genome_id=int(car["id"]),
                            x=float(car["x"]),
                            y=float(car["y"]),
                            heading=float(car["heading"]),
                            fitness=float(car["fitness"]),
                        )
                        for car in data["cars"]
                    ),
                )
            # any other type: ignored, so a newer backend can add messages

    def _handle_raw(self, raw: str | bytes) -> None:
        try:
            self.apply_message(json.loads(raw))
        except (KeyError, TypeError, ValueError):
            _log.warning("ignoring malformed viewer message: %.200r", raw)

    def _set_connected(self, connected: bool) -> None:
        with self._lock:
            self._state = replace(self._state, connected=connected)
            if connected:
                self._disconnected_since = None
            elif self._disconnected_since is None:
                self._disconnected_since = self._clock()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            if self._stop.is_set():
                return
            self._sleep(self._retry_delay_s)
