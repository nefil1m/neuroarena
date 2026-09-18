"""Speed presets and the `Pacer` that slows a training batch to a chosen speed — Phase 8's
"speed control paces the trainer" requirement. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`. Speed is dashboard-only runtime
state (not a `RunConfig` field), so nothing here is persisted."""

from __future__ import annotations

import time
from collections.abc import Callable

from neuroarena.sim.game import TICK_DT

SPEED_PRESETS: dict[str, float | None] = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
    "8x": 8.0,
    "max": None,  # unpaced: as fast as the CPU allows
}
DEFAULT_SPEED_PRESET = "max"


class Pacer:
    """Callable installed as `NeatTrainer.set_pace`'s hook: called once per simulation round
    (one `TICK_DT` of simulated time for every active car), it sleeps so that successive
    rounds are `TICK_DT / speed` of wall-clock time apart. It can only slow a run down — if a
    round already takes longer than its target, it does not sleep. `speed()` is read on every
    call, so a change of speed takes effect on the very next round. Runs on the training
    thread."""

    def __init__(
        self,
        speed: Callable[[], float | None],
        *,
        tick_dt: float = TICK_DT,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        max_lag_s: float = 0.25,
    ) -> None:
        self._speed = speed
        self._tick_dt = tick_dt
        self._clock = clock
        self._sleep = sleep
        self._max_lag_s = max_lag_s
        self._deadline: float | None = None
        self._last_speed: float | None = None

    def __call__(self) -> None:
        speed = self._speed()
        if speed is None:
            self._deadline = None
            self._last_speed = None
            return
        now = self._clock()
        if self._deadline is None or speed != self._last_speed:
            self._deadline = now
        self._last_speed = speed
        self._deadline += self._tick_dt / speed
        delay = self._deadline - now
        if delay > 0:
            self._sleep(delay)
        elif delay < -self._max_lag_s:
            # Fell too far behind (a slow round, a debugger pause): resync to "now" rather
            # than racing through the following rounds with no sleep to catch up.
            self._deadline = now
