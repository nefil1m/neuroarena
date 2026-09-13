"""Phase 3 `Environment` for the car game: wires Phase 1's `Game` to the Phase 0
`Environment` Protocol by adding the episode semantics `Game.tick()` deliberately left out
(reset, `terminated`, `truncated`, `info`) — see `game.py`'s module docstring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuroarena.interfaces.spaces import Box, Space
from neuroarena.sim.collision import distance_to_boundary
from neuroarena.sim.game import TICK_DT, Game
from neuroarena.sim.observation import SensorConfig, build_observation, observation_space_for
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.sim.track import Track, boundary_segments, track_loop_length

_CRASH_EPSILON = 1e-6


@dataclass(frozen=True)
class CarEnvironmentConfig:
    """Starting defaults; Phase 5 wires these through `RunConfig`."""

    sensor_config: SensorConfig = SensorConfig()
    physics_constants: PhysicsConstants = PhysicsConstants()
    max_episode_steps: int = 3000


class CarEnvironment:
    """Satisfies `neuroarena.interfaces.protocols.Environment` for the car game. One
    instance is one episode on one `Track`; call `reset()` to start a new episode on the
    same track (a fresh `Track`/`track_id` needs a new `CarEnvironment`)."""

    def __init__(
        self,
        track: Track,
        track_id: str,
        config: CarEnvironmentConfig | None = None,
    ) -> None:
        self._track = track
        self._track_id = track_id
        self._config = config if config is not None else CarEnvironmentConfig()
        self._boundary = boundary_segments(track)
        self._loop_length = track_loop_length(track)
        self._car_radius = self._config.physics_constants.car_width / 2
        self.observation_space: Space = observation_space_for(self._config.sensor_config)
        self.action_space: Space = Box(-1.0, 1.0, (2,))
        self._game = Game(track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._step_count = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        # No randomness in this env today (deterministic track + physics), so `seed` is
        # accepted for Protocol conformance and unused — see the Phase 3 doc.
        self._game = Game(self._track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._step_count = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        steering, throttle = float(action[0]), float(action[1])
        state = self._game.tick(steering, throttle)
        self._progress += state.car.speed * TICK_DT
        self._last_action = (steering, throttle)
        self._step_count += 1

        crashed = distance_to_boundary((state.car.x, state.car.y), self._boundary) <= (
            self._car_radius + _CRASH_EPSILON
        )
        truncated = self._step_count >= self._config.max_episode_steps

        info: dict[str, Any] = {
            "crashed": crashed,
            "track_id": self._track_id,
            "progress": self._progress,
            "lap_progress": self._progress / self._loop_length,
        }
        return self._observation(), crashed, truncated, info

    def _observation(self) -> np.ndarray:
        return build_observation(
            self._game.car,
            self._boundary,
            self._last_action,
            self._config.sensor_config,
            self._config.physics_constants,
        )
