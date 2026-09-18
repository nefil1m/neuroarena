"""Phase 3 `Environment` for the car game: wires Phase 1's `Game` to the Phase 0
`Environment` Protocol by adding the episode semantics `Game.tick()` deliberately left out
(reset, `terminated`, `truncated`, `info`) — see `game.py`'s module docstring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from neuroarena.interfaces.spaces import Box, Space
from neuroarena.sim.collision import distance_to_boundary
from neuroarena.sim.game import Game
from neuroarena.sim.observation import SensorConfig, build_observation, observation_space_for
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.sim.track import (
    Track,
    boundary_segments,
    centerline_path,
    project_onto_centerline,
    track_loop_length,
)

if TYPE_CHECKING:
    from neuroarena.config import RunConfig

_CRASH_EPSILON = 1e-6


@dataclass(frozen=True)
class CarEnvironmentConfig:
    """Starting defaults; also buildable from a `RunConfig` via `from_run_config` (Phase 5)."""

    sensor_config: SensorConfig = SensorConfig()
    physics_constants: PhysicsConstants = PhysicsConstants()
    max_episode_steps: int = 3000

    @classmethod
    def from_run_config(cls, config: RunConfig) -> CarEnvironmentConfig:
        """Reads the three car-specific knobs Phase 5 put on `RunConfig`
        (`sensor_config`, `physics_constants`, `max_episode_steps`) into a
        `CarEnvironmentConfig`. `RunConfig.track_id` is not read here — which `Track`
        object a run uses is a separate `CarEnvironment.__init__` argument, resolved by
        whatever code loads the track by id (see the Phase 5 implementation plan's "Not
        built in this plan" note)."""
        return cls(
            sensor_config=config.sensor_config,
            physics_constants=config.physics_constants,
            max_episode_steps=config.max_episode_steps,
        )


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
        # TODO(phase-4): `Game.__init__` computes this same boundary again internally, and
        # both this class and `observation.py` do an O(segments) scan per ray/check per
        # tick. Fine at this track size; worth a broad-phase lookup if NEAT's population
        # scale makes per-step cost matter.
        self._centerline = centerline_path(track)
        self._loop_length = track_loop_length(track)
        self._car_radius = self._config.physics_constants.car_width / 2
        self.observation_space: Space = observation_space_for(self._config.sensor_config)
        self.action_space: Space = Box(-1.0, 1.0, (2,))
        self._game = Game(track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._last_raw_position = self._project(self._game.car.x, self._game.car.y)
        self._step_count = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        # No randomness in this env today (deterministic track + physics), so `seed` is
        # accepted for Protocol conformance and unused. Per the Phase 0 doc: the trainer
        # owns seeding and calls `env.reset(seed=...)`; this env has nothing to seed.
        self._game = Game(self._track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._last_raw_position = self._project(self._game.car.x, self._game.car.y)
        self._step_count = 0
        return self._observation()

    def visual_state(self) -> Mapping[str, float]:
        """Satisfies `Visualizable`. One read of `self._game.car` (a frozen state replaced
        atomically each tick), so a viewer thread never sees a torn pose."""
        car = self._game.car
        return {"x": car.x, "y": car.y, "heading": car.heading}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        """Advance one tick. `info` carries `crashed` (bool, mirrors `terminated`),
        `track_id` (str, which stored map this episode runs on), `progress` (float, signed
        continuous distance travelled along the track's centerline from the spawn point,
        via `project_onto_centerline` — unwrapped, so it keeps increasing lap after lap and
        can go negative if the car drives backward past the start), and `lap_progress`
        (float, `progress / loop_length`, so `1.0` means one lap). Calling `step()` again
        after `terminated` is `True` is undefined by this env — it keeps ticking and keeps
        reporting `terminated=True`, matching Gymnasium's own convention of leaving this to
        the caller."""
        steering = max(-1.0, min(1.0, float(action[0])))
        throttle = max(-1.0, min(1.0, float(action[1])))
        state = self._game.tick(steering, throttle)

        raw_position = self._project(state.car.x, state.car.y)
        delta = raw_position - self._last_raw_position
        if delta < -self._loop_length / 2:
            delta += self._loop_length
        elif delta > self._loop_length / 2:
            delta -= self._loop_length
        self._progress += delta
        self._last_raw_position = raw_position

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

    def _project(self, x: float, y: float) -> float:
        return project_onto_centerline((x, y), self._centerline)

    def _observation(self) -> np.ndarray:
        return build_observation(
            self._game.car,
            self._boundary,
            self._last_action,
            self._config.sensor_config,
            self._config.physics_constants,
        )
