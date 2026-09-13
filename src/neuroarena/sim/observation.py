"""Assembles the car's Phase 3 observation vector: raycasts + normalised speed + last
action. See the Phase 3 doc for the numeric constants and normalisation rules this encodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroarena.interfaces.spaces import Box
from neuroarena.sim.physics import CarState, PhysicsConstants
from neuroarena.sim.raycast import cast_rays
from neuroarena.sim.track import Segment


@dataclass(frozen=True)
class SensorConfig:
    """Sensor layout — configurable at run creation (Phase 3 doc); the values below are the
    platform-standard default. A non-default `ray_angles_deg` changes `observation_space`'s
    shape, which is what the Phase 0 compatibility check compares."""

    ray_angles_deg: tuple[float, ...] = (-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0)
    range_base: float = 300.0
    range_k: float = 200.0
    max_range: float = 600.0


def observation_space_for(sensor_config: SensorConfig) -> Box:
    size = len(sensor_config.ray_angles_deg) + 3  # rays + speed + 2 last-action components
    return Box(-1.0, 1.0, (size,))


def speed_norm(speed: float, constants: PhysicsConstants) -> float:
    """Signed, single scale by max forward speed (Phase 3 doc): forward reads `[0, 1]`,
    reverse reads down to about `-constants.max_reverse_speed / constants.max_speed`."""
    return speed / constants.max_speed


def build_observation(
    car: CarState,
    boundary: list[Segment],
    last_action: tuple[float, float],
    sensor_config: SensorConfig,
    constants: PhysicsConstants,
) -> np.ndarray:
    """Assembles the observation vector: `len(sensor_config.ray_angles_deg)` raycast
    proximities (in `sensor_config.ray_angles_deg` order), then normalised speed, then the
    last commanded `(steering, throttle)` — e.g. for the default 7-ray config, slots
    `[0:7]` are rays, `[7]` is speed, `[8:10]` is `(steering, throttle)`."""
    norm_speed = speed_norm(car.speed, constants)
    ray_range = max(
        0.0,
        min(
            sensor_config.range_base + sensor_config.range_k * norm_speed,
            sensor_config.max_range,
        ),
    )
    car_radius = constants.car_width / 2
    proximities = cast_rays(
        (car.x, car.y),
        car.heading,
        sensor_config.ray_angles_deg,
        ray_range,
        boundary,
        car_radius,
    )
    values = [*proximities, norm_speed, last_action[0], last_action[1]]
    return np.array(values, dtype=np.float32)
