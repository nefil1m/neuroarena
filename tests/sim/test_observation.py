import numpy as np
import pytest

from neuroarena.interfaces.spaces import Box
from neuroarena.sim.observation import (
    SensorConfig,
    build_observation,
    observation_space_for,
    speed_norm,
)
from neuroarena.sim.physics import CarState, PhysicsConstants


def test_speed_norm_forward_reverse_and_rest() -> None:
    constants = PhysicsConstants()
    assert speed_norm(750.0, constants) == pytest.approx(1.0)
    assert speed_norm(-300.0, constants) == pytest.approx(-300.0 / 750.0)
    assert speed_norm(0.0, constants) == pytest.approx(0.0)


def test_observation_space_for_default_sensor_config() -> None:
    assert observation_space_for(SensorConfig()) == Box(-1.0, 1.0, (10,))


def test_observation_space_for_custom_ray_count() -> None:
    config = SensorConfig(ray_angles_deg=(0.0, 90.0))
    assert observation_space_for(config) == Box(-1.0, 1.0, (5,))


def test_build_observation_shape_and_dtype() -> None:
    car = CarState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    obs = build_observation(car, [], (0.0, 0.0), SensorConfig(), PhysicsConstants())
    assert obs.shape == (10,)
    assert obs.dtype == np.float32


def test_build_observation_embeds_speed_and_last_action_in_the_last_three_slots() -> None:
    car = CarState(x=0.0, y=0.0, heading=0.0, speed=375.0)
    obs = build_observation(car, [], (0.5, -0.25), SensorConfig(), PhysicsConstants())
    assert obs[-3] == pytest.approx(375.0 / 750.0)
    assert obs[-2] == pytest.approx(0.5)
    assert obs[-1] == pytest.approx(-0.25)


def test_build_observation_range_grows_with_speed() -> None:
    constants = PhysicsConstants()
    car_radius = constants.car_width / 2
    # Just past the base (speed=0) range of 300, so only a faster car's extended range sees it.
    wall_x = 300.0 + car_radius + 50.0
    boundary = [((wall_x, -1000.0), (wall_x, 1000.0))]
    forward_index = SensorConfig().ray_angles_deg.index(0.0)

    slow = build_observation(
        CarState(x=0.0, y=0.0, heading=0.0, speed=0.0),
        boundary,
        (0.0, 0.0),
        SensorConfig(),
        constants,
    )
    fast = build_observation(
        CarState(x=0.0, y=0.0, heading=0.0, speed=750.0),
        boundary,
        (0.0, 0.0),
        SensorConfig(),
        constants,
    )
    assert slow[forward_index] == pytest.approx(0.0)
    assert fast[forward_index] > 0.0
