import numpy as np
import pytest

from neuroarena.config import RunConfig
from neuroarena.interfaces.protocols import Environment
from neuroarena.interfaces.spaces import Box
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.sim.track import Facing, GridCell, TileKind, Track, track_loop_length


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


def _track() -> Track:
    return Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)


def test_car_environment_satisfies_the_environment_protocol() -> None:
    assert isinstance(CarEnvironment(_track(), track_id="abc123"), Environment)


def test_observation_and_action_space_descriptors() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    assert env.observation_space == Box(-1.0, 1.0, (10,))
    assert env.action_space == Box(-1.0, 1.0, (2,))


def test_reset_returns_expected_shape_and_zeroed_last_action() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    obs = env.reset()
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert obs[-2] == 0.0
    assert obs[-1] == 0.0


def test_step_returns_environment_protocol_shaped_tuple() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    obs, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert obs.shape == (10,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["track_id"] == "abc123"
    assert info["crashed"] is terminated


def test_terminated_fires_on_boundary_contact() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    terminated = False
    info: dict[str, object] = {}
    for _ in range(600):
        _, terminated, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated
    assert info["crashed"] is True


def test_truncated_fires_at_max_episode_steps() -> None:
    config = CarEnvironmentConfig(max_episode_steps=5)
    env = CarEnvironment(_track(), track_id="abc123", config=config)
    env.reset()
    truncated = False
    for _ in range(5):
        _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not terminated  # stationary car (zero throttle) never crashes
    assert truncated


def test_progress_increases_when_driving_forward() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    _, terminated1, _, info1 = env.step(np.array([0.0, 1.0], dtype=np.float32))
    _, terminated2, _, info2 = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert not terminated1 and not terminated2
    assert info2["progress"] > info1["progress"]


def test_lap_progress_is_progress_over_loop_length() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    _, terminated, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert not terminated
    assert info["lap_progress"] == pytest.approx(info["progress"] / track_loop_length(_track()))


def test_reset_clears_progress_from_a_previous_episode() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    env.reset()
    _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not terminated
    assert info["progress"] == pytest.approx(0.0)


def test_from_run_config_uses_platform_defaults_by_default() -> None:
    car_config = CarEnvironmentConfig.from_run_config(RunConfig())
    assert car_config == CarEnvironmentConfig()


def test_from_run_config_carries_a_non_default_sensor_config() -> None:
    sensor_config = SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0))
    car_config = CarEnvironmentConfig.from_run_config(RunConfig(sensor_config=sensor_config))
    assert car_config.sensor_config == sensor_config


def test_from_run_config_carries_physics_constants_and_episode_limit() -> None:
    physics = PhysicsConstants(max_speed=500.0)
    car_config = CarEnvironmentConfig.from_run_config(
        RunConfig(physics_constants=physics, max_episode_steps=100)
    )
    assert car_config.physics_constants == physics
    assert car_config.max_episode_steps == 100


def test_from_run_config_env_has_a_narrower_observation_space() -> None:
    sensor_config = SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0))  # 3 rays, not 7
    car_config = CarEnvironmentConfig.from_run_config(RunConfig(sensor_config=sensor_config))
    env = CarEnvironment(_track(), track_id="abc123", config=car_config)
    assert env.observation_space == Box(-1.0, 1.0, (6,))  # 3 rays + speed + 2 last-action
