from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.spaces import Box
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from tests.interfaces.doubles import DummyObjective


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


def _make_car_env(config: RunConfig) -> CarEnvironment:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    car_config = CarEnvironmentConfig.from_run_config(config)
    return CarEnvironment(track, track_id="phase-5-e2e", config=car_config)


def test_a_run_config_with_several_knobs_set_trains_end_to_end() -> None:
    config = RunConfig(
        sensor_config=SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0)),
        max_episode_steps=50,
        max_generation_steps=500,
        max_generations=3,
        population_size=6,
        neat_hyperparameters={"compatibility_threshold": 4.0},
    )
    trainer = NeatTrainer(lambda: _make_car_env(config), DummyObjective(), config)

    # The narrower sensor_config actually reshaped the env this trainer is training on:
    # 3 rays + speed + 2 last-action = 6, not the platform-default 10.
    assert trainer._env.observation_space == Box(-1.0, 1.0, (6,))
    # The hyperparameter override actually reached the neat.Config the trainer built.
    assert trainer._neat_config.species_set_config.compatibility_threshold == 4.0

    updates = list(trainer.run())

    # max_generations stopped the run at exactly 3 generations.
    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert u.population_size == 6
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness
