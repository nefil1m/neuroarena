from neuroarena.config import SCHEMA_VERSION, RunConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


def test_defaults():
    c = RunConfig()
    assert c.master_seed == 0
    assert c.schema_version == SCHEMA_VERSION


def test_fields_are_settable():
    assert RunConfig(master_seed=42).master_seed == 42


def test_equality_is_field_wise():
    assert RunConfig(master_seed=1) == RunConfig(master_seed=1)
    assert RunConfig(master_seed=1) != RunConfig(master_seed=2)


def test_max_generation_steps_default() -> None:
    assert RunConfig().max_generation_steps == 300_000


def test_max_generation_steps_is_settable() -> None:
    assert RunConfig(max_generation_steps=1000).max_generation_steps == 1000


def test_population_size_default() -> None:
    assert RunConfig().population_size == 150


def test_track_id_defaults_to_none() -> None:
    assert RunConfig().track_id is None


def test_track_id_is_settable() -> None:
    assert RunConfig(track_id="abc123").track_id == "abc123"


def test_headless_defaults_to_true() -> None:
    assert RunConfig().headless is True


def test_sim_speed_default() -> None:
    assert RunConfig().sim_speed == 1.0


def test_max_episode_steps_default() -> None:
    assert RunConfig().max_episode_steps == 3000


def test_max_generations_and_target_fitness_default_to_none() -> None:
    config = RunConfig()
    assert config.max_generations is None
    assert config.target_fitness is None


def test_max_generations_and_target_fitness_are_settable() -> None:
    config = RunConfig(max_generations=50, target_fitness=1000.0)
    assert config.max_generations == 50
    assert config.target_fitness == 1000.0


def test_sensor_config_defaults_to_the_platform_standard() -> None:
    assert RunConfig().sensor_config == SensorConfig()


def test_physics_constants_defaults_to_the_platform_standard() -> None:
    assert RunConfig().physics_constants == PhysicsConstants()


def test_sensor_config_and_physics_constants_are_settable() -> None:
    config = RunConfig(
        sensor_config=SensorConfig(ray_angles_deg=(-30.0, 30.0)),
        physics_constants=PhysicsConstants(max_speed=500.0),
    )
    assert config.sensor_config.ray_angles_deg == (-30.0, 30.0)
    assert config.physics_constants.max_speed == 500.0


def test_neat_hyperparameters_defaults_to_empty_dict() -> None:
    assert RunConfig().neat_hyperparameters == {}


def test_neat_hyperparameters_is_settable() -> None:
    config = RunConfig(neat_hyperparameters={"weight_mutate_rate": 0.9})
    assert config.neat_hyperparameters == {"weight_mutate_rate": 0.9}


def test_neat_hyperparameters_defaults_are_independent_between_instances() -> None:
    a, b = RunConfig(), RunConfig()
    a.neat_hyperparameters["x"] = 1
    assert b.neat_hyperparameters == {}


def test_checkpoint_every_n_generations_default() -> None:
    assert RunConfig().checkpoint_every_n_generations == 10


def test_champion_retention_cap_defaults_to_none() -> None:
    assert RunConfig().champion_retention_cap is None


def test_champion_retention_cap_is_settable() -> None:
    assert RunConfig(champion_retention_cap=20).champion_retention_cap == 20
