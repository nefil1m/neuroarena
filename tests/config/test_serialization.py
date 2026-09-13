import json

import pytest

from neuroarena.config import SCHEMA_VERSION, RunConfig
from neuroarena.config.serialization import (
    UnknownSchemaVersionError,
    dumps,
    loads,
    migrate,
)
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


def test_roundtrip_preserves_value():
    c = RunConfig(master_seed=7)
    assert loads(dumps(c)) == c


def test_dumps_emits_json_with_schema_version():
    raw = json.loads(dumps(RunConfig()))
    assert raw["schema_version"] == SCHEMA_VERSION


def test_loads_rejects_future_version():
    text = json.dumps({"schema_version": SCHEMA_VERSION + 1, "master_seed": 0})
    with pytest.raises(UnknownSchemaVersionError):
        loads(text)


def test_loads_rejects_version_below_one():
    text = json.dumps({"schema_version": 0, "master_seed": 0})
    with pytest.raises(UnknownSchemaVersionError):
        loads(text)


def test_loads_ignores_unknown_keys():
    text = json.dumps({"schema_version": SCHEMA_VERSION, "master_seed": 5, "future_field": 123})
    assert loads(text) == RunConfig(master_seed=5)


def test_migrate_is_identity_at_current_version():
    raw = {"schema_version": SCHEMA_VERSION, "master_seed": 1}
    assert migrate(dict(raw)) == raw


def test_roundtrip_reconstructs_sensor_config_as_a_real_dataclass() -> None:
    config = RunConfig(sensor_config=SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0)))
    restored = loads(dumps(config))
    assert isinstance(restored.sensor_config, SensorConfig)
    assert restored.sensor_config.ray_angles_deg == (-40.0, 0.0, 40.0)
    assert isinstance(restored.sensor_config.ray_angles_deg, tuple)


def test_roundtrip_reconstructs_physics_constants_as_a_real_dataclass() -> None:
    config = RunConfig(physics_constants=PhysicsConstants(max_speed=900.0))
    restored = loads(dumps(config))
    assert isinstance(restored.physics_constants, PhysicsConstants)
    assert restored.physics_constants.max_speed == 900.0


def test_roundtrip_preserves_value_with_nested_defaults() -> None:
    assert loads(dumps(RunConfig(master_seed=7))) == RunConfig(master_seed=7)


def test_loads_reconstructs_partial_sensor_config_using_its_own_default() -> None:
    # A hand-edited/corrupted config missing `ray_angles_deg` from `sensor_config` should
    # fall back to `SensorConfig`'s own default for that field, not raise a bare KeyError.
    text = json.dumps({"schema_version": SCHEMA_VERSION, "sensor_config": {"range_base": 100.0}})
    restored = loads(text)
    assert isinstance(restored.sensor_config, SensorConfig)
    assert restored.sensor_config.range_base == 100.0
    assert restored.sensor_config.ray_angles_deg == SensorConfig().ray_angles_deg


def test_loads_ignores_unknown_key_inside_sensor_config() -> None:
    text = json.dumps({"schema_version": SCHEMA_VERSION, "sensor_config": {"not_a_real_field": 1}})
    restored = loads(text)
    assert restored.sensor_config == SensorConfig()


def test_loads_ignores_unknown_key_inside_physics_constants() -> None:
    text = json.dumps(
        {"schema_version": SCHEMA_VERSION, "physics_constants": {"not_a_real_field": 1}}
    )
    restored = loads(text)
    assert restored.physics_constants == PhysicsConstants()


def test_loads_rejects_non_dict_sensor_config() -> None:
    text = json.dumps({"schema_version": SCHEMA_VERSION, "sensor_config": None})
    with pytest.raises(TypeError, match="sensor_config"):
        loads(text)


def test_loads_rejects_non_dict_physics_constants() -> None:
    text = json.dumps({"schema_version": SCHEMA_VERSION, "physics_constants": None})
    with pytest.raises(TypeError, match="physics_constants"):
        loads(text)
