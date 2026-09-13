from __future__ import annotations

import dataclasses
import json
from typing import Any

from neuroarena.config.run_config import SCHEMA_VERSION, RunConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


class UnknownSchemaVersionError(ValueError):
    """Config declares a schema_version this build cannot read."""


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate a decoded config dict toward SCHEMA_VERSION.

    Identity today (only v1 exists). The first `vN -> vN+1` step is added here
    at the first schema bump; see phase-0-architecture.md open questions.
    `loads` stamps `schema_version` to `SCHEMA_VERSION` on the dict this
    returns, so each migration step only needs to transform the other fields.
    """
    return raw


def dumps(config: RunConfig) -> str:
    return json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True)


def loads(text: str) -> RunConfig:
    raw = json.loads(text)
    version = raw.get("schema_version")
    if not isinstance(version, int) or not (1 <= version <= SCHEMA_VERSION):
        raise UnknownSchemaVersionError(
            f"schema_version {version!r} is not readable by this build "
            f"(supports 1..{SCHEMA_VERSION})"
        )
    if version < SCHEMA_VERSION:
        raw = migrate(raw)
        raw["schema_version"] = SCHEMA_VERSION
    known = {f.name for f in dataclasses.fields(RunConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if "sensor_config" in kwargs:
        kwargs["sensor_config"] = _sensor_config_from_dict(kwargs["sensor_config"])
    if "physics_constants" in kwargs:
        kwargs["physics_constants"] = _physics_constants_from_dict(kwargs["physics_constants"])
    return RunConfig(**kwargs)


def _known_fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _sensor_config_from_dict(raw: Any) -> SensorConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"sensor_config must be a dict, got {type(raw).__name__}: {raw!r}")
    # Same tolerant-unknown-key filtering as the outer `known` filter above, applied to the
    # nested dict — an unknown key inside `sensor_config` is silently dropped rather than
    # raising a `TypeError` about an unexpected keyword argument.
    kwargs = {k: v for k, v in raw.items() if k in _known_fields(SensorConfig)}
    if "ray_angles_deg" in kwargs:
        # `json.loads` decodes a JSON array as a `list`, but `SensorConfig.ray_angles_deg` is
        # typed `tuple[float, ...]` — reconstruct it as a tuple explicitly, or a round-tripped
        # RunConfig would carry a list where a tuple is expected (breaks equality/hashing
        # elsewhere `SensorConfig` is compared or hashed). Only convert when the key is
        # actually present — a partial dict missing it should let `SensorConfig`'s own
        # default apply, not raise a `KeyError`.
        kwargs["ray_angles_deg"] = tuple(kwargs["ray_angles_deg"])
    return SensorConfig(**kwargs)


def _physics_constants_from_dict(raw: Any) -> PhysicsConstants:
    if not isinstance(raw, dict):
        raise TypeError(f"physics_constants must be a dict, got {type(raw).__name__}: {raw!r}")
    kwargs = {k: v for k, v in raw.items() if k in _known_fields(PhysicsConstants)}
    return PhysicsConstants(**kwargs)
