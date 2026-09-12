import json

import pytest

from neuroarena.config import SCHEMA_VERSION, RunConfig
from neuroarena.config.serialization import (
    UnknownSchemaVersionError,
    dumps,
    loads,
    migrate,
)


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
