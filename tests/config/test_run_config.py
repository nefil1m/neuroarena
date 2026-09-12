from neuroarena.config import SCHEMA_VERSION, RunConfig


def test_defaults():
    c = RunConfig()
    assert c.master_seed == 0
    assert c.schema_version == SCHEMA_VERSION


def test_fields_are_settable():
    assert RunConfig(master_seed=42).master_seed == 42


def test_equality_is_field_wise():
    assert RunConfig(master_seed=1) == RunConfig(master_seed=1)
    assert RunConfig(master_seed=1) != RunConfig(master_seed=2)
