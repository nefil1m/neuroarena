import pytest

from neuroarena.interfaces import Box
from neuroarena.interfaces.compat import IncompatibleDescriptorsError, check_compatibility
from tests.interfaces.doubles import DummyEnvironment, DummyModel


def test_matching_descriptors_do_not_raise():
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    check_compatibility(model, env)


def test_observation_mismatch_raises_naming_observation():
    env = DummyEnvironment()
    model = DummyModel(Box(-1.0, 1.0, (7,)), env.action_space)
    with pytest.raises(IncompatibleDescriptorsError, match="observation"):
        check_compatibility(model, env)


def test_action_mismatch_raises_naming_action():
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, Box(-1.0, 1.0, (3,)))
    with pytest.raises(IncompatibleDescriptorsError, match="action"):
        check_compatibility(model, env)


def test_error_message_includes_both_descriptor_reprs():
    env = DummyEnvironment()
    model = DummyModel(Box(0.0, 1.0, (3,)), env.action_space)
    with pytest.raises(IncompatibleDescriptorsError) as exc:
        check_compatibility(model, env)
    msg = str(exc.value)
    assert "0.0" in msg and "-1.0" in msg
