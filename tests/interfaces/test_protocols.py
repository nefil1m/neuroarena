import numpy as np
import pytest

from neuroarena.interfaces import (
    Environment,
    Model,
    Objective,
    Trainer,
    TrainingUpdate,
)
from tests.interfaces.doubles import DummyEnvironment, DummyModel


def test_dummy_environment_satisfies_environment_protocol():
    assert isinstance(DummyEnvironment(), Environment)


def test_dummy_model_satisfies_model_protocol():
    from neuroarena.interfaces import Box

    assert isinstance(DummyModel(Box(-1.0, 1.0, (3,)), Box(-1.0, 1.0, (2,))), Model)


def test_bare_object_is_not_an_environment():
    class NotEnv:
        pass

    assert not isinstance(NotEnv(), Environment)


def test_environment_step_is_a_four_tuple_without_reward():
    env = DummyEnvironment()
    env.reset(seed=0)
    result = env.step(np.zeros(2, dtype=np.float32))
    assert len(result) == 4
    obs, terminated, truncated, info = result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_objective_and_trainer_are_runtime_checkable():
    # Protocols with no implementation still expose isinstance via runtime_checkable.
    assert not isinstance(object(), Objective)
    assert not isinstance(object(), Trainer)


def test_training_update_is_frozen_and_defaults_schema_version():
    u = TrainingUpdate(
        progress_index=1,
        best_fitness=1.0,
        mean_fitness=0.5,
        worst_fitness=0.0,
        population_size=50,
        champion_metrics={"distance": 12.0},
        sim_time=3.0,
        wall_time=0.2,
    )
    assert u.schema_version == 1
    with pytest.raises(Exception):  # noqa: B017
        u.progress_index = 2  # type: ignore[misc]
