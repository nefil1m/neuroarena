from typing import Any

import neat
import numpy as np

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.backends.neat.model import GenomeModel
from neuroarena.interfaces.protocols import Model
from neuroarena.interfaces.spaces import Box


def _genome(config: neat.Config) -> Any:
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    return genome


def test_genome_model_satisfies_the_model_protocol() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    assert isinstance(model, Model)
    assert model.observation_space == obs_space
    assert model.action_space == action_space


def test_act_returns_action_shaped_output_within_bounds() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    action = model.act(np.zeros(10, dtype=np.float32))
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_reset_is_a_no_op_and_does_not_raise() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    model.reset()
