import neat

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.interfaces.spaces import Box


def test_config_sizes_inputs_and_outputs_to_the_given_spaces() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.genome_config.num_inputs == 10
    assert config.genome_config.num_outputs == 2
    assert config.genome_config.input_keys == [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    assert config.genome_config.output_keys == [0, 1]
    assert config.pop_size == 25


def test_config_sizes_to_a_non_default_sensor_shape() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (5,)), Box(-1.0, 1.0, (2,)), population_size=10)
    assert config.genome_config.num_inputs == 5
    assert len(config.genome_config.input_keys) == 5


def test_config_uses_tanh_output_activation() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.genome_config.activation_default == "tanh"


def test_config_produces_a_working_genome_and_network() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=1)
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    output = net.activate([0.0] * 10)
    assert len(output) == 2
    assert all(-1.0 <= v <= 1.0 for v in output)
