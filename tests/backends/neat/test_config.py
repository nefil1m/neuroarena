import neat
import pytest

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


def test_hyperparameter_override_reaches_genome_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"weight_mutate_rate": 0.99},
    )
    assert config.genome_config.weight_mutate_rate == 0.99


def test_hyperparameter_override_reaches_species_set_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"compatibility_threshold": 5.5},
    )
    assert config.species_set_config.compatibility_threshold == 5.5


def test_hyperparameter_override_reaches_stagnation_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"max_stagnation": 42},
    )
    assert config.stagnation_config.max_stagnation == 42


def test_hyperparameter_override_reaches_reproduction_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"elitism": 4},
    )
    assert config.reproduction_config.elitism == 4


def test_hyperparameter_override_reaches_top_level_neat_section() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"fitness_threshold": 500.0},
    )
    assert config.fitness_threshold == 500.0


def test_unknown_hyperparameter_override_raises() -> None:
    with pytest.raises(ValueError, match="unknown NEAT hyperparameter"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"not_a_real_parameter": 1},
        )


def test_reserved_key_override_raises() -> None:
    with pytest.raises(ValueError, match="pop_size"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"pop_size": 999},
        )


def test_no_overrides_behaves_exactly_as_before() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.pop_size == 25


def test_internal_id_allocator_override_is_rejected() -> None:
    # `node_indexer` is neat-python's own internal node-key allocator, not a declared
    # hyperparameter — a naive `hasattr` check would accept it (and corrupt structural
    # mutation's id bookkeeping); it must be rejected the same way an unknown key is.
    with pytest.raises(ValueError, match="unknown NEAT hyperparameter"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"node_indexer": 99},
        )


def test_internal_method_override_is_rejected() -> None:
    # `save` is a bound method on `neat.Config`, not a hyperparameter — a naive `hasattr`
    # check would accept it and silently clobber the method with a non-callable value.
    with pytest.raises(ValueError, match="unknown NEAT hyperparameter"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"save": 1},
        )
