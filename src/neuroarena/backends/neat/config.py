"""Builds a `neat.Config` sized to a given `Environment`'s observation/action spaces, from
a bundled stock-`neat-python` template. See `../../../docs/phases/phase-4-learning-backend-neat.md`
for why the template's own defaults are used as-is (untuned starting point, tunable once
there's a real training loop to observe) and why output activation is `tanh` (smooth
[-1, 1] mapping, no separate clipping step)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping

import neat

from neuroarena.interfaces.spaces import Box

DEFAULT_CONFIG_TEXT = """\
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
pop_size              = 150
reset_on_extinction   = True
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.5

enabled_default            = True
enabled_mutate_rate        = 0.01
enabled_rate_to_true_add   = 0.0
enabled_rate_to_false_add  = 0.0

feed_forward            = True
initial_connection      = full

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = 10
num_outputs             = 2

single_structural_mutation = False
structural_mutation_surer  = default

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism             = 2
survival_threshold  = 0.2
min_species_size    = 2
"""
"""Stock neat-python example config (activation swapped to tanh per the Phase 4 decision).
Written to a temp file at build time — see `build_neat_config` — rather than shipped as a
packaged data file, so it stays a plain, diffable, version-controlled Python string."""

_RESERVED_OVERRIDE_KEYS = {"pop_size", "num_inputs", "num_outputs", "input_keys", "output_keys"}
"""These are set from `build_neat_config`'s own `population_size`/space arguments, which are
dedicated Phase 5 RunConfig knobs in their own right (`population_size`) or derived from the
Environment (`num_inputs`/`num_outputs`/...) — never from the generic `neat_hyperparameters`
override dict, so a caller can't accidentally fight the dedicated knob with a same-named
override."""

_TOP_LEVEL_HYPERPARAMETERS = {
    "fitness_criterion",
    "fitness_threshold",
    "reset_on_extinction",
    "no_fitness_termination",
}
"""The `[NEAT]` section's own keys, stored directly on `neat.Config` (not on one of its four
sub-configs) — see `neat.Config.__init__`."""

_SUBCONFIG_ATTRS = (
    "genome_config",
    "species_set_config",
    "stagnation_config",
    "reproduction_config",
)


def build_neat_config(
    observation_space: Box,
    action_space: Box,
    population_size: int,
    hyperparameter_overrides: Mapping[str, float | int | bool | str] | None = None,
) -> neat.Config:
    """A `neat.Config` whose genome shape matches `observation_space`/`action_space`, whose
    population size matches `population_size`, and whose hyperparameters match the bundled
    template's defaults except where `hyperparameter_overrides` says otherwise (Phase 5's
    `RunConfig.neat_hyperparameters` knob — see the Phase 5 doc's "NEAT hyperparameters
    become configurable" requirement)."""
    fd, path = tempfile.mkstemp(suffix=".cfg", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(DEFAULT_CONFIG_TEXT)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            path,
        )
    finally:
        os.unlink(path)

    num_inputs = observation_space.shape[0]
    num_outputs = action_space.shape[0]
    config.genome_config.num_inputs = num_inputs
    config.genome_config.num_outputs = num_outputs
    config.genome_config.input_keys = [-i - 1 for i in range(num_inputs)]
    config.genome_config.output_keys = list(range(num_outputs))
    config.pop_size = population_size

    if hyperparameter_overrides:
        _apply_hyperparameter_overrides(config, hyperparameter_overrides)

    return config


def _apply_hyperparameter_overrides(
    config: neat.Config, overrides: Mapping[str, float | int | bool | str]
) -> None:
    for key, value in overrides.items():
        if key in _RESERVED_OVERRIDE_KEYS:
            raise ValueError(
                f"{key!r} is set via build_neat_config's own population_size/space "
                "arguments, not hyperparameter_overrides"
            )
        if key in _TOP_LEVEL_HYPERPARAMETERS:
            setattr(config, key, value)
            continue
        for attr in _SUBCONFIG_ATTRS:
            subconfig = getattr(config, attr)
            if hasattr(subconfig, key):
                setattr(subconfig, key, value)
                break
        else:
            raise ValueError(f"unknown NEAT hyperparameter override: {key!r}")
