"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating
each genome in isolation on a shared `Environment` under a per-generation step budget.
See `../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this
implements, in particular why track switches only take effect at a generation boundary
and why a force-truncated genome is scored on partial progress rather than penalised."""

from __future__ import annotations

import copy
import itertools
import numbers
import pickle
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import neat

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.backends.neat.evaluation import EvaluationResult, StepBudget, evaluate_genome
from neuroarena.backends.neat.model import GenomeModel
from neuroarena.interfaces.compat import IncompatibleDescriptorsError
from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.interfaces.spaces import Box

if TYPE_CHECKING:
    from neuroarena.config import RunConfig
    from neuroarena.interfaces.protocols import Environment, Objective

# 2: adds `ancestors`, `total_sim_steps`, `champion_dir` and the observation/action
# descriptors, and drops the (unpicklable-from-3.14) `itertools.count` species indexer.
_CHECKPOINT_SCHEMA_VERSION = 2


class NeatTrainer:
    """Satisfies `Trainer`. `make_env` is rebuilt once per generation (not per genome) so
    every genome within one generation runs on the same `Environment` instance — cheap
    (Phase 3's `CarEnvironment.reset()` already gives each genome a fresh `Game`) and the
    natural point for a live track switch (Phase 5) to take effect."""

    def __init__(
        self,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
        *,
        population_size: int | None = None,
        neat_config: neat.Config | None = None,
        champion_dir: Path | None = None,
    ) -> None:
        self._make_env = make_env
        self._objective = objective
        self._config = config
        self._champion_dir = champion_dir
        if champion_dir is not None:
            champion_dir.mkdir(parents=True, exist_ok=True)

        resolved_population_size = (
            population_size if population_size is not None else config.population_size
        )

        env = make_env()
        if neat_config is None:
            # `build_neat_config` sizes a genome to concrete continuous bounds; NEAT (as
            # used by this backend) has no notion of a `Discrete` observation/action space.
            observation_space, action_space = env.observation_space, env.action_space
            if not isinstance(observation_space, Box) or not isinstance(action_space, Box):
                raise TypeError("NeatTrainer requires Box observation/action spaces")
            neat_config = build_neat_config(
                observation_space,
                action_space,
                resolved_population_size,
                hyperparameter_overrides=config.neat_hyperparameters or None,
            )
        self._neat_config = neat_config
        self._population = neat.Population(self._neat_config)
        self._env = env
        self._generation = 0
        self._total_sim_steps = 0.0
        self._wall_start = time.perf_counter()

    def run(self) -> Iterator[TrainingUpdate]:
        while True:
            update = self._run_one_generation()
            yield update
            if self._should_auto_stop(update):
                return

    def _should_auto_stop(self, update: TrainingUpdate) -> bool:
        """Phase 5's run-level automatic stop: checked once per generation boundary,
        alongside the existing per-generation step ceiling. Distinct from
        `Objective.should_stop()`, which ends one episode, not the run."""
        max_generations = self._config.max_generations
        if max_generations is not None and update.progress_index + 1 >= max_generations:
            return True
        target_fitness = self._config.target_fitness
        if target_fitness is not None and update.best_fitness >= target_fitness:
            return True
        return False

    def _run_one_generation(self) -> TrainingUpdate:
        self._env = self._make_env()
        step_budget = StepBudget(remaining=self._config.max_generation_steps)
        fitnesses: dict[int, float] = {}
        champion = _ChampionTracker()

        def fitness_function(genomes: list[tuple[int, Any]], neat_config: neat.Config) -> None:
            for genome_id, genome in genomes:
                model = GenomeModel(
                    genome, neat_config, self._env.observation_space, self._env.action_space
                )
                seed = self._config.master_seed + genome_id
                result = evaluate_genome(model, self._env, self._objective, step_budget, seed)
                genome.fitness = result.fitness
                fitnesses[genome_id] = result.fitness
                champion.consider(genome, result)

        self._population.run(fitness_function, 1)

        steps_used = self._config.max_generation_steps - step_budget.remaining
        self._total_sim_steps += steps_used

        if self._champion_dir is not None and champion.genome is not None:
            path = self._champion_dir / f"gen_{self._generation:05d}.pkl"
            path.write_bytes(pickle.dumps(champion.genome))

        values = list(fitnesses.values())
        update = TrainingUpdate(
            progress_index=self._generation,
            best_fitness=max(values),
            mean_fitness=sum(values) / len(values),
            worst_fitness=min(values),
            population_size=len(values),
            champion_metrics=champion.metrics(),
            sim_time=self._total_sim_steps,
            wall_time=time.perf_counter() - self._wall_start,
        )
        self._generation += 1
        return update

    def save_checkpoint(self, path: Path) -> None:
        """Pickles the NEAT-specific state (Phase 4 doc: "population, genomes, innovation
        history"), plus Python's global `random` state so a resumed run's mutation/crossover
        draws continue the same sequence rather than silently diverging. Checkpoint file
        format/retention is Phase 6's concern; this is a working save/load pair, not a claim
        on the eventual platform format."""
        # `DefaultSpeciesSet.indexer` is an `itertools.count`, which pickle supports only under
        # a DeprecationWarning today and not at all from Python 3.14 on (this project allows
        # `>=3.12`). It carries no information beyond "next unused species id", so it is
        # stripped from a shallow copy of the species set here — leaving the live object
        # untouched — and re-derived from the restored species keys in `load_checkpoint`.
        # `DefaultReproduction.genome_indexer` gets the same treatment for free: `reproduction`
        # is not pickled at all, only its `ancestors` lineage map, which is real data.
        species = copy.copy(self._population.species)
        species.indexer = None
        # `DefaultGenomeConfig.node_indexer` is the third such counter, lazily created the first
        # time a structural mutation needs a new node key. Same treatment, same re-derivation.
        neat_config = copy.copy(self._neat_config)
        neat_config.genome_config = copy.copy(self._neat_config.genome_config)
        neat_config.genome_config.node_indexer = None
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "neat_config": neat_config,
            "population": self._population.population,
            "species": species,
            "ancestors": self._population.reproduction.ancestors,
            "generation": self._generation,
            "random_state": random.getstate(),
            # Cumulative simulated steps must survive a resume, or a resumed run's reported
            # `sim_time` would regress to zero. `wall_time` is deliberately NOT persisted: it
            # measures this process's elapsed time, not the run's.
            "total_sim_steps": self._total_sim_steps,
            "champion_dir": None if self._champion_dir is None else str(self._champion_dir),
            # Phase 0: "a checkpoint records the observation- and action-space descriptors the
            # model was trained against"; `load_checkpoint` refuses a mismatched environment.
            "observation_space": self._env.observation_space,
            "action_space": self._env.action_space,
        }
        path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
    ) -> NeatTrainer:
        # Checkpoints are produced by `save_checkpoint` above and read back on the same
        # trusted local filesystem (this backend has no notion of loading a checkpoint from
        # an untrusted/remote source) — pickle is used because NEAT genomes/species/config
        # objects aren't trivially JSON-serializable, matching the existing champion-genome
        # pickling in `_run_one_generation`.
        payload = pickle.loads(path.read_bytes())
        version = payload.get("schema_version")
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unreadable NEAT checkpoint schema_version {version!r} "
                f"(supports {_CHECKPOINT_SCHEMA_VERSION})"
            )
        neat_config = payload["neat_config"]
        # `cls(...)` below runs `__init__`, which unconditionally builds a throwaway
        # `neat.Population(neat_config)` (initial_state=None) — genome weight/bias
        # initialization there consumes draws from the global `random` module. That
        # population is discarded a line later, but the RNG draws it consumed are not
        # undone, so restoring the saved state before `cls(...)` would let it drift again
        # before the real, resumed population is built. Restore the saved state right here,
        # immediately before constructing the REAL population, so the global RNG is exactly
        # at `payload["random_state"]` when the first real mutation/crossover draw happens.
        trainer = cls(make_env, objective, config, neat_config=neat_config)
        _check_descriptors(payload, trainer._env)
        random.setstate(payload["random_state"])
        trainer._population = neat.Population(
            neat_config,
            initial_state=(payload["population"], payload["species"], payload["generation"]),
        )
        # `neat.Population.__init__` always builds a FRESH `DefaultReproduction` and therefore a
        # fresh `genome_indexer = count(1)`, even when resuming from `initial_state`. Left alone,
        # the next generation's offspring get ids that are already in use by the restored
        # elites, and since `DefaultReproduction.reproduce` writes elites and offspring into the
        # same dict, an elite is silently overwritten and the population shrinks below
        # `pop_size`. Re-derive both counters from the restored keys: monotone by construction,
        # so no counter needs to live in the payload (see `save_checkpoint`).
        population = trainer._population
        population.reproduction.genome_indexer = itertools.count(
            max(payload["population"], default=0) + 1
        )
        population.reproduction.ancestors = payload["ancestors"]
        population.species.indexer = itertools.count(
            max((s.key for s in population.species.species.values()), default=0) + 1
        )
        # NEAT seeds `node_indexer` lazily from a single genome's own node keys, so leaving it
        # `None` would let a resumed run hand out node keys that already exist in *other*
        # genomes. Seed it from the whole restored population instead.
        neat_config.genome_config.node_indexer = itertools.count(
            max(
                (key for genome in payload["population"].values() for key in genome.nodes),
                default=0,
            )
            + 1
        )
        trainer._generation = payload["generation"]
        trainer._total_sim_steps = payload["total_sim_steps"]
        champion_dir = payload["champion_dir"]
        if champion_dir is not None:
            # The `Trainer` protocol's `load_checkpoint` signature takes no `champion_dir`, so
            # the path rides along in the payload; mirror `__init__`'s directory creation.
            trainer._champion_dir = Path(champion_dir)
            trainer._champion_dir.mkdir(parents=True, exist_ok=True)
        return trainer


def _check_descriptors(payload: dict[str, Any], env: Environment) -> None:
    """Phase 0's resume-safety gate: a checkpoint records the observation/action descriptors it
    was trained against, and loading it into a shape-mismatched environment is refused. Descriptor
    equality is the full field-wise comparison, exactly as `interfaces.compat.check_compatibility`
    applies it to a `Model` — that function takes a `Model`, which a checkpoint payload is not,
    so the same two comparisons are spelled out here rather than reused."""
    if payload["observation_space"] != env.observation_space:
        raise IncompatibleDescriptorsError(
            f"observation space mismatch: checkpoint {payload['observation_space']!r} "
            f"!= environment {env.observation_space!r}"
        )
    if payload["action_space"] != env.action_space:
        raise IncompatibleDescriptorsError(
            f"action space mismatch: checkpoint {payload['action_space']!r} "
            f"!= environment {env.action_space!r}"
        )


class _ChampionTracker:
    """Tracks the best-fitness genome seen so far within one generation's fitness_function
    call, and the `info` dict from its evaluation, for `TrainingUpdate.champion_metrics`."""

    def __init__(self) -> None:
        self.genome: Any = None
        self._fitness = float("-inf")
        self._info: dict[str, Any] = {}

    def consider(self, genome: Any, result: EvaluationResult) -> None:
        if result.fitness > self._fitness:
            self._fitness = result.fitness
            self.genome = genome
            self._info = result.final_info

    def metrics(self) -> dict[str, float]:
        # Derived from whatever `info` the champion's own evaluation returned, never from a
        # hard-coded key set: Phase 0 says the base interface knows nothing about `info`'s keys,
        # so this backend must work unchanged against a future second game's Environment.
        # Non-numeric values are dropped, since `TrainingUpdate.champion_metrics` is
        # `dict[str, float]` — for Phase 3's car that excludes `track_id`, an identifier rather
        # than a metric, and keeps exactly `crashed` / `progress` / `lap_progress`. The test is
        # `numbers.Real` rather than `bool | int | float` so that an environment reporting numpy
        # scalars (`np.float32`, `np.bool_`) is covered too, while arrays and strings are not.
        return {k: float(v) for k, v in self._info.items() if isinstance(v, numbers.Real)}
