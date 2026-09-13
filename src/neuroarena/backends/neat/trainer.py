"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating
each genome in isolation on a shared `Environment` under a per-generation step budget.
See `../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this
implements, in particular why track switches only take effect at a generation boundary
and why a force-truncated genome is scored on partial progress rather than penalised."""

from __future__ import annotations

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
from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.interfaces.spaces import Box

if TYPE_CHECKING:
    from neuroarena.config import RunConfig
    from neuroarena.interfaces.protocols import Environment, Objective

_CHECKPOINT_SCHEMA_VERSION = 1


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
        population_size: int = 150,
        neat_config: neat.Config | None = None,
        champion_dir: Path | None = None,
    ) -> None:
        self._make_env = make_env
        self._objective = objective
        self._config = config
        self._champion_dir = champion_dir
        if champion_dir is not None:
            champion_dir.mkdir(parents=True, exist_ok=True)

        env = make_env()
        if neat_config is None:
            # `build_neat_config` sizes a genome to concrete continuous bounds; NEAT (as
            # used by this backend) has no notion of a `Discrete` observation/action space.
            observation_space, action_space = env.observation_space, env.action_space
            if not isinstance(observation_space, Box) or not isinstance(action_space, Box):
                raise TypeError("NeatTrainer requires Box observation/action spaces")
            neat_config = build_neat_config(observation_space, action_space, population_size)
        self._neat_config = neat_config
        self._population = neat.Population(self._neat_config)
        self._env = env
        self._generation = 0
        self._total_sim_steps = 0.0
        self._wall_start = time.perf_counter()

    def run(self) -> Iterator[TrainingUpdate]:
        while True:
            yield self._run_one_generation()

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
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "neat_config": self._neat_config,
            "population": self._population.population,
            "species": self._population.species,
            "generation": self._generation,
            "random_state": random.getstate(),
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
        random.setstate(payload["random_state"])
        trainer._population = neat.Population(
            neat_config,
            initial_state=(payload["population"], payload["species"], payload["generation"]),
        )
        trainer._generation = payload["generation"]
        return trainer


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
        # Only float-valued info keys belong in TrainingUpdate.champion_metrics (Phase 0:
        # `dict[str, float]`) — `track_id` (str) is an identifier, not a metric, and is
        # excluded. Car-env-specific keys (Phase 3's `info` contract); a future second
        # game's Objective/info would need its own key set here.
        return {
            "crashed": float(self._info.get("crashed", 0.0)),
            "progress": float(self._info.get("progress", 0.0)),
            "lap_progress": float(self._info.get("lap_progress", 0.0)),
        }
