"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating
each genome in isolation on a shared `Environment` under a per-generation step budget.
See `../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this
implements, in particular why track switches only take effect at a generation boundary
and why a force-truncated genome is scored on partial progress rather than penalised."""

from __future__ import annotations

import pickle
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
        # Task 7. Declared here (rather than left off the class) so `NeatTrainer`
        # structurally satisfies the runtime_checkable `Trainer` protocol now.
        raise NotImplementedError("NeatTrainer checkpointing lands in Task 7")

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
    ) -> NeatTrainer:
        # Task 7. Declared here (rather than left off the class) so `NeatTrainer`
        # structurally satisfies the runtime_checkable `Trainer` protocol now.
        raise NotImplementedError("NeatTrainer checkpointing lands in Task 7")


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
