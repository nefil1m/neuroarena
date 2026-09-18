import random
import threading
import time
import typing
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress, NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.compat import IncompatibleDescriptorsError
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate
from neuroarena.interfaces.spaces import Box, Space
from tests.interfaces.doubles import DummyEnvironment, DummyObjective


def test_neat_trainer_satisfies_the_trainer_protocol() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    assert isinstance(trainer, Trainer)


def test_run_yields_a_training_update_per_generation() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    updates = []
    run = trainer.run()
    for _ in range(3):
        updates.append(next(run))
    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert isinstance(u, TrainingUpdate)
        assert u.population_size == 5
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness


def test_generation_ceiling_bounds_total_steps_per_generation() -> None:
    # DummyEnvironment truncates each episode after 5 steps; 5 genomes x 5 steps = 25 steps
    # if unbounded. Cap the generation well below that and confirm it still completes.
    config = RunConfig(max_generation_steps=7)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    update = next(trainer.run())
    assert update.population_size == 5
    assert update.worst_fitness <= 5.0
    # The ceiling must actually bound the generation: `sim_time` counts environment steps
    # elapsed (this backend's documented convention), so it can never exceed the budget.
    # Without this the assertion above would also pass if the budget were ignored entirely.
    assert update.sim_time <= 7


def test_a_genome_finishing_shortens_the_whole_generation() -> None:
    class SucceedsAfterOneStep(DummyObjective):
        def should_stop(self) -> bool:
            return self._steps >= 1

    trainer = NeatTrainer(DummyEnvironment, SucceedsAfterOneStep(), RunConfig(), population_size=5)
    update = next(trainer.run())
    # Every genome's Objective is an independent deep copy of the same SucceedsAfterOneStep, so
    # every genome would succeed on its own first step. Round-robin processes genomes in order
    # within a round: the first genome takes its one step, calls should_stop() (True), and
    # immediately triggers the collective stop — every other genome is force-truncated before
    # it ever takes a step of its own. Exactly one env.step() is taken across the WHOLE
    # generation, regardless of genome ordering, so sim_time is exactly 1.0 — far below the 25
    # (5 genomes x DummyEnvironment's natural 5-step length) a fully-independent run would take.
    assert update.sim_time == 1.0


def test_champion_dir_saves_one_genome_file_per_generation(tmp_path: Path) -> None:
    champion_dir = tmp_path / "champions"
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(),
        population_size=5,
        champion_dir=champion_dir,
    )
    run = trainer.run()
    next(run)
    next(run)
    saved = sorted(p.name for p in champion_dir.iterdir())
    assert saved == ["gen_00000.pkl", "gen_00001.pkl"]


def test_checkpoint_round_trip_resumes_generation_count(tmp_path: Path) -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    run = trainer.run()
    next(run)
    next(run)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    restored_update = next(restored.run())
    assert restored_update.progress_index == 2  # continues from generation 2, not 0
    assert restored_update.population_size == 5


def test_checkpoint_round_trip_restores_exact_random_state(tmp_path: Path) -> None:
    # A regression test for a bug where `load_checkpoint` restored the saved global
    # `random` state too early: `NeatTrainer.__init__` (invoked internally by
    # `load_checkpoint`) builds a throwaway `neat.Population` whose genome
    # initialization consumes `random` draws, silently advancing the RNG past the
    # restored state before the real, resumed population replaced it. The global
    # `random` state right after `load_checkpoint` returns must exactly equal the state
    # captured at `save_checkpoint` time, so a resumed run's mutation/crossover draws
    # continue the exact same sequence rather than diverging.
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    run = trainer.run()
    next(run)
    next(run)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    state_at_save = random.getstate()

    # Perturb the global RNG between save and load so a load that fails to restore
    # state would leave behind some other (wrong) state rather than accidentally
    # matching by coincidence.
    random.random()

    NeatTrainer.load_checkpoint(checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig())
    assert random.getstate() == state_at_save


def test_checkpoint_round_trip_keeps_minting_fresh_genome_ids(tmp_path: Path) -> None:
    # A regression test for a bug where `load_checkpoint` rebuilt `neat.Population` with a
    # FRESH `DefaultReproduction`, resetting its `genome_indexer` to `count(1)`. New offspring
    # were then minted with IDs already in use by the restored elites, and since
    # `DefaultReproduction.reproduce` writes elites and offspring into the SAME dict, an elite
    # was silently overwritten and the population shrank below `pop_size`.
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=20)
    run = trainer.run()
    next(run)
    next(run)
    pre_keys = set(trainer._population.population)
    pre_max = max(pre_keys)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    restored_run = restored.run()
    first = next(restored_run)
    second = next(restored_run)

    # (a) the population never shrinks: the collision above only surfaces in the generation
    # AFTER the first post-resume reproduction, so both updates are checked.
    assert first.population_size == 20
    assert second.population_size == 20
    # (b) every genome alive after resuming is either one carried over from before the
    # checkpoint or a strictly-newer ID — never a recycled ID colliding with an old one.
    for key in restored._population.population:
        assert key in pre_keys or key > pre_max


def test_load_checkpoint_reconstructs_monotone_id_counters(tmp_path: Path) -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=20)
    run = trainer.run()
    next(run)
    next(run)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    population = restored._population
    assert next(population.reproduction.genome_indexer) > max(population.population)
    assert next(population.species.indexer) > max(
        s.key for s in population.species.species.values()
    )
    node_keys = {key for genome in population.population.values() for key in genome.nodes}
    assert next(restored._neat_config.genome_config.node_indexer) > max(node_keys)
    # Lineage is real data, not a counter, so it is persisted rather than re-derived: every
    # live genome still has its recorded ancestry after the round trip.
    assert set(population.population) <= set(population.reproduction.ancestors)


def test_save_checkpoint_emits_no_deprecation_warning(tmp_path: Path) -> None:
    # Pickling `itertools.count` objects (NEAT's genome/species indexers) raises a
    # DeprecationWarning today and a TypeError on Python 3.14; the checkpoint payload must
    # therefore carry no raw `itertools` objects.
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.save_checkpoint(tmp_path / "checkpoint.pkl")
    assert [w for w in caught if "itertools" in str(w.message)] == []


def test_checkpoint_round_trip_restores_champion_dir(tmp_path: Path) -> None:
    champion_dir = tmp_path / "champions"
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(),
        population_size=5,
        champion_dir=champion_dir,
    )
    next(trainer.run())
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    next(restored.run())
    assert sorted(p.name for p in champion_dir.iterdir()) == ["gen_00000.pkl", "gen_00001.pkl"]


def test_checkpoint_round_trip_recreates_a_missing_champion_dir(tmp_path: Path) -> None:
    champion_dir = tmp_path / "champions"
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(),
        population_size=5,
        champion_dir=champion_dir,
    )
    next(trainer.run())
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    (champion_dir / "gen_00000.pkl").unlink()
    champion_dir.rmdir()

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    next(restored.run())
    assert sorted(p.name for p in champion_dir.iterdir()) == ["gen_00001.pkl"]


def test_checkpoint_round_trip_continues_cumulative_sim_time(tmp_path: Path) -> None:
    config = RunConfig(max_generation_steps=10)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    run = trainer.run()
    next(run)
    before = next(run).sim_time
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig(max_generation_steps=10)
    )
    after = next(restored.run()).sim_time
    assert after > before  # cumulative sim_time continues, it does not restart from zero


def test_load_checkpoint_refuses_a_shape_mismatched_environment(tmp_path: Path) -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    class WiderEnvironment(DummyEnvironment):
        observation_space: Space = Box(-1.0, 1.0, (7,))

    with pytest.raises(IncompatibleDescriptorsError, match="observation space"):
        NeatTrainer.load_checkpoint(
            checkpoint_path, WiderEnvironment, DummyObjective(), RunConfig()
        )

    class DifferentActionEnvironment(DummyEnvironment):
        action_space: Space = Box(-1.0, 1.0, (4,))

    with pytest.raises(IncompatibleDescriptorsError, match="action space"):
        NeatTrainer.load_checkpoint(
            checkpoint_path, DifferentActionEnvironment, DummyObjective(), RunConfig()
        )


def test_champion_metrics_come_from_the_environments_own_info(tmp_path: Path) -> None:
    # The backend must not assume car-ness: champion metrics are whatever numeric keys the
    # environment's `info` dict actually carries, not a hard-coded car-specific key set.
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    update = next(trainer.run())
    assert set(update.champion_metrics) == {"t"}
    assert update.champion_metrics["t"] > 0.0


def test_champion_metrics_keep_only_the_numeric_info_values() -> None:
    class MixedInfoEnvironment(DummyEnvironment):
        def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, typing.Any]]:
            observation, terminated, truncated, _ = super().step(action)
            info: dict[str, typing.Any] = {
                "flag": True,
                "count": 3,
                "ratio": 0.25,
                "numpy_scalar": np.float32(1.5),
                "name": "not-a-metric",
                "vector": np.zeros(2),
            }
            return observation, terminated, truncated, info

    trainer = NeatTrainer(MixedInfoEnvironment, DummyObjective(), RunConfig(), population_size=5)
    update = next(trainer.run())
    assert update.champion_metrics == {
        "flag": 1.0,
        "count": 3.0,
        "ratio": 0.25,
        "numpy_scalar": 1.5,
    }


def test_checkpoint_rejects_unknown_schema_version(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema_version"):
        NeatTrainer.load_checkpoint(path, DummyEnvironment, DummyObjective(), RunConfig())


def test_population_size_falls_back_to_config_when_not_given() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(population_size=12))
    update = next(trainer.run())
    assert update.population_size == 12


def test_explicit_population_size_overrides_config() -> None:
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(population_size=12),
        population_size=4,
    )
    update = next(trainer.run())
    assert update.population_size == 4


def test_neat_hyperparameters_reach_the_built_neat_config() -> None:
    config = RunConfig(neat_hyperparameters={"compatibility_threshold": 7.5})
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    # White-box check: NeatTrainer has no public accessor for the underlying neat.Config,
    # and this is exactly the wiring this task adds, so the test reaches into `_neat_config`.
    assert trainer._neat_config.species_set_config.compatibility_threshold == 7.5


def test_run_stops_after_max_generations() -> None:
    config = RunConfig(max_generations=3)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1, 2]


def test_run_stops_once_target_fitness_is_reached() -> None:
    # DummyEnvironment truncates every episode after exactly 5 steps, DummyObjective's
    # fitness is steps survived, so every genome's fitness is 5.0 every generation —
    # best_fitness reaches 5.0 on generation 0 already.
    config = RunConfig(target_fitness=5.0)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0]


def test_run_with_neither_stop_condition_set_keeps_running() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    run = trainer.run()
    updates = [next(run) for _ in range(5)]
    assert [u.progress_index for u in updates] == [0, 1, 2, 3, 4]


def test_both_stop_conditions_set_whichever_triggers_first_wins() -> None:
    # target_fitness is unreachable (DummyObjective never exceeds 5.0), so max_generations
    # must be what stops the run.
    config = RunConfig(max_generations=2, target_fitness=1_000_000.0)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1]


def test_target_fitness_wins_over_a_distant_max_generations() -> None:
    # Mirror of the above: max_generations is set far higher than the run could ever reach
    # first, so target_fitness must be what stops the run.
    config = RunConfig(max_generations=1_000_000, target_fitness=5.0)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0]


def test_each_genome_gets_its_own_environment_instance() -> None:
    constructed: list[DummyEnvironment] = []

    def make_env() -> DummyEnvironment:
        env = DummyEnvironment()
        constructed.append(env)
        return env

    trainer = NeatTrainer(make_env, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    # Two non-genome instances (one in __init__ for descriptors/checkpoint metadata, one at
    # the start of _run_one_generation) plus one per genome — never one instance reused
    # across all five, which concurrent round-robin evaluation requires.
    assert len(constructed) == 7
    assert len(set(map(id, constructed))) == 7


def test_each_genome_gets_its_own_objective_instance() -> None:
    seen_ids: list[int] = []

    class TrackingObjective(DummyObjective):
        def reset(self) -> None:
            super().reset()
            seen_ids.append(id(self))

    trainer = NeatTrainer(DummyEnvironment, TrackingObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    # The `Objective` passed to `NeatTrainer.__init__` is never reset/used directly during a
    # generation — only per-genome deep copies of it are, so concurrent genomes never share
    # one mutable Objective instance the way sequential evaluation could get away with.
    assert len(seen_ids) == 5
    assert len(set(seen_ids)) == 5


def test_live_changeable_fields_are_real_run_config_fields() -> None:
    import dataclasses

    from neuroarena.backends.neat.trainer import _LIVE_CHANGEABLE_FIELDS

    real_fields = {f.name for f in dataclasses.fields(RunConfig)}
    assert _LIVE_CHANGEABLE_FIELDS <= real_fields


def test_update_config_rejects_unsupported_fields() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    with pytest.raises(ValueError, match="population_size"):
        trainer.update_config({"population_size": 10})
    with pytest.raises(ValueError, match="neat_hyperparameters"):
        trainer.update_config({"neat_hyperparameters": {"compatibility_threshold": 1.0}})


def test_update_config_applies_starting_the_next_generation_not_immediately() -> None:
    config = RunConfig(max_generation_steps=300_000)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    run = trainer.run()
    gen0 = next(run)
    # DummyEnvironment truncates every genome at 5 steps, DummyObjective never should_stop()s,
    # so all 5 genomes run their full natural length under the original, generous budget.
    assert gen0.sim_time == 25.0

    trainer.update_config({"max_generation_steps": 7})
    gen1 = next(run)
    # Round-robin distributes a tight budget=7 across 5 genomes: round 1 (7->2, all 5 step),
    # round 2 (2->0, only 2 of the 5 get a second step before the budget runs out) = 7 total.
    assert gen1.sim_time - gen0.sim_time == 7.0


def test_update_config_merges_multiple_pending_updates_before_they_apply() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    trainer.update_config({"max_generations": 2})
    # Deliberately unreachable on its own (DummyObjective's fitness never exceeds 5.0) — if
    # this call silently clobbered the max_generations update above instead of merging with
    # it, the run would never stop and this test would hang until pytest's own timeout.
    trainer.update_config({"target_fitness": 1_000_000.0})
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1]


def test_update_config_does_not_affect_a_generation_already_in_progress() -> None:
    started = threading.Event()
    release = threading.Event()

    class PausesOnFirstUpdateCall(DummyObjective):
        _paused = False

        def update(
            self,
            observation: np.ndarray,
            action: np.ndarray,
            terminated: bool,
            truncated: bool,
            info: dict[str, typing.Any],
        ) -> None:
            super().update(observation, action, terminated, truncated, info)
            if not PausesOnFirstUpdateCall._paused:
                PausesOnFirstUpdateCall._paused = True
                started.set()
                assert release.wait(timeout=5), "test deadlocked waiting for release"

    config = RunConfig(max_generation_steps=300_000)
    trainer = NeatTrainer(DummyEnvironment, PausesOnFirstUpdateCall(), config, population_size=5)
    run = trainer.run()

    results: list[TrainingUpdate] = []

    def drive_generation_zero() -> None:
        results.append(next(run))

    thread = threading.Thread(target=drive_generation_zero)
    thread.start()
    assert started.wait(timeout=5), "generation 0 never reached its first genome step"

    # Called from this (main) thread while generation 0 is paused mid-step in the background
    # thread above — this is the concurrent-call scenario update_config must handle safely.
    trainer.update_config({"max_generation_steps": 7})
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "background generation never finished"

    gen0 = results[0]
    # Generation 0's step_budget was already captured from the ORIGINAL config before the
    # pause; the concurrent update must not retroactively shrink a generation in progress.
    assert gen0.sim_time == 25.0

    gen1 = next(run)
    # The staged update only takes effect starting the next generation boundary.
    assert gen1.sim_time - gen0.sim_time == 7.0


def test_progress_snapshot_is_none_before_run_starts() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    assert trainer.progress_snapshot() is None


def test_progress_snapshot_is_none_again_after_a_generation_completes() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    next(trainer.run())
    assert trainer.progress_snapshot() is None


def test_progress_snapshot_reflects_live_state_during_a_generation() -> None:
    class SlowObjective(DummyObjective):
        def update(
            self,
            observation: np.ndarray,
            action: np.ndarray,
            terminated: bool,
            truncated: bool,
            info: dict[str, typing.Any],
        ) -> None:
            super().update(observation, action, terminated, truncated, info)
            time.sleep(0.02)

    trainer = NeatTrainer(DummyEnvironment, SlowObjective(), RunConfig(), population_size=3)
    snapshots: list[GenerationProgress | None] = []

    def _poll() -> None:
        for _ in range(100):
            snapshots.append(trainer.progress_snapshot())
            time.sleep(0.005)

    poller = threading.Thread(target=_poll)
    poller.start()
    next(trainer.run())
    poller.join()

    live = [s for s in snapshots if s is not None]
    assert live, "expected at least one snapshot while the generation was running"
    assert all(s.population_size == 3 for s in live)
    assert all(0 <= s.active_genomes_remaining <= 3 for s in live)
    assert all(0 <= s.elapsed_steps <= s.step_ceiling for s in live)
    assert all(s.generation == 0 for s in live)


class _VisibleEnvironment(DummyEnvironment):
    def visual_state(self) -> Mapping[str, float]:
        return {"x": float(self._t), "y": 0.0, "heading": 0.0}


class _SlowObjective(DummyObjective):
    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, typing.Any],
    ) -> None:
        super().update(observation, action, terminated, truncated, info)
        time.sleep(0.02)


def _poll_visual_snapshots(trainer: NeatTrainer) -> list[BatchVisuals | None]:
    snapshots: list[BatchVisuals | None] = []

    def _poll() -> None:
        for _ in range(100):
            snapshots.append(trainer.visual_snapshot())
            time.sleep(0.005)

    poller = threading.Thread(target=_poll)
    poller.start()
    next(trainer.run())
    poller.join(timeout=30)
    return snapshots


def test_visual_snapshot_is_none_before_run_and_after_a_generation() -> None:
    trainer = NeatTrainer(_VisibleEnvironment, DummyObjective(), RunConfig(), population_size=4)
    assert trainer.visual_snapshot() is None
    next(trainer.run())
    assert trainer.visual_snapshot() is None


def test_visual_snapshot_reports_each_live_genomes_state_and_fitness() -> None:
    trainer = NeatTrainer(_VisibleEnvironment, _SlowObjective(), RunConfig(), population_size=3)
    live = [s for s in _poll_visual_snapshots(trainer) if s is not None]
    assert live, "expected at least one snapshot while the generation was running"
    populated = [s for s in live if s.genomes]
    assert populated, "expected at least one snapshot with live genomes"
    for snapshot in populated:
        assert snapshot.generation == 0
        assert snapshot.population_size == 3
        assert len(snapshot.genomes) <= 3
        for genome in snapshot.genomes:
            assert set(genome.state) == {"x", "y", "heading"}
            assert genome.fitness >= 0.0


def test_visual_snapshot_omits_genomes_whose_env_is_not_visualizable() -> None:
    trainer = NeatTrainer(DummyEnvironment, _SlowObjective(), RunConfig(), population_size=3)
    live = [s for s in _poll_visual_snapshots(trainer) if s is not None]
    assert live
    assert all(s.genomes == () for s in live)


def test_set_pace_is_called_once_per_round_during_a_generation() -> None:
    calls: list[int] = []
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=3)
    trainer.set_pace(lambda: calls.append(1))
    next(trainer.run())
    # DummyEnvironment truncates after 5 steps: 5 rounds, and the last leaves nobody to pace.
    assert len(calls) == 4
