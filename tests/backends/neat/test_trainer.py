from pathlib import Path

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate
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
