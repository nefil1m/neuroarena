from pathlib import Path

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.sim.car_env import CarEnvironment
from neuroarena.sim.track import Facing, GridCell, TileKind, Track
from tests.interfaces.doubles import DummyObjective


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _make_car_env() -> CarEnvironment:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    return CarEnvironment(track, track_id="phase-4-smoke-test")


def test_neat_trainer_runs_headless_on_the_real_car_environment(tmp_path: Path) -> None:
    config = RunConfig(max_generation_steps=2000)
    trainer = NeatTrainer(
        _make_car_env,
        DummyObjective(),
        config,
        population_size=8,
        champion_dir=tmp_path / "champions",
    )
    run = trainer.run()
    updates = [next(run) for _ in range(3)]

    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert u.population_size == 8
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness
        assert set(u.champion_metrics) == {"crashed", "progress", "lap_progress"}
    saved = sorted((tmp_path / "champions").iterdir())
    assert len(saved) == 3
