import numpy as np

from neuroarena.sim.objectives import ProgressObjective


def _obs() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


def _act() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_fitness_accumulates_progress_deltas() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 5.0, "lap_progress": 0.1})
    objective.update(_obs(), _act(), False, False, {"progress": 8.0, "lap_progress": 0.16})
    assert objective.fitness() == 8.0  # cumulative = final progress, since it started at 0


def test_step_reward_is_the_latest_delta() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 5.0, "lap_progress": 0.1})
    objective.update(_obs(), _act(), False, False, {"progress": 8.0, "lap_progress": 0.16})
    assert objective.step_reward() == 3.0


def test_should_stop_true_once_lap_progress_reaches_one() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 100.0, "lap_progress": 0.99})
    assert objective.should_stop() is False
    objective.update(_obs(), _act(), False, False, {"progress": 102.0, "lap_progress": 1.0})
    assert objective.should_stop() is True


def test_reset_clears_accumulated_state() -> None:
    objective = ProgressObjective()
    objective.update(_obs(), _act(), False, False, {"progress": 50.0, "lap_progress": 0.5})
    objective.reset()
    assert objective.fitness() == 0.0
    assert objective.should_stop() is False
    objective.update(_obs(), _act(), False, False, {"progress": 3.0, "lap_progress": 0.03})
    assert objective.step_reward() == 3.0  # delta from the post-reset baseline (0), not the old one
