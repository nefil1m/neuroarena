from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.dashboard.ws_protocol import (
    generation_message,
    progress_message,
    status_message,
)
from neuroarena.interfaces.protocols import TrainingUpdate


def test_progress_message_shape() -> None:
    snapshot = GenerationProgress(
        generation=3,
        population_size=10,
        active_genomes_remaining=4,
        elapsed_steps=120,
        step_ceiling=300,
        best_fitness_so_far=2.5,
    )
    msg = progress_message(snapshot)
    assert msg["type"] == "progress"
    assert msg["schema_version"] == 1
    assert msg["data"] == {
        "generation": 3,
        "population_size": 10,
        "active_genomes_remaining": 4,
        "elapsed_steps": 120,
        "step_ceiling": 300,
        "best_fitness_so_far": 2.5,
    }


def test_generation_message_carries_the_full_training_update() -> None:
    update = TrainingUpdate(
        progress_index=5,
        best_fitness=9.0,
        mean_fitness=5.0,
        worst_fitness=1.0,
        population_size=10,
        champion_metrics={"progress": 9.0},
        sim_time=300.0,
        wall_time=1.2,
    )
    msg = generation_message(update)
    assert msg["type"] == "generation"
    assert msg["schema_version"] == 1
    assert msg["data"]["progress_index"] == 5
    assert msg["data"]["champion_metrics"] == {"progress": 9.0}
    assert msg["data"]["schema_version"] == 1  # TrainingUpdate's own field, inside data


def test_status_message_shape() -> None:
    msg = status_message("crashed", "RuntimeError: boom")
    assert msg == {
        "type": "status",
        "schema_version": 1,
        "data": {"state": "crashed", "detail": "RuntimeError: boom"},
    }
