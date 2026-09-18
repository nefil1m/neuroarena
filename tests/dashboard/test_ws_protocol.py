from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress, GenomeVisual
from neuroarena.dashboard.ws_protocol import (
    generation_message,
    progress_message,
    status_message,
    viewer_frame_message,
    viewer_run_message,
    viewer_view_message,
)
from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.render.view_settings import ViewSettings


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


def test_viewer_frame_message_shape_and_skips_genomes_without_a_full_pose() -> None:
    snapshot = BatchVisuals(
        generation=4,
        population_size=10,
        genomes=(
            GenomeVisual(7, {"x": 1.0, "y": 2.0, "heading": 0.5}, 3.5),
            GenomeVisual(8, {"x": 1.0}, 1.0),  # incomplete: skipped
        ),
    )
    assert viewer_frame_message(snapshot, "2x") == {
        "type": "frame",
        "schema_version": 1,
        "data": {
            "generation": 4,
            "population_size": 10,
            "speed": "2x",
            "cars": [{"id": 7, "x": 1.0, "y": 2.0, "heading": 0.5, "fitness": 3.5}],
        },
    }


def test_viewer_run_message_shape() -> None:
    assert viewer_run_message("running", "track1", "model1") == {
        "type": "run",
        "schema_version": 1,
        "data": {"state": "running", "track_id": "track1", "model_id": "model1"},
    }


def test_viewer_view_message_shape() -> None:
    message = viewer_view_message(ViewSettings(zoom=2.0))
    assert message["type"] == "view"
    assert message["schema_version"] == 1
    assert message["data"] == ViewSettings(zoom=2.0).to_dict()
