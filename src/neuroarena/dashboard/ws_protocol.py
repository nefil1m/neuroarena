"""Pure builders for the WebSocket envelope Phase 7's dashboard pushes — see
`../../../docs/phases/phase-7-dashboard-control-panel.md`'s WebSocket message schema
decision. The `frame`/`run`/`view` builders below are used only by the viewer's own
`/ws/viewer` endpoint; the panel's `/ws` keeps exactly `progress`/`generation`/`status`.
No FastAPI/asyncio/threading here, so these are unit-testable in isolation."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress
from neuroarena.interfaces.protocols import TrainingUpdate
from neuroarena.render.view_settings import ViewSettings

_ENVELOPE_SCHEMA_VERSION = 1


def progress_message(snapshot: GenerationProgress) -> dict[str, Any]:
    return {
        "type": "progress",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {
            "generation": snapshot.generation,
            "population_size": snapshot.population_size,
            "active_genomes_remaining": snapshot.active_genomes_remaining,
            "elapsed_steps": snapshot.elapsed_steps,
            "step_ceiling": snapshot.step_ceiling,
            "best_fitness_so_far": snapshot.best_fitness_so_far,
        },
    }


def generation_message(update: TrainingUpdate) -> dict[str, Any]:
    return {
        "type": "generation",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": dataclasses.asdict(update),
    }


def status_message(state: str, detail: str | None) -> dict[str, Any]:
    return {
        "type": "status",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {"state": state, "detail": detail},
    }


_POSE_KEYS = frozenset({"x", "y", "heading"})


def viewer_frame_message(snapshot: BatchVisuals, speed: str) -> dict[str, Any]:
    cars = [
        {
            "id": genome.genome_id,
            "x": genome.state["x"],
            "y": genome.state["y"],
            "heading": genome.state["heading"],
            "fitness": genome.fitness,
        }
        for genome in snapshot.genomes
        if _POSE_KEYS <= genome.state.keys()
    ]
    return {
        "type": "frame",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {
            "generation": snapshot.generation,
            "population_size": snapshot.population_size,
            "speed": speed,
            "cars": cars,
        },
    }


def viewer_run_message(state: str, track_id: str | None, model_id: str | None) -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": {"state": state, "track_id": track_id, "model_id": model_id},
    }


def viewer_view_message(settings: ViewSettings) -> dict[str, Any]:
    return {
        "type": "view",
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "data": settings.to_dict(),
    }
