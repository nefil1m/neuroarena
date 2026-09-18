"""Pure builders for the WebSocket envelope Phase 7's dashboard pushes — see
`../../../docs/phases/phase-7-dashboard-control-panel.md`'s WebSocket message schema
decision. No FastAPI/asyncio/threading here, so these are unit-testable in isolation."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.interfaces.protocols import TrainingUpdate

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
