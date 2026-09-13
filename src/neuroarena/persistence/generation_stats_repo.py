"""`generation_stats` table: one row per `TrainingUpdate` a run's `Trainer.run()` yields —
this *is* the run history (Phase 0's "run history & per-generation records" requirement),
enough to redraw a fitness curve and compare generations without reloading a checkpoint."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from neuroarena.interfaces.protocols import TrainingUpdate


@dataclass(frozen=True)
class GenerationStatRecord:
    model_id: str
    run_id: str
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    population_size: int
    champion_metrics: dict[str, float]
    sim_time: float
    wall_time: float


def record_generation_stat(
    conn: sqlite3.Connection, *, model_id: str, run_id: str, update: TrainingUpdate
) -> None:
    conn.execute(
        "INSERT INTO generation_stats "
        "(model_id, run_id, generation, best_fitness, mean_fitness, worst_fitness, "
        " population_size, champion_metrics, sim_time, wall_time) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            model_id,
            run_id,
            update.progress_index,
            update.best_fitness,
            update.mean_fitness,
            update.worst_fitness,
            update.population_size,
            json.dumps(update.champion_metrics),
            update.sim_time,
            update.wall_time,
        ),
    )
    conn.commit()


def list_generation_stats_for_model(
    conn: sqlite3.Connection, model_id: str
) -> list[GenerationStatRecord]:
    rows = conn.execute(
        # `id` breaks generation ties deterministically — a resume from an older-than-latest
        # checkpoint replays generations that already have rows.
        "SELECT * FROM generation_stats WHERE model_id = ? ORDER BY generation, id",
        (model_id,),
    ).fetchall()
    return [
        GenerationStatRecord(
            model_id=row["model_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            best_fitness=row["best_fitness"],
            mean_fitness=row["mean_fitness"],
            worst_fitness=row["worst_fitness"],
            population_size=row["population_size"],
            champion_metrics=json.loads(row["champion_metrics"]),
            sim_time=row["sim_time"],
            wall_time=row["wall_time"],
        )
        for row in rows
    ]
