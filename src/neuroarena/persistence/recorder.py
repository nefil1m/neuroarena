"""Wraps any `Trainer.run()` iterator to persist its progress. Backend-agnostic — takes
only the `Trainer` Protocol (Phase 0), never imports `NeatTrainer` — and independently
reconstructs the champion-file naming convention `NeatTrainer` already uses
(`champion_dir/gen_<generation:05d>.pkl`) rather than reading any private attribute, so it
works unchanged for Phase 10's future deep-RL backend (with `champion_dir=None`, since
nothing has said deep RL captures per-step champions the same way). See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" and "Checkpoint
cadence and retention" sections."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuroarena.persistence import (
    checkpoints_repo,
    generation_stats_repo,
    runs_repo,
    settings_history_repo,
)
from neuroarena.persistence.runs_repo import RunRecord

if TYPE_CHECKING:
    from neuroarena.interfaces.protocols import Trainer


def run_and_record(
    conn: sqlite3.Connection,
    trainer: Trainer,
    *,
    model_id: str,
    track_id: str | None,
    starting_generation: int,
    resume_dir: Path,
    champion_dir: Path | None,
    checkpoint_every_n_generations: int,
    champion_retention_cap: int | None,
    initial_settings_diff: dict[str, Any],
) -> RunRecord:
    resume_dir.mkdir(parents=True, exist_ok=True)
    run_record = runs_repo.create_run(
        conn, model_id=model_id, track_id=track_id, starting_generation=starting_generation
    )
    settings_history_repo.record_settings_entry(
        conn,
        model_id=model_id,
        run_id=run_record.run_id,
        generation=starting_generation,
        diff=initial_settings_diff,
    )
    try:
        for update in trainer.run():
            generation_stats_repo.record_generation_stat(
                conn, model_id=model_id, run_id=run_record.run_id, update=update
            )
            generation = update.progress_index

            if generation % checkpoint_every_n_generations == 0:
                path = resume_dir / f"gen_{generation:05d}.pkl"
                trainer.save_checkpoint(path)
                checkpoints_repo.record_checkpoint(
                    conn,
                    model_id=model_id,
                    run_id=run_record.run_id,
                    kind="resume",
                    generation=generation,
                    file_path=path,
                )

            if champion_dir is not None:
                champion_path = champion_dir / f"gen_{generation:05d}.pkl"
                checkpoints_repo.record_checkpoint(
                    conn,
                    model_id=model_id,
                    run_id=run_record.run_id,
                    kind="champion",
                    generation=generation,
                    file_path=champion_path,
                )
                if champion_retention_cap is not None:
                    _prune_champions(conn, model_id, champion_retention_cap)
    except Exception:
        runs_repo.update_run_status(
            conn, run_record.run_id, status="crashed", ended_at=datetime.now(UTC).isoformat()
        )
        raise

    runs_repo.update_run_status(
        conn, run_record.run_id, status="completed", ended_at=datetime.now(UTC).isoformat()
    )
    return run_record


def _prune_champions(conn: sqlite3.Connection, model_id: str, cap: int) -> None:
    records = checkpoints_repo.list_checkpoints(conn, model_id, kind="champion")
    excess_count = len(records) - cap
    if excess_count <= 0:
        return
    for record in records[:excess_count]:  # oldest first — list_checkpoints orders by generation
        Path(record.file_path).unlink(missing_ok=True)
        checkpoints_repo.delete_checkpoint(conn, record.checkpoint_id)
