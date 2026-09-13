"""Wraps any `Trainer.run()` iterator to persist its progress. Backend-agnostic — takes
only the `Trainer` Protocol (Phase 0), never imports `NeatTrainer` — and independently
reconstructs the champion-file naming convention `NeatTrainer` already uses
(`champion_dir/gen_<generation:05d>.pkl`) rather than reading any private attribute, so it
works unchanged for Phase 10's future deep-RL backend (with `champion_dir=None`, since
nothing has said deep RL captures per-step champions the same way). See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" and "Checkpoint
cadence and retention" sections."""

from __future__ import annotations

import dataclasses
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
    except KeyboardInterrupt:
        # Ctrl-C is the normal way a human ends a long headless run — it is a
        # BaseException, so `except Exception` below never sees it and the run would
        # otherwise stay at "running" forever. `runs_repo`'s own docstring reserves
        # "stopped" for exactly this.
        _finish(conn, run_record, status="stopped")
        raise
    except Exception:
        _finish(conn, run_record, status="crashed")
        raise

    return _finish(conn, run_record, status="completed")


def _finish(conn: sqlite3.Connection, run_record: RunRecord, *, status: str) -> RunRecord:
    """Writes the run's terminal status and returns an up-to-date copy of the record —
    `update_run_status` touches only the DB, so the frozen dataclass the caller was handed
    at `create_run` time would otherwise still read `status="running"`."""
    ended_at = datetime.now(UTC).isoformat()
    runs_repo.update_run_status(conn, run_record.run_id, status=status, ended_at=ended_at)
    return dataclasses.replace(run_record, status=status, ended_at=ended_at)


def _prune_champions(conn: sqlite3.Connection, model_id: str, cap: int) -> None:
    records = checkpoints_repo.list_checkpoints(conn, model_id, kind="champion")
    excess_count = len(records) - cap
    if excess_count <= 0:
        return
    to_prune = records[:excess_count]  # oldest first — list_checkpoints orders by generation
    surviving_paths = {record.file_path for record in records[excess_count:]}
    for record in to_prune:
        # A resume from an older-than-latest checkpoint replays generations, so two rows can
        # point at the same champion file. Never unlink a file a surviving row still needs —
        # drop only the duplicate row.
        if record.file_path not in surviving_paths:
            Path(record.file_path).unlink(missing_ok=True)
        checkpoints_repo.delete_checkpoint(conn, record.checkpoint_id)
