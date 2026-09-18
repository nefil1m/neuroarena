"""`RunManager` owns the single in-process `Trainer` this dashboard backend drives — Phase
7's "embeds the trainer in-process... one training run per backend process" process model.
Runs `persistence.recorder.run_and_record` on a background thread (via
`persistence.launch.prepare_run` for setup) so the FastAPI process stays responsive to
REST/WebSocket requests while a run is in progress. See
`../../../docs/phases/phase-7-dashboard-control-panel.md`.

Threading model: every field below is touched from two threads — the FastAPI
request-handling thread (reads status, requests config updates/stop) and the background
training thread this class spawns (writes status/latest_update as the run progresses).
Every shared field is a single Python attribute (str, int, a frozen dataclass reference, or
`None`) — safe to read/write under CPython's GIL without an extra lock, the same reasoning
`NeatTrainer.progress_snapshot`/`BatchProgress` already document. No field here is ever
read-modify-written across the two threads (the only compound state mutation,
`NeatTrainer.update_config`'s own pending-update merge, already has its own lock inside
`NeatTrainer`)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import GenerationProgress
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate
from neuroarena.persistence import recorder
from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import PreparedRun, prepare_run

DEFAULT_POLL_INTERVAL_MS = 200


class RunAlreadyActiveError(RuntimeError):
    """Raised by `RunManager.start` when a run is already active — one run per process."""


class NoActiveRunError(RuntimeError):
    """Raised when an operation (stop, config update) needs an active run but none exists."""


@dataclass(frozen=True)
class RunManagerStatus:
    status: str  # "idle" | "running" | "completed" | "crashed" | "stopped"
    model_id: str | None
    generation: int | None


class RunManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._trainer: Trainer | None = None
        self._model_id: str | None = None
        self._status: str = "idle"
        self._latest_update: TrainingUpdate | None = None
        self._stop_requested = False
        self._poll_interval_ms = DEFAULT_POLL_INTERVAL_MS

    def status(self) -> RunManagerStatus:
        generation = self._latest_update.progress_index if self._latest_update is not None else None
        return RunManagerStatus(status=self._status, model_id=self._model_id, generation=generation)

    def poll_interval_ms(self) -> int:
        return self._poll_interval_ms

    def set_poll_interval_ms(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"poll interval must be >= 1ms, got {value}")
        self._poll_interval_ms = value

    def progress_snapshot(self) -> GenerationProgress | None:
        if self._trainer is None:
            return None
        snapshot_fn = getattr(self._trainer, "progress_snapshot", None)
        return None if snapshot_fn is None else snapshot_fn()

    def latest_update(self) -> TrainingUpdate | None:
        return self._latest_update

    def update_config(self, partial: Mapping[str, Any]) -> None:
        if self._trainer is None:
            raise NoActiveRunError("no run is currently active")
        self._trainer.update_config(partial)

    def request_stop(self) -> None:
        if self._status != "running":
            raise NoActiveRunError("no run is currently active")
        self._stop_requested = True

    def start(
        self,
        *,
        track_id: str,
        resume_model_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        if self._status == "running":
            raise RunAlreadyActiveError("a run is already active in this backend process")
        conn = connect(self._data_dir / "neuroarena.db")
        try:
            prepared = prepare_run(
                conn,
                track_id=track_id,
                resume_model_id=resume_model_id,
                data_dir=self._data_dir,
                **(overrides or {}),
            )
        finally:
            conn.close()

        self._trainer = prepared.trainer
        self._model_id = prepared.model.model_id
        self._status = "running"
        self._latest_update = None
        self._stop_requested = False
        thread = threading.Thread(target=self._run, args=(prepared,), daemon=True)
        thread.start()
        return prepared.model.model_id

    def _run(self, prepared: PreparedRun) -> None:
        thread_conn = connect(self._data_dir / "neuroarena.db")
        try:
            record = recorder.run_and_record(
                thread_conn,
                prepared.trainer,
                model_id=prepared.model.model_id,
                track_id=prepared.config.track_id,
                starting_generation=prepared.starting_generation,
                resume_dir=prepared.model_dir / "resume",
                champion_dir=prepared.model_dir / "champion",
                checkpoint_every_n_generations=prepared.config.checkpoint_every_n_generations,
                champion_retention_cap=prepared.config.champion_retention_cap,
                initial_settings_diff=prepared.settings_diff,
                on_update=self._handle_update,
                should_stop=lambda: self._stop_requested,
            )
            self._status = record.status
        except Exception:
            self._status = "crashed"
        finally:
            thread_conn.close()

    def _handle_update(self, update: TrainingUpdate) -> None:
        self._latest_update = update
