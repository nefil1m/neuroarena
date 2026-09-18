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
`None`) — plain reads/writes are safe under CPython's GIL, the same reasoning
`NeatTrainer.progress_snapshot`/`BatchProgress` already document. The speed preset is a
single `str` attribute read by the `Pacer` on the training thread each round. The one compound
operation, `start()`'s check-then-transition to "running" (called concurrently from
FastAPI's threadpool), is guarded by `_start_lock` so at most one run is ever reserved; the
slot is reserved under the lock *before* `prepare_run`'s slow DB/FS work and released back
to the previous status if setup fails. The worker thread always ends in a terminal status
(`record.status` or "crashed"). `NeatTrainer.update_config`'s own pending-update merge has
its own lock inside `NeatTrainer`."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import BatchVisuals, GenerationProgress
from neuroarena.dashboard.pacing import DEFAULT_SPEED_PRESET, SPEED_PRESETS, Pacer
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate
from neuroarena.persistence import recorder, runs_repo
from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import PreparedRun, prepare_run

_log = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_MS = 200
DEFAULT_SHUTDOWN_TIMEOUT_S = 10.0


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
        self._speed_preset = DEFAULT_SPEED_PRESET
        self._pacer = Pacer(self.speed_multiplier)
        self._track_id: str | None = None
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def status(self) -> RunManagerStatus:
        generation = self._latest_update.progress_index if self._latest_update is not None else None
        return RunManagerStatus(status=self._status, model_id=self._model_id, generation=generation)

    def poll_interval_ms(self) -> int:
        return self._poll_interval_ms

    def set_poll_interval_ms(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"poll interval must be >= 1ms, got {value}")
        self._poll_interval_ms = value

    def speed_preset(self) -> str:
        return self._speed_preset

    def set_speed_preset(self, preset: str) -> None:
        if preset not in SPEED_PRESETS:
            raise ValueError(f"unknown speed preset {preset!r}; choose from {list(SPEED_PRESETS)}")
        self._speed_preset = preset

    def speed_multiplier(self) -> float | None:
        """The current preset as a multiplier of real time; `None` means unpaced ("max").
        Always `None` while a stop is requested, so a paced run sprints to its generation
        boundary (where `should_stop` is checked) instead of taking minutes to stop.
        `speed_preset()` is unaffected."""
        if self._stop_requested:
            return None
        return SPEED_PRESETS[self._speed_preset]

    def current_track_id(self) -> str | None:
        return self._track_id

    def visual_snapshot(self) -> BatchVisuals | None:
        """Every still-active car of the in-progress generation (see
        `NeatTrainer.visual_snapshot`), or `None` unless a run is running — a finished run's
        stale trainer must not keep producing frames."""
        if self._trainer is None or self._status != "running":
            return None
        snapshot_fn = getattr(self._trainer, "visual_snapshot", None)
        return None if snapshot_fn is None else snapshot_fn()

    def progress_snapshot(self) -> GenerationProgress | None:
        if self._trainer is None:
            return None
        snapshot_fn = getattr(self._trainer, "progress_snapshot", None)
        return None if snapshot_fn is None else snapshot_fn()

    def latest_update(self) -> TrainingUpdate | None:
        return self._latest_update

    def update_config(self, partial: Mapping[str, Any]) -> None:
        if self._trainer is None or self._status != "running":
            raise NoActiveRunError("no run is currently active")
        self._trainer.update_config(partial)

    def request_stop(self) -> None:
        if self._status != "running":
            raise NoActiveRunError("no run is currently active")
        self._stop_requested = True

    def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        """Called on backend shutdown. The worker is a daemon thread, so interpreter
        finalization would kill it mid-generation and leave its `runs` row at "running"
        forever. Ask it to stop and wait (bounded) for it to finalize the row itself; if it
        is still alive after `timeout`, mark the row "stopped" here. No-op when idle."""
        thread = self._thread
        if self._status != "running" or thread is None:
            return
        self._stop_requested = True
        thread.join(timeout)
        if not thread.is_alive() or self._model_id is None:
            return
        _log.warning("training worker did not stop within %.1fs; finalizing run row", timeout)
        conn = connect(self._data_dir / "neuroarena.db")
        try:
            ended_at = datetime.now(UTC).isoformat()
            for run in runs_repo.list_runs_for_model(conn, self._model_id):
                if run.status == "running":
                    runs_repo.update_run_status(
                        conn, run.run_id, status="stopped", ended_at=ended_at
                    )
        finally:
            conn.close()

    def start(
        self,
        *,
        track_id: str,
        resume_model_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        with self._start_lock:
            if self._status == "running":
                raise RunAlreadyActiveError("a run is already active in this backend process")
            previous_status = self._status
            previous_trainer = self._trainer
            previous_track_id = self._track_id
            self._status = "running"  # reserve the slot before the slow setup below
            # ...and drop the previous run's identity so the viewer never sees it as current
            self._track_id = track_id
            self._trainer = None
        try:
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
            self._pacer = Pacer(self.speed_multiplier)  # fresh deadline for each run
            set_pace = getattr(prepared.trainer, "set_pace", None)
            if set_pace is not None:
                set_pace(self._pacer)
            self._track_id = track_id
            self._model_id = prepared.model.model_id
            self._latest_update = None
            self._stop_requested = False
            thread = threading.Thread(target=self._run, args=(prepared,), daemon=True)
            self._thread = thread
            thread.start()
        except BaseException:
            with self._start_lock:
                self._status = previous_status
                self._trainer = previous_trainer
                self._track_id = previous_track_id
            raise
        return prepared.model.model_id

    def _run(self, prepared: PreparedRun) -> None:
        thread_conn: sqlite3.Connection | None = None
        final_status = "crashed"
        try:
            thread_conn = connect(self._data_dir / "neuroarena.db")
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
            final_status = record.status
        except Exception:
            _log.exception("training run crashed")
        finally:
            # Any exit path (including BaseException) leaves a terminal status.
            self._status = final_status
            if thread_conn is not None:
                thread_conn.close()

    def _handle_update(self, update: TrainingUpdate) -> None:
        self._latest_update = update
