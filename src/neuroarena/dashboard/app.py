"""FastAPI app factory for Phase 7's dashboard backend — REST routes over `RunManager`
plus the `/ws` push endpoint. See `../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from neuroarena.dashboard.run_manager import NoActiveRunError, RunAlreadyActiveError, RunManager
from neuroarena.dashboard.ws_protocol import generation_message, progress_message, status_message
from neuroarena.persistence import checkpoints_repo, models_repo, settings_history_repo
from neuroarena.persistence.db import connect
from neuroarena.tracks import store as tracks_store
from neuroarena.tracks.store import UnknownTrackError


@dataclasses.dataclass
class StartRunRequest:
    track_id: str
    resume_model_id: str | None = None
    population_size: int | None = None
    max_generations: int | None = None
    target_fitness: float | None = None


@dataclasses.dataclass
class ConfigUpdateRequest:
    max_generation_steps: int | None = None
    max_generations: int | None = None
    target_fitness: float | None = None


@dataclasses.dataclass
class PollIntervalRequest:
    interval_ms: int


def create_app(run_manager: RunManager, data_dir: Path) -> FastAPI:

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        # Blocking (bounded) join — run off the event loop so it can't stall other handlers.
        await asyncio.to_thread(run_manager.shutdown)

    app = FastAPI(title="neuroarena dashboard", lifespan=lifespan)

    @app.get("/api/tracks")
    def get_tracks() -> list[dict[str, object]]:
        return tracks_store.list_tracks(tracks_dir=data_dir / "tracks")

    @app.get("/api/models")
    def get_models() -> list[dict[str, Any]]:
        conn = connect(data_dir / "neuroarena.db")
        try:
            return [
                {"model_id": m.model_id, "backend": m.backend, "created_at": m.created_at}
                for m in models_repo.list_models(conn)
            ]
        finally:
            conn.close()

    @app.get("/api/models/{model_id}/settings")
    def get_model_settings(model_id: str) -> dict[str, Any]:
        """Includes `current_generation` alongside the reconstructed `RunConfig` fields —
        the spec's resume-flow requirement is to "surface current generation vs.
        max_generations/target_fitness so they can be raised before starting", which needs
        this number, not just the settings themselves."""
        conn = connect(data_dir / "neuroarena.db")
        try:
            models_repo.get_model(conn, model_id)
            config = settings_history_repo.reconstruct_run_config(conn, model_id)
            latest = checkpoints_repo.latest_checkpoint(conn, model_id, kind="resume")
            return {
                **dataclasses.asdict(config),
                "current_generation": None if latest is None else latest.generation,
            }
        except models_repo.UnknownModelError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model_id: {model_id}") from exc
        finally:
            conn.close()

    @app.post("/api/runs", status_code=201)
    def start_run(request: StartRunRequest) -> dict[str, Any]:
        overrides = {
            key: value
            for key, value in {
                "population_size": request.population_size,
                "max_generations": request.max_generations,
                "target_fitness": request.target_fitness,
            }.items()
            if value is not None
        }
        try:
            model_id = run_manager.start(
                track_id=request.track_id,
                resume_model_id=request.resume_model_id,
                overrides=overrides,
            )
        except RunAlreadyActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, UnknownTrackError, models_repo.UnknownModelError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"model_id": model_id, "status": "running"}

    @app.get("/api/runs/current")
    def get_current_run() -> dict[str, Any]:
        status = run_manager.status()
        return {
            "status": status.status,
            "model_id": status.model_id,
            "generation": status.generation,
        }

    @app.post("/api/runs/current/stop", status_code=202)
    def stop_run() -> dict[str, str]:
        try:
            run_manager.request_stop()
        except NoActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "stopping"}

    @app.patch("/api/runs/current/config")
    def update_config(request: ConfigUpdateRequest) -> dict[str, str]:
        partial = {k: v for k, v in dataclasses.asdict(request).items() if v is not None}
        try:
            run_manager.update_config(partial)
        except NoActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted"}

    @app.get("/api/poll-interval")
    def get_poll_interval() -> dict[str, int]:
        return {"interval_ms": run_manager.poll_interval_ms()}

    @app.patch("/api/poll-interval")
    def set_poll_interval(request: PollIntervalRequest) -> dict[str, int]:
        try:
            run_manager.set_poll_interval_ms(request.interval_ms)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"interval_ms": run_manager.poll_interval_ms()}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        last_generation_sent: int | None = None
        last_status_sent: str | None = None
        try:
            while True:
                status = run_manager.status()
                if status.status != last_status_sent:
                    await websocket.send_json(status_message(status.status, None))
                    last_status_sent = status.status

                update = run_manager.latest_update()
                if update is not None and update.progress_index != last_generation_sent:
                    await websocket.send_json(generation_message(update))
                    last_generation_sent = update.progress_index

                snapshot = run_manager.progress_snapshot()
                if snapshot is not None:
                    await websocket.send_json(progress_message(snapshot))

                await asyncio.sleep(run_manager.poll_interval_ms() / 1000)
        except WebSocketDisconnect:
            pass

    return app
