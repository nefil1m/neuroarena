"""Training-launch setup: resolves a `track_id` + config overrides (fresh or resumed) into
a ready-to-run `Trainer`. Extracted from `cli.py`'s `run()` (Phase 6) so Phase 7's dashboard
can reuse the exact same setup path non-blockingly — `prepare_run` only builds the trainer;
it never drives `Trainer.run()` itself (that's the caller's job: `cli.run()` calls
`recorder.run_and_record` synchronously, Phase 7's `RunManager` runs it on a background
thread). See `../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.protocols import Trainer
from neuroarena.persistence import checkpoints_repo, models_repo, settings_history_repo
from neuroarena.persistence.models_repo import ModelRecord
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.objectives import ProgressObjective
from neuroarena.tracks.store import load as load_track

DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class PreparedRun:
    trainer: Trainer
    model: ModelRecord
    config: RunConfig
    starting_generation: int
    settings_diff: dict[str, Any]
    model_dir: Path


def prepare_run(
    conn: sqlite3.Connection,
    *,
    track_id: str,
    population_size: int | None = None,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int | None = None,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> PreparedRun:
    """`None` means "the caller did not specify this", never "no value" — see `cli.run`'s
    original docstring for the fresh-vs-resume fallback rule this preserves exactly."""
    track_record = load_track(track_id, tracks_dir=data_dir / "tracks")
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "population_size": population_size,
            "max_generations": max_generations,
            "target_fitness": target_fitness,
            "checkpoint_every_n_generations": checkpoint_every_n_generations,
            "champion_retention_cap": champion_retention_cap,
        }.items()
        if value is not None
    }

    checkpoint_root = data_dir / "checkpoints"
    resume_state: tuple[models_repo.ModelRecord, Path] | None = None
    if resume_model_id is not None:
        resume_model = models_repo.get_model(conn, resume_model_id)
        latest = checkpoints_repo.latest_checkpoint(conn, resume_model_id, kind="resume")
        if latest is None:
            raise ValueError(f"model {resume_model_id!r} has no resume checkpoint to resume from")
        previous_config = settings_history_repo.reconstruct_run_config(conn, resume_model_id)
        if (
            "population_size" in overrides
            and overrides["population_size"] != previous_config.population_size
        ):
            raise ValueError(
                "population_size cannot be changed on resume — NeatTrainer.load_checkpoint"
                " always continues with the checkpoint's own population size"
                f" ({previous_config.population_size})"
            )
        config = dataclasses.replace(previous_config, track_id=track_id, **overrides)
        diff = settings_history_repo.compute_diff(previous_config, config)
        # `NeatTrainer` pickles its generation counter *after* incrementing, so the restored
        # segment's first `generation_stats` row is the checkpoint's generation + 1 — which is
        # what `starting_generation` ("where this segment began") has to record.
        starting_generation = latest.generation + 1
        resume_state = (resume_model, Path(latest.file_path))
    else:
        config = dataclasses.replace(RunConfig(track_id=track_id), **overrides)
        diff = settings_history_repo.compute_diff(None, config)
        starting_generation = 0

    if config.checkpoint_every_n_generations < 1:
        raise ValueError(
            "checkpoint_every_n_generations must be >= 1, got"
            f" {config.checkpoint_every_n_generations}"
        )

    def make_env() -> CarEnvironment:
        return CarEnvironment(
            track_record.track, track_id, CarEnvironmentConfig.from_run_config(config)
        )

    objective = ProgressObjective()

    if resume_state is not None:
        model, checkpoint_path = resume_state
        trainer: Trainer = NeatTrainer.load_checkpoint(checkpoint_path, make_env, objective, config)
    else:
        env = make_env()
        model = models_repo.create_model(
            conn,
            backend="neat",
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        model_dir = checkpoint_root / model.model_id
        trainer = NeatTrainer(make_env, objective, config, champion_dir=model_dir / "champion")

    model_dir = checkpoint_root / model.model_id
    return PreparedRun(
        trainer=trainer,
        model=model,
        config=config,
        starting_generation=starting_generation,
        settings_diff=diff,
        model_dir=model_dir,
    )
