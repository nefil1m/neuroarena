"""Entry point: `uv run neuroarena-train`. The first working training-launch entrypoint in
this codebase — closes the "no phase has named a training-launch entrypoint yet" gap
Phase 5's implementation plan explicitly left open, the same way `tracks/cli.py` closed
the equivalent gap for track generation (Phase 2). Resolves `track_id` into a `Track` +
`make_env`, builds a `ProgressObjective`/`NeatTrainer`, and drives `run()` through
`persistence.recorder.run_and_record`. See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" section."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.persistence import checkpoints_repo, models_repo, recorder, settings_history_repo
from neuroarena.persistence.db import connect
from neuroarena.persistence.runs_repo import RunRecord
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.objectives import ProgressObjective
from neuroarena.tracks.store import UnknownTrackError
from neuroarena.tracks.store import load as load_track

DEFAULT_DATA_DIR = Path("data")


def run(
    *,
    track_id: str,
    population_size: int | None = None,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int | None = None,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> RunRecord:
    """`None` means "the caller did not specify this", never "no value": an unspecified knob
    falls back to `RunConfig`'s own default on a fresh run, and to the model's most recent
    recorded settings on a resume (the spec's "opening a model loads its most recent settings
    by default"). Passing a value explicitly overrides that fallback."""
    conn = connect(data_dir / "neuroarena.db")
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

    # Resolve the effective config first — everything below (including `make_env`) reads it,
    # and nothing here writes to the DB yet, so a rejected invocation leaves no rows behind.
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
    checkpoint_root = data_dir / "checkpoints"

    if resume_state is not None:
        model, checkpoint_path = resume_state
        trainer = NeatTrainer.load_checkpoint(checkpoint_path, make_env, objective, config)
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
    return recorder.run_and_record(
        conn,
        trainer,
        model_id=model.model_id,
        track_id=track_id,
        starting_generation=starting_generation,
        resume_dir=model_dir / "resume",
        champion_dir=model_dir / "champion",
        checkpoint_every_n_generations=config.checkpoint_every_n_generations,
        champion_retention_cap=config.champion_retention_cap,
        initial_settings_diff=diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a NEAT training run against a saved track."
    )
    parser.add_argument("--track-id", required=True, help="a track_id from `neuroarena-track-gen`")
    # Every knob defaults to None — "not specified", so `run()` can fall back to RunConfig's
    # own defaults on a fresh run and to the model's last-known settings on a resume.
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--target-fitness", type=float, default=None)
    parser.add_argument("--checkpoint-every-n-generations", type=int, default=None)
    parser.add_argument("--champion-retention-cap", type=int, default=None)
    parser.add_argument(
        "--resume", dest="resume_model_id", default=None, help="a model_id to resume"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    try:
        record = run(
            track_id=args.track_id,
            population_size=args.population_size,
            max_generations=args.max_generations,
            target_fitness=args.target_fitness,
            checkpoint_every_n_generations=args.checkpoint_every_n_generations,
            champion_retention_cap=args.champion_retention_cap,
            resume_model_id=args.resume_model_id,
            data_dir=args.data_dir,
        )
    # Bad input (unknown track/model, no resume checkpoint, rejected knob) is a user mistake,
    # not a bug — argparse's own usage + message + exit(2) beats a raw traceback. The two
    # KeyError subclasses get a label because `str(KeyError("x"))` is just `"'x'"`.
    except UnknownTrackError as exc:
        parser.error(f"unknown track_id: {exc}")
    except models_repo.UnknownModelError as exc:
        parser.error(f"unknown model_id: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"model_id={record.model_id} run_id={record.run_id} status={record.status}")


if __name__ == "__main__":
    main()
