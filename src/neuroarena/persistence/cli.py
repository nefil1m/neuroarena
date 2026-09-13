"""Entry point: `uv run neuroarena-train`. The first working training-launch entrypoint in
this codebase — closes the "no phase has named a training-launch entrypoint yet" gap
Phase 5's implementation plan explicitly left open, the same way `tracks/cli.py` closed
the equivalent gap for track generation (Phase 2). Resolves `track_id` into a `Track` +
`make_env`, builds a `ProgressObjective`/`NeatTrainer`, and drives `run()` through
`persistence.recorder.run_and_record`. See
`../../../docs/phases/phase-6-persistence.md`'s "Launch entrypoint" section."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.persistence import checkpoints_repo, models_repo, recorder, settings_history_repo
from neuroarena.persistence.db import connect
from neuroarena.persistence.runs_repo import RunRecord
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.objectives import ProgressObjective
from neuroarena.tracks.store import load as load_track

DEFAULT_DATA_DIR = Path("data")


def run(
    *,
    track_id: str,
    population_size: int = 150,
    max_generations: int | None = None,
    target_fitness: float | None = None,
    checkpoint_every_n_generations: int = 10,
    champion_retention_cap: int | None = None,
    resume_model_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> RunRecord:
    conn = connect(data_dir / "neuroarena.db")
    track_record = load_track(track_id, tracks_dir=data_dir / "tracks")
    config = RunConfig(
        population_size=population_size,
        track_id=track_id,
        max_generations=max_generations,
        target_fitness=target_fitness,
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
    )

    def make_env() -> CarEnvironment:
        return CarEnvironment(
            track_record.track, track_id, CarEnvironmentConfig.from_run_config(config)
        )

    objective = ProgressObjective()
    checkpoint_root = data_dir / "checkpoints"

    if resume_model_id is not None:
        model = models_repo.get_model(conn, resume_model_id)
        latest = checkpoints_repo.latest_checkpoint(conn, resume_model_id, kind="resume")
        if latest is None:
            raise ValueError(f"model {resume_model_id!r} has no resume checkpoint to resume from")
        trainer = NeatTrainer.load_checkpoint(Path(latest.file_path), make_env, objective, config)
        starting_generation = latest.generation
        previous_config = settings_history_repo.reconstruct_run_config(conn, resume_model_id)
        diff = settings_history_repo.compute_diff(previous_config, config)
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
        starting_generation = 0
        diff = settings_history_repo.compute_diff(None, config)

    model_dir = checkpoint_root / model.model_id
    return recorder.run_and_record(
        conn,
        trainer,
        model_id=model.model_id,
        track_id=track_id,
        starting_generation=starting_generation,
        resume_dir=model_dir / "resume",
        champion_dir=model_dir / "champion",
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
        initial_settings_diff=diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a NEAT training run against a saved track."
    )
    parser.add_argument("--track-id", required=True, help="a track_id from `neuroarena-track-gen`")
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--target-fitness", type=float, default=None)
    parser.add_argument("--checkpoint-every-n-generations", type=int, default=10)
    parser.add_argument("--champion-retention-cap", type=int, default=None)
    parser.add_argument(
        "--resume", dest="resume_model_id", default=None, help="a model_id to resume"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

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
    print(f"model_id={record.model_id} run_id={record.run_id} status={record.status}")


if __name__ == "__main__":
    main()
