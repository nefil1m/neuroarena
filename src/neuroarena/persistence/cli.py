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

from neuroarena.persistence import models_repo, recorder
from neuroarena.persistence.db import connect
from neuroarena.persistence.launch import prepare_run
from neuroarena.persistence.runs_repo import RunRecord
from neuroarena.tracks.store import UnknownTrackError

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
    by default"). Passing a value explicitly overrides that fallback. Setup (config
    resolution, trainer construction) is `persistence.launch.prepare_run` — shared with
    Phase 7's dashboard, which needs the same setup without this function's blocking
    `run_and_record` call below."""
    conn = connect(data_dir / "neuroarena.db")
    prepared = prepare_run(
        conn,
        track_id=track_id,
        population_size=population_size,
        max_generations=max_generations,
        target_fitness=target_fitness,
        checkpoint_every_n_generations=checkpoint_every_n_generations,
        champion_retention_cap=champion_retention_cap,
        resume_model_id=resume_model_id,
        data_dir=data_dir,
    )
    return recorder.run_and_record(
        conn,
        prepared.trainer,
        model_id=prepared.model.model_id,
        track_id=track_id,
        starting_generation=prepared.starting_generation,
        resume_dir=prepared.model_dir / "resume",
        champion_dir=prepared.model_dir / "champion",
        checkpoint_every_n_generations=prepared.config.checkpoint_every_n_generations,
        champion_retention_cap=prepared.config.champion_retention_cap,
        initial_settings_diff=prepared.settings_diff,
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
