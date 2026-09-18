# neuroarena

A 2D simulation platform where games and learning models are decoupled: a
driving game, a NEAT trainer that learns to play it, run persistence with
resume, and a local web dashboard to control training. The pieces talk through
small interfaces (`Environment`, `Model`, `Objective`, `Trainer`), so other
model types or games can be swapped in.

See `docs/OVERVIEW.md` for the pitch and settled decisions, and
`docs/PHASES.md` for the build plan.

## What it does

- **Playable driving game.** A kinematic-bicycle car on a tile-based track,
  drawn with real sprite art by [`arcade`](https://api.arcade.academy/) (follow
  camera, kerbs, start/finish decal). You drive with the keyboard; a human is
  just another policy producing `(steering, throttle)`.
- **Procedural tracks.** Closed-loop tracks generated from a size, a
  complexity (turn frequency) and a seed, saved with a `track_id`.
- **Sensors and a game-agnostic interface.** The car sees the world through
  raycast sensors (a fixed-size observation vector) and acts with continuous
  steering and throttle. The simulation is plain Python/NumPy and runs headless.
- **NEAT training.** Each generation's whole population runs as a batch in
  lockstep, every genome on its own isolated copy of the track. A generation
  ends when every genome has finished or crashed, when the step ceiling is
  reached, or as soon as one genome succeeds. Run-level stop conditions: a maximum number of generations and/or a target fitness.
- **Persistence.** Every run, its settings history, per-generation stats,
  resume checkpoints and champion checkpoints are stored in a local SQLite
  database plus files under `data/`. Any model can be resumed later, inheriting
  its most recent settings.
- **Web dashboard.** A local control panel (FastAPI backend + React frontend):
  start a new run or resume a saved model, watch live metrics over a WebSocket,
  edit a few settings mid-run, stop a run, set the dashboard's own update
  rate, open a game window that shows the whole generation live, and set the
  training speed.

Not there yet: run-history charts and model inspection in the dashboard.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). The dashboard
frontend also needs Node.js and npm.

```bash
uv sync
```

## Quick tour

```bash
# 1. Generate a track (saved under data/tracks/ and printed as a track_id)
uv run neuroarena-track-gen --size 30 --seed 1

# 2. Train on it — from the command line...
uv run neuroarena-train --track-id <track_id> --population-size 150

# ...or from the dashboard (see "Dashboard" below)
uv run neuroarena-dashboard

# 3. Drive a track yourself
uv run neuroarena-play
```

## Commands

### `neuroarena-play`

Opens the game window on the bundled sample track. Drive with the arrow keys
or `WASD`.

### `neuroarena-track-gen`

Generates a procedural closed-loop track.

```bash
uv run neuroarena-track-gen --size N --seed S [--complexity 0..1] [--tracks-dir DIR] [--preview]
```

| Option | Meaning |
|---|---|
| `--size` | target tile count (required) |
| `--seed` | random seed (required) |
| `--complexity` | turn-frequency bias, 0–1 |
| `--tracks-dir` | where to save the track (default `data/tracks`) |
| `--preview` | open the game window to drive the generated track |

### `neuroarena-train`

Launches a NEAT training run against a saved track and records it.

```bash
uv run neuroarena-train --track-id ID [--population-size N] [--max-generations N] \
    [--target-fitness F] [--checkpoint-every-n-generations N] \
    [--champion-retention-cap N] [--resume MODEL_ID] [--data-dir DIR]
```

| Option | Meaning |
|---|---|
| `--track-id` | a `track_id` from `neuroarena-track-gen` (required) |
| `--population-size` | genomes per generation (default 150) |
| `--max-generations` | stop after this many generations |
| `--target-fitness` | stop once a genome reaches this fitness |
| `--checkpoint-every-n-generations` | resume-checkpoint cadence (default 10) |
| `--champion-retention-cap` | keep at most this many champion checkpoints |
| `--resume` | resume a `model_id` (printed when a run starts), inheriting its latest settings |
| `--data-dir` | database and checkpoint location (default `data`) |

Resuming starts from the model's last checkpoint, so generations after it are
replayed. With the default cadence, a run stopped at generation 3 resumes from
the generation-0 checkpoint.

### `neuroarena-dashboard`

The web control panel. It runs as two processes during development.

Backend (binds `127.0.0.1:8000`, no auth):

```bash
uv run neuroarena-dashboard [--data-dir DIR] [--port N]
```

Frontend (Vite dev server on `:5173`, proxies `/api` and `/ws` to the backend):

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>. From the panel you can:

- pick a track and start a run (population size, maximum generations and target
  fitness are optional), or resume a saved model with its last settings
  pre-filled;
- watch live status, the in-progress generation (active genomes, steps, best
  fitness so far) and each completed generation's best/mean/worst fitness;
- change maximum generation steps, maximum generations and target fitness while
  a run is going (they apply at the next generation boundary);
- stop the run (it ends after its current generation);
- change how often the backend pushes updates (default 200 ms);
- open a game window — a separate `arcade` viewer, on the machine running the backend — that
  shows every car of the current generation live on the real track art (the best car
  highlighted), with three camera modes (whole track, follow the best car, follow a chosen
  rank), zoom, and a small overlay (generation, cars alive, speed, camera);
- set the simulation speed: `0.25x`, `0.5x`, `1x` (real time), `2x`, `4x`, `8x`, or `Max`.
  Slower speeds pace training so the game window is watchable, and make training take longer;
  `Max` (the default every time the backend starts) runs as fast as the CPU allows.

The game window needs a display on the same machine as the backend; the panel itself can be used remotely. To run the viewer by hand: `uv run python -m neuroarena.render.viewer --url ws://127.0.0.1:8000/ws/viewer`.

The backend allows one run at a time. The dashboard and the CLI share the same
`data/` directory, so models trained from either show up in both.

## Project layout

```
src/neuroarena/
  interfaces/   Environment / Model / Objective / Trainer protocols, spaces
  config/       versioned RunConfig
  sim/          physics, track, raycast sensors, observation, car environment
  tracks/       track generation CLI and track store
  render/       arcade game window, viewer window, input, sprite assets
  backends/neat NEAT trainer, batch evaluation, model
  persistence/  SQLite repos, recorder, training CLI
  dashboard/    FastAPI app, run manager, WebSocket protocol, CLI
frontend/       React + TypeScript dashboard UI
docs/           overview, build plan, requirements and implementation plans
```

## Development

```bash
uv run pytest -v             # tests
uv run ruff check .          # lint
uv run ruff format .         # format (add --check to only verify)
uv run mypy src tests        # static type check (strict)
```

Frontend checks, from `frontend/`: `npx tsc -b`, `npm run build`, `npm run lint`.

### Pre-commit hook

A local pre-commit hook runs lint, format and type checks (via `uv run`, so it
matches the versions pinned in `uv.lock`) before every commit. On a fresh clone:

```bash
uv run pre-commit install
uv run pre-commit run --all-files    # run it manually
```

### Interface example

`examples/quickstart.py` is a small runnable script that defines a minimal
environment and model, checks they satisfy the protocols and are compatible,
and round-trips a `RunConfig` through the versioned JSON codec:

```bash
uv run python examples/quickstart.py
```
