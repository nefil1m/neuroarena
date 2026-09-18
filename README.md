# neuroarena

A 2D (later possibly 3D) simulation platform where games and learning models
are decoupled: build a driving game, train models to play it, swap in
different model types or games, and control everything — including
headless/background training — from a web dashboard.

See `docs/OVERVIEW.md` for the full pitch and settled decisions, and
`docs/PHASES.md` for the build plan.

## Status

**Phase 0 (Architecture Scaffolding) is complete.** This is a
pure-declaration interface layer only: `typing.Protocol` definitions
(`Environment`, `Model`, `Objective`, `Trainer`), space descriptors
(`Box`/`Discrete`) whose field-wise equality is the model/environment
compatibility rule, and a versioned JSON `RunConfig` envelope. There is
**no game and no trainer yet** — those land in later phases (see
`docs/phases/`). Nothing here renders or trains anything; what you can run
today is the test suite and a small script proving the pieces compose (see
Quickstart below).

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running the tests

```bash
uv run pytest -v
```

## Linting, formatting, and type checking

```bash
uv run ruff check .          # lint
uv run ruff format .         # format (drop --check to write changes)
uv run mypy src tests        # static type check (strict)
```

### Pre-commit hook

A local pre-commit hook runs all three (via `uv run`, so it always matches
the versions pinned in `uv.lock`) before every commit. On a fresh clone:

```bash
uv run pre-commit install
```

To run it manually against the whole repo without committing:

```bash
uv run pre-commit run --all-files
```

## Quickstart

Phase 0 has no game to launch, but `examples/quickstart.py` is a small,
runnable script that exercises the actual interface layer: it defines a
minimal Environment and Model, checks that both satisfy their Protocols,
runs `check_compatibility` on a matching pair (passes) and a mismatched pair
(raises with a descriptive error), and round-trips a `RunConfig` through the
versioned JSON codec.

```bash
uv run python examples/quickstart.py
```

## Training

`neuroarena-train` (Phase 6) launches a real NEAT training run against a saved track (generate one first with `neuroarena-track-gen`) and persists its progress — run history, checkpoints, and settings history — to a local SQLite database plus a `data/checkpoints/` directory:

```bash
uv run neuroarena-train --track-id <id-from-track-gen> --population-size 150
```

Resume a model's training later, inheriting its most recent settings, with `--resume <model_id>` (printed by the command above).

## Dashboard

`neuroarena-dashboard` (Phase 7) is the local web control panel: start or resume a training run, watch live metrics, and change a few settings mid-run. It has a FastAPI backend and a React frontend, run as two processes during development.

Backend (binds `127.0.0.1:8000` by default, no auth):

```bash
uv run neuroarena-dashboard [--data-dir DIR] [--port N]
```

Frontend (Vite dev server on `:5173`, proxies `/api` and `/ws` to the backend):

```bash
cd frontend && npm install && npm run dev
```
