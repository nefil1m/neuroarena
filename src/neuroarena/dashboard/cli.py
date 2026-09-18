"""Entry point: `uv run neuroarena-dashboard`. Starts the FastAPI dashboard backend —
local-first, binds `127.0.0.1` only, no auth (spec). See
`../../../docs/phases/phase-7-dashboard-control-panel.md`."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from neuroarena.dashboard.app import create_app
from neuroarena.dashboard.run_manager import RunManager

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PORT = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the neuroarena dashboard backend.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    run_manager = RunManager(data_dir=args.data_dir)
    app = create_app(run_manager=run_manager, data_dir=args.data_dir)
    # 127.0.0.1, not 0.0.0.0: local-first, no auth (spec) — never bind every interface.
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
