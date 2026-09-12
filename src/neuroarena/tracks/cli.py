"""Entry point: `uv run neuroarena-track-gen`. Generates a map, saves it, and optionally
previews it by reusing Phase 1's Game + PlayWindow + AssetManifest renderer — drivable,
not just a static render (see the Phase 1 doc's revision history: only actually driving a
track caught the heading/kerb bugs a screenshot review had missed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.render.window import PlayWindow
from neuroarena.sim.game import Game
from neuroarena.sim.track_generation import generate_track
from neuroarena.tracks.store import DEFAULT_TRACKS_DIR, TrackRecord, save

MANIFEST_PATH = Path(__file__).parents[1] / "render/assets/road_01/manifest.json"


def run(
    size: int, complexity: float, seed: int, tracks_dir: Path = DEFAULT_TRACKS_DIR
) -> TrackRecord:
    track = generate_track(size, complexity, seed)
    return save(track, requested_size=size, complexity=complexity, seed=seed, tracks_dir=tracks_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a procedural closed-loop track.")
    parser.add_argument("--size", type=int, required=True, help="target tile count")
    parser.add_argument("--complexity", type=float, default=0.5, help="turn-frequency bias, 0-1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tracks-dir", type=Path, default=DEFAULT_TRACKS_DIR)
    parser.add_argument(
        "--preview", action="store_true", help="open an arcade window to drive the result"
    )
    args = parser.parse_args()

    record = run(args.size, args.complexity, args.seed, args.tracks_dir)
    drift_pct = 100 * (record.actual_size - record.requested_size) / record.requested_size
    print(
        f"generated track_id={record.track_id} actual_size={record.actual_size} "
        f"(requested {record.requested_size}, drift {drift_pct:+.1f}%) -> {args.tracks_dir}"
    )

    if args.preview:
        game = Game(record.track)
        manifest = AssetManifest.load(MANIFEST_PATH)
        PlayWindow(game, manifest)
        arcade.run()


if __name__ == "__main__":
    main()
