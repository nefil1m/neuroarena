"""Entry point: `uv run neuroarena-play`. Wires the sample track, the Road_01 asset
manifest, and the arcade window together and runs the game loop."""

from __future__ import annotations

from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.render.scene import DEFAULT_MANIFEST_PATH
from neuroarena.render.window import PlayWindow
from neuroarena.sim.game import Game
from neuroarena.sim.track_io import loads as load_track

TRACK_PATH = Path(__file__).parents[1] / "sim/tracks/track_01.json"


def main() -> None:
    track = load_track(TRACK_PATH.read_text())
    game = Game(track)
    manifest = AssetManifest.load(DEFAULT_MANIFEST_PATH)
    PlayWindow(game, manifest)
    arcade.run()


if __name__ == "__main__":
    main()
