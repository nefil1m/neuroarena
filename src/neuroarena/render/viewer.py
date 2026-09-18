"""The Phase 8 game viewer: an `arcade` window, run as its own process (opened by the
dashboard backend), that draws every still-active car of a training run's current generation
live on the real track art. It only draws — what to follow, the camera, and the overlay text
are decided by `view_model.py`; the backend connection is `viewer_client.py`. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`. Run manually with
`uv run python -m neuroarena.render.viewer --url ws://127.0.0.1:8000/ws/viewer`."""

from __future__ import annotations

import argparse
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.render.scene import DEFAULT_MANIFEST_PATH, TrackScene, heading_to_sprite_angle
from neuroarena.render.view_model import (
    CarView,
    ViewModel,
    overlay_lines,
    rank_cars,
    rank_of,
    track_bounds,
)
from neuroarena.render.viewer_client import ViewerClient, ViewerState
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.tracks import store as tracks_store

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DEFAULT_URL = "ws://127.0.0.1:8000/ws/viewer"
GHOST_ALPHA = 150
RING_RADIUS = 90.0
RING_WIDTH = 6
BEST_RING_COLOR = arcade.color.YELLOW
FOLLOWED_RING_COLOR = arcade.color.WHITE
OVERLAY_ROWS = 6


class ViewerWindow(arcade.Window):
    def __init__(self, client: ViewerClient, data_dir: Path) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "neuroarena — game viewer")
        self._client = client
        self._tracks_dir = data_dir / "tracks"
        self._manifest = AssetManifest.load(DEFAULT_MANIFEST_PATH)
        # The viewer draws the sprite at the default car size; a run configured with custom
        # physics constants would draw slightly off-size ghosts (accepted, see the plan notes).
        self._constants = PhysicsConstants()
        self.background_color = arcade.color.DARK_SPRING_GREEN
        self._camera = arcade.Camera2D()
        self._scene: TrackScene | None = None
        self._view: ViewModel | None = None
        self._loaded_track_id: str | None = None
        self._track_error: str | None = None
        self._sprites: list[arcade.Sprite] = []
        self._sprite_list: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self._overlay = [
            arcade.Text("", 14, WINDOW_HEIGHT - 26 - row * 22, arcade.color.WHITE, 15)
            for row in range(OVERLAY_ROWS)
        ]

    def _ensure_scene(self, track_id: str | None) -> None:
        if track_id is None or track_id == self._loaded_track_id:
            return
        self._loaded_track_id = track_id
        self._scene = None
        self._view = None
        self._track_error = None
        try:
            record = tracks_store.load(track_id, tracks_dir=self._tracks_dir)
            self._scene = TrackScene(record.track, self._manifest)
            self._view = ViewModel(track_bounds(record.track), (self.width, self.height))
        except Exception as exc:  # any load failure shows an overlay; it must not kill the viewer
            self._track_error = f"Could not load track {track_id}: {exc}"

    def _ensure_sprites(self, count: int) -> None:
        while len(self._sprites) < count:
            sprite = arcade.Sprite(str(self._manifest.car))
            sprite.width = self._constants.car_width
            sprite.height = self._constants.car_length
            sprite.alpha = 0
            self._sprites.append(sprite)
            self._sprite_list.append(sprite)

    def _place_cars(self, ranked: list[CarView]) -> None:
        """Sprite `k` shows the car at rank `len - 1 - k`, so the best car is the last sprite
        in the list and is drawn on top. Unused sprites are made fully transparent."""
        self._ensure_sprites(len(ranked))
        for index, sprite in enumerate(self._sprites):
            if index >= len(ranked):
                sprite.alpha = 0
                continue
            rank_index = len(ranked) - 1 - index
            car = ranked[rank_index]
            sprite.center_x, sprite.center_y = car.x, car.y
            sprite.angle = heading_to_sprite_angle(car.heading)
            sprite.alpha = 255 if rank_index == 0 else GHOST_ALPHA

    def _draw_rings(self, ranked: list[CarView], followed: CarView | None) -> None:
        if not ranked:
            return
        best = ranked[0]
        arcade.draw_circle_outline(best.x, best.y, RING_RADIUS, BEST_RING_COLOR, RING_WIDTH)
        if followed is not None and followed.genome_id != best.genome_id:
            arcade.draw_circle_outline(
                followed.x, followed.y, RING_RADIUS, FOLLOWED_RING_COLOR, RING_WIDTH
            )

    def _draw_overlay(self, state: ViewerState, followed_rank: int | None) -> None:
        lines = overlay_lines(
            connected=state.connected,
            run_state=state.run_state,
            generation=state.generation,
            alive=len(state.cars),
            population_size=state.population_size,
            speed=state.speed,
            settings=state.settings,
            followed_rank=followed_rank,
        )
        if self._track_error is not None:
            lines.append(self._track_error)
        for row, text in enumerate(self._overlay):
            text.text = lines[row] if row < len(lines) else ""
            text.draw()

    def on_draw(self) -> None:
        state = self._client.state()
        self._ensure_scene(state.track_id)
        self.clear()
        followed_rank: int | None = None
        if self._scene is not None and self._view is not None:
            ranked = rank_cars(state.cars)
            followed = self._view.followed_car(state.cars, state.settings)
            camera = self._view.camera(followed, state.settings)
            self._camera.position = (camera.center_x, camera.center_y)
            self._camera.zoom = camera.scale
            self._place_cars(ranked)
            with self._camera.activate():
                self._scene.draw()
                self._sprite_list.draw()
                self._draw_rings(ranked, followed)
            followed_rank = None if followed is None else rank_of(state.cars, followed)
        self._draw_overlay(state, followed_rank)
        if self._client.should_exit():
            arcade.exit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="neuroarena game viewer (opened by the dashboard)."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="the dashboard's /ws/viewer URL")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    client = ViewerClient(args.url)
    client.start()
    ViewerWindow(client, args.data_dir)
    try:
        arcade.run()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
