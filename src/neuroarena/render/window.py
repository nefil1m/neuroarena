"""arcade window: draws the track/decor/car and runs a fixed-step accumulator loop
decoupled from the display framerate. The static track drawing lives in `scene.py`.
Imports `neuroarena.sim`; never the reverse."""

from __future__ import annotations

import arcade

from neuroarena.render.input import KeyboardInput
from neuroarena.render.manifest import AssetManifest
from neuroarena.render.scene import TrackScene, heading_to_sprite_angle
from neuroarena.sim.game import TICK_DT, Game

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class PlayWindow(arcade.Window):
    def __init__(self, game: Game, manifest: AssetManifest) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "neuroarena — Phase 1")
        self.game = game
        self.manifest = manifest
        self.input = KeyboardInput()
        self._accumulator = 0.0

        self.background_color = arcade.color.DARK_SPRING_GREEN
        self.scene = TrackScene(game.track, manifest)
        self.car_sprite = arcade.Sprite(str(manifest.car))
        self.car_sprite.width = game.constants.car_width
        self.car_sprite.height = game.constants.car_length

        self.camera = arcade.Camera2D()
        self._sync_car_sprite()

    def _sync_car_sprite(self) -> None:
        self.car_sprite.center_x = self.game.car.x
        self.car_sprite.center_y = self.game.car.y
        self.car_sprite.angle = heading_to_sprite_angle(self.game.car.heading)
        self.camera.position = (self.car_sprite.center_x, self.car_sprite.center_y)

    def on_update(self, delta_time: float) -> None:
        steering, throttle = self.input.poll()
        self._accumulator += delta_time
        while self._accumulator >= TICK_DT:
            self.game.tick(steering, throttle)
            self._accumulator -= TICK_DT
        self._sync_car_sprite()

    def on_draw(self) -> None:
        self.clear()
        with self.camera.activate():
            self.scene.draw()
            arcade.draw_sprite(self.car_sprite)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_press(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_release(symbol)
