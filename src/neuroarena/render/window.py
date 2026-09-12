"""arcade window: draws the track/decor/car and runs a fixed-step accumulator loop
decoupled from the display framerate. Imports `neuroarena.sim`; never the reverse."""

from __future__ import annotations

import math

import arcade

from neuroarena.render.input import KeyboardInput
from neuroarena.render.manifest import AssetManifest
from neuroarena.sim.game import TICK_DT, Game
from neuroarena.sim.track import DRIVABLE_WIDTH

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BACKGROUND_MARGIN_CELLS = 2


def _heading_to_sprite_angle(heading: float) -> float:
    """`arcade.Sprite.angle` rotates clockwise from the texture's native orientation.
    The car art's native (angle=0) nose points world-heading -90° (down) — empirically
    confirmed by rendering it at a few angles and inspecting where the nose landed."""
    return -90.0 - math.degrees(heading)


class PlayWindow(arcade.Window):
    def __init__(self, game: Game, manifest: AssetManifest) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "neuroarena — Phase 1")
        self.game = game
        self.manifest = manifest
        self.input = KeyboardInput()
        self._accumulator = 0.0

        self.background_color = arcade.color.DARK_SPRING_GREEN
        self.background_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.tile_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.decor_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.car_sprite = arcade.Sprite(str(manifest.car))
        self.car_sprite.width = game.constants.car_width
        self.car_sprite.height = game.constants.car_length

        self.camera = arcade.Camera2D()

        self._build_background()
        self._build_track()
        self._build_decor()
        self._sync_car_sprite()

    def _build_background(self) -> None:
        cell_size = self.game.track.cell_size
        xs = [cell[0] for cell in self.game.track.cells]
        ys = [cell[1] for cell in self.game.track.cells]
        margin = BACKGROUND_MARGIN_CELLS
        for gx in range(min(xs) - margin, max(xs) + margin + 1):
            for gy in range(min(ys) - margin, max(ys) + margin + 1):
                sprite = arcade.Sprite(str(self.manifest.background["grass"]))
                sprite.width = cell_size
                sprite.height = cell_size
                sprite.center_x, sprite.center_y = gx * cell_size, gy * cell_size
                self.background_sprites.append(sprite)

    def _build_track(self) -> None:
        cell_size = self.game.track.cell_size
        for cell, kind in self.game.track.cells.items():
            resolved = self.manifest.resolve(kind)
            cx, cy = self.game.track.cell_center(cell)
            for layer_path in resolved.paths:
                sprite = arcade.Sprite(str(layer_path))
                # Scale by width only, preserving aspect ratio — the vendored art isn't
                # always an exact cell_size square (some tiles carry a little vertical
                # bleed), and squashing it to force a square would visibly distort it.
                scale = cell_size / sprite.width
                sprite.width *= scale
                sprite.height *= scale
                sprite.center_x, sprite.center_y = cx, cy
                sprite.angle = resolved.rotate_degrees
                self.tile_sprites.append(sprite)

    def _build_decor(self) -> None:
        cx, cy = self.game.track.cell_center(self.game.track.start_cell)
        sprite = arcade.Sprite(str(self.manifest.decor["start_finish"]))
        # Native art is a wide banner (long axis = the checkered cross-lane line, short axis
        # = along-travel thickness) sized to the drivable width, not the full kerb-to-kerb cell.
        native_aspect = sprite.height / sprite.width
        sprite.width = DRIVABLE_WIDTH
        sprite.height = DRIVABLE_WIDTH * native_aspect
        sprite.center_x, sprite.center_y = cx, cy
        # Native (angle=0) long axis is horizontal, which already crosses a vertical (N/S)
        # road correctly; a horizontal (E/W) road needs the band rotated 90° to still cross
        # it perpendicular to travel.
        sprite.angle = 90.0 - math.degrees(self.game.track.start_facing.heading_radians)
        self.decor_sprites.append(sprite)

    def _sync_car_sprite(self) -> None:
        self.car_sprite.center_x = self.game.car.x
        self.car_sprite.center_y = self.game.car.y
        self.car_sprite.angle = _heading_to_sprite_angle(self.game.car.heading)
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
            self.background_sprites.draw()
            self.tile_sprites.draw()
            self.decor_sprites.draw()
            arcade.draw_sprite(self.car_sprite)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_press(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.input.on_key_release(symbol)
