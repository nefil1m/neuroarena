"""The static part of the game's picture — grass background, asphalt tiles, kerbs and the
start/finish decal — shared by the play window and the viewer window. Imports `neuroarena.sim`
and `arcade`; never the reverse. A `TrackScene` allocates GPU `SpriteList`s, so it must be
created after an `arcade.Window` exists."""

from __future__ import annotations

import math
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.sim.track import DRIVABLE_WIDTH, Segment, Track, boundary_segments

DEFAULT_MANIFEST_PATH = Path(__file__).parent / "assets/road_01/manifest.json"
BACKGROUND_MARGIN_CELLS = 2
KERB_COLOR = arcade.color.WHITE_SMOKE
KERB_LINE_WIDTH = 10.0
KERB_ARC_STEPS = 24  # smoother than the collision resolver's default — this is visual only


def heading_to_sprite_angle(heading: float) -> float:
    """`arcade.Sprite.angle` rotates clockwise from the texture's native orientation.
    The car art's native (angle=0) nose points world-heading +90° (up) — confirmed by
    actually driving it: the first cut had this 180° off and drove the car tail-first."""
    return 90.0 - math.degrees(heading)


class TrackScene:
    def __init__(self, track: Track, manifest: AssetManifest) -> None:
        self.track = track
        self.manifest = manifest
        self.background_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.tile_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.decor_sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self.kerb_segments: list[Segment] = boundary_segments(track, arc_steps=KERB_ARC_STEPS)
        self._build_background()
        self._build_track()
        self._build_decor()

    def _build_background(self) -> None:
        cell_size = self.track.cell_size
        xs = [cell[0] for cell in self.track.cells]
        ys = [cell[1] for cell in self.track.cells]
        margin = BACKGROUND_MARGIN_CELLS
        for gx in range(min(xs) - margin, max(xs) + margin + 1):
            for gy in range(min(ys) - margin, max(ys) + margin + 1):
                sprite = arcade.Sprite(str(self.manifest.background["grass"]))
                sprite.width = cell_size
                sprite.height = cell_size
                sprite.center_x, sprite.center_y = gx * cell_size, gy * cell_size
                self.background_sprites.append(sprite)

    def _build_track(self) -> None:
        # One undirected, edge-to-edge asphalt fill per cell — no per-TileKind art or
        # rotation needed, so this can't reintroduce either the seam or the wrong-corner
        # bugs a rotated bitmap kerb had. Kerbs are drawn separately in `draw` from the
        # same boundary geometry the collision resolver uses (see `kerb_segments`).
        cell_size = self.track.cell_size
        for cell in self.track.cells:
            cx, cy = self.track.cell_center(cell)
            sprite = arcade.Sprite(str(self.manifest.surface))
            sprite.width = cell_size
            sprite.height = cell_size
            sprite.center_x, sprite.center_y = cx, cy
            self.tile_sprites.append(sprite)

    def _build_decor(self) -> None:
        cx, cy = self.track.cell_center(self.track.start_cell)
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
        sprite.angle = 90.0 - math.degrees(self.track.start_facing.heading_radians)
        self.decor_sprites.append(sprite)

    def draw(self) -> None:
        """Draws the whole static scene; the caller has already activated its camera."""
        self.background_sprites.draw()
        self.tile_sprites.draw()
        for (x1, y1), (x2, y2) in self.kerb_segments:
            arcade.draw_line(x1, y1, x2, y2, KERB_COLOR, KERB_LINE_WIDTH)
        self.decor_sprites.draw()
