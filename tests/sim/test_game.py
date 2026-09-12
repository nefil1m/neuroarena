import math

from neuroarena.sim.game import Game
from neuroarena.sim.physics import CarState
from neuroarena.sim.track import Facing, TileKind, Track


def _track() -> Track:
    return Track(
        cells={
            (0, 0): TileKind.CURVE_NE,
            (1, 0): TileKind.STRAIGHT_EW,
            (2, 0): TileKind.STRAIGHT_EW,
            (3, 0): TileKind.CURVE_NW,
            (3, 1): TileKind.STRAIGHT_NS,
            (3, 2): TileKind.CURVE_SW,
            (2, 2): TileKind.STRAIGHT_EW,
            (1, 2): TileKind.STRAIGHT_EW,
            (0, 2): TileKind.CURVE_SE,
            (0, 1): TileKind.STRAIGHT_NS,
        },
        start_cell=(1, 0),
        start_facing=Facing.E,
    )


def test_car_spawns_at_start_cell_and_facing() -> None:
    game = Game(_track())
    cx, cy = game.track.cell_center((1, 0))
    assert game.car.x == cx
    assert game.car.y == cy
    assert math.isclose(game.car.heading, Facing.E.heading_radians)
    assert game.car.speed == 0.0


def test_repeated_ticks_are_deterministic() -> None:
    inputs = [(0.0, 1.0)] * 30 + [(0.5, 1.0)] * 30 + [(-1.0, -1.0)] * 10
    game_a, game_b = Game(_track()), Game(_track())
    for steering, throttle in inputs:
        state_a = game_a.tick(steering, throttle)
        state_b = game_b.tick(steering, throttle)
        assert state_a.car == state_b.car


def test_tick_advances_speed_under_throttle() -> None:
    game = Game(_track())
    before = game.car.speed
    after = game.tick(0.0, 1.0).car.speed
    assert after > before


def test_car_cannot_cross_track_boundary() -> None:
    # Place the car overlapping (1, 0)'s north kerb, heading straight into it (head-on).
    game = Game(_track())
    cx, cy = game.track.cell_center((1, 0))
    half_drivable = 326.0 / 2
    car_radius = game.constants.car_width / 2
    game.car = CarState(x=cx, y=cy + half_drivable - 1.0, heading=math.pi / 2, speed=100.0)

    state = game.tick(0.0, 0.0)

    assert (cy + half_drivable) - state.car.y >= car_radius - 1e-6
    assert math.isclose(state.car.speed, 0.0, abs_tol=1e-6)
