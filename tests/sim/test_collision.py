import math

import pytest

from neuroarena.sim.collision import distance_to_boundary, resolve_collision
from neuroarena.sim.physics import CarState

WALL = ((-100.0, 0.0), (100.0, 0.0))
RADIUS = 10.0


def test_car_far_from_any_wall_is_untouched() -> None:
    previous = CarState(x=0, y=500, heading=0.3, speed=42.0)
    state = CarState(x=0, y=498, heading=0.3, speed=42.0)
    result = resolve_collision(state, previous, [WALL], RADIUS)
    assert result == state


def test_head_on_hit_stops_the_car_at_the_wall() -> None:
    # Car approaching from above, overlapping the wall, heading straight down into it.
    previous = CarState(x=0, y=8, heading=-math.pi / 2, speed=50.0)
    state = CarState(x=0, y=5, heading=-math.pi / 2, speed=50.0)
    result = resolve_collision(state, previous, [WALL], RADIUS)
    assert math.isclose(result.y, RADIUS)
    assert math.isclose(result.x, 0.0, abs_tol=1e-9)
    assert math.isclose(result.speed, 0.0, abs_tol=1e-9)


def test_car_does_not_cross_the_wall_even_after_a_small_step_past_it() -> None:
    # Previous position was above the wall (the valid side); this tick's raw position
    # already nudged slightly past the wall line — the push-out must still send it back
    # up, not further down, or the car would tunnel through on a low-speed graze.
    previous = CarState(x=0, y=1.5, heading=-math.pi / 2, speed=50.0)
    state = CarState(x=0, y=-0.5, heading=-math.pi / 2, speed=50.0)
    result = resolve_collision(state, previous, [WALL], RADIUS)
    assert result.y >= RADIUS - 1e-9


def test_grazing_hit_at_45_degrees_keeps_half_the_speed() -> None:
    previous = CarState(x=-1, y=8, heading=-math.pi / 4, speed=100.0)
    state = CarState(x=0, y=5, heading=-math.pi / 4, speed=100.0)  # down-right, 45° off normal
    result = resolve_collision(state, previous, [WALL], RADIUS)
    assert math.isclose(result.speed, 50.0, rel_tol=1e-6)
    assert result.speed > 0  # tangential motion survives — it isn't a dead stop


def test_moving_away_from_an_overlapped_wall_keeps_its_speed() -> None:
    # Overlapping the wall but already heading away from it (e.g. just after a bounce-back).
    previous = CarState(x=0, y=4, heading=math.pi / 2, speed=50.0)
    state = CarState(x=0, y=5, heading=math.pi / 2, speed=50.0)
    result = resolve_collision(state, previous, [WALL], RADIUS)
    assert result.speed == 50.0
    assert math.isclose(result.y, RADIUS)


def test_distance_to_boundary_zero_when_on_the_wall() -> None:
    assert distance_to_boundary((0.0, 0.0), [WALL]) == pytest.approx(0.0)


def test_distance_to_boundary_positive_when_clear() -> None:
    assert distance_to_boundary((0.0, 50.0), [WALL]) == pytest.approx(50.0)


def test_distance_to_boundary_is_the_nearest_of_several_segments() -> None:
    near = ((-10.0, 10.0), (10.0, 10.0))
    far = ((-10.0, 100.0), (10.0, 100.0))
    assert distance_to_boundary((0.0, 0.0), [far, near]) == pytest.approx(10.0)
