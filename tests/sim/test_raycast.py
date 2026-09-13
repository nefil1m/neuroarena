import math

import pytest

from neuroarena.sim.raycast import cast_ray, cast_rays


def test_cast_ray_hits_wall_directly_ahead() -> None:
    boundary = [((100.0, -50.0), (100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance == pytest.approx(100.0)


def test_cast_ray_returns_none_beyond_max_dist() -> None:
    boundary = [((300.0, -50.0), (300.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_ignores_walls_behind_the_origin() -> None:
    boundary = [((-100.0, -50.0), (-100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_respects_segment_span() -> None:
    # Wall segment ends well short of the ray's y=0 line -> no intersection.
    boundary = [((100.0, 10.0), (100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_at_45_degrees() -> None:
    boundary = [((100.0, -200.0), (100.0, 200.0))]
    distance = cast_ray((0.0, 0.0), math.pi / 4, 200.0, boundary)
    assert distance == pytest.approx(100.0 * math.sqrt(2))


def test_cast_rays_returns_one_proximity_per_angle() -> None:
    boundary = [((100.0, -50.0), (100.0, 50.0))]
    proximities = cast_rays((0.0, 0.0), 0.0, (-10.0, 0.0, 10.0), 200.0, boundary, car_radius=0.0)
    assert len(proximities) == 3
    assert proximities[1] == pytest.approx(1.0 - 100.0 / 200.0)


def test_cast_rays_clamps_no_hit_to_zero() -> None:
    proximities = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, [], car_radius=0.0)
    assert proximities == [0.0]


def test_cast_rays_reads_one_when_bumper_touches_wall() -> None:
    boundary = [((50.0, -50.0), (50.0, 50.0))]
    proximities = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, boundary, car_radius=50.0)
    assert proximities[0] == pytest.approx(1.0)


def test_cast_rays_heading_rotates_the_fan() -> None:
    # A wall to the north (+y): only visible once heading points the ray that way.
    boundary = [((-50.0, 100.0), (50.0, 100.0))]
    facing_east = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, boundary, car_radius=0.0)
    facing_north = cast_rays((0.0, 0.0), math.pi / 2, (0.0,), 200.0, boundary, car_radius=0.0)
    assert facing_east == [0.0]
    assert facing_north[0] == pytest.approx(1.0 - 100.0 / 200.0)


def test_cast_rays_handles_zero_max_dist_without_dividing_by_zero() -> None:
    boundary = [((10.0, -50.0), (10.0, 50.0))]
    proximities = cast_rays((0.0, 0.0), 0.0, (0.0,), 0.0, boundary, car_radius=10.0)
    assert proximities == [1.0]  # wall sits exactly at the bumper (raw=10=car_radius)

    proximities_no_hit = cast_rays((0.0, 0.0), 0.0, (0.0,), 0.0, [], car_radius=10.0)
    assert proximities_no_hit == [0.0]
