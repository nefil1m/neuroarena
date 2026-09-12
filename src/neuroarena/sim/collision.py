"""Hard boundary collider: the car cannot cross the track's collision geometry.

The car body is approximated as a bounding circle (radius = half the car's width) rather
than a full oriented rectangle — a deliberate v1 simplification (implementation-level, not
a requirements change; see the Phase 1 doc's boundary-collision requirement, which only
fixes that crossing is blocked, not the exact collision shape).
"""

from __future__ import annotations

import math
from dataclasses import replace

from neuroarena.sim.physics import CarState
from neuroarena.sim.track import Point, Segment


def resolve_collision(
    state: CarState, previous: CarState, boundary: list[Segment], car_radius: float
) -> CarState:
    """Push the car out of any wall segment it overlaps, and kill the velocity component
    driving it into that wall. The kinematic model has no independent lateral velocity
    (see physics.py) — speed is always along `heading` — so a grazing hit only loses the
    head-on portion of that scalar speed and the car keeps moving, while a head-on hit
    zeroes it outright.

    `previous` (the car's state before this tick's movement, assumed already valid) decides
    which side of a wall is "inside": a wall is a zero-width line, so at low relative speed a
    single step can nudge the current position a hair across it, which would otherwise flip
    the push-out direction and shove the car further outside instead of back in.
    """
    x, y, heading, speed = state.x, state.y, state.heading, state.speed
    for segment in boundary:
        closest = _closest_point_on_segment((x, y), segment)
        dx, dy = x - closest[0], y - closest[1]
        distance = math.hypot(dx, dy)
        if distance >= car_radius:
            continue

        nx, ny = _inside_direction(previous, segment, dx, dy, distance)
        x = closest[0] + nx * car_radius
        y = closest[1] + ny * car_radius

        hx, hy = math.cos(heading), math.sin(heading)
        vx, vy = speed * hx, speed * hy
        into_wall = -(vx * nx + vy * ny)
        if into_wall > 0:
            vx += into_wall * nx
            vy += into_wall * ny
            speed = vx * hx + vy * hy

    return replace(state, x=x, y=y, speed=speed)


def _inside_direction(
    previous: CarState, segment: Segment, dx: float, dy: float, distance: float
) -> Point:
    ref_closest = _closest_point_on_segment((previous.x, previous.y), segment)
    rdx, rdy = previous.x - ref_closest[0], previous.y - ref_closest[1]
    rdist = math.hypot(rdx, rdy)
    if rdist > 0.0:
        return (rdx / rdist, rdy / rdist)
    if distance > 0.0:
        return (dx / distance, dy / distance)
    return (0.0, 0.0)


def _closest_point_on_segment(point: Point, segment: Segment) -> Point:
    (ax, ay), (bx, by) = segment
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq == 0.0:
        return (ax, ay)
    t = ((point[0] - ax) * abx + (point[1] - ay) * aby) / length_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * abx, ay + t * aby)
