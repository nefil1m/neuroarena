"""Boundary-proximity raycasting for the car's Phase 3 observation. Pure geometry — no
knowledge of Track/TileKind, just points, an origin, and an angle (see the Phase 3 doc,
"Raycasts")."""

from __future__ import annotations

import math
from collections.abc import Sequence

from neuroarena.sim.track import Point, Segment


def cast_ray(
    origin: Point, angle_rad: float, max_dist: float, boundary: list[Segment]
) -> float | None:
    """Distance from `origin` to the nearest boundary intersection within `max_dist` along
    `angle_rad` (world-frame radians, matching `CarState.heading`'s convention), or `None`
    if nothing is that close."""
    direction = (math.cos(angle_rad), math.sin(angle_rad))
    nearest: float | None = None
    for segment in boundary:
        hit = _ray_segment_intersection(origin, direction, segment)
        if hit is not None and hit <= max_dist and (nearest is None or hit < nearest):
            nearest = hit
    return nearest


def cast_rays(
    origin: Point,
    heading: float,
    angles_deg: Sequence[float],
    max_dist: float,
    boundary: list[Segment],
    car_radius: float,
) -> list[float]:
    """One proximity value per angle in `angles_deg` (degrees, relative to `heading`):
    `1 - effective_distance / max_dist`, clamped to `[0, 1]`. `effective_distance` measures
    from the car's bumper (its bounding circle, radius `car_radius` — see `collision.py`),
    not its center, so `1.0` means the same "touching the wall" state
    `collision.distance_to_boundary` reports."""
    proximities = []
    for deg in angles_deg:
        angle = heading + math.radians(deg)
        raw = cast_ray(origin, angle, max_dist + car_radius, boundary)
        if raw is None:
            proximities.append(0.0)
        else:
            effective = max(0.0, raw - car_radius)
            proximities.append(max(0.0, min(1.0, 1.0 - effective / max_dist)))
    return proximities


def _ray_segment_intersection(origin: Point, direction: Point, segment: Segment) -> float | None:
    """Parametric ray/segment intersection. `direction` must be a unit vector, so the
    returned value is a true Euclidean distance. `None` if parallel, behind `origin`, or
    outside the segment's span."""
    ox, oy = origin
    dx, dy = direction
    (ax, ay), (bx, by) = segment
    sx, sy = bx - ax, by - ay
    denom = sx * dy - sy * dx
    if abs(denom) < 1e-12:
        return None
    ex, ey = ax - ox, ay - oy
    t = (sx * ey - sy * ex) / denom
    s = (dx * ey - dy * ex) / denom
    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None
