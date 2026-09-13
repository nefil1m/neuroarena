"""Grid-based closed-loop track representation, validation, and collision geometry.

A track is a closed loop of grid cells, each tagged with a `TileKind`. Two adjacent cells
are connected exactly when each opens an edge toward the other (see `TileKind.open_edges`).
Collision geometry (`boundary_segments`) is derived from the same cells — there is no
separate freeform geometry model (see the Phase 1 doc, "Track representation").
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

CELL_SIZE = 512.0
"""World-unit size of one grid cell, tied to the vendored art's native tile pixel size."""

DRIVABLE_WIDTH = 326.0
"""World-unit width of the drivable corridor, tied to the vendored art's kerb geometry."""

GridCell = tuple[int, int]
Point = tuple[float, float]
Segment = tuple[Point, Point]


class Facing(Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"

    @property
    def delta(self) -> GridCell:
        return _FACING_DELTA[self]

    @property
    def opposite(self) -> Facing:
        return _FACING_OPPOSITE[self]

    @property
    def heading_radians(self) -> float:
        """World heading (0 = +x/East, increasing counter-clockwise) a car faces on spawn."""
        return _FACING_HEADING[self]


_FACING_DELTA: dict[Facing, GridCell] = {
    Facing.N: (0, 1),
    Facing.E: (1, 0),
    Facing.S: (0, -1),
    Facing.W: (-1, 0),
}
_FACING_OPPOSITE: dict[Facing, Facing] = {
    Facing.N: Facing.S,
    Facing.S: Facing.N,
    Facing.E: Facing.W,
    Facing.W: Facing.E,
}
_FACING_HEADING: dict[Facing, float] = {
    Facing.E: 0.0,
    Facing.N: math.pi / 2,
    Facing.W: math.pi,
    Facing.S: -math.pi / 2,
}


class TileKind(Enum):
    STRAIGHT_NS = "STRAIGHT_NS"
    STRAIGHT_EW = "STRAIGHT_EW"
    CURVE_NE = "CURVE_NE"
    CURVE_NW = "CURVE_NW"
    CURVE_SE = "CURVE_SE"
    CURVE_SW = "CURVE_SW"

    @property
    def open_edges(self) -> frozenset[Facing]:
        return _OPEN_EDGES[self]

    @property
    def is_curve(self) -> bool:
        return self not in (TileKind.STRAIGHT_NS, TileKind.STRAIGHT_EW)


_OPEN_EDGES: dict[TileKind, frozenset[Facing]] = {
    TileKind.STRAIGHT_NS: frozenset({Facing.N, Facing.S}),
    TileKind.STRAIGHT_EW: frozenset({Facing.E, Facing.W}),
    TileKind.CURVE_NE: frozenset({Facing.N, Facing.E}),
    TileKind.CURVE_NW: frozenset({Facing.N, Facing.W}),
    TileKind.CURVE_SE: frozenset({Facing.S, Facing.E}),
    TileKind.CURVE_SW: frozenset({Facing.S, Facing.W}),
}

_EDGES_TO_KIND: dict[frozenset[Facing], TileKind] = {
    edges: kind for kind, edges in _OPEN_EDGES.items()
}


def tile_kind_for_open_edges(a: Facing, b: Facing) -> TileKind:
    """Inverse of `TileKind.open_edges`: the TileKind whose two open edges are exactly {a, b}."""
    try:
        return _EDGES_TO_KIND[frozenset({a, b})]
    except KeyError:
        raise ValueError(f"no TileKind connects {a.value} and {b.value}") from None


class TrackValidationError(ValueError):
    """Raised when a track's cells do not form a single, edge-consistent closed loop."""


@dataclass(frozen=True)
class Track:
    cells: dict[GridCell, TileKind]
    start_cell: GridCell
    start_facing: Facing
    cell_size: float = CELL_SIZE

    def __post_init__(self) -> None:
        validate(self)

    def cell_center(self, cell: GridCell) -> Point:
        cx, cy = cell
        return (cx * self.cell_size, cy * self.cell_size)


def validate(track: Track) -> None:
    cells = track.cells
    if len(cells) < 4:
        raise TrackValidationError("a 90°-turn-only closed loop needs at least 4 cells")
    if track.start_cell not in cells:
        raise TrackValidationError(f"start_cell {track.start_cell} is not part of the track")
    start_kind = cells[track.start_cell]
    if track.start_facing not in start_kind.open_edges:
        raise TrackValidationError(
            f"start_facing {track.start_facing.value} is not an open edge of start_cell "
            f"{track.start_cell} ({start_kind.value}); open edges are "
            f"{sorted(f.value for f in start_kind.open_edges)}"
        )

    for cell, kind in cells.items():
        for facing in kind.open_edges:
            dx, dy = facing.delta
            neighbor = (cell[0] + dx, cell[1] + dy)
            neighbor_kind = cells.get(neighbor)
            if neighbor_kind is None:
                raise TrackValidationError(
                    f"cell {cell} ({kind.value}) opens {facing.value} onto {neighbor}, "
                    "which is not part of the track"
                )
            if facing.opposite not in neighbor_kind.open_edges:
                raise TrackValidationError(
                    f"cell {cell} ({kind.value}) opens {facing.value} onto {neighbor} "
                    f"({neighbor_kind.value}), which does not open back {facing.opposite.value}"
                )

    _assert_single_loop(cells)


def _assert_single_loop(cells: dict[GridCell, TileKind]) -> None:
    # Per-edge validation above already guarantees every cell has exactly two open edges,
    # each matched by a consenting neighbor — so the cell graph is 2-regular and decomposes
    # into disjoint simple cycles. Walking from an arbitrary cell until we return to it
    # confirms there is exactly one cycle (i.e. it covers every cell).
    start = next(iter(cells))
    visited: set[GridCell] = set()
    previous: GridCell | None = None
    current = start
    while True:
        visited.add(current)
        neighbors = [
            (current[0] + facing.delta[0], current[1] + facing.delta[1])
            for facing in cells[current].open_edges
        ]
        next_cell = neighbors[0] if neighbors[1] == previous else neighbors[1]
        if next_cell == start:
            break
        if next_cell in visited:
            raise TrackValidationError(
                f"cell {next_cell} revisited before returning to start {start} — "
                "the track is not a single simple loop"
            )
        previous, current = current, next_cell
    if len(visited) != len(cells):
        raise TrackValidationError(
            f"track has {len(cells)} cells but the loop from {start} only reaches "
            f"{len(visited)} of them — there is more than one closed loop"
        )


def boundary_segments(
    track: Track, drivable_width: float = DRIVABLE_WIDTH, arc_steps: int = 8
) -> list[Segment]:
    """The collision polyline: two walls (inner/outer kerb) per cell, straight for
    `STRAIGHT_*` tiles and arced for `CURVE_*` tiles, forming the drivable corridor."""
    segments: list[Segment] = []
    for cell, kind in track.cells.items():
        center = track.cell_center(cell)
        if kind.is_curve:
            segments.extend(_curve_walls(center, kind, track.cell_size, drivable_width, arc_steps))
        else:
            segments.extend(_straight_walls(center, kind, track.cell_size, drivable_width))
    return segments


def _straight_walls(
    center: Point, kind: TileKind, cell_size: float, drivable_width: float
) -> list[Segment]:
    half = cell_size / 2
    drivable_half = drivable_width / 2
    cx, cy = center
    if kind is TileKind.STRAIGHT_NS:
        return [
            ((cx - drivable_half, cy - half), (cx - drivable_half, cy + half)),
            ((cx + drivable_half, cy - half), (cx + drivable_half, cy + half)),
        ]
    return [
        ((cx - half, cy - drivable_half), (cx + half, cy - drivable_half)),
        ((cx - half, cy + drivable_half), (cx + half, cy + drivable_half)),
    ]


def _curve_walls(
    center: Point,
    kind: TileKind,
    cell_size: float,
    drivable_width: float,
    arc_steps: int,
) -> list[Segment]:
    half = cell_size / 2
    drivable_half = drivable_width / 2
    inner_radius = half - drivable_half
    outer_radius = half + drivable_half
    a, b = kind.open_edges
    pivot = (
        center[0] + (a.delta[0] + b.delta[0]) * half,
        center[1] + (a.delta[1] + b.delta[1]) * half,
    )
    edge_a = (center[0] + a.delta[0] * half, center[1] + a.delta[1] * half)
    edge_b = (center[0] + b.delta[0] * half, center[1] + b.delta[1] * half)
    angle_a = math.atan2(edge_a[1] - pivot[1], edge_a[0] - pivot[0])
    angle_b = math.atan2(edge_b[1] - pivot[1], edge_b[0] - pivot[0])
    # Sweep the short way (±90°) from angle_a, not naively between min(a,b) and max(a,b) —
    # atan2's ±π branch cut would otherwise pick the outer 270° sweep for some corners.
    delta = (angle_b - angle_a + math.pi) % (2 * math.pi) - math.pi
    angle_end = angle_a + delta
    return [
        *_arc(pivot, inner_radius, angle_a, angle_end, arc_steps),
        *_arc(pivot, outer_radius, angle_a, angle_end, arc_steps),
    ]


def _arc(
    pivot: Point, radius: float, angle_from: float, angle_to: float, steps: int
) -> Iterator[Segment]:
    points = [
        (
            pivot[0] + radius * math.cos(angle_from + (angle_to - angle_from) * i / steps),
            pivot[1] + radius * math.sin(angle_from + (angle_to - angle_from) * i / steps),
        )
        for i in range(steps + 1)
    ]
    return iter(zip(points, points[1:], strict=False))


def track_loop_length(track: Track) -> float:
    """Total path length of the loop's centerline — radius `cell_size / 2`, the midpoint
    between the inner/outer collision walls `boundary_segments` derives (independent of
    `drivable_width`, which offsets both walls equally). A straight tile contributes
    `cell_size`; a curve tile contributes a quarter-circle arc, `(pi / 2) * (cell_size / 2)`.
    Used to turn Phase 3's raw `progress` into `lap_progress`, a fraction of one lap."""
    straight_count = sum(1 for kind in track.cells.values() if not kind.is_curve)
    curve_count = len(track.cells) - straight_count
    return straight_count * track.cell_size + curve_count * (math.pi / 2) * (track.cell_size / 2)


def centerline_path(track: Track, arc_steps: int = 8) -> list[Point]:
    """An ordered polyline approximating the loop's centerline (radius `cell_size / 2`, the
    same centerline `track_loop_length` measures analytically), starting where the car
    spawns and proceeding around the loop in the `start_facing` direction. Used by Phase 3's
    `CarEnvironment` to measure how far around the track the car has actually travelled, by
    projecting its real position onto this path (`project_onto_centerline`) rather than
    measuring raw distance driven — a car spinning in place would inflate raw distance
    without ever advancing here, since the projection tracks net position, not motion."""
    order = _ordered_cells(track)
    n = len(order)
    points: list[Point] = []
    for i, cell in enumerate(order):
        kind = track.cells[cell]
        exit_facing = _facing_between(cell, order[(i + 1) % n])
        entry_facing = next(f for f in kind.open_edges if f != exit_facing)
        piece = (
            _centerline_arc(
                track.cell_center(cell), track.cell_size, entry_facing, exit_facing, arc_steps
            )
            if kind.is_curve
            else _centerline_straight(
                track.cell_center(cell), track.cell_size, entry_facing, exit_facing
            )
        )
        points.extend(piece if i == 0 else piece[1:])
    return points


def project_onto_centerline(point: Point, centerline: list[Point]) -> float:
    """Arc length from `centerline[0]` to the nearest point on the closed polyline
    `centerline` (wrapping from the last point back to the first) to `point`, wrapped to
    `[0, total_length)` where `total_length` is this polyline's own length (very close to,
    but not exactly, `track_loop_length`'s analytic value, since this is a discretized
    approximation of the same centerline)."""
    segments = list(zip(centerline, centerline[1:] + centerline[:1], strict=True))
    best_distance = math.inf
    best_arc_length = 0.0
    cumulative = 0.0
    for a, b in segments:
        abx, aby = b[0] - a[0], b[1] - a[1]
        length = math.hypot(abx, aby)
        t = (
            0.0
            if length == 0.0
            else max(
                0.0,
                min(1.0, ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / (length * length)),
            )
        )
        closest = (a[0] + t * abx, a[1] + t * aby)
        distance = math.hypot(point[0] - closest[0], point[1] - closest[1])
        if distance < best_distance:
            best_distance = distance
            best_arc_length = cumulative + t * length
        cumulative += length
    return best_arc_length


def _ordered_cells(track: Track) -> list[GridCell]:
    """Cells in traversal order starting at `start_cell`, first step toward `start_facing`,
    then following the loop (mirrors the walk `_assert_single_loop` already does, but seeded
    to start at a specific cell/direction instead of an arbitrary one)."""
    order = [track.start_cell]
    dx, dy = track.start_facing.delta
    current = (track.start_cell[0] + dx, track.start_cell[1] + dy)
    previous = track.start_cell
    while current != track.start_cell:
        order.append(current)
        neighbors = [
            (current[0] + facing.delta[0], current[1] + facing.delta[1])
            for facing in track.cells[current].open_edges
        ]
        next_cell = neighbors[0] if neighbors[1] == previous else neighbors[1]
        previous, current = current, next_cell
    return order


def _facing_between(a: GridCell, b: GridCell) -> Facing:
    delta = (b[0] - a[0], b[1] - a[1])
    for facing in Facing:
        if facing.delta == delta:
            return facing
    raise ValueError(f"{a} and {b} are not grid-adjacent")


def _centerline_straight(
    center: Point, cell_size: float, entry_facing: Facing, exit_facing: Facing
) -> list[Point]:
    half = cell_size / 2
    entry_point = (
        center[0] + entry_facing.delta[0] * half,
        center[1] + entry_facing.delta[1] * half,
    )
    exit_point = (center[0] + exit_facing.delta[0] * half, center[1] + exit_facing.delta[1] * half)
    return [entry_point, exit_point]


def _centerline_arc(
    center: Point, cell_size: float, entry_facing: Facing, exit_facing: Facing, arc_steps: int
) -> list[Point]:
    half = cell_size / 2
    pivot = (
        center[0] + (entry_facing.delta[0] + exit_facing.delta[0]) * half,
        center[1] + (entry_facing.delta[1] + exit_facing.delta[1]) * half,
    )
    entry_point = (
        center[0] + entry_facing.delta[0] * half,
        center[1] + entry_facing.delta[1] * half,
    )
    exit_point = (center[0] + exit_facing.delta[0] * half, center[1] + exit_facing.delta[1] * half)
    angle_entry = math.atan2(entry_point[1] - pivot[1], entry_point[0] - pivot[0])
    angle_exit = math.atan2(exit_point[1] - pivot[1], exit_point[0] - pivot[0])
    delta = (angle_exit - angle_entry + math.pi) % (2 * math.pi) - math.pi
    angle_end = angle_entry + delta
    return [
        (
            pivot[0] + half * math.cos(angle_entry + (angle_end - angle_entry) * i / arc_steps),
            pivot[1] + half * math.sin(angle_entry + (angle_end - angle_entry) * i / arc_steps),
        )
        for i in range(arc_steps + 1)
    ]
