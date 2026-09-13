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
