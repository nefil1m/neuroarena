import math

import pytest

from neuroarena.sim.track import (
    CELL_SIZE,
    DRIVABLE_WIDTH,
    Facing,
    GridCell,
    TileKind,
    Track,
    TrackValidationError,
    boundary_segments,
    tile_kind_for_open_edges,
)


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    from neuroarena.sim.track import TileKind as K

    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def test_valid_loop_constructs() -> None:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    assert track.cell_size == CELL_SIZE
    assert len(track.cells) == 10


def test_dangling_open_edge_is_rejected() -> None:
    cells = _rounded_rectangle()
    del cells[(0, 1)]  # leaves (0,0) and (0,2) each opening onto a missing cell
    with pytest.raises(TrackValidationError):
        Track(cells=cells, start_cell=(1, 0), start_facing=Facing.E)


def test_mismatched_neighbor_opening_is_rejected() -> None:
    from neuroarena.sim.track import TileKind as K

    cells = _rounded_rectangle()
    cells[(0, 1)] = K.STRAIGHT_EW  # opens E/W, but (0,0)/(0,2) both expect it to open N/S
    with pytest.raises(TrackValidationError):
        Track(cells=cells, start_cell=(1, 0), start_facing=Facing.E)


def test_two_disjoint_loops_are_rejected() -> None:
    from neuroarena.sim.track import TileKind as K

    cells = _rounded_rectangle()
    # A second, disconnected 4-cell loop far away from the first.
    cells[(10, 10)] = K.CURVE_NE
    cells[(11, 10)] = K.CURVE_NW
    cells[(11, 11)] = K.CURVE_SW
    cells[(10, 11)] = K.CURVE_SE
    with pytest.raises(TrackValidationError):
        Track(cells=cells, start_cell=(1, 0), start_facing=Facing.E)


def test_start_cell_must_be_in_track() -> None:
    with pytest.raises(TrackValidationError):
        Track(cells=_rounded_rectangle(), start_cell=(99, 99), start_facing=Facing.E)


def test_start_facing_must_be_an_open_edge_of_start_cell() -> None:
    # (1, 0) is STRAIGHT_EW (open edges {E, W}); Facing.N is not one of them.
    with pytest.raises(TrackValidationError):
        Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.N)


def test_boundary_segments_count_matches_tile_mix() -> None:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    segments = boundary_segments(track, arc_steps=8)
    straight_count = sum(1 for k in track.cells.values() if not k.is_curve)
    curve_count = sum(1 for k in track.cells.values() if k.is_curve)
    assert len(segments) == straight_count * 2 + curve_count * 2 * 8


def test_straight_wall_offsets_from_drivable_width() -> None:
    # (1, 0) is STRAIGHT_EW — the road runs along x, so its kerb walls run along y = cy ± half
    # drivable width (top/bottom of the corridor), spanning the cell's full x extent.
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    segments = boundary_segments(track)
    cx, cy = track.cell_center((1, 0))  # (512, 0)
    half = CELL_SIZE / 2
    half_drivable = DRIVABLE_WIDTH / 2
    expected = {
        ((cx - half, cy - half_drivable), (cx + half, cy - half_drivable)),
        ((cx - half, cy + half_drivable), (cx + half, cy + half_drivable)),
    }
    assert expected <= set(segments)


def test_curve_arc_endpoints_meet_adjacent_straight_walls() -> None:
    # CURVE_NE at (0,0) opens N (toward (0,1), a STRAIGHT_NS cell) and E (toward (1,0), a
    # STRAIGHT_EW cell). Its inner arc should start/end exactly where those cells' inner
    # walls cross the shared cell boundary.
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    segments = boundary_segments(track, arc_steps=8)
    half = CELL_SIZE / 2
    half_drivable = DRIVABLE_WIDTH / 2
    points = [p for seg in segments for p in seg]
    on_shared_n_edge = (half_drivable, half)  # meets (0,1)'s inner wall at y = half
    on_shared_e_edge = (half, half_drivable)  # meets (1,0)'s inner wall at x = half
    for expected in (on_shared_n_edge, on_shared_e_edge):
        assert any(
            math.isclose(p[0], expected[0], abs_tol=1e-6)
            and math.isclose(p[1], expected[1], abs_tol=1e-6)
            for p in points
        ), f"no arc point near {expected}"


def test_tile_kind_for_open_edges_matches_known_kinds() -> None:
    assert tile_kind_for_open_edges(Facing.N, Facing.S) == TileKind.STRAIGHT_NS
    assert tile_kind_for_open_edges(Facing.S, Facing.N) == TileKind.STRAIGHT_NS  # order-independent
    assert tile_kind_for_open_edges(Facing.E, Facing.W) == TileKind.STRAIGHT_EW
    assert tile_kind_for_open_edges(Facing.N, Facing.E) == TileKind.CURVE_NE
    assert tile_kind_for_open_edges(Facing.N, Facing.W) == TileKind.CURVE_NW
    assert tile_kind_for_open_edges(Facing.S, Facing.E) == TileKind.CURVE_SE
    assert tile_kind_for_open_edges(Facing.S, Facing.W) == TileKind.CURVE_SW


def test_tile_kind_for_open_edges_rejects_same_facing_twice() -> None:
    with pytest.raises(ValueError):
        tile_kind_for_open_edges(Facing.N, Facing.N)
