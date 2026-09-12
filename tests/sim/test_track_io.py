from pathlib import Path

import pytest

from neuroarena.sim import track_io
from neuroarena.sim.track import Facing, TileKind, Track

FIXTURE = Path(__file__).parents[2] / "src/neuroarena/sim/tracks/track_01.json"


def _sample_track() -> Track:
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


def test_round_trip_preserves_track() -> None:
    original = _sample_track()
    restored = track_io.loads(track_io.dumps(original))
    assert restored == original


def test_unknown_schema_version_raises() -> None:
    text = track_io.dumps(_sample_track()).replace('"schema_version": 1', '"schema_version": 99')
    with pytest.raises(track_io.UnknownSchemaVersionError):
        track_io.loads(text)


def test_track_01_fixture_loads() -> None:
    track = track_io.loads(FIXTURE.read_text())
    assert track.start_cell == (1, 0)
    assert track.start_facing is Facing.E
    assert len(track.cells) == 10
