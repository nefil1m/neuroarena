import pytest

from neuroarena.sim.track import Track
from neuroarena.sim.track_generation import (
    SIZE_DRIFT,
    generate_track,
)


def test_generated_track_is_a_valid_track() -> None:
    track = generate_track(size=12, complexity=0.5, seed=1)
    assert isinstance(track, Track)  # Track.__post_init__ already validates a single closed loop


def test_same_seed_is_deterministic() -> None:
    a = generate_track(size=16, complexity=0.5, seed=42)
    b = generate_track(size=16, complexity=0.5, seed=42)
    assert a.cells == b.cells
    assert a.start_cell == b.start_cell
    assert a.start_facing == b.start_facing


def test_different_seed_gives_a_different_track() -> None:
    a = generate_track(size=16, complexity=0.5, seed=1)
    b = generate_track(size=16, complexity=0.5, seed=2)
    assert a.cells != b.cells


def test_actual_size_within_drift_tolerance() -> None:
    for seed in range(10):
        track = generate_track(size=20, complexity=0.5, seed=seed)
        drift = abs(len(track.cells) - 20) / 20
        assert drift <= SIZE_DRIFT, f"seed={seed} produced {len(track.cells)} tiles"


def test_complexity_zero_and_one_both_produce_valid_tracks() -> None:
    assert isinstance(generate_track(size=12, complexity=0.0, seed=3), Track)
    assert isinstance(generate_track(size=12, complexity=1.0, seed=3), Track)


def test_rejects_size_below_minimum() -> None:
    with pytest.raises(ValueError):
        generate_track(size=3, complexity=0.5, seed=1)


def test_rejects_out_of_range_complexity() -> None:
    with pytest.raises(ValueError):
        generate_track(size=12, complexity=1.5, seed=1)
