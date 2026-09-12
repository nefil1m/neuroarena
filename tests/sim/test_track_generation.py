import time

import pytest

from neuroarena.sim.track import Track
from neuroarena.sim.track_generation import (
    SIZE_DRIFT,
    TrackGenerationError,
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


def test_sizes_spread_across_drift_window_not_clustered_at_top() -> None:
    # Before the per-attempt sampling fix, the walk always saturated at the largest reachable
    # even value <= max_length, so every seed produced the same (top-of-window) tile count.
    # Now each attempt samples its own target, so tile counts should show real spread.
    for size in (14, 20, 40):
        counts = {
            len(generate_track(size=size, complexity=0.5, seed=seed).cells) for seed in range(20)
        }
        assert len(counts) > 1, f"size={size} produced a single clustered count: {counts}"


def test_same_seed_still_deterministic_with_per_attempt_sampling() -> None:
    a = generate_track(size=14, complexity=0.5, seed=7)
    b = generate_track(size=14, complexity=0.5, seed=7)
    assert a.cells == b.cells
    assert a.start_cell == b.start_cell
    assert a.start_facing == b.start_facing


def test_start_cell_is_never_a_curve_when_a_straight_tile_exists() -> None:
    for size, seed in [(12, 1), (16, 2), (20, 3), (24, 4), (30, 5), (40, 6)]:
        track = generate_track(size=size, complexity=0.5, seed=seed)
        start_kind = track.cells[track.start_cell]
        assert not start_kind.is_curve, (
            f"size={size} seed={seed} started on a curve tile {start_kind}"
        )


def test_minimum_size_track_is_valid_even_with_all_curve_fallback() -> None:
    # A 4-cell loop is a 2x2 block of curve tiles — no straight tile exists, so start_cell
    # must fall back to path[0] (a curve). This should still produce a valid Track.
    track = generate_track(size=4, complexity=0.5, seed=1)
    assert len(track.cells) == 4
    assert track.cells[track.start_cell].is_curve  # confirms the fallback path was exercised


def test_rejects_size_with_no_even_tile_count_in_drift_window() -> None:
    # size=5's ±15% drift window collapses to the single value [5, 5], and a 90°-only closed
    # loop always has an even tile count (the grid graph is bipartite by (x+y) parity, so
    # every cycle has even length) — so no attempt could ever succeed. This must fail fast
    # with a message naming the reason, not silently exhaust the attempt budget: pin
    # max_attempts/step_budget_factor absurdly high and assert it still returns quickly, and
    # that the message explains the even-tile-count constraint rather than reporting
    # exhausted attempts.
    start = time.monotonic()
    with pytest.raises(TrackGenerationError, match="even tile count") as excinfo:
        generate_track(
            size=5, complexity=0.5, seed=1, max_attempts=100_000, step_budget_factor=10_000
        )
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"took {elapsed:.2f}s — looks like it exhausted attempts, not fast-fail"
    assert "attempts" not in str(excinfo.value)
