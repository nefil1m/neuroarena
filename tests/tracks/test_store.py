from pathlib import Path

import pytest

from neuroarena.sim.track_generation import generate_track
from neuroarena.tracks.store import UnknownTrackError, list_tracks, load, save


def test_save_writes_a_file_and_a_manifest_entry(tmp_path: Path) -> None:
    track = generate_track(size=8, complexity=0.3, seed=5)
    record = save(track, requested_size=8, complexity=0.3, seed=5, tracks_dir=tmp_path)

    assert (tmp_path / f"{record.track_id}.json").is_file()
    entries = list_tracks(tmp_path)
    assert any(e["track_id"] == record.track_id for e in entries)


def test_load_round_trips_the_track(tmp_path: Path) -> None:
    track = generate_track(size=8, complexity=0.3, seed=5)
    record = save(track, requested_size=8, complexity=0.3, seed=5, tracks_dir=tmp_path)

    loaded = load(record.track_id, tracks_dir=tmp_path)
    assert loaded.track_id == record.track_id
    assert loaded.track.cells == track.cells
    assert loaded.track.start_cell == track.start_cell
    assert loaded.requested_size == 8
    assert loaded.actual_size == len(track.cells)


def test_load_unknown_id_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownTrackError):
        load("does-not-exist", tracks_dir=tmp_path)


def test_manifest_accumulates_across_multiple_saves(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        track = generate_track(size=8, complexity=0.3, seed=seed)
        save(track, requested_size=8, complexity=0.3, seed=seed, tracks_dir=tmp_path)

    assert len(list_tracks(tmp_path)) == 3
