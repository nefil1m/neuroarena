from pathlib import Path

from neuroarena.tracks.cli import run


def test_run_generates_and_saves_a_track(tmp_path: Path) -> None:
    record = run(size=8, complexity=0.4, seed=3, tracks_dir=tmp_path)
    assert (tmp_path / f"{record.track_id}.json").is_file()
    assert record.requested_size == 8
    assert record.seed == 3
