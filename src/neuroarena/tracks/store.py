"""File-based storage for generated maps: one JSON record per track under a directory,
plus a manifest for enumeration without opening every file. Deliberately plain files, not
SQLite — see the Phase 2 doc's "Map identity and storage": Phase 6 (which owns SQLite)
doesn't exist yet at this point in build order, and tracks are static artifacts like
checkpoints, not metrics that belong in a queryable table."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from neuroarena.sim.track import Track
from neuroarena.sim.track_io import dumps as track_dumps
from neuroarena.sim.track_io import loads as track_loads

RECORD_SCHEMA_VERSION = 1
DEFAULT_TRACKS_DIR = Path("data/tracks")
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class TrackRecord:
    track_id: str
    track: Track
    requested_size: int
    actual_size: int
    complexity: float
    seed: int
    generated_at: str


class UnknownTrackError(KeyError):
    """Raised when a track_id has no matching file in the given tracks_dir."""


def save(
    track: Track,
    *,
    requested_size: int,
    complexity: float,
    seed: int,
    tracks_dir: Path = DEFAULT_TRACKS_DIR,
) -> TrackRecord:
    tracks_dir.mkdir(parents=True, exist_ok=True)
    record = TrackRecord(
        track_id=uuid.uuid4().hex,
        track=track,
        requested_size=requested_size,
        actual_size=len(track.cells),
        complexity=complexity,
        seed=seed,
        generated_at=datetime.now(UTC).isoformat(),
    )
    path = tracks_dir / f"{record.track_id}.json"
    path.write_text(_dumps_record(record))
    _append_manifest_entry(tracks_dir, record, path)
    return record


def load(track_id: str, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> TrackRecord:
    path = tracks_dir / f"{track_id}.json"
    if not path.is_file():
        raise UnknownTrackError(track_id)
    return _loads_record(path.read_text())


def list_tracks(tracks_dir: Path = DEFAULT_TRACKS_DIR) -> list[dict[str, object]]:
    manifest_path = tracks_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    raw = json.loads(manifest_path.read_text())
    entries: list[dict[str, object]] = raw["tracks"]
    return entries


def _dumps_record(record: TrackRecord) -> str:
    raw = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "track_id": record.track_id,
        "requested_size": record.requested_size,
        "actual_size": record.actual_size,
        "complexity": record.complexity,
        "seed": record.seed,
        "generated_at": record.generated_at,
        "track": json.loads(track_dumps(record.track)),
    }
    return json.dumps(raw, indent=2)


def _loads_record(text: str) -> TrackRecord:
    raw = json.loads(text)
    version = raw.get("schema_version")
    if version != RECORD_SCHEMA_VERSION:
        raise ValueError(f"unreadable track record schema_version {version!r}")
    return TrackRecord(
        track_id=raw["track_id"],
        track=track_loads(json.dumps(raw["track"])),
        requested_size=raw["requested_size"],
        actual_size=raw["actual_size"],
        complexity=raw["complexity"],
        seed=raw["seed"],
        generated_at=raw["generated_at"],
    )


def _append_manifest_entry(tracks_dir: Path, record: TrackRecord, path: Path) -> None:
    manifest_path = tracks_dir / MANIFEST_FILENAME
    entries = list_tracks(tracks_dir)
    entries.append(
        {
            "track_id": record.track_id,
            "path": path.name,
            "requested_size": record.requested_size,
            "actual_size": record.actual_size,
            "complexity": record.complexity,
            "seed": record.seed,
            "generated_at": record.generated_at,
        }
    )
    manifest_path.write_text(json.dumps({"tracks": entries}, indent=2))
