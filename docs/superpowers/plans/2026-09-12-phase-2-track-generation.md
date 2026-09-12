# Phase 2 — Track Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Procedurally generate closed-loop tracks in Phase 1's grid/`TileKind` representation, store them as identifiable files, and provide a CLI to generate/preview/save one — replacing the hand-authored fixture as the platform's source of tracks.

**Architecture:** A pure generator (`neuroarena.sim.track_generation`) runs a self-avoiding random walk with backtracking and returns a `Track` — no I/O, no `arcade`, matching `sim`'s existing purity. A new `neuroarena.tracks` package (storage + CLI) sits alongside `sim`/`render`: it gives a generated `Track` an identity (`track_id`) and writes it to a plain-file store with a manifest, and its CLI wires generation + storage + Phase 1's existing `Game`/`PlayWindow` renderer together for a preview. Nothing here touches `sim/game.py`, `sim/physics.py`, or the renderer's drawing code — Phase 1's runtime is reused unmodified, only pointed at a different `Track`.

**Tech Stack:** Python 3.12+, standard-library `random`/`uuid`/`json` (no new dependencies), `pytest`, `ruff`, `mypy` strict — matching Phase 0/1.

**Spec:** `docs/phases/phase-2-track-generation.md` (FINALIZED 2026-09-12)

## Global Constraints

- `size` is the target **tile count** of the closed loop, not a world-unit dimension. Minimum feasible size is 4 tiles; requesting less is rejected. Actual tile count is accepted within **±15%** of requested `size`.
- `complexity` is a **turn-frequency bias** in `[0, 1]`: the probability of turning vs. continuing straight at each step of the walk.
- Generation is **deterministic**: the same `(size, complexity, seed)` always produces the same track.
- Self-intersection is prevented **by construction** — the walk never revisits a cell (except the closing step back to `start`), never checked after the fact.
- Storage is **plain JSON files** under a `tracks_dir` directory plus a manifest file — **not SQLite**. Phase 6 (which owns SQLite) is implemented after this phase in build order and later references these files by `track_id`; this phase must be self-sufficient without it.
- The generation control surface **for this phase is a CLI only**, no browser UI — Phase 7's dashboard backend doesn't exist yet in build order. Phase 9's reserved track editor wraps this same generation function later.
- `neuroarena.sim` (including the new `track_generation.py`) imports no `arcade`, no `gymnasium` — enforced by the existing import-hygiene test (`tests/sim/test_import_hygiene.py`).
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy` clean before every commit; commit at the end of each task.

---

## File structure

```
src/neuroarena/sim/track.py             # MODIFY: add tile_kind_for_open_edges()
src/neuroarena/sim/track_generation.py  # NEW: generate_track() — self-avoiding walk, retry-until-valid closure
src/neuroarena/tracks/__init__.py       # NEW
src/neuroarena/tracks/store.py          # NEW: TrackRecord, save()/load()/list_tracks(), manifest.json
src/neuroarena/tracks/cli.py            # NEW: neuroarena-track-gen entry point (generate, save, optional --preview)

tests/sim/test_track.py                 # MODIFY: add tile_kind_for_open_edges tests
tests/sim/test_track_generation.py      # NEW
tests/tracks/__init__.py                # NEW
tests/tracks/test_store.py              # NEW
tests/tracks/test_cli.py                # NEW

pyproject.toml                          # MODIFY: add neuroarena-track-gen to [project.scripts]
.gitignore                              # MODIFY: add data/ — generated tracks are runtime artifacts, not checked in
```

`src/neuroarena/sim/tracks/track_01.json` (Phase 1's hand-authored fixture) is untouched and stays the `neuroarena-play` default; generated maps live in a separate runtime directory (default `data/tracks/`) so the two are never confused.

---

### Task 1 — `tile_kind_for_open_edges`: the inverse lookup the generator needs

The walk produces, per cell, the two directions its path connects (where it came from, where it goes next) — this is the piece that turns that pair back into a `TileKind`, the same lookup `track.py`'s own validation logic implicitly relies on but doesn't expose.

**Files:**
- Modify: `src/neuroarena/sim/track.py` (add near `_OPEN_EDGES`, after the `TileKind` class)
- Test: `tests/sim/test_track.py` (add to the existing file)

**Interfaces:**
- Produces: `tile_kind_for_open_edges(a: Facing, b: Facing) -> TileKind`, raises `ValueError` if no `TileKind` connects the two.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/sim/test_track.py
from neuroarena.sim.track import tile_kind_for_open_edges


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_track.py -k tile_kind_for_open_edges -v`
Expected: FAIL with `ImportError` (`tile_kind_for_open_edges` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# add to src/neuroarena/sim/track.py, directly below the _OPEN_EDGES dict
_EDGES_TO_KIND: dict[frozenset[Facing], TileKind] = {
    edges: kind for kind, edges in _OPEN_EDGES.items()
}


def tile_kind_for_open_edges(a: Facing, b: Facing) -> TileKind:
    """Inverse of `TileKind.open_edges`: the TileKind whose two open edges are exactly {a, b}."""
    try:
        return _EDGES_TO_KIND[frozenset({a, b})]
    except KeyError:
        raise ValueError(f"no TileKind connects {a.value} and {b.value}") from None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_track.py -k tile_kind_for_open_edges -v`
Expected: PASS (7 cases)

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/sim/track.py tests/sim/test_track.py
git commit -m "feat(phase-2): add tile_kind_for_open_edges lookup to Track"
```

---

### Task 2 — `generate_track`: self-avoiding walk with retry-until-valid closure

**Files:**
- Create: `src/neuroarena/sim/track_generation.py`
- Test: `tests/sim/test_track_generation.py`

**Interfaces:**
- Consumes: `Facing`, `GridCell`, `TileKind`, `Track`, `tile_kind_for_open_edges` from `neuroarena.sim.track` (Task 1).
- Produces: `generate_track(size: int, complexity: float, seed: int, *, max_attempts: int = 500, step_budget_factor: int = 25) -> Track`; `TrackGenerationError(RuntimeError)`; module constants `MIN_TRACK_SIZE = 4`, `SIZE_DRIFT = 0.15` (later tasks/phases read these rather than re-declaring the tolerance).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_track_generation.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_track_generation.py -v`
Expected: FAIL with `ModuleNotFoundError` (`track_generation` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/sim/track_generation.py
"""Procedural closed-loop track generation: a self-avoiding random walk over the grid,
biased by `complexity`, backtracking out of dead ends, retried until a loop closes near
the requested size. See the Phase 2 doc, "Generation algorithm"."""

from __future__ import annotations

import math
import random

from neuroarena.sim.track import Facing, GridCell, TileKind, Track, tile_kind_for_open_edges

MIN_TRACK_SIZE = 4
"""Smallest closed loop expressible with 90°-only turns: a 2x2 block of curve tiles."""

SIZE_DRIFT = 0.15
"""Accepted fractional drift between requested and actual tile count (Phase 2 doc)."""

_CLOCKWISE = [Facing.N, Facing.E, Facing.S, Facing.W]
_TURN_RIGHT: dict[Facing, Facing] = {
    f: _CLOCKWISE[(i + 1) % 4] for i, f in enumerate(_CLOCKWISE)
}
_TURN_LEFT: dict[Facing, Facing] = {
    f: _CLOCKWISE[(i - 1) % 4] for i, f in enumerate(_CLOCKWISE)
}


class TrackGenerationError(RuntimeError):
    """Raised when no valid closed loop near the requested size could be generated."""


def generate_track(
    size: int,
    complexity: float,
    seed: int,
    *,
    max_attempts: int = 500,
    step_budget_factor: int = 25,
) -> Track:
    if size < MIN_TRACK_SIZE:
        raise ValueError(
            f"size must be >= {MIN_TRACK_SIZE} (a 90°-only closed loop's minimum); got {size}"
        )
    if not 0.0 <= complexity <= 1.0:
        raise ValueError(f"complexity must be in [0, 1]; got {complexity}")

    rng = random.Random(seed)
    min_close_length = max(MIN_TRACK_SIZE, math.ceil(size * (1 - SIZE_DRIFT)))
    max_length = math.floor(size * (1 + SIZE_DRIFT))
    step_budget = max(size * step_budget_factor, 200)

    for _ in range(max_attempts):
        path = _walk_attempt(rng, min_close_length, max_length, complexity, step_budget)
        if path is not None:
            cells = _path_to_cells(path)
            return Track(cells=cells, start_cell=path[0], start_facing=_direction(path[0], path[1]))

    raise TrackGenerationError(
        f"failed to close a loop within {min_close_length}-{max_length} tiles "
        f"of size={size} after {max_attempts} attempts"
    )


def _walk_attempt(
    rng: random.Random,
    min_close_length: int,
    max_length: int,
    complexity: float,
    step_budget: int,
) -> list[GridCell] | None:
    """One self-avoiding random-walk attempt: returns an ordered cycle of cells (the last
    cell connects back to the first) or None if it dead-ended without closing in budget."""
    start: GridCell = (0, 0)
    heading = rng.choice(_CLOCKWISE)
    path: list[GridCell] = [start]
    visited: set[GridCell] = {start}
    choice_stack: list[list[Facing]] = [_ordered_choices(rng, heading, complexity)]
    steps = 0

    while steps < step_budget:
        steps += 1
        current = path[-1]
        choices = choice_stack[-1]

        if not choices:
            path.pop()
            choice_stack.pop()
            if not path:
                return None  # backtracked past the start — this attempt cannot close
            visited.discard(current)
            continue

        next_heading = choices.pop(0)
        dx, dy = next_heading.delta
        candidate = (current[0] + dx, current[1] + dy)

        if candidate == start:
            if len(path) >= min_close_length:
                return path
            continue  # too short to close yet — try the next queued heading instead
        if candidate in visited:
            continue
        if len(path) >= max_length:
            continue  # at the length cap — only a closing move is allowed from here

        path.append(candidate)
        visited.add(candidate)
        choice_stack.append(_ordered_choices(rng, next_heading, complexity))

    return None  # exhausted the step budget without closing


def _ordered_choices(rng: random.Random, heading: Facing, complexity: float) -> list[Facing]:
    """Candidate next headings from `heading` — never a reversal, since {N,E,S,W} only
    connect via straight-through or 90°-turn TileKinds, never a U-turn. `complexity` biases
    whether a turn or the straight continuation is tried first."""
    turns = [_TURN_LEFT[heading], _TURN_RIGHT[heading]]
    rng.shuffle(turns)
    if rng.random() < complexity:
        return [*turns, heading]
    return [heading, *turns]


def _path_to_cells(path: list[GridCell]) -> dict[GridCell, TileKind]:
    cells: dict[GridCell, TileKind] = {}
    n = len(path)
    for i, cell in enumerate(path):
        prev_cell = path[i - 1]
        next_cell = path[(i + 1) % n]
        cells[cell] = tile_kind_for_open_edges(
            _direction(cell, prev_cell), _direction(cell, next_cell)
        )
    return cells


def _direction(frm: GridCell, to: GridCell) -> Facing:
    delta = (to[0] - frm[0], to[1] - frm[1])
    for facing in _CLOCKWISE:
        if facing.delta == delta:
            return facing
    raise AssertionError(f"{frm} -> {to} is not a unit grid step")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_track_generation.py -v`
Expected: PASS (7 cases). If `test_actual_size_within_drift_tolerance` or the determinism/validity tests flake across seeds, tune `step_budget_factor` or `max_attempts` upward before treating the algorithm itself as broken — Phase 1's physics constants went through the same empirical-tuning step after first passing their own tests.

- [ ] **Step 5: Run the full test suite and import-hygiene check**

Run: `uv run pytest -v && uv run mypy`
Expected: all pass, including `tests/sim/test_import_hygiene.py` (confirms `track_generation.py` didn't accidentally import `arcade`/`gymnasium`).

- [ ] **Step 6: Commit**

```bash
git add src/neuroarena/sim/track_generation.py tests/sim/test_track_generation.py
git commit -m "feat(phase-2): procedural track generation via self-avoiding random walk"
```

---

### Task 3 — Map identity and file storage

**Files:**
- Create: `src/neuroarena/tracks/__init__.py` (empty)
- Create: `src/neuroarena/tracks/store.py`
- Create: `tests/tracks/__init__.py` (empty)
- Create: `tests/tracks/test_store.py`

**Interfaces:**
- Consumes: `Track` from `neuroarena.sim.track`; `dumps`/`loads` from `neuroarena.sim.track_io`.
- Produces: `TrackRecord` (frozen dataclass: `track_id: str`, `track: Track`, `requested_size: int`, `actual_size: int`, `complexity: float`, `seed: int`, `generated_at: str`); `save(track: Track, *, requested_size: int, complexity: float, seed: int, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> TrackRecord`; `load(track_id: str, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> TrackRecord`; `list_tracks(tracks_dir: Path = DEFAULT_TRACKS_DIR) -> list[dict[str, object]]`; `UnknownTrackError(KeyError)`; `DEFAULT_TRACKS_DIR = Path("data/tracks")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracks/test_store.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/tracks/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError` (`neuroarena.tracks` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/tracks/store.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/tracks/test_store.py -v && uv run mypy`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/tracks/__init__.py src/neuroarena/tracks/store.py tests/tracks/__init__.py tests/tracks/test_store.py
git commit -m "feat(phase-2): file-based storage for generated tracks (track_id + manifest)"
```

---

### Task 4 — CLI: generate, save, optionally preview

**Files:**
- Create: `src/neuroarena/tracks/cli.py`
- Create: `tests/tracks/test_cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry)
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Consumes: `generate_track` (Task 2); `save`, `DEFAULT_TRACKS_DIR`, `TrackRecord` (Task 3); `Game` from `neuroarena.sim.game`; `AssetManifest` from `neuroarena.render.manifest`; `PlayWindow` from `neuroarena.render.window` (all existing, Phase 1).
- Produces: `run(size: int, complexity: float, seed: int, tracks_dir: Path) -> TrackRecord` (pure generate+save, no `arcade`, unit-testable); `main() -> None` (argparse + optional preview); console entry point `neuroarena-track-gen`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracks/test_cli.py
from pathlib import Path

from neuroarena.tracks.cli import run


def test_run_generates_and_saves_a_track(tmp_path: Path) -> None:
    record = run(size=8, complexity=0.4, seed=3, tracks_dir=tmp_path)
    assert (tmp_path / f"{record.track_id}.json").is_file()
    assert record.requested_size == 8
    assert record.seed == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/tracks/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` (`neuroarena.tracks.cli` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/tracks/cli.py
"""Entry point: `uv run neuroarena-track-gen`. Generates a map, saves it, and optionally
previews it by reusing Phase 1's Game + PlayWindow + AssetManifest renderer — drivable,
not just a static render (see the Phase 1 doc's revision history: only actually driving a
track caught the heading/kerb bugs a screenshot review had missed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import arcade

from neuroarena.render.manifest import AssetManifest
from neuroarena.render.window import PlayWindow
from neuroarena.sim.game import Game
from neuroarena.sim.track_generation import generate_track
from neuroarena.tracks.store import DEFAULT_TRACKS_DIR, TrackRecord, save

MANIFEST_PATH = Path(__file__).parents[1] / "render/assets/road_01/manifest.json"


def run(size: int, complexity: float, seed: int, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> TrackRecord:
    track = generate_track(size, complexity, seed)
    return save(track, requested_size=size, complexity=complexity, seed=seed, tracks_dir=tracks_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a procedural closed-loop track.")
    parser.add_argument("--size", type=int, required=True, help="target tile count")
    parser.add_argument("--complexity", type=float, default=0.5, help="turn-frequency bias, 0-1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tracks-dir", type=Path, default=DEFAULT_TRACKS_DIR)
    parser.add_argument("--preview", action="store_true", help="open an arcade window to drive the result")
    args = parser.parse_args()

    record = run(args.size, args.complexity, args.seed, args.tracks_dir)
    drift_pct = 100 * (record.actual_size - record.requested_size) / record.requested_size
    print(
        f"generated track_id={record.track_id} actual_size={record.actual_size} "
        f"(requested {record.requested_size}, drift {drift_pct:+.1f}%) -> {args.tracks_dir}"
    )

    if args.preview:
        game = Game(record.track)
        manifest = AssetManifest.load(MANIFEST_PATH)
        PlayWindow(game, manifest)
        arcade.run()


if __name__ == "__main__":
    main()
```

```toml
# pyproject.toml — add alongside the existing neuroarena-play entry
[project.scripts]
neuroarena-play = "neuroarena.render.play:main"
neuroarena-track-gen = "neuroarena.tracks.cli:main"
```

```gitignore
# .gitignore — add near the Python section
# --- Runtime data ---
data/
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/tracks/test_cli.py -v && uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all PASS.

- [ ] **Step 5: Reinstall the entry point and verify the CLI runs**

Run: `uv sync && uv run neuroarena-track-gen --size 20 --complexity 0.5 --seed 1`
Expected: prints a line like `generated track_id=... actual_size=... (requested 20, drift +x.x%) -> data/tracks`, and `data/tracks/<track_id>.json` plus `data/tracks/manifest.json` exist.

- [ ] **Step 6: Manual verification — preview a generated track**

Run: `uv run neuroarena-track-gen --size 20 --complexity 0.6 --seed 2 --preview`
Expected: an `arcade` window opens showing a closed drivable loop (same renderer as `neuroarena-play`); confirm the loop has no visible gaps/self-crossings and the car can be driven around the full lap back to the start. If it can't, that's a generation bug, not a rendering one — the renderer is unmodified from Phase 1.

- [ ] **Step 7: Commit**

```bash
git add src/neuroarena/tracks/cli.py tests/tracks/test_cli.py pyproject.toml .gitignore
git commit -m "feat(phase-2): CLI to generate, save, and preview a procedural track"
```

---

## Revision history
- 2026-09-12 — Plan written, following Phase 2's FINALIZED requirements. Four tasks: a `tile_kind_for_open_edges` lookup, the self-avoiding-walk generator, file-based map storage with a manifest, and a CLI tying generation/storage/Phase 1's existing renderer together. No changes to `sim/game.py`, `sim/physics.py`, or the renderer's drawing code — the CLI's `--preview` reuses `Game`/`PlayWindow`/`AssetManifest` exactly as `neuroarena-play` does, just pointed at a generated `Track` instead of the hand-authored fixture.
