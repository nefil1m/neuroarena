# Phase 1 — Playable Game Core Implementation Plan

**Goal:** A kinematic car, manually controllable, driving on one hand-authored closed-loop track, drawn by an `arcade`-based renderer with real vendored art. No episode/reset concept — Phase 1 is a standalone playable game, not yet a training `Environment` (that's Phase 3).

**Architecture:** `neuroarena.sim` holds all physics/track/collision logic — pure Python + `numpy`, no `arcade` import, no episode concept. `neuroarena.render` imports `sim` (never the reverse), and is the only place `arcade` is imported. A `Game` object in `sim` exposes one entry point, `tick(steering, throttle) -> GameState`, called once per fixed `TICK_DT` step; the renderer owns the display loop and calls `tick` as many times as elapsed wall-clock time demands (fixed-timestep accumulator), then draws whatever `GameState` currently holds.

**Tech Stack:** Python 3.12+, `numpy`, `arcade` (new dependency this phase), `pytest`, `ruff`, `mypy` strict.

**Spec:** `docs/phases/phase-1-game-core.md` (FINALIZED 2026-09-12)

## Global constraints

- `neuroarena.sim` imports no `arcade`, no `gymnasium`, no `neuroarena.render` — enforced by an import-hygiene test mirroring Phase 0's.
- Physics runs at a fixed `TICK_DT = 1/60` s, always — never scaled by display framerate or by a future sim-speed setting (Phase 5 owns pacing, not timestep size).
- Track data (`TileKind` grid + `start_cell` + `cell_size`) is serialized as JSON with a top-level integer `schema_version`, matching the `RunConfig` codec convention (`neuroarena/config/serialization.py`).
- Track/collision geometry and render assets are separate layers: nothing in `sim` references a texture path, and nothing in `render/manifest.py` encodes collision geometry.
- Physics constants live in one `PhysicsConstants` dataclass with the Phase-1-doc defaults, passed in rather than hardcoded, so they stay tunable without code changes.
- Car geometry (`car_width ≈ 148`, `car_length ≈ 300` world units) is a `sim`-level fact (it drives collision), not just a render-scale fact — defined once, read by both collision and the renderer's scale-to-sprite step.
- Commit at the end of every task. Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy` before each commit.

## File structure

```
src/neuroarena/sim/__init__.py
src/neuroarena/sim/track.py            # TileKind, Track, connectivity/boundary derivation, validation
src/neuroarena/sim/track_io.py         # JSON schema_version codec for Track
src/neuroarena/sim/physics.py          # PhysicsConstants, CarState, step_car()
src/neuroarena/sim/collision.py        # car-vs-boundary hard collider resolution
src/neuroarena/sim/game.py             # Game: owns Track + CarState, tick(steering, throttle) -> GameState
src/neuroarena/sim/tracks/track_01.json  # the one hand-authored Phase 1 track

src/neuroarena/render/manifest.py      # AssetManifest: load road_01/manifest.json, resolve TileKind -> textures incl. rotation
src/neuroarena/render/window.py        # arcade.Window subclass: camera, tile/decor/car drawing
src/neuroarena/render/input.py         # keyboard -> (steering, throttle)
src/neuroarena/render/play.py          # entry point: wires Game + AssetManifest + Window + input, runs the loop

tests/sim/__init__.py
tests/sim/test_track.py
tests/sim/test_track_io.py
tests/sim/test_physics.py
tests/sim/test_collision.py
tests/sim/test_game.py
tests/render/__init__.py
tests/render/test_manifest.py
```

`render/window.py` and `render/play.py` are exercised by manual play-testing (`uv run neuroarena-play`), not unit tests — an `arcade.Window` needs a display/GL context; per Phase 1's requirement this is also the human-playable deliverable, so "run it and drive around" *is* the acceptance test for those two files.

---

### Task 1 — Track representation, validation, JSON codec

**Produces:** `TileKind` (6-member enum), `Track` (frozen dataclass: `cells: dict[tuple[int,int], TileKind]`, `start_cell: tuple[int,int]`, `start_facing: Facing`, `cell_size: float`), a `validate(track)` that checks the cell sequence forms a single closed loop with edge-consistent tile kinds (each cell's open edges connect to a neighbor's open edge), and `boundary_segments(track) -> list[Segment]` deriving the collision polyline from the tile kinds (straight = two parallel wall segments; curve = two quarter-circle-ish wall arcs approximated as short line segments). JSON codec (`track_io.py`) with `schema_version = 1`, `dumps`/`loads` mirroring `config/serialization.py`. One hand-authored track fixture (`sim/tracks/track_01.json`) — a small closed loop using all four curve kinds at least once (e.g. a rounded rectangle: 2 straights per side, one curve per corner).

**Tests:** valid loop passes validation; a broken loop (dangling open edge) fails; boundary segment count/shape matches tile count; JSON round-trip preserves the track exactly; loading an unknown `schema_version` raises.

### Task 2 — Car physics (kinematic bicycle model)

**Produces:** `PhysicsConstants` (the Phase-1-doc defaults as a frozen dataclass), `CarState(x, y, heading, speed)`, `step_car(state, steering, throttle, dt, constants) -> CarState` implementing the bicycle model plus brake-first-then-reverse throttle semantics (forward + negative throttle decelerates toward 0 before reversing; already-stopped + negative throttle accelerates backward up to `max_reverse_speed`).

**Tests:** straight-line accel reaches `max_speed` asymptotically and never exceeds it; full negative throttle from cruising speed decelerates without going negative-then-reverse in the same step; steering at max angle with the doc's constants produces the expected ~257-unit turn radius (numeric check against the sanity-check math already in the doc); zero controls at zero speed is a fixed point.

### Task 3 — Collision (hard boundary collider)

**Produces:** given a car's OBB (from `CarState` + car geometry constants) and `boundary_segments(track)`, detect crossing and resolve it as a hard stop (clamp position to the boundary, zero the outward velocity component) rather than a bounce — matches the doc's "stop/bounce at the kerb" resolved as stop.

**Tests:** a car driving straight into a wall stops at the wall, not past it; a car fully inside the track with no wall nearby is untouched; grazing a wall at an angle kills only the outward-normal component of velocity, not tangential speed (so it can still slide along the wall, not stick dead).

### Task 4 — Game loop

**Produces:** `Game(track, constants)` holding current `CarState`, spawning at `start_cell`/`start_facing`; `tick(steering, throttle) -> GameState` (a small snapshot: car state + track reference) that internally calls `step_car` then the collision resolver, advancing exactly one `TICK_DT`. No episode/reset/terminated concept — Phase 3 owns that later.

**Tests:** repeated `tick` calls with fixed inputs are deterministic (same result from the same start state — this is the determinism guarantee the doc ties to fixed `TICK_DT`); spawn position/heading matches `start_cell`/`start_facing`.

### Task 5 — Asset manifest loader

**Produces:** `AssetManifest.load(path)` parsing `render/assets/road_01/manifest.json`; `resolve(tile_kind) -> ResolvedTile` returning either a composite path or an ordered layer-path list, each tagged with a `rotate_degrees` (0 for the two base tiles, the manifest's derived-tile value otherwise) so the renderer applies one rotation step uniformly regardless of composite-vs-layered. Also resolves `car`, `decor.start_finish`, `background.grass` paths. Validates every referenced file exists on disk at load time (fail fast on a broken manifest) — no image-library dependency needed since this stays at the path/metadata level.

**Tests:** all six `TileKind`s resolve to something; the two base tiles resolve at `rotate_degrees == 0`; each derived tile resolves to its documented rotation; a manifest referencing a missing file raises at load time.

### Task 6 — Renderer (arcade window)

**Produces:** an `arcade.Window` that: builds a `arcade.Sprite`/texture per resolved tile (rotating layered or composite textures per Task 5's metadata) and places one per `Track` cell at `cell_size`-spaced world coordinates; draws the `start_finish` decor sprite at `start_cell`; draws the car as a single sprite whose `angle` is set from `CarState.heading` every frame (continuous rotation, not a frame set); a camera that follows the car (centered, no easing needed for v1 — simplest thing that works, note anything fancier as a later polish item, not a new open question). Runs a fixed-step accumulator in `on_update`: enough `Game.tick` calls to consume elapsed wall time at `TICK_DT` each, independent of the display's actual frame rate.

**Verification:** manual — run it, confirm the track renders with correct tile orientations and no visible gaps/overlaps, confirm the car rotates smoothly and the camera tracks it. If WSL2 OpenGL proves unreliable (the doc's flagged risk), fall back to Pygame here and record that as a revision-history entry, not a silent swap.

### Task 7 — Input adapter + play entry point

**Produces:** `render/input.py` mapping held keys (arrow keys or WASD) to continuous `steering ∈ [-1, 1]` / `throttle ∈ [-1, 1]` (simple direct mapping for v1 — no analog easing needed, a human can already feel it as "playable" with digital-in/continuous-out); `render/play.py` as the script entry point (`uv run neuroarena-play` via a `[project.scripts]` entry in `pyproject.toml`) that constructs `Game` from `sim/tracks/track_01.json`, `AssetManifest` from `render/assets/road_01/manifest.json`, and the `Window`, and starts `arcade.run()`.

**Verification:** manual — play the game: drive around the full loop, confirm brake/reverse feels right, confirm the hard collider stops the car at the kerb instead of letting it through.

---

## Revision history
- 2026-09-12 — Plan written, following Phase 1's FINALIZED requirements. Seven tasks: track representation/validation/codec, car physics, collision, game loop, asset manifest loader, arcade renderer, input + play entry point. `sim` stays arcade-free and episode-free per the doc and per `WORKFLOW.md`'s "stay within the current phase" (no `Environment`/`reset`/`terminated` — that's Phase 3).
- 2026-09-12 — All seven tasks implemented and verified (unit tests for Tasks 1-5, headless-rendered screenshots inspected for Tasks 6-7 in lieu of an interactive display). See the Phase 1 doc's matching revision-history entry for the four implementation-level findings (bounding-circle collision, composite-over-layers manifest preference, aspect-preserving tile scale, empirically-derived sprite rotation).
- 2026-09-12 — Manual play-testing (Task 7's actual stated verification method, run for the first time) found what the headless screenshots in the previous entry had missed: a 180°-backwards car heading and every curve corner's kerb on the wrong edges. Task 5/6 now resolve the road surface to one uniform asphalt fill and draw kerbs procedurally from `boundary_segments` instead of per-`TileKind` bitmap art. Full story and rationale in the Phase 1 doc's matching revision-history entry.
