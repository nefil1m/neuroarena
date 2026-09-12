# Phase 2 — Track Generation

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Procedural tracks parameterised by size, complexity, and seed, renderable as a real track.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Parameters:** size, complexity, seed. Generation is deterministic from the seed.
- **Output serves two consumers:** (a) collision geometry (drivable region + boundary) for the simulation, and (b) enough structure for the renderer to dress it as a real track — road surface, edge lines for kerbs, start/finish, spawn points, scenery hints.
- **Same track representation** as Phase 1's static-track path: a closed-loop sequence of grid cells, each a `TileKind` (straight or 90°-turn curve — see Phase 1), sharing the fixed `cell_size` world-unit constant. Generation means producing a valid closed loop of grid moves, not freeform geometry. Phase 2 becomes the source of tracks, superseding the hand-authored one.
- **Ray-range default is tied to a few car-lengths** (see Phase 3) so raycast sensors stay informative across generated track widths — a long fixed range saturates on narrow tracks and blinds the model.
- **Track lifecycle is user-driven** (per `../OVERVIEW.md`): training continues on a track until the user regenerates or switches. It is not a per-run success signal.

## Open questions
- Generation algorithm for the grid-move walk (e.g. self-avoiding random walk with a loop-closure step vs a template/piece-based approach) — the representation is now grid/`TileKind`-based (settled in Phase 1), not freeform spline/cellular/tile-choice-in-the-abstract.
- How "complexity" maps to concrete geometry now that turns are fixed at 90°: likely turn *frequency* and path length/shape, rather than curvature or track-width variance (width is presumably constant, tied to the fixed `cell_size`) — width variance may not apply at all under this representation.
- Avoiding self-intersecting loops when the random walk revisits a cell.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT
- 2026-09-12 — Adopted Phase 1's grid/`TileKind` track representation (closed loop, 90°-turn tiles only) in place of the previously-abstract "collision geometry" wording, and resolved the closed-loop-vs-point-to-point open question in favor of closed loops only — prompted by a review of available tile-based racing art (straight/curve tiles only, no diagonals) and a deliberate choice to keep track generation simple. Reworded the generation-algorithm and complexity open questions to reflect that generation is now a grid-move walk problem, not a freeform-geometry one, and added self-intersection avoidance as an explicit open question.
