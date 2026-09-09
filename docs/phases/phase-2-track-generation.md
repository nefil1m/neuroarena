# Phase 2 — Track Generation

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Procedural tracks parameterised by size, complexity, and seed, renderable as a real track.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Parameters:** size, complexity, seed. Generation is deterministic from the seed.
- **Output serves two consumers:** (a) collision geometry (drivable region + boundary) for the simulation, and (b) enough structure for the renderer to dress it as a real track — road surface, edge lines for kerbs, start/finish, spawn points, scenery hints.
- **Same track representation** as Phase 1's static-track path; Phase 2 becomes the source of tracks, superseding the hand-authored one.
- **Ray-range default is tied to a few car-lengths** (see Phase 3) so raycast sensors stay informative across generated track widths — a long fixed range saturates on narrow tracks and blinds the model.
- **Track lifecycle is user-driven** (per `../OVERVIEW.md`): training continues on a track until the user regenerates or switches. It is not a per-run success signal.

## Open questions
- Generation algorithm (spline loop, cellular, tile-based, …).
- How "complexity" maps to concrete geometry (curvature, track-width variance, number/tightness of turns).
- Whether tracks are always closed loops or can be point-to-point.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT
