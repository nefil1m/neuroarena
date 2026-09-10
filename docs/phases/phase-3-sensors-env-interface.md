# Phase 3 — Sensors & Environment Interface

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Implement the car game's observation and action spaces and wire the game to the Phase 0 Environment interface.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Observation vector (v1): 10 floats, normalised** — 7 raycasts + normalised speed + last action (previous steering, previous throttle).
- **Raycasts:** 7 rays in a forward-dense fan `[−75, −45, −20, 0, 20, 45, 75]°` relative to car heading. Value is **proximity** `1 − d/range` (0 = clear, 1 = boundary at the bumper). Rays detect the **track boundary only** in v1; other-car detection is a flagged add-on delivered alongside Phase 5 collisions.
- **Ray range is speed-scaled:** `range = base + k · speed_norm` (see farther when moving faster).
- **Sensor layout is configurable at run creation** (ray count, angles, range params); the values above are the platform-standard default. A non-default sensor config is flagged in the UI as breaking model transfer to other games and breaking resume of an existing model.
- **Action space:** continuous `Box`, 2 dimensions, each `[−1, 1]` — steering, throttle. Exposed as a typed descriptor per the Phase 0 contract.
- The car game implements the Phase 0 Environment interface. Its observation and action descriptors are exactly what the cross-game strict-match rule (Phase 0) compares.
- **The car Environment enforces a configurable maximum episode length** (step count and/or sim-time) and sets `truncated` when it is hit, per the Phase 0 flag contract (`terminated` = crash / finish / off-track; `truncated` = budget hit). The limit is a config parameter (Phase 5), never hard-coded. This is the single-agent "stuck forever" guard; the population-level per-generation ceiling is Phase 4.

## Open questions
- Concrete `base` and `k` for the speed-scaled range, and the `max_range` cap.
- Normalisation ranges for speed and for the raycast base distance.
- Whether "last action" is the raw commanded action or the post-physics realised action.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added the requirement that the car Environment enforces a configurable max episode length and sets `truncated`. This enforcement was briefly placed in the Phase 0 base interface earlier the same day, then moved here: Phase 0 fixes only the meaning of the `terminated` / `truncated` flags, and the standalone game (Phase 1) has no episode concept.
