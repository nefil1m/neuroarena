# Phase 1 — Playable Game Core

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Kinematic car with manual control on one static track, drawn by a sprite-based renderer.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Car physics — kinematic bicycle model:** state is `(x, y, heading, speed)`. Controls are continuous: `steering ∈ [−1, 1]` and `throttle ∈ [−1, 1]`. No tire slip / drift in v1 (a dynamic model can slot in later behind the same interface).
- **One hand-authored static track** for this phase; procedural generation is Phase 2. The track is defined by collision geometry (a drivable region plus its boundary). Leaving the boundary is "off track"; boundary collision is always on.
- **Renderer built on the `arcade` library**, living only in the render layer and reading simulation state. It is swappable and never required for the sim to run. Pygame is the documented fallback (e.g. if WSL2 OpenGL proves unreliable).
- **It must look like a real game:** a rotating car sprite, a road surface, kerbs, scenery, and a camera that follows the car. Placeholder art now; real assets (supplied later) drop into the renderer with no simulation changes. Collision geometry and visual assets are separate layers.
- **Human-playable:** a keyboard/gamepad input adapter produces the same `(steering, throttle)` output a model does — a human is just another policy. This is a permanent feature, not a bootstrapping throwaway.

## Open questions
- File format for the hand-authored track, and how it relates to Phase 2's generated output.
- The asset-slot specification: which sprite/texture slots the renderer exposes for later real art.
- Simulation tick rate and its relationship to adjustable simulation speed (Phase 5).

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT
