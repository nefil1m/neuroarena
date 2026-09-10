# Phase 8 — Dashboard: Live Canvas

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
An in-browser live top-down view of training on the track.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- The sim streams state to the browser over the existing dashboard API (Phase 7); the browser renders a live top-down view on the track. Because genomes are evaluated in isolation (Phase 4), this shows the genome(s) currently being evaluated, optionally with the trajectories of genomes already evaluated this generation overlaid — not a shared-space swarm.
- This is a **second rendering path** (browser-side) reading the same sim-state stream the dashboard already carries. The local `arcade` renderer is unaffected.
- **Independent phase** — reorderable relative to Phases 9 and 10, and other work may be inserted before it.

## Open questions
- Browser rendering technology (canvas 2D, WebGL, SVG) and how much simulation detail to send.
- State rate over the wire, and behaviour under high sim-speed runs.
- How to present isolated per-genome evaluations as a coherent "watch the generation learn" view — live single genome, overlaid trajectories, or a fast montage.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Created by splitting the original Phase 7 (Web Dashboard) into three phases; seeded requirements from the platform decisions pass.
- 2026-09-10 — Reframed for the platform scope cut (see `../OVERVIEW.md`): with genomes evaluated in isolation rather than in a shared simulation space, the live view is no longer "a running population" / swarm on one track. It now shows the currently-evaluating genome(s), optionally with this generation's completed trajectories overlaid. Added an open question on how to make that read as "watch the generation learn".
