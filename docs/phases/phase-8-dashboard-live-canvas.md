# Phase 8 — Dashboard: Live Canvas

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
An in-browser live top-down view of a running population.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- The sim streams state to the browser over the existing dashboard API (Phase 7); the browser renders a live top-down view of the population on the track.
- This is a **second rendering path** (browser-side) reading the same sim-state stream the dashboard already carries. The local `arcade` renderer is unaffected.
- **Independent phase** — reorderable relative to Phases 9 and 10, and other work may be inserted before it.

## Open questions
- Browser rendering technology (canvas 2D, WebGL, SVG) and how much simulation detail to send.
- State rate over the wire, and behaviour under high sim-speed runs.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Created by splitting the original Phase 7 (Web Dashboard) into three phases; seeded requirements from the platform decisions pass
