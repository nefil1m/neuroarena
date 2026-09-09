# Phase 9 — Dashboard: Advanced

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
The rich dashboard slice: cross-run comparison, model/network inspection, and a track editor.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Cross-run comparison charts** — fitness curves and similar, across runs from Phase 6 history.
- **NEAT network-topology visualisation.**
- **Model inspection** — the per-model settings history and lineage from Phase 6.
- **Track editor / regeneration UI** — create, tweak, and regenerate tracks from Phase 2 parameters. This is a new interaction surface and may be split into its own phase later.
- **Independent phase** — reorderable relative to Phases 8 and 10.

## Open questions
- Whether the track editor edits generation parameters only or allows direct geometry editing.
- Charting/graph library choices; how much of this is generic vs backend-specific.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Created by splitting the original Phase 7 (Web Dashboard) into three phases; seeded requirements from the platform decisions pass
