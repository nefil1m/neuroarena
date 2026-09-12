# Phase 9 — Dashboard: Advanced

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
The rich dashboard slice: cross-run comparison, model/network inspection, and a track editor.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Cross-run comparison charts** — fitness curves and similar, across runs from Phase 6 history.
- **Generation drill-down.** From a run's fitness curve, select a past generation to see its stats and — where a champion checkpoint was saved (Phase 6) — replay that champion on the track (recorded headless run or live).
- **NEAT network-topology visualisation.**
- **Model inspection** — the per-model settings history and lineage from Phase 6.
- **Track editor / regeneration UI** — a browser control surface (new-map button, size/complexity sliders, seed input) wrapping Phase 2's generation function directly: this phase adds the UI only, not the generator, storage, or map identity, which Phase 2 already owns. No real game art is assumed for these controls — generic form/slider UI is sufficient. Lists and lets the user pick among stored maps (Phase 2's manifest / Phase 6's `track_id` references), triggers generation with a size/complexity/seed, and shows drift when the actual tile count differs from the requested one (Phase 2's ±15% tolerance). This is a new interaction surface and may be split into its own phase later.
- **Independent phase** — reorderable relative to Phases 8 and 10.

## Open questions
- Whether the track editor edits generation parameters only or allows direct geometry editing.
- Charting/graph library choices; how much of this is generic vs backend-specific.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Created by splitting the original Phase 7 (Web Dashboard) into three phases; seeded requirements from the platform decisions pass.
- 2026-09-10 — Added a "generation drill-down" requirement: pick a generation off the fitness curve to inspect its stats and replay its saved champion. Follows the Phase 0 "run history & per-generation records" requirement, prompted by a request to compare and replay generations of choice after a headless run.
- 2026-09-12 — Clarified the track editor bullet following Phase 2's exploration pass: this phase wraps Phase 2's generation function/CLI in a browser UI (button, sliders, map list) — it does not own the algorithm, storage, or map identity, all settled in Phase 2. Until this phase is built, the same generation is available via Phase 2's CLI dev tool.
