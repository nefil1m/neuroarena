# Phase 5 — Training Controls

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Configurable training/run behaviour: population size, spawn mode, deviation limit, collisions, simulation speed, success criteria.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Config knobs** (per `../OVERVIEW.md`): population size; generation spawn mode (**batch** vs **sequential**, as defined in `../OVERVIEW.md`); inheritance between runs; max deviation from track before death; car-to-car collisions on/off (track-boundary collision is always on); adjustable simulation speed; headless vs rendered.
- **Pluggable per-run success criteria** (e.g. finish line crossed / N checkpoints).
- **Each knob is classified** as either set-at-run-creation or live-changeable while a run is in progress. The dashboard (Phase 7) exposes the live-changeable subset.
- **Settings are changeable between runs and on resume.** A model saved under one settings set can be resumed under a different (e.g. harder) set and continue training from its saved state — see Phase 6. This is valid as long as the observation and action descriptors are unchanged; changing sensor layout (Phase 3) starts a fresh model rather than a resume.

## Open questions
- The exact partition of knobs into set-at-creation vs live-changeable.
- Semantics of changing certain knobs mid-run (e.g. population size in batch mode).
- A concrete definition of "inheritance between runs."

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT
