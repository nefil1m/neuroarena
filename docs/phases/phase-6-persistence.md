# Phase 6 — Persistence

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Checkpointing, resume (including resume under changed settings), and run history — SQLite for metadata/history, files for checkpoints.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Storage split:** SQLite holds run metadata and history; model checkpoints are files.
- **Config-independent checkpoints:** a checkpoint holds the model state independent of the run configuration, so a model can be resumed under a different settings set and continue training from its saved state. Valid while the observation and action descriptors are unchanged; a sensor-layout change (Phase 3) starts a fresh model, not a resume.
- **Settings history per model:** a model carries an ordered log of every configuration change over its entire training lifetime — which settings changed, and the point in training (generation / step / wall-time) each change took effect.
- **Opening a model loads its most recent settings by default.** The user can then change them, which appends a new entry to the settings history.
- **Track/map reference, not track storage.** Maps are generated and stored as plain files by Phase 2 (`tracks/<track_id>.json` plus a manifest) — that storage is a **deliberate permanent design choice**, not a placeholder to migrate into SQLite later: tracks are static geometry artifacts, the same shape as checkpoints in the platform's existing "SQLite for metadata, files for artifacts" split (`../OVERVIEW.md`), not metrics that belong in a queryable table. Phase 6 does not re-store map contents; it adds a `track_id` column/reference to run and model metadata (settings history included, per below) so "which map was this trained on" is answerable from SQLite alone, resolved to a file via Phase 2's manifest only when the map itself (not just its id) is needed. Whether SQLite eventually also mirrors Phase 2's manifest (id → path → params) as a queryable table, superseding the plain-file manifest for enumeration/search, is an open question for this phase's own exploration, not decided here.
- **Settings history per model includes track/map selection.** Track switching (Phase 5, live-changeable) is a config change like any other and appends a settings-history entry the same way population size or deviation limit would — no separate mechanism.
- **Run history is browsable from the dashboard** (Phase 9 for comparison views).
- **Per-generation stats log:** the `TrainingUpdate` stream a run emits (Phase 0) is persisted as the run's history — enough to redraw the fitness curve and compare generations by their stats without reloading a checkpoint.
- **Optional per-generation champion checkpoints** (config-gated, per Phase 4): the best genome/policy of a generation, stored so a chosen generation can be replayed later (Phase 9). Needs a retention policy — keeping every champion for a long run is unbounded.

## Open questions
- Checkpoint file format and per-backend contents (NEAT: population + innovation history; deep RL: policy + optimizer state).
- Where the settings history lives (inline in the checkpoint, in SQLite, or both) and its granularity.
- Checkpoint cadence (every N generations / steps / on demand) and retention policy.
- Retention for per-generation champion checkpoints specifically (keep all / last N / milestone generations only).
- Schema and versioning for forward compatibility.
- Whether SQLite grows its own queryable track index (id/params/path) mirroring Phase 2's plain-file manifest, or that manifest stays the only index and SQLite holds bare `track_id` references only.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass (config-independent resume, per-model settings history); status → DRAFT.
- 2026-09-10 — Added a per-generation stats log (persist the `TrainingUpdate` stream as run history) and optional config-gated per-generation champion checkpoints with a retention policy, following the Phase 0 "run history & per-generation records" requirement. Prompted by a request to compare and replay generations of choice after a headless run.
- 2026-09-12 — Added a track/map-reference requirement, following Phase 2's exploration pass which gave maps a `track_id` and plain-file storage. Decided map storage stays files permanently (not a placeholder for SQLite) since tracks are static artifacts like checkpoints, not metrics — Phase 6 references Phase 2's files by `track_id` rather than duplicating them, and settings history tracks map switches the same way it tracks any other live config change (Phase 5). Left open whether SQLite later grows its own queryable track index alongside Phase 2's manifest.
