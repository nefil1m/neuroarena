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
- **Run history is browsable from the dashboard** (Phase 9 for comparison views).
- **Per-generation stats log:** the `TrainingUpdate` stream a run emits (Phase 0) is persisted as the run's history — enough to redraw the fitness curve and compare generations by their stats without reloading a checkpoint.
- **Optional per-generation champion checkpoints** (config-gated, per Phase 4): the best genome/policy of a generation, stored so a chosen generation can be replayed later (Phase 9). Needs a retention policy — keeping every champion for a long run is unbounded.

## Open questions
- Checkpoint file format and per-backend contents (NEAT: population + innovation history; deep RL: policy + optimizer state).
- Where the settings history lives (inline in the checkpoint, in SQLite, or both) and its granularity.
- Checkpoint cadence (every N generations / steps / on demand) and retention policy.
- Retention for per-generation champion checkpoints specifically (keep all / last N / milestone generations only).
- Schema and versioning for forward compatibility.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass (config-independent resume, per-model settings history); status → DRAFT.
- 2026-09-10 — Added a per-generation stats log (persist the `TrainingUpdate` stream as run history) and optional config-gated per-generation champion checkpoints with a retention policy, following the Phase 0 "run history & per-generation records" requirement. Prompted by a request to compare and replay generations of choice after a headless run.
