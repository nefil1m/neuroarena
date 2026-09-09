# Phase 7 — Dashboard: Control Panel

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
The first dashboard slice: run control and live scalar metrics, working while the sim runs headless.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Stack:** FastAPI + WebSocket backend, React frontend. The trainer and dashboard communicate only over the HTTP/WebSocket API — no shared memory.
- **Features:**
  - Start / stop / pause a training run.
  - Live scalar metrics: generation #, best & mean fitness, alive count, sim-time vs wall-time, current sim speed.
  - Live edit of the live-changeable config subset (defined in Phase 5).
  - Basic run list + history view (data from Phase 6).
- **No in-browser rendering in this phase** — a run is watched locally via the `arcade` renderer. The in-browser live view is Phase 8.
- **Local-first:** binds to localhost, single user, no auth. The architecture must not preclude adding auth + remote deployment later ("local-first, remote-capable by construction").

## Open questions
- Metric push cadence and the WebSocket message schema.
- How the dashboard discovers and attaches to a running trainer process.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created (as "Phase 7 — Web Dashboard", covering the whole dashboard)
- 2026-09-09 — Split: this doc is now the control-panel slice only (live canvas → Phase 8, advanced → Phase 9); seeded requirements from the platform decisions pass; status → DRAFT
