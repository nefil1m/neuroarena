# Phase 4 — Learning Backend: NEAT

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
NEAT neuroevolution end-to-end on the car game, running both headless and rendered.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **NEAT is one of several peer learning backends** behind the Phase 0 Model/Trainer interface. It consumes the same Environment and config as any other backend. Deep RL is Phase 10; more backends may follow.
- **Genome → network** maps the 10-float observation to the 2-float continuous action (e.g. `tanh` output neurons mapped to `[−1, 1]`).
- **Runs headless** (fast, no rendering) and with the `arcade` renderer attached. Rendering is never required.
- **Population evaluated in a shared simulation space** (per `../OVERVIEW.md`). Batch and sequential spawn modes are defined in Phase 5.
- **Fitness comes from the game / pluggable success criteria** (Phase 5), never baked into the backend.
- Checkpoint / resume and the settings-history requirement (Phase 6) apply, with NEAT-specific state (population, genomes, innovation history).

## Open questions
- Which NEAT implementation (`neat-python` vs a custom or other library).
- Output activation and the exact mapping from network outputs to `[−1, 1]` controls.
- Speciation and hyperparameter defaults.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created (as "Phase 4 — First Learning Model (NEAT)")
- 2026-09-09 — Renamed to "Learning Backend: NEAT" (peer of Phase 10); seeded requirements from the platform decisions pass; status → DRAFT
