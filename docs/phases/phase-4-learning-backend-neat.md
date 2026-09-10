# Phase 4 — Learning Backend: NEAT

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
NEAT neuroevolution end-to-end on the car game, running both headless and rendered.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **NEAT is one of several peer learning backends** behind the Phase 0 Model/Trainer interface. It consumes the same Environment and config as any other backend. Deep RL is Phase 10; more backends may follow.
- **Genome → network** maps the 10-float observation to the 2-float continuous action (e.g. `tanh` output neurons mapped to `[−1, 1]`).
- **Runs headless** (fast, no rendering) and with the `arcade` renderer attached. Rendering is never required.
- **Each genome is evaluated in isolation** on its own copy of the track — no other cars, no car-to-car interaction. This keeps per-genome fitness clean and lets evaluations parallelise trivially. The generational loop is **batch**: evaluate the whole population, then select and breed the next generation.
- **Fitness comes from the `Objective`** (Phase 0 interface; concrete objectives in Phase 5), read via `Objective.fitness()`. Never baked into the backend.
- **Per-generation records.** The trainer emits a `TrainingUpdate` per generation (fitness distribution, champion metrics, species count). Config-gated, it also saves the generation's champion genome, so a chosen generation can be re-simulated later — the sim is deterministic given `(seed, genome)`, so the genome plus seed is enough, no stored trajectory. Storage and retention are Phase 6; the inspection UI is Phase 9.
- **Generations always terminate.** The trainer enforces the per-generation ceiling from config (Phase 0 / Phase 5): when it fires, any individual still running is force-truncated and the generation advances. Together with the per-episode `truncated` limit, a generation cannot hang on an individual that neither crashes nor finishes.
- Checkpoint / resume and the settings-history requirement (Phase 6) apply, with NEAT-specific state (population, genomes, innovation history).

## Open questions
- Which NEAT implementation (`neat-python` vs a custom or other library).
- Output activation and the exact mapping from network outputs to `[−1, 1]` controls.
- Speciation and hyperparameter defaults.
- Whether an individual force-truncated by the generation ceiling is scored on its partial progress or penalised.
- Champion-genome capture cadence: every generation, every N, or only on improvement.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created (as "Phase 4 — First Learning Model (NEAT)").
- 2026-09-09 — Renamed to "Learning Backend: NEAT" (peer of Phase 10); seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added the requirement that generations always terminate: the trainer force-truncates any individual still running when the per-generation ceiling fires. Added the open question of how a force-truncated individual is scored.
- 2026-09-10 — Followed the platform scope cut (see `../OVERVIEW.md`): replaced "population evaluated in a shared simulation space" with per-genome isolated evaluation (own track copy, no car-to-car interaction) for clean fitness and trivial parallelism, and removed the batch-vs-sequential distinction — the loop is batch only. Sequential spawn mode was dropped from scope.
- 2026-09-10 — Added per-generation records: the trainer emits a `TrainingUpdate` per generation and, config-gated, saves the champion genome for later deterministic re-simulation. This follows the Phase 0 "run history & per-generation records" requirement; the request was for a way to compare and replay generations of choice after a headless run. Storage/retention is Phase 6, the inspection UI Phase 9.
