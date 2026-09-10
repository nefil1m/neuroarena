# Phase 10 — Learning Backend: Deep RL

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
A deep reinforcement learning backend (PPO via stable-baselines3) on the same Model interface, proving the game/model decoupling.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **A peer learning backend** behind the Phase 0 Model/Trainer interface — same Environment, config, and cross-game rules as NEAT (Phase 4). Both backends are committed; more may be added as further phases.
- **PPO via stable-baselines3** as the initial algorithm. The continuous action space (Phase 3) was chosen partly so PPO needs no per-backend action adapter.
- **Runs headless and rendered.** Checkpoint / resume and the settings-history requirement (Phase 6) apply, with RL-specific state (policy + optimizer).

## Open questions
- **Revalidate the Phase 0 "local `Protocol`, not `gymnasium`" decision here.** Confirm the `gymnasium.Env` adapter that wraps the Environment stays thin, and that we are not re-implementing large parts of `gymnasium.wrappers` / `gymnasium.vector`. If we are, reconsider adopting `gymnasium` in the core.
- Single-env vs vectorised-env execution, and how that maps onto the shared-space simulation used for NEAT.
- Reward shaping vs reusing the pluggable success criteria that provide NEAT's fitness.
- Whether recorded human laps (Phase 1) are used for any pretraining / imitation.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created (as "Phase 8 — Second Model Backend (RL)").
- 2026-09-09 — Renumbered to Phase 10 and renamed "Learning Backend: Deep RL" (peer of Phase 4); seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added a revalidation checkpoint for the Phase 0 decision to mirror the Gym contract with a local `Protocol` rather than depend on `gymnasium`; this backend is where the `gymnasium.Env` adapter gets written and where that trade-off is re-checked.
