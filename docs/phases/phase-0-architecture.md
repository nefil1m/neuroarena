# Phase 0 — Architecture Scaffolding

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Define the Game/Model/Trainer interfaces and the shared config schema — general enough for many games, not car-specific.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Environment interface follows the Gymnasium `Env` contract:** `reset() -> observation`, `step(action) -> (observation, terminated, truncated, info)`, plus an observation-space descriptor, an action-space descriptor, and a score/fitness signal surfaced via `info` and/or a pluggable success-criteria component.
- **Episodes are bounded.** The Environment enforces a maximum episode length (step count and/or sim-time) and reports a hit limit via `truncated` — kept distinct from `terminated`, which is an in-world end such as a crash or a finish. The limit is a field in the shared config schema, never hard-coded. This is what stops a single agent running forever; the per-generation ceiling below is what stops a whole batch generation hanging on one such agent.
- **The interface must not assume "car-ness."** No raycasts, track, or steering in the base interface — those are car-game specifics that live behind it. It must fit future games too: Flappy Bird, a Mario-style platformer, Galaga, bullet hell (none implemented now, see Phase 11).
- **Model contract:** a pure function from observation vector to action vector. No game knowledge. A model records the observation-space and action-space descriptors it was trained against.
- **Cross-game compatibility rule — strict descriptor match:** a model may only be loaded into a game whose observation- and action-space descriptors are identical to the model's. The platform blocks a mismatched load at run creation. No adapter layer, no canonical shared space.
- **Trainer contract:** learning backends (NEAT, deep RL, and possibly more later) are pluggable and consume the same Environment + Model interfaces and the same config.
- **Simulation is decoupled from rendering:** the sim exposes state; a renderer is an optional consumer and is never on the training path.
- **Shared config schema:** covers all run-creation parameters (see Phase 5), is serializable (for persistence in Phase 6 and for the trainer↔dashboard API in Phase 7), and is versioned for forward compatibility. It carries both the per-episode limit above and a **per-generation ceiling** (wall-clock and/or total steps) that a batch trainer uses to force-close a generation in which some individual never terminates or truncates on its own.

## Open questions
- Whether the Environment interface subclasses / depends on `gymnasium.Env` directly or is a local protocol that mirrors it.
- Serialization format (JSON vs other) and schema-versioning approach for configs and checkpoints.
- Exact shape of the space descriptors (reuse Gymnasium `Box` / `Discrete`, or a local equivalent) — this is what the strict-match rule compares.
- Whether the per-episode limit is expressed in steps, sim-seconds, or both, and whether the per-generation ceiling is measured in wall-clock or sim-time (they diverge once simulation speed is adjustable — Phase 5).

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added a bounded-episode requirement (Environment enforces a max episode length, reports it via `truncated`, limit lives in config) and a per-generation ceiling field in the shared config schema. Before this nothing in the interface guaranteed an episode or a batch generation would ever end.
