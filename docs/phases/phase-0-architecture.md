# Phase 0 — Architecture Scaffolding

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Define the Game/Model/Trainer interfaces and the shared config schema — general enough for many games, not car-specific.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Environment interface is a local `Protocol` that mirrors the Gymnasium `Env` contract — it does not subclass or import `gymnasium`.** Shape: `reset(*, seed=None) -> observation`, `step(action) -> (observation, terminated, truncated, info)`, plus an observation-space descriptor, an action-space descriptor, and a score/fitness signal surfaced via `info` and/or a pluggable success-criteria component. The core (`sim`, interfaces, NEAT backend) stays free of `gymnasium`; a thin `gymnasium.Env` adapter is written only where a real Gym env is required — planned for the Phase 10 deep-RL backend (stable-baselines3), which is the single place `gymnasium` is imported. Space descriptors are local classes (`Box`, `Discrete`), and their equality is what the strict cross-game match rule compares.
  - **Revalidate this at Phase 10.** The choice optimises for a `gymnasium`-free core and full control of the match semantics, at the cost of one adapter. Before building the RL backend, confirm the adapter stays thin and we are not re-implementing large parts of `gymnasium.wrappers` / `gymnasium.vector`; if we are, reconsider adopting `gymnasium` in the core then.
- **`terminated` vs `truncated` are distinct by contract.** `terminated` is an in-world end (crash, finish line, off-track); `truncated` is an end forced from outside the world (a step or time budget was hit). Phase 0 fixes only the *meaning* of the two flags so every backend can rely on it. *Enforcing* a specific episode limit is the env implementation's job (Phase 3, for the car), and the limit value is a config parameter (Phase 5); the standalone game milestone (Phase 1) has no episode concept at all. The population-level per-generation ceiling is Phase 4 / Phase 5.
- **The interface must not assume "car-ness."** No raycasts, track, or steering in the base interface — those are car-game specifics that live behind it. It must fit future games too: Flappy Bird, a Mario-style platformer, Galaga, bullet hell (none implemented now, see Phase 11).
- **Model contract:** a pure function from observation vector to action vector. No game knowledge. A model records the observation-space and action-space descriptors it was trained against.
- **Cross-game compatibility rule — strict descriptor match:** a model may only be loaded into a game whose observation- and action-space descriptors are identical to the model's. The platform blocks a mismatched load at run creation. No adapter layer, no canonical shared space.
- **Trainer contract:** learning backends (NEAT, deep RL, and possibly more later) are pluggable and consume the same Environment + Model interfaces and the same config.
- **Simulation is decoupled from rendering:** the sim exposes state; a renderer is an optional consumer and is never on the training path.
- **Shared config schema:** Phase 0 fixes the schema's *shape*, its serializability (for persistence in Phase 6 and the trainer↔dashboard API in Phase 7), and its versioning for forward compatibility. The concrete parameter list — including the episode limit and the per-generation ceiling — is owned by Phase 5; the schema only has to accommodate it.

## Open questions
- Serialization format (JSON vs other) and schema-versioning approach for configs and checkpoints.
- Exact fields of the local `Box` / `Discrete` descriptors and what their equality compares — shape only, or shape + bounds + dtype. This is the definition of the strict-match rule.
- Whether `reset` / `step` are a `typing.Protocol` (structural, no inheritance) or an ABC the games subclass.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added a bounded-episode requirement (Environment enforces a max episode length, reports it via `truncated`, limit lives in config) and a per-generation ceiling field in the shared config schema. Before this nothing in the interface guaranteed an episode or a batch generation would ever end.
- 2026-09-10 — Resolved the Gymnasium open question: the Environment interface is a local `Protocol` mirroring Gym's contract, with local `Box` / `Discrete` descriptors, and a thin `gymnasium.Env` adapter to be written for the Phase 10 RL backend (the only place `gymnasium` is imported). Alternatives considered: subclass `gymnasium.Env`, or reuse `gymnasium.spaces` as descriptors — both rejected to keep the core dependency-free and to own the strict-match semantics; flagged for revalidation at Phase 10. Same pass trimmed the bounded-episode requirement added earlier today: Phase 0 now fixes only the `terminated` vs `truncated` contract; episode-limit *enforcement* moved to Phase 3 (car env) and the config fields to Phase 5, since the standalone game (Phase 1) has no episode or timeout concept and the base interface should not carry one.
