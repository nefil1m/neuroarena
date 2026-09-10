# Phase 11 — Second Game (Stretch)

Status: **NOT STARTED** — requirements not yet finalized. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Another game on the same interfaces and infrastructure; validates that the Phase 0 interfaces are genuinely game-agnostic.

## Context
Candidate games: Flappy Bird, a Mario-style platformer, Galaga, bullet hell. Trained models are **not** expected to move between games — the Phase 0 compatibility check is resume-safety within one game, not cross-game transfer. The value of a second game is confirming the Phase 0 interfaces did not accidentally bake in car-specific assumptions, plus variety and the chance to reuse the renderer, trainer, dashboard, and persistence. The Phase 0 interface must stay general enough that a second game slots in without redesigning it.

## Requirements
_Not yet explored. Fill this in during requirements exploration sessions, before writing an implementation plan. Reference `../OVERVIEW.md` for related open questions and carry over anything relevant here._

## Open questions
_Phase-specific questions go here as they come up during exploration._

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created (as "Phase 9 — Second Game (Stretch)").
- 2026-09-09 — Renumbered to Phase 11; added candidate-games context from the platform decisions pass.
- 2026-09-10 — Repurposed for the platform scope cut (see `../OVERVIEW.md`): the goal changed from "cross-game model testing" to "validate the Phase 0 interfaces are game-agnostic", since cross-game model transfer was dropped. A trained model is not expected to run in a structurally different game; the second game's value is interface validation, infrastructure reuse, and variety.
