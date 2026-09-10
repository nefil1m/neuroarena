# Phases — Car AI Project

Status: **FINALIZED** — the phase breakdown and ordering below are settled as of 2026-09-10. Individual phase docs are still explored and promoted to FINALIZED one at a time, but the set of phases and their sequence now changes only by deliberate revision (see `WORKFLOW.md`), not a drive-by edit.

## How to read this
Each row links to a doc under `/phases/`. Those docs hold each phase's detailed requirements and, once settled, its implementation plan. Actual build work happens inside a phase doc, never here. Phases 0–10 now carry requirements seeded from the 2026-09-09 platform decisions pass (status: DRAFT); Phases 11–12 are stubs (NOT STARTED).

The phase set and its ordering are settled. Two flexibilities are part of that decision, not open questions: additional learning backends (peers of Phases 4 and 10 behind the Model interface) may be inserted later, and the three dashboard phases (7, 8, 9) may be reordered relative to each other and to Phase 10, or have other work inserted between them.

## Phase list (draft)

| # | Phase | One-line goal | Doc |
|---|---|---|---|
| 0 | Architecture scaffolding | Game/Model/Trainer interfaces + shared config schema — Gym-`Env`-shaped, not car-specific | `phases/phase-0-architecture.md` |
| 1 | Playable game core | Kinematic car + manual control on one static track, sprite renderer (`arcade`) | `phases/phase-1-game-core.md` |
| 2 | Track generation | Procedural tracks (size/complexity/seed), renderable as a real track | `phases/phase-2-track-generation.md` |
| 3 | Sensors & env interface | 10-float observation, continuous action, game wired to the Environment interface | `phases/phase-3-sensors-env-interface.md` |
| 4 | Learning backend: NEAT | NEAT end-to-end, headless + rendered | `phases/phase-4-learning-backend-neat.md` |
| 5 | Training controls | Population size, batch/sequential spawn, deviation limit, collisions, sim speed, success criteria | `phases/phase-5-training-controls.md` |
| 6 | Persistence | Checkpoint / resume (incl. resume under changed settings) + run history — SQLite + files | `phases/phase-6-persistence.md` |
| 7 | Dashboard — control panel | Start/stop/pause, live scalar metrics, live-safe config edits, basic run history | `phases/phase-7-dashboard-control-panel.md` |
| 8 | Dashboard — live canvas | In-browser live top-down view of a running population | `phases/phase-8-dashboard-live-canvas.md` |
| 9 | Dashboard — advanced | Cross-run comparison charts, NEAT topology viz, track editor UI, model inspection | `phases/phase-9-dashboard-advanced.md` |
| 10 | Learning backend: deep RL | PPO (stable-baselines3) on the same Model interface | `phases/phase-10-learning-backend-deep-rl.md` |
| 11 | Second game (stretch) | Another game on the same interfaces; cross-game model testing | `phases/phase-11-second-game.md` |
| 12 | 3D (stretch) | Explore what 3D would require — not scoped yet | `phases/phase-12-3d-stretch.md` |

## Revision history
- 2026-09-09 — Initial draft phase list, carried over from early conversation.
- 2026-09-09 — Split the single "Phase 7 — Web Dashboard" into three reorderable phases: 7 (control panel), 8 (in-browser live canvas), 9 (advanced views), because one phase was too large for the dashboard. Renamed the two learning phases from "First Learning Model (NEAT)" / "Second Model Backend (RL)" to peer "Learning Backend" phases (4 NEAT, 10 deep RL), since both are equal consumers of one Model interface rather than a primary and a follow-up. Renumbered the stretch phases: second game 9 → 11, 3D 10 → 12. Seeded carried-over requirements into Phases 0–10 (supersedes 7a46fd6).
- 2026-09-10 — Promoted from DRAFT to FINALIZED. The phase set and ordering are now locked and change only by deliberate revision; individual phase-doc statuses are unchanged. This is the deliberate promotion `WORKFLOW.md` requires before Phase 0 implementation can begin (supersedes 60c7011).
