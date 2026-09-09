# Phases — Car AI Project

Status: **DRAFT** — this phase list is a first pass from early conversation. Phases may be split, merged, reordered, added, or dropped once requirements exploration (see `OVERVIEW.md`) progresses. Don't treat the numbering below as committed.

## How to read this
Each row links to a doc under `/phases/`. Those docs hold each phase's detailed requirements and, once settled, its implementation plan. Actual build work happens inside a phase doc, never here. Phases 0–10 now carry requirements seeded from the 2026-09-09 platform decisions pass (status: DRAFT); Phases 11–12 are stubs (NOT STARTED).

Numbering is still DRAFT and reorderable. Learning backends (Phases 4, 10, …) are peers behind the Model interface — more may be inserted. The three dashboard phases (7, 8, 9) can be reordered relative to each other and to Phase 10, or have other work inserted between them.

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
- 2026-09-09 — Initial draft phase list, carried over from early conversation
- 2026-09-09 — Split the dashboard into three phases (7–9), renamed the learning-backend phases (4, 10) as peers, renumbered stretch phases to 11–12, seeded requirements into Phases 0–10 (supersedes 7a46fd6)
