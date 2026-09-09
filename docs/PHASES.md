# Phases — Car AI Project

Status: **DRAFT** — this phase list is a first pass from early conversation. Phases may be split, merged, reordered, added, or dropped once requirements exploration (see `OVERVIEW.md`) progresses. Don't treat the numbering below as committed.

## How to read this
Each row links to a doc under `/phases/`. That doc is currently a stub (status: NOT STARTED) — it will hold that phase's detailed requirements and, once those are settled, its implementation plan. Actual build work happens inside a phase doc, never here.

## Phase list (draft)

| # | Phase | One-line goal | Doc |
|---|---|---|---|
| 0 | Architecture scaffolding | Define the Game/Model/Trainer interfaces and shared config schema | `phases/phase-0-architecture.md` |
| 1 | Playable game core | Car physics + manual control on one static track | `phases/phase-1-game-core.md` |
| 2 | Track generation | Procedural tracks with size/complexity/seed params | `phases/phase-2-track-generation.md` |
| 3 | Sensors & env interface | Observation/action space; game wired to the Environment interface | `phases/phase-3-sensors-env-interface.md` |
| 4 | First learning model | NEAT end-to-end, headless + rendered | `phases/phase-4-first-model-neat.md` |
| 5 | Training controls | Population size, batch/sequential, deviation limits, collisions, speed, success criteria | `phases/phase-5-training-controls.md` |
| 6 | Persistence | Checkpointing, resume, run history | `phases/phase-6-persistence.md` |
| 7 | Web dashboard | FastAPI + WebSocket backend, React frontend, live control & metrics | `phases/phase-7-dashboard.md` |
| 8 | Second model backend | Deep RL backend on the same Model interface | `phases/phase-8-second-model-backend.md` |
| 9 | Second game (stretch) | Another 2D game on the same interfaces; cross-game model testing | `phases/phase-9-second-game.md` |
| 10 | 3D (stretch) | Explore what 3D would require — not scoped yet | `phases/phase-10-3d-stretch.md` |

## Revision history
- 2026-09-09 — Initial draft phase list, carried over from early conversation
