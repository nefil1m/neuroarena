# Project Overview — Car AI Game Platform

Status: **DRAFT** — the feature set and decisions below are a starting point carried over from initial conversation, not finalized. Expect this document to change substantially during requirements exploration. Nothing here should be treated as committed until its status says so.

## Pitch
A 2D (later possibly 3D) simulation platform where games and learning models are decoupled: build a driving game, train models to play it, swap in different model types or games, and control everything — including headless/background training — from a web dashboard.

## High-level feature set (draft)
- Top-down 2D car driving game with configurable procedural track generation (size, complexity, seed)
- Learning models decoupled from the game via a defined interface — a model doesn't know which game it's driving, a game doesn't know which model is driving it
- First model backend: NEAT (neuroevolution) — population-based, no manual reward shaping required
- Planned second model backend: deep RL (e.g. PPO via stable-baselines3), to prove the decoupling actually works
- Configurable training/run behavior: population size, batch vs sequential run spawning, inheritance between runs, max deviation from track before death, collisions between cars on/off, adjustable simulation speed, headless vs rendered
- Pluggable success criteria (finish line crossed / N checkpoints / manual track regeneration)
- Persistence: checkpointing and resuming training runs, run history
- Control interface: local web dashboard that works with the simulation running headless, with live metrics, live config changes, and an optional live canvas view
- Future/stretch: additional 2D games on the same interfaces, testing models across games, 3D

## Key decisions so far (tentative — see status column)
| Decision | Current answer | Status |
|---|---|---|
| Game engine / stack | Python + Pygame for the game core | tentative |
| First learning approach | NEAT (neuroevolution) | tentative |
| Control interface | Local web dashboard: FastAPI + WebSocket backend, React frontend | tentative |
| Persistence | SQLite for run metadata/history, files for model checkpoints | tentative |
| Rendering | Headless by default; rendering is an optional consumer of simulation state, not required for training | tentative |

None of these are FINALIZED. They were working assumptions from the initial conversation and should be revisited during requirements exploration — including the alternatives Claude should surface for each, per `WORKFLOW.md` (e.g. Unity/Godot vs custom Pygame, NEAT vs RL from the start, dashboard vs other control UI).

## Open questions (non-exhaustive — expand freely during exploration)
- Engine choice: stick with custom Pygame, or use Godot/Unity? (trade-offs not yet discussed in depth)
- Observation space design: raycasting sensor count/angles, what else the model perceives
- Action space: discrete vs continuous controls
- What has to match for "a model trained in one game" to be evaluatable in another game?
- How much of the dashboard is v1 vs a later phase
- Is this ever meant to run anywhere but locally?

## Links
- Workflow & standards: `WORKFLOW.md`
- Phase list: `PHASES.md`
- Individual phase docs: `/phases/`

## Revision history
- 2026-09-09 — Initial draft, seeded from early conversation
