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
| 5 | Training controls | Population size, sim speed, success criteria, timeouts, run-level stop conditions, NEAT hyperparameters | `phases/phase-5-training-controls.md` |
| 6 | Persistence | Checkpoint / resume (incl. resume under changed settings) + run history — SQLite + files | `phases/phase-6-persistence.md` |
| 7 | Dashboard — control panel | Start/stop/pause, live scalar metrics, live-safe config edits, basic run history | `phases/phase-7-dashboard-control-panel.md` |
| 8 | Dashboard — game viewer | Game window (extended `arcade` renderer) showing a run's whole batch live, opened and controlled from the panel | `phases/phase-8-dashboard-game-viewer.md` |
| 9 | Dashboard — advanced | Cross-run comparison charts, NEAT topology viz, track editor UI, model inspection | `phases/phase-9-dashboard-advanced.md` |
| 10 | Learning backend: deep RL | PPO (stable-baselines3) on the same Model interface | `phases/phase-10-learning-backend-deep-rl.md` |
| 11 | Second game (stretch) | Another game on the same interfaces; validates they are genuinely game-agnostic | `phases/phase-11-second-game.md` |
| 12 | 3D (stretch) | Explore what 3D would require — not scoped yet | `phases/phase-12-3d-stretch.md` |

## Revision history
- 2026-09-09 — Initial draft phase list, carried over from early conversation.
- 2026-09-09 — Split the single "Phase 7 — Web Dashboard" into three reorderable phases: 7 (control panel), 8 (in-browser live canvas), 9 (advanced views), because one phase was too large for the dashboard. Renamed the two learning phases from "First Learning Model (NEAT)" / "Second Model Backend (RL)" to peer "Learning Backend" phases (4 NEAT, 10 deep RL), since both are equal consumers of one Model interface rather than a primary and a follow-up. Renumbered the stretch phases: second game 9 → 11, 3D 10 → 12. Seeded carried-over requirements into Phases 0–10 (supersedes 7a46fd6).
- 2026-09-10 — Promoted from DRAFT to FINALIZED. The phase set and ordering are now locked and change only by deliberate revision; individual phase-doc statuses are unchanged. This is the deliberate promotion `WORKFLOW.md` requires before Phase 0 implementation can begin (supersedes 60c7011).
- 2026-09-10 — Adjusted two phase one-liners for the scope cut recorded in `OVERVIEW.md` the same day (dropped cross-game model transfer, sequential spawn mode, and shared-space collisions). Phase 5 no longer lists batch/sequential spawn or a collisions toggle; Phase 11's goal is now "validate the interfaces are game-agnostic" rather than "cross-game model testing". Phase set and ordering unchanged.
- 2026-09-12 — Dropped "deviation limit" from Phase 5's one-liner: Phase 3's exploration pass removed the corresponding "max deviation from track before death" knob (its referent didn't survive resolving how `terminated`/off-track actually works — see Phase 3 and Phase 5's Revision history). Phase set and ordering unchanged.
- 2026-09-13 — Expanded Phase 5's one-liner following its exploration pass: added run-level stop conditions and NEAT hyperparameters, both new knobs this pass added to Phase 5's scope (see its Revision history). Phase set and ordering unchanged.
- 2026-09-18 — **Deliberate revision of Phase 8:** from "Dashboard — live canvas" (in-browser top-down view) to "Dashboard — game viewer" (a native `arcade` window showing the whole concurrent batch, spawned and controlled from the panel). Reason: a browser view means either duplicating the game's rendering in TypeScript or streaming frames from an offscreen `arcade` (GL risk under WSL2, encoding cost); extending the existing renderer avoids both. Phase set, numbering and ordering unchanged; file renamed to `phases/phase-8-dashboard-game-viewer.md`. Details and rejected alternatives are in that doc's Revision history.
