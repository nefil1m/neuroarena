# Project Overview — Car AI Game Platform

Status: **FINALIZED** — the platform-level decisions in this document are settled and left DRAFT on 2026-09-10. Changing any of them now requires a deliberate revision (see `WORKFLOW.md`), not a drive-by edit. Per-phase requirements exploration continues in the individual phase docs and does not reopen this document unless it contradicts a platform-level decision here.

## Pitch
A 2D (later possibly 3D) simulation platform where games and learning models are decoupled: build a driving game, train models to play it, swap in different model types or games, and control everything — including headless/background training — from a web dashboard.

## High-level feature set (draft)
- Top-down 2D car driving game with configurable procedural track generation (size, complexity, seed)
- The game must look like a real game: sprite-based rendering (car, road, kerbs, scenery, follow-camera), user-supplied art dropping in later without simulation changes, and human play via keyboard/gamepad using the same action interface a model uses
- Car physics: kinematic bicycle model (no tire slip in v1). Action space: continuous steering + throttle, each in [−1, 1]. Observation: 10 floats — 7 speed-scaled proximity raycasts + normalized speed + last action
- Learning models decoupled from the game via a defined interface — a model doesn't know which game it's driving, a game doesn't know which model is driving it
- Two learning backends, each its own phase and more may be added: NEAT (neuroevolution, population-based, no manual reward shaping) and deep RL (PPO via stable-baselines3), the second proving the decoupling holds
- Cross-game model use: strict observation/action descriptor match — the platform blocks loading a model into a game whose descriptors differ; no adapter layer, no canonical shared space
- Configurable training/run behavior:
  - Population size
  - Generation spawn mode — **batch**: evaluate a full generation concurrently, advance to the next generation only once every individual has died or finished; vs **sequential**: no discrete generations, as soon as any individual dies immediately spawn a replacement bred from the current best genome, keeping the population continuously topped up
  - Inheritance between runs
  - Max deviation from track before death
  - Car-to-car collisions on/off — all individuals in a generation are simulated in the same shared space simultaneously; collision with track bounds is always on (that's what defines going off-track), this toggle only controls whether cars can collide with each other
  - Adjustable simulation speed
  - Headless vs rendered
  - Episode time limit and per-generation time/step ceiling — a bounded episode length so a single car can't run forever, plus a ceiling on a whole generation so batch mode always advances even if some individual never dies or finishes
- Pluggable per-run success criteria (finish line crossed / N checkpoints)
- Track lifecycle is a separate, user-driven decision, not a per-run success signal: training keeps running on the current track until the user manually decides they've trained all they usefully can on it and regenerates/switches to a new one
- Persistence: SQLite for run metadata/history, files for checkpoints. Checkpoints hold model state independent of run config, so a model trained under easy settings can be resumed under harder settings (valid while the observation/action descriptors are unchanged). Each model carries a settings history — the ordered log of every config change over its training lifetime; opening a model loads its most recent settings by default
- Control interface: local web dashboard that works with the simulation running headless. FastAPI + WebSocket backend, React frontend; trainer and dashboard talk only over that API. Local-first (localhost, single user, no auth in v1), remote-capable by construction. Built as three reorderable phases: control panel + live metrics, then in-browser live canvas, then advanced views (comparison charts, NEAT topology, track editor, model inspection)
- Future/stretch: more games on the same interfaces (candidates so far: Flappy Bird, a Mario-style platformer, Galaga, bullet hell), testing models across games, 3D. None implemented now; the Phase 0 interface must stay general enough to add them without redesign

## Key decisions (settled 2026-09-09 unless noted)
| Decision | Answer | Status |
|---|---|---|
| Engine / simulation | Custom simulation in plain Python/NumPy; runs headless, fast, parallelisable. The simulation module's only real dependencies are Python and NumPy — see Sim / render split | decided |
| Renderer | `arcade` library (OpenGL sprite/camera/tilemap), lives only in the render layer, swappable; Pygame is the documented fallback | decided |
| Sim / render split | The simulation module — the code that computes car physics, collisions, raycasts, and observations and steps the world forward — imports no graphics or windowing library (`arcade`, Pygame, or otherwise). The renderer is a separate module that imports the sim, reads its state, and draws it. The dependency arrow points one way, renderer → sim, and never back; delete the renderer package and the sim still runs identically. A renderer is never on the training path | decided |
| Physics model | Kinematic bicycle model `(x, y, heading, speed)`, no tire slip in v1 | decided |
| Action space | Continuous: steering & throttle, each `[−1, 1]` | decided |
| Observation space | 10 floats: 7 speed-scaled proximity raycasts (forward fan, walls only in v1) + normalized speed + last action | decided |
| Learning backends | Both committed, each its own phase, more may be added: NEAT (Phase 4) and deep RL / PPO via stable-baselines3 (Phase 10) | decided |
| Cross-game compatibility | Strict observation/action descriptor match; platform blocks mismatched loads; no adapters, no canonical space | decided |
| Control interface | Local web dashboard: FastAPI + WebSocket backend, React frontend; three reorderable phases (7–9) | decided |
| Local vs remote | Local-first (localhost, single user, no auth in v1); architecture stays remote-capable by construction | decided |
| Persistence | SQLite for run metadata/history, files for checkpoints; checkpoints are config-independent and carry a per-model settings-change history | decided |

These are settled at the platform level. Per-phase specifics (numeric sensor constants, NEAT library choice, checkpoint format, track-generation algorithm, etc.) remain open and belong to the individual phase docs; resolving them there does not require revising this document unless a phase decision contradicts a platform-level decision above.

## Open questions (non-exhaustive — expand freely during exploration)
The six original open questions (engine, observation space, action space, cross-game match, dashboard scope, local vs remote) were resolved in the 2026-09-09 decisions pass — see Key decisions above and the Revision history below for what changed and why. Remaining open items, all owned by phase docs:
- Concrete numeric parameters for the observation space — speed-scaled ray-range constants, `max_range` cap, normalization ranges (Phase 3)
- NEAT implementation choice: `neat-python` vs custom/other (Phase 4)
- Checkpoint file format and where the per-model settings history is stored (Phase 6)
- Procedural track-generation algorithm and how "complexity" maps to geometry (Phase 2)
- The partition of training knobs into set-at-run-creation vs live-changeable (Phase 5)

## Links
- Workflow & standards: `WORKFLOW.md`
- Phase list: `PHASES.md`
- Individual phase docs: `/phases/`

## Revision history
- 2026-09-09 — Initial draft, seeded from early conversation.
- 2026-09-09 — Made three previously vague training terms precise. "Batch vs sequential run spawning" now means: **batch** evaluates a whole generation concurrently and advances only once every individual has died or finished; **sequential** has no discrete generations and breeds an immediate replacement from the current best on each death. Car collisions: all individuals share one simulation space simultaneously, track-boundary collision is always on, and the toggle governs only car-to-car collisions. Manual track regeneration was removed from the "success criteria" list — it is now a separate user-driven track-lifecycle decision, not a per-run success signal. Reason: the original phrasings were ambiguous enough to build the wrong thing (supersedes 7a46fd6).
- 2026-09-09 — Resolved the six original open questions and promoted the Key decisions table from "tentative" to "decided". Engine: custom Python/NumPy simulation with an `arcade` renderer in a separate layer (previously the tentative answer was "Python + Pygame for the game core"). Physics: kinematic bicycle model. Action space: continuous steering + throttle. Observation: 10 floats (7 speed-scaled proximity raycasts + speed + last action). Cross-game: strict observation/action descriptor match, no adapters, no canonical space. Both learning backends (NEAT and deep RL) committed as peer phases — RL had only been "planned". Dashboard split into three reorderable phases; local-first, remote-capable by construction. Persistence: config-independent checkpoints plus a per-model settings history. Added new requirements: the game must look like a real game, be human-playable, and keep art out of the simulation core (supersedes 7a46fd6).
- 2026-09-10 — Promoted from DRAFT to FINALIZED at the platform level. No decision content changed; this is the deliberate promotion `WORKFLOW.md` requires before Phase 0 implementation can begin. Platform decisions had been stable since the 2026-09-09 pass, with only phase-level specifics still open (supersedes 60c7011).
- 2026-09-10 — Reworded the "zero rendering dependencies" decision, which read as broader than intended. It is specifically a constraint on the simulation module's import list: the physics / collision / raycast / observation / step code imports no graphics or windowing library (only Python and NumPy). The `arcade` renderer is a separate module that imports the sim one-way; deleting it leaves the sim running identically. No decision changed — only the wording.
- 2026-09-10 — Added timeout controls to the configurable training/run behaviour: a per-episode time limit and a per-generation time/step ceiling. Previously nothing guaranteed a generation would ever end — an individual that neither crashes nor finishes could stall batch mode forever. Interface details are Phase 0 (the Environment signals a hit episode limit via `truncated`); the generation ceiling is enforced by the NEAT trainer (Phase 4) and is a Phase 5 config knob.
