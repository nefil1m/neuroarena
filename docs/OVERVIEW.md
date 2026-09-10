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
- Model / config compatibility check: a checkpoint records the observation/action shape it was trained against; the platform refuses to load it into an environment whose shape differs (e.g. after a sensor-layout change). This is resume-safety within a game, not a mechanism for moving trained models between different games — that is not a goal
- Configurable training/run behavior:
  - Population size
  - Inheritance between runs
  - Max deviation from track before death
  - Adjustable simulation speed
  - Headless vs rendered
  - Episode time limit and per-generation time/step ceiling — a bounded episode length so a single car can't run forever, plus a ceiling on a whole generation so it always advances even if some individual never dies or finishes
- Generational (batch) evaluation: evaluate a full generation, then select and breed the next. Each individual is evaluated in isolation on its own copy of the track — no car-to-car interaction, which keeps per-genome fitness clean and the evaluations trivially parallel. Track-boundary collision (which defines going off-track) is always on
- Pluggable per-run success criteria (finish line crossed / N checkpoints)
- Track lifecycle is a separate, user-driven decision, not a per-run success signal: training keeps running on the current track until the user manually decides they've trained all they usefully can on it and regenerates/switches to a new one
- Persistence: SQLite for run metadata/history, files for checkpoints. Checkpoints hold model state independent of run config, so a model trained under easy settings can be resumed under harder settings (valid while the observation/action descriptors are unchanged). Each model carries a settings history — the ordered log of every config change over its training lifetime; opening a model loads its most recent settings by default
- Run history: every run's per-generation metrics stream is recorded, so after a headless run the fitness curve can be redrawn and any generation compared by its stats. Optionally each generation's champion is checkpointed so it can be replayed later (the sim is deterministic given seed + genome)
- Control interface: local web dashboard that works with the simulation running headless. FastAPI + WebSocket backend, React frontend; trainer and dashboard talk only over that API. Local-first (localhost, single user, no auth in v1), remote-capable by construction. Built as three reorderable phases: control panel + live metrics, then in-browser live canvas, then advanced views (comparison charts, NEAT topology, track editor, model inspection)
- Future/stretch: more games on the same interfaces and infrastructure — renderer, trainer, dashboard, persistence — (candidates so far: Flappy Bird, a Mario-style platformer, Galaga, bullet hell), and 3D. A second game's purpose is to confirm the Phase 0 interfaces are genuinely game-agnostic and to add variety, not to run trained models across games. None implemented now; the Phase 0 interface must stay general enough to add them without redesign

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
| Population evaluation | Generational (batch) loop; each individual evaluated in isolation on its own copy of the track; no car-to-car collisions | decided |
| Model / config compatibility | A checkpoint records the observation/action shape it trained against; loading it into a shape-mismatched environment is refused. Resume-safety within one game — not cross-game model transfer, which is not a goal | decided |
| Control interface | Local web dashboard: FastAPI + WebSocket backend, React frontend; three reorderable phases (7–9) | decided |
| Local vs remote | Local-first (localhost, single user, no auth in v1); architecture stays remote-capable by construction | decided |
| Persistence | SQLite for run metadata/history, files for checkpoints; checkpoints are config-independent and carry a per-model settings-change history | decided |

These are settled at the platform level. Per-phase specifics (numeric sensor constants, NEAT library choice, checkpoint format, track-generation algorithm, etc.) remain open and belong to the individual phase docs; resolving them there does not require revising this document unless a phase decision contradicts a platform-level decision above.

## Open questions (non-exhaustive — expand freely during exploration)
The six original open questions (engine, observation space, action space, cross-game match, dashboard scope, local vs remote) were resolved in the 2026-09-09 decisions pass — see Key decisions above and the Revision history below for what changed and why. The cross-game-match answer was later narrowed: cross-game model transfer is no longer a goal, and what remains is a shape-based resume-compatibility check (Revision history, 2026-09-10). Remaining open items, all owned by phase docs:
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
- 2026-09-10 — Added timeout controls to the configurable training/run behaviour: a per-episode time limit and a per-generation time/step ceiling. Previously nothing guaranteed a generation would ever end — an individual that neither crashes nor finishes could stall batch mode forever. Placement across phases: Phase 0 fixes only the `terminated` vs `truncated` flag contract; the car Environment enforces the episode limit (Phase 3); the NEAT trainer enforces the per-generation ceiling (Phase 4); both limits are Phase 5 config knobs.
- 2026-09-10 — Cut three features from platform scope after review found each added implementation and/or training friction for speculative payoff. (1) **Cross-game model transfer** dropped as a goal: the previous "strict observation/action descriptor match / no canonical space" system is replaced by a plain shape-based resume-safety check (a checkpoint records the obs/action shape it trained on; loading it into a shape-mismatched env is refused). Reason: a model trained on one game's observations is meaningless in a structurally different game, and any two games similar enough to share descriptors are effectively the same game. The decoupled Game/Model/Trainer interface stays — that is worthwhile on its own. (2) **Sequential spawn mode** dropped: the generational loop is batch only. Reason: sequential (steady-state replacement on each death) is the unusual variant, adds edge cases, and NEAT's standard loop is batch; PPO has no generations at all. (3) **Shared-space simulation with car-to-car collisions** dropped: each genome is now evaluated in isolation on its own track copy. Reason: collisions inject noise into fitness, which slows evolutionary convergence, and add physics complexity, for a mostly-spectacle benefit; isolation gives clean fitness and trivial parallelism. Isolation is a subset of shared-space, so a swarm mode remains possible later without a Phase 0 redesign. Propagated to Phases 0, 3, 4, 5, 8, 10, 11 and PHASES.md.
- 2026-09-10 — Added run history to platform scope: every run's per-generation metrics stream is recorded so the fitness curve can be redrawn and generations compared after a headless run, and optionally each generation's champion is checkpointed for later deterministic replay. Prompted by wanting to inspect and compare specific generations post-hoc. The interface reservation is Phase 0; storage is Phase 6; NEAT champion capture is Phase 4; the drill-down UI is Phase 9.
