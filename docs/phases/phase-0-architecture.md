# Phase 0 — Architecture Scaffolding

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Define the `Environment` / `Model` / `SuccessCriterion` / `Trainer` interfaces and the shared config schema — general enough for many games, not car-specific.

## Requirements
Settled through the 2026-09-10 Phase 0 exploration. Items still genuinely undecided are in **Open questions**.

### Interfaces — all `typing.Protocol`, structural
A game or backend satisfies an interface by shape alone; it does not import or subclass anything from the platform. (`Protocol` over `ABC` so a game need not inherit a base class, matching Gym-style duck typing.)

**`Environment`** — a local Protocol that mirrors the Gymnasium `Env` contract's method names, arguments, and `reset` / `terminated` / `truncated` semantics, without importing or subclassing `gymnasium`.

```
observation_space: Box | Discrete
action_space:      Box | Discrete
reset(*, seed: int | None = None) -> observation
step(action) -> (observation, terminated, truncated, info)
```

- `terminated` = an in-world end (crash, off-track, finish line). `truncated` = an end forced from outside the world (a step or time budget hit). Phase 0 fixes only the *meaning* of the two flags. *Enforcing* a particular episode limit is the env implementation's job (Phase 3 for the car), the limit is a config parameter (Phase 5), and the standalone game (Phase 1) has no episode concept. The population-level per-generation ceiling is Phase 4 / Phase 5.
- **One intentional divergence from Gym:** `step` returns no `reward`. Scoring is external — see `SuccessCriterion`. The Phase 10 `gymnasium.Env` adapter synthesises the `reward` stable-baselines3 expects from the RL reward component.
- `info` is a `str -> Any` dict of game-specific raw facts (for the car, e.g. distance, checkpoints passed, `finished`, `crashed`). The base interface says nothing about its keys.
- Not car-specific: no raycasts, track, or steering in the base interface. It must fit Flappy Bird, a platformer, Galaga, bullet hell (Phase 11) with no redesign.

**`Model`** — a policy.

```
observation_space: Box | Discrete
action_space:      Box | Discrete
reset() -> None                 # per-episode hook: no-op for feedforward NEAT nets, clears hidden state for recurrent policies
act(observation) -> action
```

- No game knowledge. A checkpoint records the observation- and action-space descriptors the model was trained against.

**`SuccessCriterion`** — decides, for one episode, when the goal is reached and what the fitness score is. Selected in the run config; driven by the **trainer**, never imported by the `Environment`.

```
reset() -> None
update(observation, action, info) -> None   # once per step
should_stop() -> bool                        # goal reached (finish line / N checkpoints / ...)
fitness() -> float                           # score for this episode
```

- Concrete criteria (finish line crossed, N checkpoints) are Phase 5; Phase 0 fixes only this interface.
- An episode ends when the env reports `terminated` or `truncated`, **or** `should_stop()` returns true.

**`Trainer`** — a learning backend (NEAT is Phase 4, deep RL Phase 10; more may follow).

```
__init__(make_env: Callable[[], Environment], config: RunConfig)
run() -> Iterator[TrainingUpdate]     # yields after each unit of progress (NEAT: a generation; RL: a rollout batch)
save_checkpoint(path) -> None
load_checkpoint(path, make_env, config) -> Trainer     # classmethod
```

- `make_env` is a **factory**, not an instance — isolated per-genome evaluation (Phase 4) spins up many environments.
- `run()` is an iterator so the CLI or dashboard stays in control: it pulls a `TrainingUpdate` after each step and can stop or pause between yields.
- Checkpoint *contents* are per-backend (Phase 4 / Phase 10); checkpoint *file format* is Phase 6. Every checkpoint carries a `schema_version`.

**`TrainingUpdate`** — the value `run()` yields. Minimal Phase 0 shape: progress index (generation or step count), best and mean fitness so far, current population / batch size, elapsed sim-time and wall-time. Versioned; Phase 7 may extend it.

### Space descriptors
- Local classes `Box(low, high, shape, dtype)` and `Discrete(n)` — no `gymnasium` import.
- Equality compares **all fields** (`Box`: shape + bounds + dtype; `Discrete`: `n`). This equality *is* the compatibility check below.

### Model / config compatibility check
- At run creation, loading a checkpoint into an environment whose `observation_space` or `action_space` is not equal (per the descriptor equality above) to the checkpoint's recorded descriptors is refused.
- Resume-safety within one game — e.g. catching a sensor-layout change (Phase 3) that alters the observation shape. Not cross-game model transfer, which is not a goal. No adapter layer, no canonical space.

### Decoupling rules
- The `sim` package imports no `gymnasium`, no renderer, no trainer.
- Renderer → sim is one-way (see `../OVERVIEW.md`); a renderer is never on the training path.
- A `Trainer` consumes only the `Environment`, `Model`, `SuccessCriterion`, and `RunConfig` interfaces.
- `gymnasium` is imported in exactly one place: the `gymnasium.Env` adapter for the Phase 10 RL backend. **Revalidate the local-Protocol choice at Phase 10** — confirm the adapter stays thin and we are not re-implementing `gymnasium.wrappers` / `gymnasium.vector`; if we are, reconsider adopting `gymnasium` in the core then.

### Config & serialization
- `RunConfig` is a serializable, versioned object. Format: **JSON with a top-level integer `schema_version`** — human-readable and diffable; msgpack / protobuf would be premature.
- Phase 0 fixes that `RunConfig` exists, is serializable, and is versioned, plus the `schema_version` field and a forward-migration hook. Its concrete fields (population size, deviation limit, sim speed, timeouts, success-criterion selection, …) are owned by Phase 5.

### Project scaffolding
- Python 3.12+ (final floor is whatever `arcade` and `stable-baselines3` both support).
- **`uv`** for dependency management + locking; **`pytest`** for tests; **`ruff`** for lint + format.
- `src/` layout, one package `neuroarena/`. Phase 0 delivers `neuroarena/interfaces/` (Protocols + descriptors), `neuroarena/config/` (`RunConfig` + JSON serialization + version hook), and their tests. `neuroarena/sim/`, `neuroarena/backends/`, `neuroarena/render/` arrive in later phases.

## Open questions
- Exact fields of `TrainingUpdate` — settled once Phase 7 knows what the dashboard displays.
- Whether `SuccessCriterion.update` needs `terminated` / `truncated` passed explicitly or can read all it needs from `info`.
- The schema-migration mechanism (a registry of `vN -> vN+1` functions vs something lighter) — deferred to the first real bump.
- Whether `make_env` carries the seed, or the trainer seeds each env via `reset(seed=...)`.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added a bounded-episode requirement (Environment enforces a max episode length, reports it via `truncated`, limit lives in config) and a per-generation ceiling field in the shared config schema. Before this nothing in the interface guaranteed an episode or a batch generation would ever end.
- 2026-09-10 — Resolved the Gymnasium open question: the Environment interface is a local `Protocol` mirroring Gym's contract, with local `Box` / `Discrete` descriptors, and a thin `gymnasium.Env` adapter to be written for the Phase 10 RL backend (the only place `gymnasium` is imported). Alternatives considered: subclass `gymnasium.Env`, or reuse `gymnasium.spaces` as descriptors — both rejected to keep the core dependency-free and to own the match semantics; flagged for revalidation at Phase 10. Same pass trimmed the bounded-episode requirement added earlier today: Phase 0 now fixes only the `terminated` vs `truncated` contract; episode-limit *enforcement* moved to Phase 3 (car env) and the config fields to Phase 5, since the standalone game (Phase 1) has no episode or timeout concept and the base interface should not carry one.
- 2026-09-10 — Replaced the "strict cross-game descriptor match / no canonical space" requirement with a plain **model/config compatibility check**: a checkpoint records its obs/action descriptors, and loading it into a shape-mismatched environment is refused. Reason: cross-game model transfer was cut from platform scope (see `../OVERVIEW.md`) — a trained model is meaningless in a structurally different game — so the check is now only resume-safety within one game. The "must not assume car-ness" requirement stays, its rationale changed from "so models transfer" to "so a second game can reuse the interfaces and infrastructure".
- 2026-09-10 — Consolidated the settled requirements from the Phase 0 exploration pass. Added full Protocol signatures: `Environment` mirrors the Gym contract but `step` returns a 4-tuple with no `reward` (scoring is externalised to `SuccessCriterion`; the Phase 10 adapter synthesises `reward`); `Model` is `act` plus a per-episode `reset` hook so recurrent policies are possible later; `SuccessCriterion` is a new trainer-driven interface that keeps the env dumb about scoring; `Trainer` takes a `make_env` factory and exposes `run()` as an iterator plus checkpoint save/load. Decided: local `Box`/`Discrete` with all-field equality as the compatibility check; `RunConfig` serialised as JSON with an integer `schema_version`; `typing.Protocol` over `ABC`; scaffolding is Python 3.12+, `uv`, `pytest`, `ruff`, `src/` layout with a `neuroarena/` package. Remaining open questions narrowed to `TrainingUpdate` fields, the `SuccessCriterion.update` signature, the schema-migration mechanism, and where env seeding lives.
