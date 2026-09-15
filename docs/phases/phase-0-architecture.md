# Phase 0 — Architecture Scaffolding

Status: **FINALIZED** — requirements settled as of 2026-09-10, revised 2026-09-15 to add `Trainer.update_config` (Phase 7 needs it), implemented by `NeatTrainer` the same day (for the fields it can safely support — see the `Trainer` interface section below). Changing the requirements further needs its own deliberate revision (see `../WORKFLOW.md`). The two items under Open questions are deferred by design and do not block.

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
- **One intentional divergence from Gym:** `step` returns no `reward`. Reward is not a property of the world — it is defined by the `Objective` (below), which the trainer runs. The Phase 10 `gymnasium.Env` adapter reads `Objective.step_reward()` into the `reward` slot stable-baselines3 expects.
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

**`Objective`** — the single definition of what the agent is trying to do and how well it did. This is the interface behind OVERVIEW's user-facing "success criteria"; it is named `Objective` because it also defines the score. Selected in the run config; driven by the **trainer**, never imported by the `Environment`.

```
reset() -> None
update(observation, action, terminated, truncated, info) -> None   # once per step, with everything step() returned
should_stop() -> bool                        # goal reached early (finish line / N checkpoints / ...)
step_reward() -> float                        # dense per-step signal — consumed by RL backends (Phase 10)
fitness() -> float                           # episode score — consumed by evolutionary backends (Phase 4)
```

- One scoring source feeds both backend families: `fitness()` is the terminal/aggregate score NEAT selects on; `step_reward()` is the per-step signal the Phase 10 adapter wires into `reward`. For most objectives `fitness()` is the running sum of `step_reward()`, but that is the objective's choice.
- Concrete objectives (finish line crossed, N checkpoints, distance + speed shaping) are Phase 5; Phase 0 fixes only this interface.
- An episode ends when the env reports `terminated` or `truncated`, **or** `should_stop()` returns true.

**`Trainer`** — a learning backend (NEAT is Phase 4, deep RL Phase 10; more may follow).

```
__init__(make_env: Callable[[], Environment], objective: Objective, config: RunConfig)
run() -> Iterator[TrainingUpdate]     # yields after each unit of progress (NEAT: a generation; RL: a rollout batch)
update_config(partial: Mapping[str, Any]) -> None     # thread-safe; stages a change, applied atomically at the next progress-unit boundary
save_checkpoint(path) -> None
load_checkpoint(path, make_env, objective, config) -> Trainer     # classmethod
```

- **`update_config`** (added 2026-09-15, see Revision history) lets a live-changeable config field be updated while `run()` is already executing, without tearing the trainer down — Phase 7's dashboard needs this for live config edits. It is safe to call from another thread than the one driving `run()`: the change is staged, not applied in place, and every backend applies it atomically at the top of its own next unit of progress (NEAT: next generation boundary) so no genome/rollout ever sees a config mutate mid-evaluation. Implemented by `NeatTrainer` for `max_generation_steps`/`max_generations`/`target_fitness` — see [`../superpowers/plans/2026-09-15-neat-trainer-update-config.md`](../superpowers/plans/2026-09-15-neat-trainer-update-config.md). The rest of Phase 5's live-changeable knob list (`neat_hyperparameters`, `physics_constants`, `max_episode_steps`, `track_id`, `sim_speed`) is explicitly rejected by `NeatTrainer.update_config` today, not silently ignored — see that method's own docstring for why each is excluded.
- `make_env` is a **factory**, not an instance — isolated per-genome evaluation (Phase 4) spins up many environments.
- `run()` is an iterator so the CLI or dashboard stays in control: it pulls a `TrainingUpdate` after each step and can stop or pause between yields.
- The trainer owns seeding: it derives per-episode seeds from the run's master seed (a `RunConfig` field) and passes them to `env.reset(seed=...)`. `make_env` only constructs.
- Checkpoint *contents* are per-backend (Phase 4 / Phase 10); checkpoint *file format* is Phase 6. Every checkpoint carries a `schema_version`.

**`TrainingUpdate`** — the value `run()` yields. Minimal Phase 0 shape: progress index (generation or step count), best / mean / worst fitness so far, current population / batch size, the champion's key `info` metrics, elapsed sim-time and wall-time. Versioned; Phase 7 / Phase 9 may extend it.

### Run history & per-generation records
- The platform records the stream of `TrainingUpdate`s a run emits. That stream *is* the run history — after a headless run it drives the fitness curve and lets any generation be inspected by its stats.
- Replaying a chosen generation's champion later is a **reserved** capability, not a Phase 0 deliverable. Phase 0's only obligation is that `TrainingUpdate` and the checkpoint interface do not preclude per-progress-unit champion checkpoints. The checkpoints themselves, deterministic `(seed, genome)` replay, and the inspection UI are designed in Phases 4, 6, and 9.

### Space descriptors
- Local classes `Box(low, high, shape, dtype)` and `Discrete(n)` — no `gymnasium` import.
- Equality compares **all fields** (`Box`: shape + bounds + dtype; `Discrete`: `n`). This equality *is* the compatibility check below.

### Model / config compatibility check
- At run creation, loading a checkpoint into an environment whose `observation_space` or `action_space` is not equal (per the descriptor equality above) to the checkpoint's recorded descriptors is refused.
- Resume-safety within one game — e.g. catching a sensor-layout change (Phase 3) that alters the observation shape. Not cross-game model transfer, which is not a goal. No adapter layer, no canonical space.

### Decoupling rules
- The `sim` package imports no `gymnasium`, no renderer, no trainer.
- Renderer → sim is one-way (see `../OVERVIEW.md`); a renderer is never on the training path.
- A `Trainer` consumes only the `Environment`, `Model`, `Objective`, and `RunConfig` interfaces.
- `gymnasium` is imported in exactly one place: the `gymnasium.Env` adapter for the Phase 10 RL backend. **Revalidate the local-Protocol choice at Phase 10** — confirm the adapter stays thin and we are not re-implementing `gymnasium.wrappers` / `gymnasium.vector`; if we are, reconsider adopting `gymnasium` in the core then.

### Config & serialization
- `RunConfig` is a serializable, versioned object. Format: **JSON with a top-level integer `schema_version`** — human-readable and diffable; msgpack / protobuf would be premature.
- Phase 0 fixes that `RunConfig` exists, is serializable, and is versioned, plus the `schema_version` field and a forward-migration hook. Its concrete fields (population size, deviation limit, sim speed, timeouts, `Objective` selection, …) are owned by Phase 5.

### Project scaffolding
- Python 3.12+ (final floor is whatever `arcade` and `stable-baselines3` both support).
- **`uv`** for dependency management + locking; **`pytest`** for tests; **`ruff`** for lint + format.
- `src/` layout, one package `neuroarena/`. Phase 0 delivers `neuroarena/interfaces/` (Protocols + descriptors), `neuroarena/config/` (`RunConfig` + JSON serialization + version hook), and their tests. `neuroarena/sim/`, `neuroarena/backends/`, `neuroarena/render/` arrive in later phases.

## Open questions
Deferred by design, not blockers for FINALIZED. Resolving them happens in the phase that owns them and does not count as reopening this doc.
- Exact fields of `TrainingUpdate` beyond the minimum set above — Phase 7 / Phase 9 add what the dashboard needs. The struct is versioned, so additions are not a Phase 0 revision.
- The schema-migration mechanism (a registry of `vN -> vN+1` functions vs something lighter) — built at the first real `schema_version` bump; Phase 0 only reserves the field and the hook.

## Implementation plan
Detailed, step-by-step plan: [`../superpowers/plans/2026-09-10-phase-0-architecture.md`](../superpowers/plans/2026-09-10-phase-0-architecture.md).

Six tasks, each ending in an independently testable, committed deliverable:
1. **Project skeleton + tooling** — `pyproject.toml`, `uv` / `pytest` / `ruff`, `src/` layout, smoke test.
2. **Space descriptors** — `Box` / `Discrete` frozen dataclasses with field-wise equality.
3. **Interface Protocols** — `Environment` / `Model` / `Objective` / `Trainer` + `TrainingUpdate`, plus reusable test doubles.
4. **Compatibility check** — `check_compatibility(model, environment)` descriptor gate.
5. **`RunConfig` + serialization** — envelope dataclass, versioned JSON codec, `migrate` identity hook.
6. **Import-hygiene guard** — test that the core packages pull in no `gymnasium` / renderer / trainer / heavy deps.

## Revision history
- 2026-09-15 — Implemented `Trainer.update_config` on `NeatTrainer`, for the same day's revision above — scoped to `max_generation_steps`/`max_generations`/`target_fitness` (the fields `NeatTrainer` already re-reads fresh every generation); every other Phase-5-classified live-changeable field is explicitly rejected with an error naming why, not silently accepted. Status REVISED -> FINALIZED. See [`../superpowers/plans/2026-09-15-neat-trainer-update-config.md`](../superpowers/plans/2026-09-15-neat-trainer-update-config.md).
- 2026-09-15 — **Deliberate revision, status → REVISED** (per `../WORKFLOW.md`): added `Trainer.update_config(partial)` to the `Trainer` Protocol — a thread-safe, staged, atomically-applied-at-the-next-progress-boundary config update while `run()` is executing. Needed by Phase 7 (dashboard live config edits go through this instead of NEAT-specific glue, keeping Phase 7 backend-agnostic). Decided during Phase 7's 2026-09-14 exploration pass (recovered 2026-09-15 after that session was interrupted — see Phase 7's own Revision history). `NeatTrainer` does not implement this yet; status stays REVISED until it does.
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added a bounded-episode requirement (Environment enforces a max episode length, reports it via `truncated`, limit lives in config) and a per-generation ceiling field in the shared config schema. Before this nothing in the interface guaranteed an episode or a batch generation would ever end.
- 2026-09-10 — Resolved the Gymnasium open question: the Environment interface is a local `Protocol` mirroring Gym's contract, with local `Box` / `Discrete` descriptors, and a thin `gymnasium.Env` adapter to be written for the Phase 10 RL backend (the only place `gymnasium` is imported). Alternatives considered: subclass `gymnasium.Env`, or reuse `gymnasium.spaces` as descriptors — both rejected to keep the core dependency-free and to own the match semantics; flagged for revalidation at Phase 10. Same pass trimmed the bounded-episode requirement added earlier today: Phase 0 now fixes only the `terminated` vs `truncated` contract; episode-limit *enforcement* moved to Phase 3 (car env) and the config fields to Phase 5, since the standalone game (Phase 1) has no episode or timeout concept and the base interface should not carry one.
- 2026-09-10 — Replaced the "strict cross-game descriptor match / no canonical space" requirement with a plain **model/config compatibility check**: a checkpoint records its obs/action descriptors, and loading it into a shape-mismatched environment is refused. Reason: cross-game model transfer was cut from platform scope (see `../OVERVIEW.md`) — a trained model is meaningless in a structurally different game — so the check is now only resume-safety within one game. The "must not assume car-ness" requirement stays, its rationale changed from "so models transfer" to "so a second game can reuse the interfaces and infrastructure".
- 2026-09-10 — Consolidated the settled requirements from the Phase 0 exploration pass. Added full Protocol signatures: `Environment` mirrors the Gym contract but `step` returns a 4-tuple with no `reward` (scoring is externalised; the Phase 10 adapter synthesises `reward`); `Model` is `act` plus a per-episode `reset` hook so recurrent policies are possible later; a new trainer-driven scoring interface keeps the env dumb about scoring; `Trainer` takes a `make_env` factory and exposes `run()` as an iterator plus checkpoint save/load. Decided: local `Box`/`Discrete` with all-field equality as the compatibility check; `RunConfig` serialised as JSON with an integer `schema_version`; `typing.Protocol` over `ABC`; scaffolding is Python 3.12+, `uv`, `pytest`, `ruff`, `src/` layout with a `neuroarena/` package.
- 2026-09-10 — Unified scoring. Renamed the `SuccessCriterion` interface to `Objective` and added `step_reward()` (dense, for RL) alongside `fitness()` (episode score, for evolution), so one config-selected component is the single source of both signals — `step` still carries no `reward`, and the Phase 10 adapter reads `Objective.step_reward()`. Reason: the previous split implied the env computes reward and something else computes fitness; a single objective feeding both is a cleaner decoupling. Also added a "Run history & per-generation records" requirement — the recorded `TrainingUpdate` stream is the run history — and reserved (guaranteed not precluded) per-generation champion checkpoints + replay for design in Phases 4 / 6 / 9. `Trainer.__init__` and `load_checkpoint` now also take the `objective`.
- 2026-09-10 — Settled two of the four open questions before finalizing: `Objective.update` takes the full step tuple `(observation, action, terminated, truncated, info)`; the trainer owns seeding and derives per-episode seeds from a `RunConfig` master seed. The remaining two (`TrainingUpdate` field additions, schema-migration mechanism) are explicitly deferred to the phases that own them and do not count as reopening this doc.
- 2026-09-10 — Promoted to FINALIZED. The requirements set — interface Protocols, descriptors, compatibility check, decoupling rules, JSON config envelope, project scaffolding — is settled. Considered and rejected splitting Phase 0 into smaller phases: it is already the smallest, most cohesive phase, and volume is a plan-level concern handled by the implementation plan's task breakdown (see `../WORKFLOW.md`, "How phases are divided").
- 2026-09-10 — Wrote the implementation plan (`../superpowers/plans/2026-09-10-phase-0-architecture.md`): six TDD tasks — skeleton/tooling, `Box`/`Discrete`, the Protocols + test doubles, `check_compatibility`, `RunConfig` + JSON codec, import-hygiene guard.
