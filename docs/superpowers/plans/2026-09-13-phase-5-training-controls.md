# Phase 5 — Training Controls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire every knob Phase 5 decided training needs — population size, track selection, headless/sim-speed stubs, sensor layout, physics constants, episode/generation step limits, a run-level automatic stop (max-generations cap, target-fitness threshold), and the full set of NEAT hyperparameters — into `RunConfig` and the concrete pieces that already consume config (`CarEnvironmentConfig`, `NeatTrainer`, `build_neat_config`), so a `RunConfig` instance is the single source of truth a real training run is built from.

**Architecture:** No new packages. `RunConfig` (`neuroarena.config.run_config`) grows from its Phase 0/4 envelope (`master_seed`, `max_generation_steps`, `schema_version`) into the full Phase 5 knob set — plain scalars, two nested dataclasses (`SensorConfig`, `PhysicsConstants`, both already car-specific per Phase 3/1) carried directly on the envelope, and a generic `neat_hyperparameters: dict` for the NEAT-specific knobs, mirroring the precedent Phase 4 already set by putting its own backend-specific `max_generation_steps` stub directly on the shared envelope rather than inventing a generic sub-config layer. Three existing pieces gain a `RunConfig`-consuming wiring point: `neuroarena.sim.car_env.CarEnvironmentConfig` gains a `from_run_config` classmethod (Task 5); `neuroarena.backends.neat.config.build_neat_config` gains a hyperparameter-override mechanism (Task 3); `neuroarena.backends.neat.trainer.NeatTrainer` reads `config.population_size`/`config.neat_hyperparameters` when building its `neat.Config` (Task 4) and checks `config.max_generations`/`config.target_fitness` at each generation boundary to decide whether to keep running (Task 6). `config/serialization.py`'s `loads()` needs a real fix, not just new fields: `dataclasses.asdict` already serializes nested dataclasses correctly, but `loads()` was never taught to reconstruct them, so a naive round trip would silently leave `sensor_config`/`physics_constants` as plain dicts (Task 2 proves this with a failing test first).

**Not built in this plan (explicitly out of scope):** resolving `RunConfig.track_id` into a live `Track` via `tracks.store.load()` and assembling a full `make_env` closure. No phase doc has yet named the module a real "launch a training run" entrypoint belongs in (a CLI script, a dashboard-triggered launcher, or something else) — building that now would invent a package boundary no phase doc has decided, not just wire an existing one. `RunConfig.track_id` becomes a real field in this plan (Task 1) since the field's existence is what Phase 5's doc requires; resolving it into a `Track` is left for whichever later phase first needs to actually launch a run. `sim_speed` and `headless` are the same kind of stub: real `RunConfig` fields with no consumer yet, following the exact precedent Phase 4 set for `max_generation_steps` before Phase 5 existed to consume it — see Task 1.

Also not built: any mechanism for changing a knob on a `NeatTrainer` that is already mid-`run()`. Phase 5's doc classifies several knobs "live-changeable" (e.g. physics constants, the step/generation ceilings, `neat_hyperparameters`) — but that classification says what a *future* control surface (Phase 7's dashboard, or whatever calls it) is allowed to request; it doesn't yet exist. Today, `NeatTrainer._config` is set once in `__init__` and never reassigned, exactly like `max_generation_steps` already works post-Phase-4. Nothing in this plan adds an API for mutating a running trainer's config, appending a settings-history entry (that's Phase 6's storage to build on), or applying a change "at the next generation boundary" versus immediately — there is no live caller yet to drive such a change. This plan only makes every knob reachable through the config a *new* `NeatTrainer`/`CarEnvironmentConfig` is constructed from.

**Tech Stack:** Python 3.12+, `neat-python`, `numpy`, `pytest`, `mypy --strict`, `ruff`. No new dependencies.

**Spec:** [`../../phases/phase-5-training-controls.md`](../../phases/phase-5-training-controls.md) (FINALIZED 2026-09-13). Builds on the FINALIZED [`../../phases/phase-0-architecture.md`](../../phases/phase-0-architecture.md) (`RunConfig` envelope), [`../../phases/phase-3-sensors-env-interface.md`](../../phases/phase-3-sensors-env-interface.md) (`CarEnvironmentConfig`, `SensorConfig`), and [`../../phases/phase-4-learning-backend-neat.md`](../../phases/phase-4-learning-backend-neat.md) (`NeatTrainer`, `build_neat_config`, `max_generation_steps`).

## Global Constraints

- Python 3.12+; `uv` for deps; `pytest` for tests; `ruff` for lint+format; `mypy --strict` (see `pyproject.toml`'s `[tool.mypy]`).
- `src/` layout, package `neuroarena/`. This phase touches `src/neuroarena/config/`, `src/neuroarena/sim/car_env.py`, and `src/neuroarena/backends/neat/{config,trainer}.py` only — no new top-level packages.
- `neuroarena.backends.neat` still imports no `gymnasium`, `arcade`, `pygame`, `stable_baselines3`, or `torch`, and still imports no `neuroarena.sim.*`/`neuroarena.render.*` directly (Phase 4's import-hygiene guard, `tests/backends/neat/test_import_hygiene.py`, must keep passing unmodified).
- `neuroarena.config` gains a real (non-`TYPE_CHECKING`) import of `neuroarena.sim.observation.SensorConfig` and `neuroarena.sim.physics.PhysicsConstants` in this plan. This is a deliberate, narrow exception to keeping `config` game-agnostic — both types are pure dataclasses with zero external dependencies, and Phase 4 already established that a backend/game-specific field can live on the shared `RunConfig` envelope as a pragmatic stub (its own `max_generation_steps`). Genuinely generalizing `RunConfig` for a hypothetical second game is explicitly not required until Phase 11 needs it.
- Every new `RunConfig` field gets a default so all existing `RunConfig()` call sites (Phase 4's tests, this plan's own tests) keep working unmodified.
- `RunConfig` equality is field-wise (`@dataclass`, not `frozen` but no custom `__eq__`) — nested `SensorConfig`/`PhysicsConstants` dataclasses compare structurally too, so `RunConfig(...) == RunConfig(...)` stays meaningful after this plan.

---

## Task 1: Plain scalar `RunConfig` knobs

**Files:**
- Modify: `src/neuroarena/config/run_config.py`
- Modify: `tests/config/test_run_config.py`

**Interfaces:**
- Produces: `RunConfig.population_size: int` (default `150`, matches `NeatTrainer`'s current default), `RunConfig.track_id: str | None` (default `None`), `RunConfig.headless: bool` (default `True`), `RunConfig.sim_speed: float` (default `1.0`), `RunConfig.max_episode_steps: int` (default `3000`, matches `CarEnvironmentConfig`'s current default), `RunConfig.max_generations: int | None` (default `None`), `RunConfig.target_fitness: float | None` (default `None`). Consumed by: `population_size` and `max_generations`/`target_fitness` — Tasks 4 and 6; `max_episode_steps` — Task 5; `track_id`, `headless`, `sim_speed` — no consumer in this plan (see the plan header's "Not built in this plan").

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_run_config.py`:

```python
def test_population_size_default() -> None:
    assert RunConfig().population_size == 150


def test_track_id_defaults_to_none() -> None:
    assert RunConfig().track_id is None


def test_track_id_is_settable() -> None:
    assert RunConfig(track_id="abc123").track_id == "abc123"


def test_headless_defaults_to_true() -> None:
    assert RunConfig().headless is True


def test_sim_speed_default() -> None:
    assert RunConfig().sim_speed == 1.0


def test_max_episode_steps_default() -> None:
    assert RunConfig().max_episode_steps == 3000


def test_max_generations_and_target_fitness_default_to_none() -> None:
    config = RunConfig()
    assert config.max_generations is None
    assert config.target_fitness is None


def test_max_generations_and_target_fitness_are_settable() -> None:
    config = RunConfig(max_generations=50, target_fitness=1000.0)
    assert config.max_generations == 50
    assert config.target_fitness == 1000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_run_config.py -v`
Expected: FAIL — `TypeError: RunConfig.__init__() got an unexpected keyword argument 'track_id'` (or an `AttributeError` on whichever default-value test pytest collects first).

- [ ] **Step 3: Add the fields**

In `src/neuroarena/config/run_config.py`, update the dataclass:

```python
@dataclass
class RunConfig:
    """Serializable run configuration — the single source of truth a training run is built
    from (Phase 5). Phase 0 fixed the envelope (`master_seed`, `schema_version`); Phase 4
    added `max_generation_steps` as a stub ahead of this phase. `track_id`/`headless`/
    `sim_speed` are stub fields with no consumer yet in this codebase — see the Phase 5
    implementation plan's "Not built in this plan" note for why (no phase has named a
    training-launch entrypoint to resolve `track_id` into a `Track` or read `sim_speed`).
    """

    master_seed: int = 0
    max_generation_steps: int = 300_000
    population_size: int = 150
    track_id: str | None = None
    headless: bool = True
    sim_speed: float = 1.0
    max_episode_steps: int = 3000
    max_generations: int | None = None
    target_fitness: float | None = None
    schema_version: int = SCHEMA_VERSION
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_run_config.py tests/config/test_serialization.py -v`
Expected: PASS — `test_serialization.py` must still pass unchanged since `dumps`/`loads` operate generically over dataclass fields and every new field here is a plain scalar (`int`, `str | None`, `bool`, `float`) that JSON already round-trips without help.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/config/run_config.py tests/config/test_run_config.py
git commit -m "feat(phase-5): add plain scalar RunConfig knobs (population_size, track_id, headless, sim_speed, max_episode_steps, max_generations, target_fitness)"
```

---

## Task 2: Nested-dataclass `RunConfig` knobs (`sensor_config`, `physics_constants`) + serialization fix

**Files:**
- Modify: `src/neuroarena/config/run_config.py`
- Modify: `src/neuroarena/config/serialization.py`
- Modify: `tests/config/test_run_config.py`
- Modify: `tests/config/test_serialization.py`

**Interfaces:**
- Consumes: `neuroarena.sim.observation.SensorConfig`, `neuroarena.sim.physics.PhysicsConstants` (both existing, unmodified).
- Produces: `RunConfig.sensor_config: SensorConfig` (default `SensorConfig()`), `RunConfig.physics_constants: PhysicsConstants` (default `PhysicsConstants()`). Consumed by Task 5's `CarEnvironmentConfig.from_run_config`.

- [ ] **Step 1: Write the failing test proving the serialization bug**

Append to `tests/config/test_serialization.py`:

```python
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


def test_roundtrip_reconstructs_sensor_config_as_a_real_dataclass() -> None:
    config = RunConfig(sensor_config=SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0)))
    restored = loads(dumps(config))
    assert isinstance(restored.sensor_config, SensorConfig)
    assert restored.sensor_config.ray_angles_deg == (-40.0, 0.0, 40.0)
    assert isinstance(restored.sensor_config.ray_angles_deg, tuple)


def test_roundtrip_reconstructs_physics_constants_as_a_real_dataclass() -> None:
    config = RunConfig(physics_constants=PhysicsConstants(max_speed=900.0))
    restored = loads(dumps(config))
    assert isinstance(restored.physics_constants, PhysicsConstants)
    assert restored.physics_constants.max_speed == 900.0


def test_roundtrip_preserves_value_with_nested_defaults() -> None:
    assert loads(dumps(RunConfig(master_seed=7))) == RunConfig(master_seed=7)
```

Also add to `tests/config/test_run_config.py`:

```python
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


def test_sensor_config_defaults_to_the_platform_standard() -> None:
    assert RunConfig().sensor_config == SensorConfig()


def test_physics_constants_defaults_to_the_platform_standard() -> None:
    assert RunConfig().physics_constants == PhysicsConstants()


def test_sensor_config_and_physics_constants_are_settable() -> None:
    config = RunConfig(
        sensor_config=SensorConfig(ray_angles_deg=(-30.0, 30.0)),
        physics_constants=PhysicsConstants(max_speed=500.0),
    )
    assert config.sensor_config.ray_angles_deg == (-30.0, 30.0)
    assert config.physics_constants.max_speed == 500.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_run_config.py tests/config/test_serialization.py -v`
Expected: FAIL — `test_run_config.py`'s new tests fail with `TypeError: RunConfig.__init__() got an unexpected keyword argument 'sensor_config'` (the field doesn't exist yet).

- [ ] **Step 3: Add the fields**

In `src/neuroarena/config/run_config.py`, add the imports and two fields:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants

SCHEMA_VERSION = 1


@dataclass
class RunConfig:
    """Serializable run configuration — the single source of truth a training run is built
    from (Phase 5). Phase 0 fixed the envelope (`master_seed`, `schema_version`); Phase 4
    added `max_generation_steps` as a stub ahead of this phase. `track_id`/`headless`/
    `sim_speed` are stub fields with no consumer yet in this codebase — see the Phase 5
    implementation plan's "Not built in this plan" note for why (no phase has named a
    training-launch entrypoint to resolve `track_id` into a `Track` or read `sim_speed`).
    """

    master_seed: int = 0
    max_generation_steps: int = 300_000
    population_size: int = 150
    track_id: str | None = None
    headless: bool = True
    sim_speed: float = 1.0
    max_episode_steps: int = 3000
    max_generations: int | None = None
    target_fitness: float | None = None
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
    physics_constants: PhysicsConstants = field(default_factory=PhysicsConstants)
    schema_version: int = SCHEMA_VERSION
```

- [ ] **Step 4: Run test to verify the new `RunConfig` tests pass but serialization still fails**

Run: `uv run pytest tests/config/test_run_config.py -v`
Expected: PASS (all of Task 1 and 2's `RunConfig` tests).

Run: `uv run pytest tests/config/test_serialization.py -v`
Expected: FAIL on the three new tests — `dataclasses.asdict` already serializes `sensor_config`/`physics_constants` into nested dicts correctly (so `dumps` produces valid JSON), but `loads` currently does `RunConfig(**{k: v for k, v in raw.items() if k in known})`, which hands those nested dicts straight to `RunConfig.__init__` — so `restored.sensor_config` is a plain `dict`, not a `SensorConfig`, and `isinstance(restored.sensor_config, SensorConfig)` is `False`.

- [ ] **Step 5: Fix `loads()` to reconstruct nested dataclasses**

In `src/neuroarena/config/serialization.py`:

```python
from __future__ import annotations

import dataclasses
import json
from typing import Any

from neuroarena.config.run_config import SCHEMA_VERSION, RunConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


class UnknownSchemaVersionError(ValueError):
    """Config declares a schema_version this build cannot read."""


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate a decoded config dict toward SCHEMA_VERSION.

    Identity today (only v1 exists). The first `vN -> vN+1` step is added here
    at the first schema bump; see phase-0-architecture.md open questions.
    `loads` stamps `schema_version` to `SCHEMA_VERSION` on the dict this
    returns, so each migration step only needs to transform the other fields.
    """
    return raw


def dumps(config: RunConfig) -> str:
    return json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True)


def loads(text: str) -> RunConfig:
    raw = json.loads(text)
    version = raw.get("schema_version")
    if not isinstance(version, int) or not (1 <= version <= SCHEMA_VERSION):
        raise UnknownSchemaVersionError(
            f"schema_version {version!r} is not readable by this build "
            f"(supports 1..{SCHEMA_VERSION})"
        )
    if version < SCHEMA_VERSION:
        raw = migrate(raw)
        raw["schema_version"] = SCHEMA_VERSION
    known = {f.name for f in dataclasses.fields(RunConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if isinstance(kwargs.get("sensor_config"), dict):
        kwargs["sensor_config"] = _sensor_config_from_dict(kwargs["sensor_config"])
    if isinstance(kwargs.get("physics_constants"), dict):
        kwargs["physics_constants"] = PhysicsConstants(**kwargs["physics_constants"])
    return RunConfig(**kwargs)


def _sensor_config_from_dict(raw: dict[str, Any]) -> SensorConfig:
    # `json.loads` decodes a JSON array as a `list`, but `SensorConfig.ray_angles_deg` is
    # typed `tuple[float, ...]` — reconstruct it as a tuple explicitly, or a round-tripped
    # RunConfig would carry a list where a tuple is expected (breaks equality/hashing
    # elsewhere `SensorConfig` is compared or hashed).
    return SensorConfig(**{**raw, "ray_angles_deg": tuple(raw["ray_angles_deg"])})
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/config/ -v`
Expected: PASS — every test in `tests/config/test_run_config.py` and `tests/config/test_serialization.py`, including the three new round-trip tests and the pre-existing `test_roundtrip_preserves_value` (which now exercises the nested-dataclass defaults too, since `RunConfig()`'s `sensor_config`/`physics_constants` are no longer plain scalars).

- [ ] **Step 7: Commit**

```bash
git add src/neuroarena/config/run_config.py src/neuroarena/config/serialization.py tests/config/test_run_config.py tests/config/test_serialization.py
git commit -m "feat(phase-5): add sensor_config/physics_constants RunConfig knobs, fix loads() to reconstruct nested dataclasses"
```

---

## Task 3: `neat_hyperparameters` `RunConfig` knob + `build_neat_config` override mechanism

**Files:**
- Modify: `src/neuroarena/config/run_config.py`
- Modify: `tests/config/test_run_config.py`
- Modify: `src/neuroarena/backends/neat/config.py`
- Modify: `tests/backends/neat/test_config.py`

**Interfaces:**
- Produces: `RunConfig.neat_hyperparameters: dict[str, float | int | bool | str]` (default `{}`); `build_neat_config(observation_space, action_space, population_size, hyperparameter_overrides=None) -> neat.Config` gains a fourth, optional parameter. Consumed by Task 4's `NeatTrainer`.

- [ ] **Step 1: Write the failing `RunConfig` test**

Append to `tests/config/test_run_config.py`:

```python
def test_neat_hyperparameters_defaults_to_empty_dict() -> None:
    assert RunConfig().neat_hyperparameters == {}


def test_neat_hyperparameters_is_settable() -> None:
    config = RunConfig(neat_hyperparameters={"weight_mutate_rate": 0.9})
    assert config.neat_hyperparameters == {"weight_mutate_rate": 0.9}


def test_neat_hyperparameters_defaults_are_independent_between_instances() -> None:
    a, b = RunConfig(), RunConfig()
    a.neat_hyperparameters["x"] = 1
    assert b.neat_hyperparameters == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_run_config.py -v`
Expected: FAIL — `TypeError: RunConfig.__init__() got an unexpected keyword argument 'neat_hyperparameters'`.

- [ ] **Step 3: Add the field**

In `src/neuroarena/config/run_config.py`, add one field (after `physics_constants`, before `schema_version`):

```python
    neat_hyperparameters: dict[str, float | int | bool | str] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/ -v`
Expected: PASS. (`dataclasses.asdict`/`json.dumps` already handle a plain `dict[str, float | int | bool | str]` with no help — no `serialization.py` change needed here, unlike Task 2's nested dataclasses.)

- [ ] **Step 5: Write the failing `build_neat_config` tests**

Append to `tests/backends/neat/test_config.py`:

```python
import pytest


def test_hyperparameter_override_reaches_genome_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"weight_mutate_rate": 0.99},
    )
    assert config.genome_config.weight_mutate_rate == 0.99


def test_hyperparameter_override_reaches_species_set_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"compatibility_threshold": 5.5},
    )
    assert config.species_set_config.compatibility_threshold == 5.5


def test_hyperparameter_override_reaches_stagnation_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"max_stagnation": 42},
    )
    assert config.stagnation_config.max_stagnation == 42


def test_hyperparameter_override_reaches_reproduction_config() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"elitism": 4},
    )
    assert config.reproduction_config.elitism == 4


def test_hyperparameter_override_reaches_top_level_neat_section() -> None:
    config = build_neat_config(
        Box(-1.0, 1.0, (10,)),
        Box(-1.0, 1.0, (2,)),
        population_size=25,
        hyperparameter_overrides={"fitness_threshold": 500.0},
    )
    assert config.fitness_threshold == 500.0


def test_unknown_hyperparameter_override_raises() -> None:
    with pytest.raises(ValueError, match="unknown NEAT hyperparameter"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"not_a_real_parameter": 1},
        )


def test_reserved_key_override_raises() -> None:
    with pytest.raises(ValueError, match="pop_size"):
        build_neat_config(
            Box(-1.0, 1.0, (10,)),
            Box(-1.0, 1.0, (2,)),
            population_size=25,
            hyperparameter_overrides={"pop_size": 999},
        )


def test_no_overrides_behaves_exactly_as_before() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.pop_size == 25
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_config.py -v`
Expected: FAIL — `TypeError: build_neat_config() got an unexpected keyword argument 'hyperparameter_overrides'`.

- [ ] **Step 7: Implement the override mechanism**

In `src/neuroarena/backends/neat/config.py`, add imports and the override machinery, and thread the new parameter through `build_neat_config`:

```python
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping

import neat

from neuroarena.interfaces.spaces import Box

# ... DEFAULT_CONFIG_TEXT unchanged ...

_RESERVED_OVERRIDE_KEYS = {"pop_size", "num_inputs", "num_outputs", "input_keys", "output_keys"}
"""These are set from `build_neat_config`'s own `population_size`/space arguments, which are
dedicated Phase 5 RunConfig knobs in their own right (`population_size`) or derived from the
Environment (`num_inputs`/`num_outputs`/...) — never from the generic `neat_hyperparameters`
override dict, so a caller can't accidentally fight the dedicated knob with a same-named
override."""

_TOP_LEVEL_HYPERPARAMETERS = {
    "fitness_criterion",
    "fitness_threshold",
    "reset_on_extinction",
    "no_fitness_termination",
}
"""The `[NEAT]` section's own keys, stored directly on `neat.Config` (not on one of its four
sub-configs) — see `neat.Config.__init__`."""

_SUBCONFIG_ATTRS = ("genome_config", "species_set_config", "stagnation_config", "reproduction_config")


def build_neat_config(
    observation_space: Box,
    action_space: Box,
    population_size: int,
    hyperparameter_overrides: Mapping[str, float | int | bool | str] | None = None,
) -> neat.Config:
    """A `neat.Config` whose genome shape matches `observation_space`/`action_space`, whose
    population size matches `population_size`, and whose hyperparameters match the bundled
    template's defaults except where `hyperparameter_overrides` says otherwise (Phase 5's
    `RunConfig.neat_hyperparameters` knob — see the Phase 5 doc's "NEAT hyperparameters
    become configurable" requirement)."""
    fd, path = tempfile.mkstemp(suffix=".cfg", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(DEFAULT_CONFIG_TEXT)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            path,
        )
    finally:
        os.unlink(path)

    num_inputs = observation_space.shape[0]
    num_outputs = action_space.shape[0]
    config.genome_config.num_inputs = num_inputs
    config.genome_config.num_outputs = num_outputs
    config.genome_config.input_keys = [-i - 1 for i in range(num_inputs)]
    config.genome_config.output_keys = list(range(num_outputs))
    config.pop_size = population_size

    if hyperparameter_overrides:
        _apply_hyperparameter_overrides(config, hyperparameter_overrides)

    return config


def _apply_hyperparameter_overrides(
    config: neat.Config, overrides: Mapping[str, float | int | bool | str]
) -> None:
    for key, value in overrides.items():
        if key in _RESERVED_OVERRIDE_KEYS:
            raise ValueError(
                f"{key!r} is set via build_neat_config's own population_size/space "
                "arguments, not hyperparameter_overrides"
            )
        if key in _TOP_LEVEL_HYPERPARAMETERS:
            setattr(config, key, value)
            continue
        for attr in _SUBCONFIG_ATTRS:
            subconfig = getattr(config, attr)
            if hasattr(subconfig, key):
                setattr(subconfig, key, value)
                break
        else:
            raise ValueError(f"unknown NEAT hyperparameter override: {key!r}")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_config.py -v`
Expected: PASS — all tests, including Phase 4's original four and this task's eight new ones.

- [ ] **Step 9: Commit**

```bash
git add src/neuroarena/config/run_config.py tests/config/test_run_config.py src/neuroarena/backends/neat/config.py tests/backends/neat/test_config.py
git commit -m "feat(phase-5): RunConfig.neat_hyperparameters + build_neat_config hyperparameter overrides"
```

---

## Task 4: Wire `population_size` and `neat_hyperparameters` into `NeatTrainer`

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Modify: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `RunConfig.population_size`, `RunConfig.neat_hyperparameters` (Tasks 1, 3); `build_neat_config`'s `hyperparameter_overrides` parameter (Task 3).
- Produces: `NeatTrainer.__init__`'s `population_size` parameter becomes optional (`int | None = None`), falling back to `config.population_size` when not given explicitly — the explicit-kwarg-always-wins behavior every existing Phase 4 test relies on (they all pass `population_size=...` explicitly) is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/backends/neat/test_trainer.py`:

```python
def test_population_size_falls_back_to_config_when_not_given() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(population_size=12))
    update = next(trainer.run())
    assert update.population_size == 12


def test_explicit_population_size_overrides_config() -> None:
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(population_size=12),
        population_size=4,
    )
    update = next(trainer.run())
    assert update.population_size == 4


def test_neat_hyperparameters_reach_the_built_neat_config() -> None:
    config = RunConfig(neat_hyperparameters={"compatibility_threshold": 7.5})
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    # White-box check: NeatTrainer has no public accessor for the underlying neat.Config,
    # and this is exactly the wiring this task adds, so the test reaches into `_neat_config`.
    assert trainer._neat_config.species_set_config.compatibility_threshold == 7.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v -k "population_size_falls_back or neat_hyperparameters_reach"`
Expected: FAIL — `test_population_size_falls_back_to_config_when_not_given` fails because `NeatTrainer.__init__`'s current default is a hard-coded `population_size: int = 150`, ignoring `config.population_size` entirely, so the trainer builds a population of 150, not 12.

- [ ] **Step 3: Update `NeatTrainer.__init__`**

In `src/neuroarena/backends/neat/trainer.py`, change the constructor:

```python
    def __init__(
        self,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
        *,
        population_size: int | None = None,
        neat_config: neat.Config | None = None,
        champion_dir: Path | None = None,
    ) -> None:
        self._make_env = make_env
        self._objective = objective
        self._config = config
        self._champion_dir = champion_dir
        if champion_dir is not None:
            champion_dir.mkdir(parents=True, exist_ok=True)

        resolved_population_size = (
            population_size if population_size is not None else config.population_size
        )

        env = make_env()
        if neat_config is None:
            observation_space, action_space = env.observation_space, env.action_space
            if not isinstance(observation_space, Box) or not isinstance(action_space, Box):
                raise TypeError("NeatTrainer requires Box observation/action spaces")
            neat_config = build_neat_config(
                observation_space,
                action_space,
                resolved_population_size,
                hyperparameter_overrides=config.neat_hyperparameters or None,
            )
        self._neat_config = neat_config
        self._population = neat.Population(self._neat_config)
        self._env = env
        self._generation = 0
        self._total_sim_steps = 0.0
        self._wall_start = time.perf_counter()
```

This is the only change in this task — `_run_one_generation`, `save_checkpoint`, and `load_checkpoint` are untouched (checkpoint round-trip tests must still pass unmodified, since `load_checkpoint` already passes an explicit `neat_config` and never hits this fallback path).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: PASS — the whole file, confirming no regression in Phase 4's original tests (which all pass `population_size=` explicitly, so `resolved_population_size` takes the explicit-kwarg branch exactly as before) or Task 7's checkpoint tests.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-5): NeatTrainer reads population_size/neat_hyperparameters from RunConfig"
```

---

## Task 5: `CarEnvironmentConfig.from_run_config`

**Files:**
- Modify: `src/neuroarena/sim/car_env.py`
- Modify: `tests/sim/test_car_env.py`

**Interfaces:**
- Consumes: `RunConfig.sensor_config`, `RunConfig.physics_constants`, `RunConfig.max_episode_steps` (Tasks 1, 2).
- Produces: `CarEnvironmentConfig.from_run_config(config: RunConfig) -> CarEnvironmentConfig` (classmethod).

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/test_car_env.py`:

```python
from neuroarena.config import RunConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.physics import PhysicsConstants


def test_from_run_config_uses_platform_defaults_by_default() -> None:
    car_config = CarEnvironmentConfig.from_run_config(RunConfig())
    assert car_config == CarEnvironmentConfig()


def test_from_run_config_carries_a_non_default_sensor_config() -> None:
    sensor_config = SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0))
    car_config = CarEnvironmentConfig.from_run_config(RunConfig(sensor_config=sensor_config))
    assert car_config.sensor_config == sensor_config


def test_from_run_config_carries_physics_constants_and_episode_limit() -> None:
    physics = PhysicsConstants(max_speed=500.0)
    car_config = CarEnvironmentConfig.from_run_config(
        RunConfig(physics_constants=physics, max_episode_steps=100)
    )
    assert car_config.physics_constants == physics
    assert car_config.max_episode_steps == 100


def test_from_run_config_env_has_a_narrower_observation_space() -> None:
    sensor_config = SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0))  # 3 rays, not 7
    car_config = CarEnvironmentConfig.from_run_config(RunConfig(sensor_config=sensor_config))
    env = CarEnvironment(_track(), track_id="abc123", config=car_config)
    assert env.observation_space == Box(-1.0, 1.0, (6,))  # 3 rays + speed + 2 last-action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sim/test_car_env.py -v -k from_run_config`
Expected: FAIL — `AttributeError: type object 'CarEnvironmentConfig' has no attribute 'from_run_config'`.

- [ ] **Step 3: Implement `from_run_config`**

In `src/neuroarena/sim/car_env.py`, add an import and the classmethod:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from neuroarena.interfaces.spaces import Box, Space
from neuroarena.sim.collision import distance_to_boundary
from neuroarena.sim.game import Game
from neuroarena.sim.observation import SensorConfig, build_observation, observation_space_for
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.sim.track import (
    Track,
    boundary_segments,
    centerline_path,
    project_onto_centerline,
    track_loop_length,
)

if TYPE_CHECKING:
    from neuroarena.config import RunConfig

_CRASH_EPSILON = 1e-6


@dataclass(frozen=True)
class CarEnvironmentConfig:
    """Starting defaults; also buildable from a `RunConfig` via `from_run_config` (Phase 5)."""

    sensor_config: SensorConfig = SensorConfig()
    physics_constants: PhysicsConstants = PhysicsConstants()
    max_episode_steps: int = 3000

    @classmethod
    def from_run_config(cls, config: RunConfig) -> CarEnvironmentConfig:
        """Reads the three car-specific knobs Phase 5 put on `RunConfig`
        (`sensor_config`, `physics_constants`, `max_episode_steps`) into a
        `CarEnvironmentConfig`. `RunConfig.track_id` is not read here — which `Track`
        object a run uses is a separate `CarEnvironment.__init__` argument, resolved by
        whatever code loads the track by id (see the Phase 5 implementation plan's "Not
        built in this plan" note)."""
        return cls(
            sensor_config=config.sensor_config,
            physics_constants=config.physics_constants,
            max_episode_steps=config.max_episode_steps,
        )
```

Only the `CarEnvironmentConfig` dataclass changes; `CarEnvironment` itself is untouched — it already reads every field of whatever `CarEnvironmentConfig` it's given.

Since `RunConfig` is only referenced for a type hint here (never constructed or introspected at runtime in `car_env.py`), the import stays `TYPE_CHECKING`-only — this module still has no runtime dependency on `neuroarena.config`, matching how `interfaces/protocols.py` already imports `RunConfig` for `Trainer`'s signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sim/test_car_env.py -v`
Expected: PASS — the whole file, confirming no regression in Phase 3's existing `CarEnvironment` tests.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/sim/car_env.py tests/sim/test_car_env.py
git commit -m "feat(phase-5): CarEnvironmentConfig.from_run_config wires sensor_config/physics_constants/max_episode_steps"
```

---

## Task 6: Run-level automatic stop (`max_generations`, `target_fitness`)

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Modify: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `RunConfig.max_generations`, `RunConfig.target_fitness` (Task 1).
- Produces: `NeatTrainer.run()` now returns (raises `StopIteration`) after yielding the generation whose `TrainingUpdate` satisfies either configured stop condition, instead of running forever. With both `None` (the default), behaviour is unchanged — an infinite generator, exactly as every existing Phase 4 test already relies on.

- [ ] **Step 1: Write the failing test**

Append to `tests/backends/neat/test_trainer.py`:

```python
def test_run_stops_after_max_generations() -> None:
    config = RunConfig(max_generations=3)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1, 2]


def test_run_stops_once_target_fitness_is_reached() -> None:
    # DummyEnvironment truncates every episode after exactly 5 steps, DummyObjective's
    # fitness is steps survived, so every genome's fitness is 5.0 every generation —
    # best_fitness reaches 5.0 on generation 0 already.
    config = RunConfig(target_fitness=5.0)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0]


def test_run_with_neither_stop_condition_set_keeps_running() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    run = trainer.run()
    updates = [next(run) for _ in range(5)]
    assert [u.progress_index for u in updates] == [0, 1, 2, 3, 4]


def test_both_stop_conditions_set_whichever_triggers_first_wins() -> None:
    # target_fitness is unreachable (DummyObjective never exceeds 5.0), so max_generations
    # must be what stops the run.
    config = RunConfig(max_generations=2, target_fitness=1_000_000.0)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    updates = list(trainer.run())
    assert [u.progress_index for u in updates] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 10 uv run pytest tests/backends/neat/test_trainer.py -v -k "stops_after_max_generations or target_fitness_is_reached or both_stop_conditions"`
Expected: the `timeout` wrapper kills the process after 10s with no test having completed — `list(trainer.run())` never terminates today (`run()` is `while True: yield ...` unconditionally, with no exit condition to reach `StopIteration`), so `list(...)` hangs forever rather than returning. The shell's `timeout` command (coreutils, no new Python dependency) is what turns that hang into an observable failure instead of actually waiting forever.

- [ ] **Step 3: Implement the auto-stop check**

In `src/neuroarena/backends/neat/trainer.py`, change `run()` and add one helper method:

```python
    def run(self) -> Iterator[TrainingUpdate]:
        while True:
            update = self._run_one_generation()
            yield update
            if self._should_auto_stop(update):
                return

    def _should_auto_stop(self, update: TrainingUpdate) -> bool:
        """Phase 5's run-level automatic stop: checked once per generation boundary,
        alongside the existing per-generation step ceiling. Distinct from
        `Objective.should_stop()`, which ends one episode, not the run."""
        max_generations = self._config.max_generations
        if max_generations is not None and update.progress_index + 1 >= max_generations:
            return True
        target_fitness = self._config.target_fitness
        if target_fitness is not None and update.best_fitness >= target_fitness:
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: PASS — the whole file. In particular, Phase 4's `test_run_yields_a_training_update_per_generation` (which pulls 3 items via repeated `next(run)` from what it expects to be an unbounded generator) must still pass, since `RunConfig()`'s `max_generations`/`target_fitness` default to `None`.

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-5): NeatTrainer run-level automatic stop (max_generations, target_fitness)"
```

---

## Task 7: End-to-end integration on the real `CarEnvironment` + full suite check

**Files:**
- Create: `tests/backends/neat/test_training_controls_end_to_end.py`

**Interfaces:**
- Consumes: `NeatTrainer` (Tasks 4, 6), `CarEnvironmentConfig.from_run_config` (Task 5), `RunConfig` (Tasks 1-3), the real `neuroarena.sim.car_env.CarEnvironment` (Phase 3).
- Produces: nothing new — verification only, that every Phase 5 knob this plan wired actually changes real training behaviour together, not just in isolation against test doubles.

- [ ] **Step 1: Write the end-to-end test**

Create `tests/backends/neat/test_training_controls_end_to_end.py`:

```python
from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.spaces import Box
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.observation import SensorConfig
from neuroarena.sim.track import Facing, GridCell, Track, TileKind
from tests.interfaces.doubles import DummyObjective


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _make_car_env(config: RunConfig) -> CarEnvironment:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    car_config = CarEnvironmentConfig.from_run_config(config)
    return CarEnvironment(track, track_id="phase-5-e2e", config=car_config)


def test_a_run_config_with_several_knobs_set_trains_end_to_end() -> None:
    config = RunConfig(
        sensor_config=SensorConfig(ray_angles_deg=(-40.0, 0.0, 40.0)),
        max_episode_steps=50,
        max_generation_steps=500,
        max_generations=3,
        population_size=6,
        neat_hyperparameters={"compatibility_threshold": 4.0},
    )
    trainer = NeatTrainer(lambda: _make_car_env(config), DummyObjective(), config)

    # The narrower sensor_config actually reshaped the env this trainer is training on:
    # 3 rays + speed + 2 last-action = 6, not the platform-default 10.
    assert trainer._env.observation_space == Box(-1.0, 1.0, (6,))
    # The hyperparameter override actually reached the neat.Config the trainer built.
    assert trainer._neat_config.species_set_config.compatibility_threshold == 4.0

    updates = list(trainer.run())

    # max_generations stopped the run at exactly 3 generations.
    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert u.population_size == 6
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_training_controls_end_to_end.py -v -s`
Expected: PASS, completing in well under a minute (population 6, 3 generations, small track, a 50-step episode cap).

- [ ] **Step 3: Run the full test suite, lint, and type-check**

Run: `uv run pytest -v`
Expected: PASS, all tests (Phases 0-5).

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

Run: `uv run mypy src`
Expected: clean (`--strict`).

- [ ] **Step 4: Commit**

```bash
git add tests/backends/neat/test_training_controls_end_to_end.py
git commit -m "test(phase-5): end-to-end training-controls smoke test on the real CarEnvironment"
```

---

## Post-plan: update the phase doc

After all 7 tasks are committed, update `../../phases/phase-5-training-controls.md`'s "Implementation plan" section to link this file and summarize the tasks (matching Phase 3/4's doc pattern), and add a dated Revision history entry noting implementation is complete — test count, and any findings discovered during implementation (per `../../WORKFLOW.md`'s decision-provenance rule), the same way Phase 3's and Phase 4's 2026-09-13 entries recorded theirs. Also note explicitly, in that entry, the two things this plan deliberately did not build (`track_id` resolution into a live `Track`/`make_env`, and any consumer for `sim_speed`/`headless`) so a future session doesn't mistake their absence for an oversight.
