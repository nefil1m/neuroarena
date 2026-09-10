# Phase 0 — Architecture Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `neuroarena` package skeleton plus the platform's core interface Protocols, space descriptors, the model/config compatibility check, and the JSON `RunConfig` envelope — the code every later phase imports.

**Architecture:** Pure-declaration interface layer. `neuroarena.interfaces` holds `typing.Protocol` definitions (`Environment`, `Model`, `Objective`, `Trainer`) with no implementations, plus frozen-dataclass space descriptors (`Box`, `Discrete`) whose field-wise equality *is* the compatibility rule. `neuroarena.config` holds a small serializable `RunConfig` (its real fields land in Phase 5) with a versioned JSON codec and a forward-migration hook. Nothing here imports `gymnasium`, a renderer, or a trainer.

**Tech Stack:** Python 3.12+, `uv` (deps + lock), `pytest`, `ruff` (lint + format), `numpy`, `dataclasses`, `typing.Protocol`.

**Spec:** `docs/phases/phase-0-architecture.md` (FINALIZED 2026-09-10)

## Global Constraints

- Python `>=3.12` (final floor is whatever `arcade` and `stable-baselines3` both support).
- `neuroarena.interfaces` and `neuroarena.config` import **no** `gymnasium`, **no** renderer, **no** trainer. Only runtime dependency in Phase 0 is `numpy`.
- Interfaces are `typing.Protocol`, structural — a game or backend never subclasses a platform base class. Make them `@runtime_checkable`.
- Descriptor equality compares **all fields**: `Box` on `(low, high, shape, dtype)`, `Discrete` on `n`.
- `Environment.step` returns a **4-tuple** `(observation, terminated, truncated, info)` — **no** `reward`. `terminated` = in-world end; `truncated` = external budget hit.
- `Objective.update` takes the full step tuple: `(observation, action, terminated, truncated, info)`.
- Config is serialized as **JSON with a top-level integer `schema_version`**.
- `src/` layout, single package `neuroarena/`. Phase 0 delivers `neuroarena/interfaces/` and `neuroarena/config/` only.
- Commit at the end of every task. Run `uv run ruff check .` and `uv run ruff format --check .` before each commit.

---

## File Structure

```
pyproject.toml                              # project + tool config (uv, ruff, pytest)
.python-version                             # "3.12"
src/neuroarena/__init__.py                  # __version__
src/neuroarena/interfaces/__init__.py       # re-exports the public interface names
src/neuroarena/interfaces/spaces.py         # Box, Discrete, Space
src/neuroarena/interfaces/protocols.py      # Environment, Model, Objective, Trainer, TrainingUpdate, Observation, Action
src/neuroarena/interfaces/compat.py         # IncompatibleDescriptorsError, check_compatibility
src/neuroarena/config/__init__.py           # re-exports RunConfig, SCHEMA_VERSION, dumps, loads, migrate, UnknownSchemaVersionError
src/neuroarena/config/run_config.py         # RunConfig, SCHEMA_VERSION
src/neuroarena/config/serialization.py      # dumps, loads, migrate, UnknownSchemaVersionError
tests/__init__.py
tests/test_smoke.py
tests/interfaces/__init__.py
tests/interfaces/doubles.py                 # DummyEnvironment, DummyModel (reused by later phases' tests)
tests/interfaces/test_spaces.py
tests/interfaces/test_protocols.py
tests/interfaces/test_compat.py
tests/interfaces/test_import_hygiene.py
tests/config/__init__.py
tests/config/test_run_config.py
tests/config/test_serialization.py
```

---

### Task 1: Project skeleton + tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/neuroarena/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing
- Produces: an importable `neuroarena` package with `neuroarena.__version__ == "0.0.0"`; `uv run pytest` and `uv run ruff check .` both succeed.

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_package_imports_and_has_version():
    import neuroarena

    assert neuroarena.__version__ == "0.0.0"
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "neuroarena"
version = "0.0.0"
description = "Decoupled car-driving game and learning-model training platform"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26"]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neuroarena"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
pythonpath = ["src"]
```

- [ ] **Step 3: Create the package and test package markers**

```bash
mkdir -p src/neuroarena tests
printf '3.12\n' > .python-version
printf '__version__ = "0.0.0"\n' > src/neuroarena/__init__.py
: > tests/__init__.py
```

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`; exits 0.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Lint and format check**

Run: `uv run ruff check .` then `uv run ruff format --check .`
Expected: both exit 0 (run `uv run ruff format .` first if the check fails, then re-check).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/neuroarena/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore(phase-0): project skeleton, uv/pytest/ruff, smoke test"
```

---

### Task 2: Space descriptors (`Box`, `Discrete`)

**Files:**
- Create: `src/neuroarena/interfaces/__init__.py`
- Create: `src/neuroarena/interfaces/spaces.py`
- Create: `tests/interfaces/__init__.py`
- Test: `tests/interfaces/test_spaces.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Box(low: float, high: float, shape: tuple[int, ...], dtype: str = "float32")` — frozen dataclass; `__eq__` / `__hash__` over all four fields.
  - `Discrete(n: int)` — frozen dataclass; `__eq__` / `__hash__` over `n`.
  - `Space` — type alias `Box | Discrete`.
  - All three importable from `neuroarena.interfaces`.

- [ ] **Step 1: Write the failing test**

`tests/interfaces/__init__.py`: empty file.

`tests/interfaces/test_spaces.py`:
```python
from neuroarena.interfaces.spaces import Box, Discrete


def test_box_equal_when_all_fields_match():
    assert Box(-1.0, 1.0, (10,)) == Box(-1.0, 1.0, (10,))


def test_box_unequal_on_shape():
    assert Box(-1.0, 1.0, (10,)) != Box(-1.0, 1.0, (7,))


def test_box_unequal_on_bounds():
    assert Box(-1.0, 1.0, (10,)) != Box(0.0, 1.0, (10,))


def test_box_unequal_on_dtype():
    assert Box(-1.0, 1.0, (10,), "float32") != Box(-1.0, 1.0, (10,), "float64")


def test_box_is_hashable_and_hash_matches_equal_instances():
    assert hash(Box(-1.0, 1.0, (10,))) == hash(Box(-1.0, 1.0, (10,)))


def test_discrete_equality_and_inequality():
    assert Discrete(4) == Discrete(4)
    assert Discrete(4) != Discrete(2)


def test_box_never_equals_discrete():
    assert Box(0.0, 1.0, (1,)) != Discrete(1)


def test_box_repr_contains_type_and_shape():
    r = repr(Box(-1.0, 1.0, (10,)))
    assert "Box" in r and "10" in r
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/interfaces/test_spaces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.interfaces'`.

- [ ] **Step 3: Write the implementation**

`src/neuroarena/interfaces/spaces.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """A continuous vector space: `shape` floats, each within [low, high]."""

    low: float
    high: float
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class Discrete:
    """A single integer choice in `range(n)`."""

    n: int


Space = Box | Discrete
```

`src/neuroarena/interfaces/__init__.py`:
```python
from neuroarena.interfaces.spaces import Box, Discrete, Space

__all__ = ["Box", "Discrete", "Space"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/interfaces/test_spaces.py -v`
Expected: all 8 tests PASS. (`@dataclass(frozen=True)` supplies field-wise `__eq__`/`__hash__` and type-sensitive equality.)

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/neuroarena/interfaces/ tests/interfaces/__init__.py tests/interfaces/test_spaces.py
git commit -m "feat(phase-0): Box/Discrete space descriptors with field-wise equality"
```

---

### Task 3: Interface Protocols + `TrainingUpdate` + test doubles

**Files:**
- Create: `src/neuroarena/interfaces/protocols.py`
- Modify: `src/neuroarena/interfaces/__init__.py`
- Create: `tests/interfaces/doubles.py`
- Test: `tests/interfaces/test_protocols.py`

**Interfaces:**
- Consumes: `Box`, `Space` from Task 2.
- Produces:
  - `Observation` / `Action` — type aliases for `numpy.ndarray`.
  - `@runtime_checkable class Environment(Protocol)`: attrs `observation_space: Space`, `action_space: Space`; `reset(self, *, seed: int | None = None) -> Observation`; `step(self, action: Action) -> tuple[Observation, bool, bool, dict]`.
  - `@runtime_checkable class Model(Protocol)`: attrs `observation_space: Space`, `action_space: Space`; `reset(self) -> None`; `act(self, observation: Observation) -> Action`.
  - `@runtime_checkable class Objective(Protocol)`: `reset(self) -> None`; `update(self, observation: Observation, action: Action, terminated: bool, truncated: bool, info: dict) -> None`; `should_stop(self) -> bool`; `step_reward(self) -> float`; `fitness(self) -> float`.
  - `@dataclass(frozen=True) class TrainingUpdate`: fields `progress_index: int`, `best_fitness: float`, `mean_fitness: float`, `worst_fitness: float`, `population_size: int`, `champion_metrics: dict[str, float]`, `sim_time: float`, `wall_time: float`, `schema_version: int = 1`.
  - `@runtime_checkable class Trainer(Protocol)`: `run(self) -> Iterator[TrainingUpdate]`; `save_checkpoint(self, path: Path) -> None`; classmethod `load_checkpoint(cls, path: Path, make_env: Callable[[], Environment], objective: Objective, config: "RunConfig") -> "Trainer"`.
  - `tests/interfaces/doubles.py`: `DummyEnvironment` (3-float obs, 2-float action, `truncated` after 5 steps) and `DummyModel(observation_space, action_space)` (emits zeros). These are reused by later phases' tests.
  - All public names importable from `neuroarena.interfaces`.

- [ ] **Step 1: Write the test doubles**

`tests/interfaces/doubles.py`:
```python
from __future__ import annotations

import numpy as np

from neuroarena.interfaces.spaces import Box, Space


class DummyEnvironment:
    """Minimal Environment double: 3-float obs, 2-float action, truncates after 5 steps."""

    observation_space: Space = Box(-1.0, 1.0, (3,))
    action_space: Space = Box(-1.0, 1.0, (2,))

    def __init__(self) -> None:
        self._t = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        self._t = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict]:
        self._t += 1
        return np.zeros(3, dtype=np.float32), False, self._t >= 5, {"t": self._t}


class DummyModel:
    """Minimal Model double: always emits zeros of the action shape."""

    def __init__(self, observation_space: Space, action_space: Space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        shape = self.action_space.shape  # type: ignore[union-attr]
        return np.zeros(shape, dtype=np.float32)
```

- [ ] **Step 2: Write the failing test**

`tests/interfaces/test_protocols.py`:
```python
import numpy as np
import pytest

from neuroarena.interfaces import (
    Environment,
    Model,
    Objective,
    Trainer,
    TrainingUpdate,
)
from tests.interfaces.doubles import DummyEnvironment, DummyModel


def test_dummy_environment_satisfies_environment_protocol():
    assert isinstance(DummyEnvironment(), Environment)


def test_dummy_model_satisfies_model_protocol():
    from neuroarena.interfaces import Box

    assert isinstance(DummyModel(Box(-1.0, 1.0, (3,)), Box(-1.0, 1.0, (2,))), Model)


def test_bare_object_is_not_an_environment():
    class NotEnv:
        pass

    assert not isinstance(NotEnv(), Environment)


def test_environment_step_is_a_four_tuple_without_reward():
    env = DummyEnvironment()
    env.reset(seed=0)
    result = env.step(np.zeros(2, dtype=np.float32))
    assert len(result) == 4
    obs, terminated, truncated, info = result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_objective_and_trainer_are_runtime_checkable():
    # Protocols with no implementation still expose isinstance via runtime_checkable.
    assert not isinstance(object(), Objective)
    assert not isinstance(object(), Trainer)


def test_training_update_is_frozen_and_defaults_schema_version():
    u = TrainingUpdate(
        progress_index=1,
        best_fitness=1.0,
        mean_fitness=0.5,
        worst_fitness=0.0,
        population_size=50,
        champion_metrics={"distance": 12.0},
        sim_time=3.0,
        wall_time=0.2,
    )
    assert u.schema_version == 1
    with pytest.raises(Exception):
        u.progress_index = 2
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/interfaces/test_protocols.py -v`
Expected: FAIL — `ImportError: cannot import name 'Environment' from 'neuroarena.interfaces'`.

- [ ] **Step 4: Write the implementation**

`src/neuroarena/interfaces/protocols.py`:
```python
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from neuroarena.interfaces.spaces import Space

if TYPE_CHECKING:
    from neuroarena.config import RunConfig

Observation = np.ndarray
Action = np.ndarray


@runtime_checkable
class Environment(Protocol):
    """A game episode. Mirrors Gymnasium's Env contract, minus `reward`."""

    observation_space: Space
    action_space: Space

    def reset(self, *, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, bool, bool, dict]: ...


@runtime_checkable
class Model(Protocol):
    """A policy: observation -> action, plus a per-episode reset hook."""

    observation_space: Space
    action_space: Space

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...


@runtime_checkable
class Objective(Protocol):
    """Defines what the agent is trying to do and how well it did.

    One config-selected component, driven by the trainer. `fitness()` feeds
    evolutionary backends; `step_reward()` feeds RL backends.
    """

    def reset(self) -> None: ...

    def update(
        self,
        observation: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None: ...

    def should_stop(self) -> bool: ...

    def step_reward(self) -> float: ...

    def fitness(self) -> float: ...


@dataclass(frozen=True)
class TrainingUpdate:
    """One unit of training progress. Fields may be added later (versioned)."""

    progress_index: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    population_size: int
    champion_metrics: dict[str, float]
    sim_time: float
    wall_time: float
    schema_version: int = 1


@runtime_checkable
class Trainer(Protocol):
    """A learning backend. Consumes Environment + Model + Objective + RunConfig only."""

    def run(self) -> Iterator[TrainingUpdate]: ...

    def save_checkpoint(self, path: Path) -> None: ...

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
    ) -> Trainer: ...
```

Update `src/neuroarena/interfaces/__init__.py`:
```python
from neuroarena.interfaces.protocols import (
    Action,
    Environment,
    Model,
    Objective,
    Observation,
    Trainer,
    TrainingUpdate,
)
from neuroarena.interfaces.spaces import Box, Discrete, Space

__all__ = [
    "Action",
    "Box",
    "Discrete",
    "Environment",
    "Model",
    "Objective",
    "Observation",
    "Space",
    "Trainer",
    "TrainingUpdate",
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/interfaces/test_protocols.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/neuroarena/interfaces/ tests/interfaces/doubles.py tests/interfaces/test_protocols.py
git commit -m "feat(phase-0): Environment/Model/Objective/Trainer protocols + TrainingUpdate"
```

---

### Task 4: Model / config compatibility check

**Files:**
- Create: `src/neuroarena/interfaces/compat.py`
- Modify: `src/neuroarena/interfaces/__init__.py`
- Test: `tests/interfaces/test_compat.py`

**Interfaces:**
- Consumes: `Model`, `Environment` from Task 3; `DummyEnvironment`, `DummyModel` from Task 3; `Box` from Task 2.
- Produces:
  - `class IncompatibleDescriptorsError(ValueError)`.
  - `check_compatibility(model: Model, environment: Environment) -> None` — raises `IncompatibleDescriptorsError` naming the first mismatching space (`"observation space mismatch: ..."` or `"action space mismatch: ..."`), including both `repr`s. Returns `None` on a match.
  - Both importable from `neuroarena.interfaces`.

- [ ] **Step 1: Write the failing test**

`tests/interfaces/test_compat.py`:
```python
import pytest

from neuroarena.interfaces import Box
from neuroarena.interfaces.compat import IncompatibleDescriptorsError, check_compatibility
from tests.interfaces.doubles import DummyEnvironment, DummyModel


def test_matching_descriptors_do_not_raise():
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    check_compatibility(model, env)


def test_observation_mismatch_raises_naming_observation():
    env = DummyEnvironment()
    model = DummyModel(Box(-1.0, 1.0, (7,)), env.action_space)
    with pytest.raises(IncompatibleDescriptorsError, match="observation"):
        check_compatibility(model, env)


def test_action_mismatch_raises_naming_action():
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, Box(-1.0, 1.0, (3,)))
    with pytest.raises(IncompatibleDescriptorsError, match="action"):
        check_compatibility(model, env)


def test_error_message_includes_both_descriptor_reprs():
    env = DummyEnvironment()
    model = DummyModel(Box(0.0, 1.0, (3,)), env.action_space)
    with pytest.raises(IncompatibleDescriptorsError) as exc:
        check_compatibility(model, env)
    msg = str(exc.value)
    assert "0.0" in msg and "-1.0" in msg
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/interfaces/test_compat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.interfaces.compat'`.

- [ ] **Step 3: Write the implementation**

`src/neuroarena/interfaces/compat.py`:
```python
from __future__ import annotations

from neuroarena.interfaces.protocols import Environment, Model


class IncompatibleDescriptorsError(ValueError):
    """A model's trained descriptors do not match an environment's."""


def check_compatibility(model: Model, environment: Environment) -> None:
    """Raise IncompatibleDescriptorsError if model and environment descriptors differ.

    Resume-safety check, run at run creation. Equality is the full field-wise
    comparison defined by the space descriptors.
    """
    if model.observation_space != environment.observation_space:
        raise IncompatibleDescriptorsError(
            f"observation space mismatch: model {model.observation_space!r} "
            f"!= environment {environment.observation_space!r}"
        )
    if model.action_space != environment.action_space:
        raise IncompatibleDescriptorsError(
            f"action space mismatch: model {model.action_space!r} "
            f"!= environment {environment.action_space!r}"
        )
```

Append to `src/neuroarena/interfaces/__init__.py` imports and `__all__`:
```python
from neuroarena.interfaces.compat import IncompatibleDescriptorsError, check_compatibility
```
Add `"IncompatibleDescriptorsError"` and `"check_compatibility"` to `__all__` (keep it sorted).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/interfaces/test_compat.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/neuroarena/interfaces/ tests/interfaces/test_compat.py
git commit -m "feat(phase-0): check_compatibility descriptor gate for run creation"
```

---

### Task 5: `RunConfig` + JSON serialization

**Files:**
- Create: `src/neuroarena/config/__init__.py`
- Create: `src/neuroarena/config/run_config.py`
- Create: `src/neuroarena/config/serialization.py`
- Create: `tests/config/__init__.py`
- Test: `tests/config/test_run_config.py`
- Test: `tests/config/test_serialization.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (config is independent of `interfaces`).
- Produces:
  - `SCHEMA_VERSION: int = 1` (in `run_config.py`).
  - `@dataclass class RunConfig`: `master_seed: int = 0`, `schema_version: int = SCHEMA_VERSION`. Non-frozen (Phase 5 adds fields, some live-editable). Plain `__eq__` from `@dataclass`.
  - `dumps(config: RunConfig) -> str` — pretty, key-sorted JSON.
  - `loads(text: str) -> RunConfig` — parses; rejects `schema_version` outside `1..SCHEMA_VERSION` with `UnknownSchemaVersionError`; calls `migrate(raw)` when `schema_version < SCHEMA_VERSION`; ignores unknown keys so a newer minor field does not crash an older reader path within the same major.
  - `migrate(raw: dict) -> dict` — forward-migration hook; identity at v1.
  - `class UnknownSchemaVersionError(ValueError)`.
  - All public names importable from `neuroarena.config`.

- [ ] **Step 1: Write the failing tests**

`tests/config/__init__.py`: empty file.

`tests/config/test_run_config.py`:
```python
from neuroarena.config import SCHEMA_VERSION, RunConfig


def test_defaults():
    c = RunConfig()
    assert c.master_seed == 0
    assert c.schema_version == SCHEMA_VERSION


def test_fields_are_settable():
    assert RunConfig(master_seed=42).master_seed == 42


def test_equality_is_field_wise():
    assert RunConfig(master_seed=1) == RunConfig(master_seed=1)
    assert RunConfig(master_seed=1) != RunConfig(master_seed=2)
```

`tests/config/test_serialization.py`:
```python
import json

import pytest

from neuroarena.config import SCHEMA_VERSION, RunConfig
from neuroarena.config.serialization import (
    UnknownSchemaVersionError,
    dumps,
    loads,
    migrate,
)


def test_roundtrip_preserves_value():
    c = RunConfig(master_seed=7)
    assert loads(dumps(c)) == c


def test_dumps_emits_json_with_schema_version():
    raw = json.loads(dumps(RunConfig()))
    assert raw["schema_version"] == SCHEMA_VERSION


def test_loads_rejects_future_version():
    text = json.dumps({"schema_version": SCHEMA_VERSION + 1, "master_seed": 0})
    with pytest.raises(UnknownSchemaVersionError):
        loads(text)


def test_loads_rejects_version_below_one():
    text = json.dumps({"schema_version": 0, "master_seed": 0})
    with pytest.raises(UnknownSchemaVersionError):
        loads(text)


def test_loads_ignores_unknown_keys():
    text = json.dumps(
        {"schema_version": SCHEMA_VERSION, "master_seed": 5, "future_field": 123}
    )
    assert loads(text) == RunConfig(master_seed=5)


def test_migrate_is_identity_at_current_version():
    raw = {"schema_version": SCHEMA_VERSION, "master_seed": 1}
    assert migrate(dict(raw)) == raw
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.config'`.

- [ ] **Step 3: Write the implementation**

`src/neuroarena/config/run_config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1


@dataclass
class RunConfig:
    """Serializable run configuration.

    Phase 0 holds only the envelope. Concrete run parameters (population size,
    deviation limit, sim speed, timeouts, Objective selection, ...) are added
    in Phase 5.
    """

    master_seed: int = 0
    schema_version: int = SCHEMA_VERSION
```

`src/neuroarena/config/serialization.py`:
```python
from __future__ import annotations

import dataclasses
import json

from neuroarena.config.run_config import SCHEMA_VERSION, RunConfig


class UnknownSchemaVersionError(ValueError):
    """Config declares a schema_version this build cannot read."""


def migrate(raw: dict) -> dict:
    """Forward-migrate a decoded config dict toward SCHEMA_VERSION.

    Identity today (only v1 exists). The first `vN -> vN+1` step is added here
    at the first schema bump; see phase-0-architecture.md open questions.
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
    known = {f.name for f in dataclasses.fields(RunConfig)}
    return RunConfig(**{k: v for k, v in raw.items() if k in known})
```

`src/neuroarena/config/__init__.py`:
```python
from neuroarena.config.run_config import SCHEMA_VERSION, RunConfig
from neuroarena.config.serialization import (
    UnknownSchemaVersionError,
    dumps,
    loads,
    migrate,
)

__all__ = [
    "SCHEMA_VERSION",
    "RunConfig",
    "UnknownSchemaVersionError",
    "dumps",
    "loads",
    "migrate",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/config -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/neuroarena/config/ tests/config/
git commit -m "feat(phase-0): RunConfig envelope + versioned JSON codec"
```

---

### Task 6: Import-hygiene guard

**Files:**
- Test: `tests/interfaces/test_import_hygiene.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces`, `neuroarena.config` (all prior tasks).
- Produces: an executable check of the Global Constraint that the core packages pull in no heavy/forbidden dependency.

- [ ] **Step 1: Write the failing test**

`tests/interfaces/test_import_hygiene.py`:
```python
import sys


def _fresh_import(*modules: str) -> set[str]:
    for name in list(sys.modules):
        if name.startswith("neuroarena"):
            del sys.modules[name]
    for m in modules:
        __import__(m)
    return set(sys.modules)


def test_core_packages_do_not_import_forbidden_dependencies():
    loaded = _fresh_import("neuroarena.interfaces", "neuroarena.config")
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into the core import graph"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/interfaces/test_import_hygiene.py -v`
Expected: PASS with the code as built in Tasks 2–5. (If it FAILS, a forbidden import was introduced — remove it; do not relax the test.)

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: every test from Tasks 1–6 passes (≈28 tests).

- [ ] **Step 4: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add tests/interfaces/test_import_hygiene.py
git commit -m "test(phase-0): guard core packages against forbidden imports"
```

---

## Self-Review

**1. Spec coverage** (against `docs/phases/phase-0-architecture.md`):

| Spec requirement | Task |
|---|---|
| `Environment` Protocol, Gym-mirroring, 4-tuple `step`, no reward | Task 3 |
| `terminated` vs `truncated` meaning | Task 3 (docstring + `DummyEnvironment` behaviour); enforcement is Phase 3 |
| Not car-specific (no raycasts/track/steering in base) | Task 3 (Protocols carry none) |
| `Model` Protocol: `act` + `reset` hook, records descriptors | Task 3 |
| `Objective` Protocol: `reset`/`update(full step tuple)`/`should_stop`/`step_reward`/`fitness` | Task 3 |
| `Trainer` Protocol: `make_env` factory, `run()` iterator, checkpoint save/load, takes `objective` | Task 3 |
| `TrainingUpdate` minimal shape, versioned | Task 3 |
| Local `Box`/`Discrete`, all-field equality | Task 2 |
| Model/config compatibility check at run creation | Task 4 |
| Decoupling: core imports no `gymnasium`/renderer/trainer | Task 6 |
| Trainer owns seeding from `RunConfig` master seed | Task 3 (`load_checkpoint`/`__init__` signature) + Task 5 (`master_seed` field) |
| `RunConfig` serializable + versioned, JSON + integer `schema_version` | Task 5 |
| `schema_version` field + forward-migration hook | Task 5 (`migrate`) |
| Fields owned by Phase 5 | Task 5 (only envelope fields present) |
| Scaffolding: Python 3.12+, `uv`, `pytest`, `ruff`, `src/` layout, `neuroarena/` with `interfaces/` + `config/` | Task 1 (+ layout realised across Tasks 2–5) |
| `typing.Protocol` over `ABC` | Task 3 |
| Reserved (not built): champion checkpoints/replay, schema-migration mechanism, `TrainingUpdate` field additions | Correctly absent — `migrate` is an identity stub; `TrainingUpdate` has `schema_version` |

No gaps.

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases"/"similar to Task N". `migrate` returning `raw` unchanged is a deliberate, spec-sanctioned identity hook (documented as such), not a placeholder.

**3. Type consistency:** `Box(low, high, shape, dtype)` and `Discrete(n)` used identically in Tasks 2/3/4. `check_compatibility(model, environment)` — same name and parameter order in the Task 4 interface block, implementation, and tests. `RunConfig` fields `master_seed` / `schema_version` and `SCHEMA_VERSION` consistent across Task 5 impl and tests. `TrainingUpdate` field names match between the Task 3 interface block, implementation, and test. `dumps`/`loads`/`migrate`/`UnknownSchemaVersionError` consistent across Task 5.
