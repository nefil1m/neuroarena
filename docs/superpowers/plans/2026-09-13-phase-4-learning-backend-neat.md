# Phase 4 — Learning Backend: NEAT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NEAT neuroevolution end-to-end on the car game: a `neat-python`-backed `Model`/`Trainer` pair that consumes the existing Phase 0 `Environment`/`Objective`/`RunConfig` interfaces and the Phase 3 `CarEnvironment`, running headless with isolated per-genome evaluation, a per-generation step ceiling that guarantees every generation terminates, and per-generation champion capture.

**Architecture:** A new `neuroarena.backends.neat` package, independent of `neuroarena.sim` and any renderer. `build_neat_config()` turns a bundled stock `neat-python` config template into a `neat.Config` sized to whatever `Environment` it's pointed at. `GenomeModel` wraps one genome as a `Model`. `NeatTrainer` drives one `neat.Population` one generation at a time (`population.run(fn, 1)` in a loop), evaluating each genome in isolation against a single shared `Environment` instance rebuilt at each generation boundary (so a live track switch — Phase 5 — takes effect for the next generation, not mid-batch), under a shared per-generation step budget that force-truncates any genome still running once exhausted.

**Tech Stack:** Python 3.12+, `neat-python`, `numpy`, `pytest`, `mypy --strict`, `ruff`. No `gymnasium`, `arcade`, `pygame`, `stable_baselines3`, or `torch` anywhere in this backend.

**Spec:** [`../../phases/phase-4-learning-backend-neat.md`](../../phases/phase-4-learning-backend-neat.md) (FINALIZED 2026-09-13). Also depends on the FINALIZED [`../../phases/phase-0-architecture.md`](../../phases/phase-0-architecture.md) (interfaces) and [`../../phases/phase-3-sensors-env-interface.md`](../../phases/phase-3-sensors-env-interface.md) (`CarEnvironment`).

## Global Constraints

- Python 3.12+; `uv` for deps; `pytest` for tests; `ruff` for lint+format; `mypy --strict` (see `pyproject.toml`'s `[tool.mypy]`).
- `src/` layout, package `neuroarena/`. New code lives under `src/neuroarena/backends/neat/`.
- `neuroarena.backends.neat` imports no `gymnasium`, `arcade`, `pygame`, `stable_baselines3`, or `torch` — mirrors the existing `neuroarena.sim` import-hygiene guard (`tests/sim/test_import_hygiene.py`).
- `neuroarena.backends.neat` must not import `neuroarena.sim.*` or `neuroarena.render.*` — it consumes only the `Environment`/`Model`/`Objective`/`RunConfig` interfaces (`neuroarena.interfaces.*`, `neuroarena.config.*`), per Phase 0's "A `Trainer` consumes only the `Environment`, `Model`, `Objective`, and `RunConfig` interfaces" decoupling rule. Tests are exempt (they construct a real `CarEnvironment` to exercise the backend end-to-end).
- Action space is continuous `Box(-1, 1, (2,))`; observation space is `Box(-1, 1, (n,))` for whatever `n` the sensor config produces (10 by default). Never hard-code `10`/`2` in backend code — read shapes off `env.observation_space`/`env.action_space`.
- `Objective` is a single instance passed into the trainer once (Phase 0's `Trainer.__init__(make_env, objective, config)` signature) — call `objective.reset()` before each genome's episode, never construct a new `Objective`.
- No car-specific imports (e.g. `TICK_DT` from `neuroarena.sim.game`) in `neuroarena.backends.neat` — `TrainingUpdate.sim_time` is counted in elapsed environment steps (a float), not real seconds, documented inline, since the base `Environment` interface has no generic step-to-seconds conversion and this backend must stay usable by a future second game (Phase 11).

---

## Task 1: `RunConfig` generation-ceiling stub field + `neat-python` dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/neuroarena/config/run_config.py`
- Modify: `tests/config/test_run_config.py`

**Interfaces:**
- Produces: `RunConfig.max_generation_steps: int` (default `300_000`) — the per-generation step ceiling Phase 4's doc requires. Consumed by `NeatTrainer` (Task 6).

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_run_config.py`:

```python
def test_max_generation_steps_default() -> None:
    assert RunConfig().max_generation_steps == 300_000


def test_max_generation_steps_is_settable() -> None:
    assert RunConfig(max_generation_steps=1000).max_generation_steps == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_run_config.py -v`
Expected: FAIL — `TypeError: RunConfig.__init__() got an unexpected keyword argument 'max_generation_steps'` (or `AttributeError` on the default-value test).

- [ ] **Step 3: Add the field**

In `src/neuroarena/config/run_config.py`, update the dataclass:

```python
@dataclass
class RunConfig:
    """Serializable run configuration.

    Phase 0 holds the envelope. Phase 4 adds a minimal generation-ceiling stub ahead of
    Phase 5's full config-knob classification pass (see Phase 4 doc's Revision history,
    2026-09-13) — a stub/reserved field per `../../docs/WORKFLOW.md`'s "stay within the
    current phase" rule. Concrete run parameters beyond this (population size, sim speed,
    Objective selection, ...) are added in Phase 5.
    """

    master_seed: int = 0
    max_generation_steps: int = 300_000
    schema_version: int = SCHEMA_VERSION
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_run_config.py tests/config/test_serialization.py -v`
Expected: PASS — `test_serialization.py` must still pass unchanged since `dumps`/`loads` operate generically over dataclass fields.

- [ ] **Step 5: Add the `neat-python` dependency**

In `pyproject.toml`, change:

```toml
dependencies = ["numpy>=1.26", "arcade>=3.0"]
```

to:

```toml
dependencies = ["numpy>=1.26", "arcade>=3.0", "neat-python>=0.92,<1.0"]
```

Run: `uv sync`
Expected: `neat-python` installs into `.venv`; `uv run python -c "import neat; print(neat.__version__)"` prints a version.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/neuroarena/config/run_config.py tests/config/test_run_config.py
git commit -m "feat(phase-4): add RunConfig.max_generation_steps stub and neat-python dependency"
```

---

## Task 2: `DummyObjective` test double

**Files:**
- Modify: `tests/interfaces/doubles.py`
- Test: `tests/interfaces/test_protocols.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DummyObjective` (satisfies `neuroarena.interfaces.protocols.Objective`) — reused by Tasks 5, 6, 8's tests. `fitness()` counts steps survived; never signals early stop.

- [ ] **Step 1: Write the failing test**

Append to `tests/interfaces/test_protocols.py` (check its existing imports first and add `Objective` and `DummyObjective` to them):

```python
def test_dummy_objective_satisfies_the_objective_protocol() -> None:
    obj = DummyObjective()
    assert isinstance(obj, Objective)


def test_dummy_objective_fitness_counts_steps() -> None:
    obj = DummyObjective()
    obj.reset()
    for _ in range(3):
        obj.update(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32), False, False, {})
    assert obj.fitness() == 3.0
    assert obj.should_stop() is False


def test_dummy_objective_resets_between_episodes() -> None:
    obj = DummyObjective()
    obj.reset()
    obj.update(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32), False, False, {})
    obj.reset()
    assert obj.fitness() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/interfaces/test_protocols.py -v`
Expected: FAIL — `ImportError: cannot import name 'DummyObjective'`.

- [ ] **Step 3: Add `DummyObjective`**

Append to `tests/interfaces/doubles.py`:

```python
class DummyObjective:
    """Minimal Objective double: fitness = number of `update()` calls (steps survived).
    Never signals should_stop() — episodes only end via the env's own terminated/truncated."""

    def __init__(self) -> None:
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._steps += 1

    def should_stop(self) -> bool:
        return False

    def step_reward(self) -> float:
        return 1.0

    def fitness(self) -> float:
        return float(self._steps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/interfaces/test_protocols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/interfaces/doubles.py tests/interfaces/test_protocols.py
git commit -m "test(phase-4): add DummyObjective test double"
```

---

## Task 3: `build_neat_config` — sizing a bundled NEAT config to an `Environment`

**Files:**
- Create: `src/neuroarena/backends/__init__.py`
- Create: `src/neuroarena/backends/neat/__init__.py`
- Create: `src/neuroarena/backends/neat/config.py`
- Test: `tests/backends/__init__.py`
- Test: `tests/backends/neat/__init__.py`
- Test: `tests/backends/neat/test_config.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces.spaces.Box`, `Space` (existing).
- Produces: `build_neat_config(observation_space: Box, action_space: Box, population_size: int) -> neat.Config`, used by Tasks 4 (`GenomeModel`) and 6 (`NeatTrainer`). The returned config's `genome_config.num_inputs`/`num_outputs`/`input_keys`/`output_keys` match the given spaces; `config.pop_size == population_size`.

- [ ] **Step 1: Write the failing test**

Create `tests/backends/__init__.py` and `tests/backends/neat/__init__.py` (empty files).

Create `tests/backends/neat/test_config.py`:

```python
import neat

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.interfaces.spaces import Box


def test_config_sizes_inputs_and_outputs_to_the_given_spaces() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.genome_config.num_inputs == 10
    assert config.genome_config.num_outputs == 2
    assert config.genome_config.input_keys == [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    assert config.genome_config.output_keys == [0, 1]
    assert config.pop_size == 25


def test_config_sizes_to_a_non_default_sensor_shape() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (5,)), Box(-1.0, 1.0, (2,)), population_size=10)
    assert config.genome_config.num_inputs == 5
    assert len(config.genome_config.input_keys) == 5


def test_config_uses_tanh_output_activation() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=25)
    assert config.genome_config.activation_default == "tanh"


def test_config_produces_a_working_genome_and_network() -> None:
    config = build_neat_config(Box(-1.0, 1.0, (10,)), Box(-1.0, 1.0, (2,)), population_size=1)
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    output = net.activate([0.0] * 10)
    assert len(output) == 2
    assert all(-1.0 <= v <= 1.0 for v in output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.backends'`.

- [ ] **Step 3: Implement `build_neat_config`**

Create `src/neuroarena/backends/__init__.py` (empty).

Create `src/neuroarena/backends/neat/__init__.py` (empty).

Create `src/neuroarena/backends/neat/config.py`:

```python
"""Builds a `neat.Config` sized to a given `Environment`'s observation/action spaces, from
a bundled stock-`neat-python` template. See `../../../docs/phases/phase-4-learning-backend-neat.md`
for why the template's own defaults are used as-is (untuned starting point, tunable once
there's a real training loop to observe) and why output activation is `tanh` (smooth
[-1, 1] mapping, no separate clipping step)."""

from __future__ import annotations

import os
import tempfile

import neat

from neuroarena.interfaces.spaces import Box

DEFAULT_CONFIG_TEXT = """\
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
pop_size              = 150
reset_on_extinction   = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.5

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = 10
num_outputs             = 2

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism             = 2
survival_threshold  = 0.2
"""
"""Stock neat-python example config (activation swapped to tanh per the Phase 4 decision).
Written to a temp file at build time — see `build_neat_config` — rather than shipped as a
packaged data file, so it stays a plain, diffable, version-controlled Python string."""


def build_neat_config(observation_space: Box, action_space: Box, population_size: int) -> neat.Config:
    """A `neat.Config` whose genome shape matches `observation_space`/`action_space` and
    whose population size matches `population_size`. Everything else is the bundled
    template's defaults."""
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
    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends tests/backends
git commit -m "feat(phase-4): build_neat_config sizes a bundled NEAT config to an Environment"
```

---

## Task 4: `GenomeModel` — one genome as a `Model`

**Files:**
- Create: `src/neuroarena/backends/neat/model.py`
- Test: `tests/backends/neat/test_model.py`

**Interfaces:**
- Consumes: `build_neat_config` (Task 3); `neuroarena.interfaces.protocols.Model`; `neuroarena.interfaces.spaces.Box`.
- Produces: `GenomeModel(genome, neat_config, observation_space, action_space)` satisfying `Model`. Consumed by Task 5's `_evaluate_genome`.

- [ ] **Step 1: Write the failing test**

Create `tests/backends/neat/test_model.py`:

```python
import numpy as np

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.backends.neat.model import GenomeModel
from neuroarena.interfaces.protocols import Model
from neuroarena.interfaces.spaces import Box


def _genome(config):
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    return genome


def test_genome_model_satisfies_the_model_protocol() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    assert isinstance(model, Model)
    assert model.observation_space == obs_space
    assert model.action_space == action_space


def test_act_returns_action_shaped_output_within_bounds() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    action = model.act(np.zeros(10, dtype=np.float32))
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_reset_is_a_no_op_and_does_not_raise() -> None:
    obs_space = Box(-1.0, 1.0, (10,))
    action_space = Box(-1.0, 1.0, (2,))
    config = build_neat_config(obs_space, action_space, population_size=1)
    model = GenomeModel(_genome(config), config, obs_space, action_space)
    model.reset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.backends.neat.model'`.

- [ ] **Step 3: Implement `GenomeModel`**

Create `src/neuroarena/backends/neat/model.py`:

```python
"""Wraps one NEAT genome as a `neuroarena.interfaces.protocols.Model`: a feed-forward
network whose output layer is `tanh`-activated (per `build_neat_config`'s template), so
`act()`'s output already lies in `[-1, 1]` with no separate clipping step (Phase 4 doc)."""

from __future__ import annotations

from typing import Any

import neat
import numpy as np

from neuroarena.interfaces.spaces import Space


class GenomeModel:
    """Satisfies `Model`. No hidden state: `reset()` is a no-op since `neat-python`
    feed-forward networks carry none across steps (Phase 0's `Model.reset()` docstring:
    "no-op for feedforward NEAT nets")."""

    def __init__(
        self,
        genome: Any,
        neat_config: neat.Config,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self._network = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        raw = self._network.activate(observation.tolist())
        return np.array(raw, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/model.py tests/backends/neat/test_model.py
git commit -m "feat(phase-4): GenomeModel wraps a NEAT genome as a Model"
```

---

## Task 5: Isolated per-genome evaluation under a shared step budget

**Files:**
- Create: `src/neuroarena/backends/neat/evaluation.py`
- Test: `tests/backends/neat/test_evaluation.py`

**Interfaces:**
- Consumes: `neuroarena.interfaces.protocols.{Model, Environment, Objective}`; `tests/interfaces/doubles.py::{DummyEnvironment, DummyModel, DummyObjective}` (test-only).
- Produces: `StepBudget` (mutable `remaining: int` counter, shared across a whole generation's genomes) and `evaluate_genome(model, env, objective, step_budget, seed) -> EvaluationResult` (`EvaluationResult.fitness: float`, `EvaluationResult.final_info: dict[str, Any]`). Consumed by Task 6's `NeatTrainer`.

- [ ] **Step 1: Write the failing test**

Create `tests/backends/neat/test_evaluation.py`:

```python
from tests.interfaces.doubles import DummyEnvironment, DummyModel, DummyObjective

from neuroarena.backends.neat.evaluation import StepBudget, evaluate_genome


def test_evaluate_genome_runs_until_env_truncates() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=100)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    # DummyEnvironment truncates after 5 steps (see tests/interfaces/doubles.py).
    assert result.fitness == 5.0
    assert result.final_info == {"t": 5}
    assert budget.remaining == 95


def test_evaluate_genome_force_truncates_when_budget_runs_out_mid_episode() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=3)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 3.0
    assert budget.remaining == 0


def test_evaluate_genome_scores_zero_when_budget_is_already_exhausted() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=0)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 0.0
    assert result.final_info == {}


def test_evaluate_genome_stops_early_when_objective_says_stop() -> None:
    class StopsImmediately(DummyObjective):
        def should_stop(self) -> bool:
            return True

    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = StopsImmediately()
    budget = StepBudget(remaining=100)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 1.0
    assert budget.remaining == 99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.backends.neat.evaluation'`.

- [ ] **Step 3: Implement `evaluate_genome`**

Create `src/neuroarena/backends/neat/evaluation.py`:

```python
"""Runs one genome's episode in isolation on a shared `Environment`, under a
per-generation step budget that force-truncates it the instant the budget runs out —
see the Phase 4 doc's "Generations always terminate" and "force-truncated ... scored on
partial progress" requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuroarena.interfaces.protocols import Environment, Model, Objective


@dataclass
class StepBudget:
    """Mutable counter shared across every genome evaluated within one generation."""

    remaining: int


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    final_info: dict[str, Any] = field(default_factory=dict)


def evaluate_genome(
    model: Model,
    env: Environment,
    objective: Objective,
    step_budget: StepBudget,
    seed: int,
) -> EvaluationResult:
    """One episode for `model` on `env`, scored by `objective` (reset first, per Phase 0's
    per-episode `Objective.reset()`/`Model.reset()` hooks). Consumes one unit of
    `step_budget` per `env.step()` call; if the budget is already at zero, the genome is
    scored on its freshly-reset state with zero steps taken — this is what lets a whole
    generation's total step count never exceed the ceiling regardless of population size."""
    observation = env.reset(seed=seed)
    model.reset()
    objective.reset()
    info: dict[str, Any] = {}
    while True:
        if step_budget.remaining <= 0:
            break
        action = model.act(observation)
        observation, terminated, truncated, info = env.step(action)
        step_budget.remaining -= 1
        objective.update(observation, action, terminated, truncated, info)
        if terminated or truncated or objective.should_stop():
            break
    return EvaluationResult(fitness=objective.fitness(), final_info=info)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/evaluation.py tests/backends/neat/test_evaluation.py
git commit -m "feat(phase-4): isolated per-genome evaluation under a shared step budget"
```

---

## Task 6: `NeatTrainer.run()` — the per-generation loop

**Files:**
- Create: `src/neuroarena/backends/neat/trainer.py`
- Test: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `build_neat_config` (Task 3), `GenomeModel` (Task 4), `StepBudget`/`evaluate_genome` (Task 5), `neuroarena.interfaces.protocols.{Trainer, TrainingUpdate, Environment, Objective}`, `neuroarena.config.RunConfig`, `tests/interfaces/doubles.py::{DummyEnvironment, DummyObjective}`.
- Produces: `NeatTrainer(make_env, objective, config, *, population_size=150, champion_dir=None)` satisfying `Trainer`'s `run()` (this task) — `save_checkpoint`/`load_checkpoint` are Task 7.

- [ ] **Step 1: Write the failing test**

Create `tests/backends/neat/test_trainer.py`:

```python
from pathlib import Path

from tests.interfaces.doubles import DummyEnvironment, DummyObjective

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.interfaces.protocols import Trainer, TrainingUpdate


def test_neat_trainer_satisfies_the_trainer_protocol() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    assert isinstance(trainer, Trainer)


def test_run_yields_a_training_update_per_generation() -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    updates = []
    run = trainer.run()
    for _ in range(3):
        updates.append(next(run))
    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert isinstance(u, TrainingUpdate)
        assert u.population_size == 5
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness


def test_generation_ceiling_bounds_total_steps_per_generation() -> None:
    # DummyEnvironment truncates each episode after 5 steps; 5 genomes x 5 steps = 25 steps
    # if unbounded. Cap the generation well below that and confirm it still completes.
    config = RunConfig(max_generation_steps=7)
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), config, population_size=5)
    update = next(trainer.run())
    assert update.population_size == 5
    assert update.worst_fitness <= 5.0


def test_champion_dir_saves_one_genome_file_per_generation(tmp_path: Path) -> None:
    champion_dir = tmp_path / "champions"
    trainer = NeatTrainer(
        DummyEnvironment,
        DummyObjective(),
        RunConfig(),
        population_size=5,
        champion_dir=champion_dir,
    )
    run = trainer.run()
    next(run)
    next(run)
    saved = sorted(p.name for p in champion_dir.iterdir())
    assert saved == ["gen_00000.pkl", "gen_00001.pkl"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuroarena.backends.neat.trainer'`.

- [ ] **Step 3: Implement `NeatTrainer.run()`**

Create `src/neuroarena/backends/neat/trainer.py`:

```python
"""The NEAT `Trainer`: drives one `neat.Population` a generation at a time, evaluating
each genome in isolation on a shared `Environment` under a per-generation step budget.
See `../../../docs/phases/phase-4-learning-backend-neat.md` for the requirements this
implements, in particular why track switches only take effect at a generation boundary
and why a force-truncated genome is scored on partial progress rather than penalised."""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import neat

from neuroarena.backends.neat.config import build_neat_config
from neuroarena.backends.neat.evaluation import EvaluationResult, StepBudget, evaluate_genome
from neuroarena.backends.neat.model import GenomeModel
from neuroarena.interfaces.protocols import TrainingUpdate

if TYPE_CHECKING:
    from neuroarena.config import RunConfig
    from neuroarena.interfaces.protocols import Environment, Objective

_CHECKPOINT_SCHEMA_VERSION = 1


class NeatTrainer:
    """Satisfies `Trainer`. `make_env` is rebuilt once per generation (not per genome) so
    every genome within one generation runs on the same `Environment` instance — cheap
    (Phase 3's `CarEnvironment.reset()` already gives each genome a fresh `Game`) and the
    natural point for a live track switch (Phase 5) to take effect."""

    def __init__(
        self,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
        *,
        population_size: int = 150,
        neat_config: neat.Config | None = None,
        champion_dir: Path | None = None,
    ) -> None:
        self._make_env = make_env
        self._objective = objective
        self._config = config
        self._champion_dir = champion_dir
        if champion_dir is not None:
            champion_dir.mkdir(parents=True, exist_ok=True)

        env = make_env()
        self._neat_config = neat_config or build_neat_config(
            env.observation_space, env.action_space, population_size
        )
        self._population = neat.Population(self._neat_config)
        self._env = env
        self._generation = 0
        self._total_sim_steps = 0.0
        self._wall_start = time.perf_counter()

    def run(self) -> Iterator[TrainingUpdate]:
        while True:
            yield self._run_one_generation()

    def _run_one_generation(self) -> TrainingUpdate:
        self._env = self._make_env()
        step_budget = StepBudget(remaining=self._config.max_generation_steps)
        fitnesses: dict[int, float] = {}
        champion = _ChampionTracker()

        def fitness_function(genomes: list[tuple[int, Any]], neat_config: neat.Config) -> None:
            for genome_id, genome in genomes:
                model = GenomeModel(genome, neat_config, self._env.observation_space, self._env.action_space)
                seed = self._config.master_seed + genome_id
                result = evaluate_genome(model, self._env, self._objective, step_budget, seed)
                genome.fitness = result.fitness
                fitnesses[genome_id] = result.fitness
                champion.consider(genome, result)

        self._population.run(fitness_function, 1)

        steps_used = self._config.max_generation_steps - step_budget.remaining
        self._total_sim_steps += steps_used

        if self._champion_dir is not None and champion.genome is not None:
            path = self._champion_dir / f"gen_{self._generation:05d}.pkl"
            path.write_bytes(pickle.dumps(champion.genome))

        values = list(fitnesses.values())
        update = TrainingUpdate(
            progress_index=self._generation,
            best_fitness=max(values),
            mean_fitness=sum(values) / len(values),
            worst_fitness=min(values),
            population_size=len(values),
            champion_metrics=champion.metrics(),
            sim_time=self._total_sim_steps,
            wall_time=time.perf_counter() - self._wall_start,
        )
        self._generation += 1
        return update


class _ChampionTracker:
    """Tracks the best-fitness genome seen so far within one generation's fitness_function
    call, and the `info` dict from its evaluation, for `TrainingUpdate.champion_metrics`."""

    def __init__(self) -> None:
        self.genome: Any = None
        self._fitness = float("-inf")
        self._info: dict[str, Any] = {}

    def consider(self, genome: Any, result: EvaluationResult) -> None:
        if result.fitness > self._fitness:
            self._fitness = result.fitness
            self.genome = genome
            self._info = result.final_info

    def metrics(self) -> dict[str, float]:
        # Only float-valued info keys belong in TrainingUpdate.champion_metrics (Phase 0:
        # `dict[str, float]`) — `track_id` (str) is an identifier, not a metric, and is
        # excluded. Car-env-specific keys (Phase 3's `info` contract); a future second
        # game's Objective/info would need its own key set here.
        return {
            "crashed": float(self._info.get("crashed", 0.0)),
            "progress": float(self._info.get("progress", 0.0)),
            "lap_progress": float(self._info.get("lap_progress", 0.0)),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-4): NeatTrainer drives one generation at a time with champion capture"
```

---

## Task 7: Checkpoint save/load

**Files:**
- Modify: `src/neuroarena/backends/neat/trainer.py`
- Modify: `tests/backends/neat/test_trainer.py`

**Interfaces:**
- Consumes: `NeatTrainer` internals from Task 6 (`self._population`, `self._neat_config`, `self._generation`).
- Produces: `NeatTrainer.save_checkpoint(path: Path) -> None` and `NeatTrainer.load_checkpoint(path, make_env, objective, config) -> NeatTrainer` (classmethod), completing the `Trainer` protocol.

- [ ] **Step 1: Write the failing test**

Append to `tests/backends/neat/test_trainer.py`:

```python
def test_checkpoint_round_trip_resumes_generation_count(tmp_path: Path) -> None:
    trainer = NeatTrainer(DummyEnvironment, DummyObjective(), RunConfig(), population_size=5)
    run = trainer.run()
    next(run)
    next(run)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    restored = NeatTrainer.load_checkpoint(
        checkpoint_path, DummyEnvironment, DummyObjective(), RunConfig()
    )
    restored_update = next(restored.run())
    assert restored_update.progress_index == 2  # continues from generation 2, not 0
    assert restored_update.population_size == 5


def test_checkpoint_rejects_unknown_schema_version(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema_version"):
        NeatTrainer.load_checkpoint(path, DummyEnvironment, DummyObjective(), RunConfig())
```

Add `import pytest` to the top of `tests/backends/neat/test_trainer.py` alongside the existing imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v -k checkpoint`
Expected: FAIL — `AttributeError: 'NeatTrainer' object has no attribute 'save_checkpoint'`.

- [ ] **Step 3: Implement checkpoint save/load**

In `src/neuroarena/backends/neat/trainer.py`, add `import pickle` is already present; add these methods to `NeatTrainer` (after `_run_one_generation`):

```python
    def save_checkpoint(self, path: Path) -> None:
        """Pickles the NEAT-specific state (Phase 4 doc: "population, genomes, innovation
        history"), plus Python's global `random` state so a resumed run's mutation/crossover
        draws continue the same sequence rather than silently diverging. Checkpoint file
        format/retention is Phase 6's concern; this is a working save/load pair, not a claim
        on the eventual platform format."""
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "neat_config": self._neat_config,
            "population": self._population.population,
            "species": self._population.species,
            "generation": self._generation,
            "random_state": random.getstate(),
        }
        path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        make_env: Callable[[], Environment],
        objective: Objective,
        config: RunConfig,
    ) -> NeatTrainer:
        payload = pickle.loads(path.read_bytes())
        version = payload.get("schema_version")
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unreadable NEAT checkpoint schema_version {version!r} "
                f"(supports {_CHECKPOINT_SCHEMA_VERSION})"
            )
        random.setstate(payload["random_state"])
        neat_config = payload["neat_config"]
        trainer = cls(make_env, objective, config, neat_config=neat_config)
        trainer._population = neat.Population(
            neat_config,
            initial_state=(payload["population"], payload["species"], payload["generation"]),
        )
        trainer._generation = payload["generation"]
        return trainer
```

Add `import random` to the top of `src/neuroarena/backends/neat/trainer.py`, alongside `import pickle`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_trainer.py -v`
Expected: PASS (all tests in the file, not just checkpoint ones — confirm no regression)

- [ ] **Step 5: Commit**

```bash
git add src/neuroarena/backends/neat/trainer.py tests/backends/neat/test_trainer.py
git commit -m "feat(phase-4): NeatTrainer checkpoint save/load"
```

---

## Task 8: Import-hygiene guard + headless end-to-end smoke test

**Files:**
- Create: `tests/backends/neat/test_import_hygiene.py`
- Create: `tests/backends/neat/test_end_to_end.py`

**Interfaces:**
- Consumes: `NeatTrainer` (Tasks 6-7), the real `neuroarena.sim.car_env.CarEnvironment` (Phase 3), `tests/interfaces/doubles.py::DummyObjective`.
- Produces: nothing new — verification only.

- [ ] **Step 1: Write the import-hygiene test**

Create `tests/backends/neat/test_import_hygiene.py`:

```python
import json
import subprocess
import sys


def test_backends_neat_does_not_import_forbidden_dependencies() -> None:
    code = (
        "import sys, json; "
        "import neuroarena.backends.neat.config, neuroarena.backends.neat.model, "
        "neuroarena.backends.neat.evaluation, neuroarena.backends.neat.trainer; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into neuroarena.backends.neat's import graph"


def test_backends_neat_does_not_import_sim_or_render() -> None:
    code = (
        "import sys, json; "
        "import neuroarena.backends.neat.config, neuroarena.backends.neat.model, "
        "neuroarena.backends.neat.evaluation, neuroarena.backends.neat.trainer; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in loaded:
        assert not forbidden.startswith("neuroarena.sim"), (
            "neuroarena.backends.neat must consume only the Environment/Model/Objective "
            "interfaces, not neuroarena.sim directly"
        )
        assert not forbidden.startswith("neuroarena.render")
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_import_hygiene.py -v`
Expected: PASS (this guards against regression, so it should pass immediately given Tasks 3-7's code — if it fails, an earlier task imported something it shouldn't have; fix that task's imports before continuing)

- [ ] **Step 3: Write the end-to-end smoke test**

Create `tests/backends/neat/test_end_to_end.py`:

```python
from pathlib import Path

from tests.interfaces.doubles import DummyObjective

from neuroarena.backends.neat.trainer import NeatTrainer
from neuroarena.config import RunConfig
from neuroarena.sim.car_env import CarEnvironment
from neuroarena.sim.track import Facing, GridCell, TileKind, Track


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


def _make_car_env() -> CarEnvironment:
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    return CarEnvironment(track, track_id="phase-4-smoke-test")


def test_neat_trainer_runs_headless_on_the_real_car_environment(tmp_path: Path) -> None:
    config = RunConfig(max_generation_steps=2000)
    trainer = NeatTrainer(
        _make_car_env,
        DummyObjective(),
        config,
        population_size=8,
        champion_dir=tmp_path / "champions",
    )
    run = trainer.run()
    updates = [next(run) for _ in range(3)]

    assert [u.progress_index for u in updates] == [0, 1, 2]
    for u in updates:
        assert u.population_size == 8
        assert u.best_fitness >= u.mean_fitness >= u.worst_fitness
        assert set(u.champion_metrics) == {"crashed", "progress", "lap_progress"}
    saved = sorted((tmp_path / "champions").iterdir())
    assert len(saved) == 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backends/neat/test_end_to_end.py -v -s`
Expected: PASS, completing in well under a minute (population 8, 3 generations, small track) with no window opening (headless — no `arcade` import triggered, confirmed separately by Step 2's hygiene test).

- [ ] **Step 5: Run the full test suite, lint, and type-check**

Run: `uv run pytest -v`
Expected: PASS, all tests (Phase 0-4).

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

Run: `uv run mypy src`
Expected: clean (`--strict`, per `pyproject.toml`). Note `neat-python` ships no type stubs — if `mypy` complains about missing stubs for the `neat` import, add:

```toml
[[tool.mypy.overrides]]
module = "neat.*"
ignore_missing_imports = true
```

to `pyproject.toml`'s mypy overrides section.

- [ ] **Step 6: Commit**

```bash
git add tests/backends/neat/test_import_hygiene.py tests/backends/neat/test_end_to_end.py pyproject.toml
git commit -m "test(phase-4): import-hygiene guard and headless end-to-end NEAT smoke test"
```

---

## Post-plan: update the phase doc

After all 8 tasks are committed, update `../../phases/phase-4-learning-backend-neat.md`'s "Implementation plan" section to link this file and summarize the tasks (matching Phase 3's doc pattern), and add a dated Revision history entry noting implementation is complete, mentioning test count and any findings discovered during implementation (per `../../WORKFLOW.md`'s decision-provenance rule) — the same way Phase 3's 2026-09-13 entry recorded its centerline-projection finding.
