# Phase 3 — Sensors & Environment Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the car game a 10-float observation, a 2-float continuous action, and episode semantics (`reset`/`step`/`terminated`/`truncated`/`info`), wiring it to the Phase 0 `Environment` Protocol.

**Architecture:** Two new pure, dependency-free `sim` modules — `raycast.py` (ray/segment geometry) and `observation.py` (assembles the 10-float vector from raycasts + speed + last action) — sit alongside Phase 1's `physics.py`/`track.py`/`collision.py`. A new `CarEnvironment` class in `sim/car_env.py` wraps Phase 1's existing `Game` (unmodified) and adds exactly what `Game.tick()` deliberately left out: episode reset, `terminated` on boundary contact, `truncated` on a step budget, and an `info` dict carrying track-progress facts. Two small additions to Phase 1's own modules (`track.py` gets a loop-length function, `collision.py` gets a boundary-distance query) supply data `CarEnvironment` needs but `Game` has no reason to expose itself.

**Tech Stack:** Python 3.12+, `numpy` (already a dependency, used for the observation array), `pytest`, `ruff`, `mypy` strict — matching Phase 0/1/2. No new dependencies.

**Spec:** `docs/phases/phase-3-sensors-env-interface.md` (FINALIZED 2026-09-12)

## Global Constraints

- Observation: **10 floats** — 7 raycast proximities + normalised speed + last action (2). `observation_space`/`action_space` are `Box(-1.0, 1.0, shape)` descriptors from `neuroarena.interfaces.spaces`.
- Raycast fan (default): angles `[-75, -45, -20, 0, 20, 45, 75]°` relative to heading. Value is proximity `1 - effective_distance / range`, clamped to `[0, 1]` — a miss within `range` reads as `0`, same as a hit exactly at `range`.
- Ray range: `range = 300 + 200 * speed_norm`, clamped to `[0, 600]`.
- Speed normalisation: signed, single scale — `speed_norm = speed / max_speed` (`750`), so reverse (`max_reverse_speed = 300`) reads down to about `-0.4`.
- Last action is the **raw commanded** `(steering, throttle)` from the previous `step()` call, defaulting to `(0.0, 0.0)` right after `reset()` — never reconstructed from physics.
- `terminated` fires on **any contact with the track boundary** (Phase 1's hard collider is unchanged — the car still cannot cross it). No separate off-track state, no deviation buffer, no in-world "finish".
- `truncated` fires when a configurable `max_episode_steps` is reached (starting default `3000` ticks = 50s at `TICK_DT`; Phase 5 wires this through `RunConfig` later).
- `info` carries: `crashed: bool` (mirrors `terminated`), `track_id: str`, `progress: float` (signed, continuous, unwrapped distance travelled along the car's own path since spawn — not a geometric projection onto the track centerline), `lap_progress: float` (`progress / loop_length`, so `1.0` = one lap).
- `neuroarena.sim` (all modules, including the three new ones) imports no `arcade`, `gymnasium`, `pygame`, `stable_baselines3`, or `torch` — enforced by `tests/sim/test_import_hygiene.py`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy` clean before every commit; commit at the end of each task.

---

## File structure

```
src/neuroarena/sim/raycast.py           # NEW: cast_ray()/cast_rays() — pure ray/segment geometry
src/neuroarena/sim/observation.py       # NEW: SensorConfig, speed_norm(), build_observation(), observation_space_for()
src/neuroarena/sim/track.py             # MODIFY: add track_loop_length()
src/neuroarena/sim/collision.py         # MODIFY: add distance_to_boundary()
src/neuroarena/sim/car_env.py           # NEW: CarEnvironmentConfig, CarEnvironment (the Environment Protocol impl)

tests/sim/test_raycast.py               # NEW
tests/sim/test_observation.py           # NEW
tests/sim/test_track.py                 # MODIFY: add track_loop_length tests
tests/sim/test_collision.py             # MODIFY: add distance_to_boundary tests
tests/sim/test_car_env.py               # NEW
tests/sim/test_import_hygiene.py        # MODIFY: cover raycast.py, observation.py, car_env.py
```

Nothing in `sim/game.py` or `sim/physics.py` changes — `CarEnvironment` composes `Game` unmodified, exactly as Phase 1's own docstring anticipated ("that's Phase 3's `Environment` concern"). The renderer (`neuroarena.render`) is untouched; `CarEnvironment` has no rendering path.

---

### Task 1 — Boundary raycasting geometry

Pure ray/segment intersection math, independent of `Track`/`TileKind` — it only knows about points, segments, an origin, and an angle. `observation.py` (Task 3) is the only consumer.

**Files:**
- Create: `src/neuroarena/sim/raycast.py`
- Test: `tests/sim/test_raycast.py`

**Interfaces:**
- Consumes: `Point`, `Segment` type aliases from `neuroarena.sim.track`.
- Produces: `cast_ray(origin: Point, angle_rad: float, max_dist: float, boundary: list[Segment]) -> float | None`; `cast_rays(origin: Point, heading: float, angles_deg: Sequence[float], max_dist: float, boundary: list[Segment], car_radius: float) -> list[float]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_raycast.py
import math

import pytest

from neuroarena.sim.raycast import cast_ray, cast_rays


def test_cast_ray_hits_wall_directly_ahead() -> None:
    boundary = [((100.0, -50.0), (100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance == pytest.approx(100.0)


def test_cast_ray_returns_none_beyond_max_dist() -> None:
    boundary = [((300.0, -50.0), (300.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_ignores_walls_behind_the_origin() -> None:
    boundary = [((-100.0, -50.0), (-100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_respects_segment_span() -> None:
    # Wall segment ends well short of the ray's y=0 line -> no intersection.
    boundary = [((100.0, 10.0), (100.0, 50.0))]
    distance = cast_ray((0.0, 0.0), 0.0, 200.0, boundary)
    assert distance is None


def test_cast_ray_at_45_degrees() -> None:
    boundary = [((100.0, -200.0), (100.0, 200.0))]
    distance = cast_ray((0.0, 0.0), math.pi / 4, 200.0, boundary)
    assert distance == pytest.approx(100.0 * math.sqrt(2))


def test_cast_rays_returns_one_proximity_per_angle() -> None:
    boundary = [((100.0, -50.0), (100.0, 50.0))]
    proximities = cast_rays((0.0, 0.0), 0.0, (-10.0, 0.0, 10.0), 200.0, boundary, car_radius=0.0)
    assert len(proximities) == 3
    assert proximities[1] == pytest.approx(1.0 - 100.0 / 200.0)


def test_cast_rays_clamps_no_hit_to_zero() -> None:
    proximities = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, [], car_radius=0.0)
    assert proximities == [0.0]


def test_cast_rays_reads_one_when_bumper_touches_wall() -> None:
    boundary = [((50.0, -50.0), (50.0, 50.0))]
    proximities = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, boundary, car_radius=50.0)
    assert proximities[0] == pytest.approx(1.0)


def test_cast_rays_heading_rotates_the_fan() -> None:
    # A wall to the north (+y): only visible once heading points the ray that way.
    boundary = [((-50.0, 100.0), (50.0, 100.0))]
    facing_east = cast_rays((0.0, 0.0), 0.0, (0.0,), 200.0, boundary, car_radius=0.0)
    facing_north = cast_rays((0.0, 0.0), math.pi / 2, (0.0,), 200.0, boundary, car_radius=0.0)
    assert facing_east == [0.0]
    assert facing_north[0] == pytest.approx(1.0 - 100.0 / 200.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_raycast.py -v`
Expected: FAIL with `ModuleNotFoundError` (`neuroarena.sim.raycast` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/sim/raycast.py
"""Boundary-proximity raycasting for the car's Phase 3 observation. Pure geometry — no
knowledge of Track/TileKind, just points, an origin, and an angle (see the Phase 3 doc,
"Raycasts")."""

from __future__ import annotations

import math
from collections.abc import Sequence

from neuroarena.sim.track import Point, Segment


def cast_ray(
    origin: Point, angle_rad: float, max_dist: float, boundary: list[Segment]
) -> float | None:
    """Distance from `origin` to the nearest boundary intersection within `max_dist` along
    `angle_rad` (world-frame radians, matching `CarState.heading`'s convention), or `None`
    if nothing is that close."""
    direction = (math.cos(angle_rad), math.sin(angle_rad))
    nearest: float | None = None
    for segment in boundary:
        hit = _ray_segment_intersection(origin, direction, segment)
        if hit is not None and hit <= max_dist and (nearest is None or hit < nearest):
            nearest = hit
    return nearest


def cast_rays(
    origin: Point,
    heading: float,
    angles_deg: Sequence[float],
    max_dist: float,
    boundary: list[Segment],
    car_radius: float,
) -> list[float]:
    """One proximity value per angle in `angles_deg` (degrees, relative to `heading`):
    `1 - effective_distance / max_dist`, clamped to `[0, 1]`. `effective_distance` measures
    from the car's bumper (its bounding circle, radius `car_radius` — see `collision.py`),
    not its center, so `1.0` means the same "touching the wall" state
    `collision.distance_to_boundary` reports."""
    proximities = []
    for deg in angles_deg:
        angle = heading + math.radians(deg)
        raw = cast_ray(origin, angle, max_dist + car_radius, boundary)
        if raw is None:
            proximities.append(0.0)
        else:
            effective = max(0.0, raw - car_radius)
            proximities.append(max(0.0, min(1.0, 1.0 - effective / max_dist)))
    return proximities


def _ray_segment_intersection(
    origin: Point, direction: Point, segment: Segment
) -> float | None:
    """Parametric ray/segment intersection. `direction` must be a unit vector, so the
    returned value is a true Euclidean distance. `None` if parallel, behind `origin`, or
    outside the segment's span."""
    ox, oy = origin
    dx, dy = direction
    (ax, ay), (bx, by) = segment
    sx, sy = bx - ax, by - ay
    denom = sx * dy - sy * dx
    if abs(denom) < 1e-12:
        return None
    ex, ey = ax - ox, ay - oy
    t = (sx * ey - sy * ex) / denom
    s = (dx * ey - dy * ex) / denom
    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_raycast.py -v`
Expected: PASS (9 cases)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/neuroarena/sim/raycast.py && uv run ruff check src/neuroarena/sim/raycast.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/neuroarena/sim/raycast.py tests/sim/test_raycast.py
git commit -m "feat(phase-3): boundary raycasting geometry"
```

---

### Task 2 — Track loop length and boundary-contact detection

Two small, independent additions to Phase 1's own modules — data `CarEnvironment` (Task 4) needs but that neither `Track` nor `Game` had a reason to compute for themselves.

**Files:**
- Modify: `src/neuroarena/sim/track.py` (add near `boundary_segments`)
- Modify: `src/neuroarena/sim/collision.py` (add near `_closest_point_on_segment`)
- Test: `tests/sim/test_track.py`, `tests/sim/test_collision.py` (add to the existing files)

**Interfaces:**
- Produces: `track_loop_length(track: Track) -> float` in `neuroarena.sim.track`; `distance_to_boundary(point: Point, boundary: list[Segment]) -> float` in `neuroarena.sim.collision`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_track.py — add `track_loop_length` to the existing
# `from neuroarena.sim.track import (...)` block at the top of the file, then add:
def test_loop_length_sums_straights_and_curve_arcs() -> None:
    # _rounded_rectangle(): 6 straight cells + 4 curve cells, CELL_SIZE = 512.
    track = Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)
    expected = 6 * 512.0 + 4 * (math.pi / 2) * 256.0
    assert track_loop_length(track) == pytest.approx(expected)
```

```python
# tests/sim/test_collision.py — add `distance_to_boundary` to the existing
# `from neuroarena.sim.collision import resolve_collision` line, then add:
def test_distance_to_boundary_zero_when_on_the_wall() -> None:
    assert distance_to_boundary((0.0, 0.0), [WALL]) == pytest.approx(0.0)


def test_distance_to_boundary_positive_when_clear() -> None:
    assert distance_to_boundary((0.0, 50.0), [WALL]) == pytest.approx(50.0)


def test_distance_to_boundary_is_the_nearest_of_several_segments() -> None:
    near = ((-10.0, 10.0), (10.0, 10.0))
    far = ((-10.0, 100.0), (10.0, 100.0))
    assert distance_to_boundary((0.0, 0.0), [far, near]) == pytest.approx(10.0)
```

Both files already have `import math`, `import pytest`, and (in `test_track.py`) the `_rounded_rectangle()` fixture at the top — no new imports needed beyond the two additions to the existing `from neuroarena.sim.X import (...)` lines noted above.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_track.py tests/sim/test_collision.py -v`
Expected: FAIL with `ImportError` for both `track_loop_length` and `distance_to_boundary`.

- [ ] **Step 3: Implement**

```python
# add to src/neuroarena/sim/track.py, directly below boundary_segments()
def track_loop_length(track: Track) -> float:
    """Total path length of the loop's centerline — radius `cell_size / 2`, the midpoint
    between the inner/outer collision walls `boundary_segments` derives (independent of
    `drivable_width`, which offsets both walls equally). A straight tile contributes
    `cell_size`; a curve tile contributes a quarter-circle arc, `(pi / 2) * (cell_size / 2)`.
    Used to turn Phase 3's raw `progress` into `lap_progress`, a fraction of one lap."""
    straight_count = sum(1 for kind in track.cells.values() if not kind.is_curve)
    curve_count = len(track.cells) - straight_count
    return straight_count * track.cell_size + curve_count * (math.pi / 2) * (track.cell_size / 2)
```

```python
# add to src/neuroarena/sim/collision.py, directly below _closest_point_on_segment()
def distance_to_boundary(point: Point, boundary: list[Segment]) -> float:
    """Shortest distance from `point` to any segment in `boundary`. `resolve_collision`
    clamps a colliding car to exactly `car_radius` from the wall it hit, so comparing this
    against `car_radius` after a tick tells Phase 3's `CarEnvironment` whether that tick's
    `resolve_collision` call actually pushed the car back — i.e. whether it crashed."""
    return min(
        math.hypot(point[0] - closest[0], point[1] - closest[1])
        for closest in (_closest_point_on_segment(point, segment) for segment in boundary)
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_track.py tests/sim/test_collision.py -v`
Expected: PASS

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/neuroarena/sim/track.py src/neuroarena/sim/collision.py && uv run ruff check src/neuroarena/sim/track.py src/neuroarena/sim/collision.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/neuroarena/sim/track.py src/neuroarena/sim/collision.py tests/sim/test_track.py tests/sim/test_collision.py
git commit -m "feat(phase-3): track loop length and boundary-contact detection"
```

---

### Task 3 — Observation vector assembly and sensor config

Combines Task 1's raycasting with speed normalisation and last-action into the full 10-float observation, plus the `SensorConfig` that makes ray count/angles/range configurable at run creation (Phase 3 doc).

**Files:**
- Create: `src/neuroarena/sim/observation.py`
- Test: `tests/sim/test_observation.py`

**Interfaces:**
- Consumes: `cast_rays` (Task 1); `CarState`, `PhysicsConstants` from `neuroarena.sim.physics`; `Segment` from `neuroarena.sim.track`; `Box` from `neuroarena.interfaces.spaces`.
- Produces: `SensorConfig` (frozen dataclass: `ray_angles_deg: tuple[float, ...] = (-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0)`, `range_base: float = 300.0`, `range_k: float = 200.0`, `max_range: float = 600.0`); `speed_norm(speed: float, constants: PhysicsConstants) -> float`; `observation_space_for(sensor_config: SensorConfig) -> Box`; `build_observation(car: CarState, boundary: list[Segment], last_action: tuple[float, float], sensor_config: SensorConfig, constants: PhysicsConstants) -> np.ndarray` (dtype `float32`, shape `(len(ray_angles_deg) + 3,)`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_observation.py
import numpy as np
import pytest

from neuroarena.interfaces.spaces import Box
from neuroarena.sim.observation import (
    SensorConfig,
    build_observation,
    observation_space_for,
    speed_norm,
)
from neuroarena.sim.physics import CarState, PhysicsConstants


def test_speed_norm_forward_reverse_and_rest() -> None:
    constants = PhysicsConstants()
    assert speed_norm(750.0, constants) == pytest.approx(1.0)
    assert speed_norm(-300.0, constants) == pytest.approx(-300.0 / 750.0)
    assert speed_norm(0.0, constants) == pytest.approx(0.0)


def test_observation_space_for_default_sensor_config() -> None:
    assert observation_space_for(SensorConfig()) == Box(-1.0, 1.0, (10,))


def test_observation_space_for_custom_ray_count() -> None:
    config = SensorConfig(ray_angles_deg=(0.0, 90.0))
    assert observation_space_for(config) == Box(-1.0, 1.0, (5,))


def test_build_observation_shape_and_dtype() -> None:
    car = CarState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    obs = build_observation(car, [], (0.0, 0.0), SensorConfig(), PhysicsConstants())
    assert obs.shape == (10,)
    assert obs.dtype == np.float32


def test_build_observation_embeds_speed_and_last_action_in_the_last_three_slots() -> None:
    car = CarState(x=0.0, y=0.0, heading=0.0, speed=375.0)
    obs = build_observation(car, [], (0.5, -0.25), SensorConfig(), PhysicsConstants())
    assert obs[-3] == pytest.approx(375.0 / 750.0)
    assert obs[-2] == pytest.approx(0.5)
    assert obs[-1] == pytest.approx(-0.25)


def test_build_observation_range_grows_with_speed() -> None:
    constants = PhysicsConstants()
    car_radius = constants.car_width / 2
    # Just past the base (speed=0) range of 300, so only a faster car's extended range sees it.
    wall_x = 300.0 + car_radius + 50.0
    boundary = [((wall_x, -1000.0), (wall_x, 1000.0))]
    forward_index = SensorConfig().ray_angles_deg.index(0.0)

    slow = build_observation(
        CarState(x=0.0, y=0.0, heading=0.0, speed=0.0), boundary, (0.0, 0.0),
        SensorConfig(), constants,
    )
    fast = build_observation(
        CarState(x=0.0, y=0.0, heading=0.0, speed=750.0), boundary, (0.0, 0.0),
        SensorConfig(), constants,
    )
    assert slow[forward_index] == pytest.approx(0.0)
    assert fast[forward_index] > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_observation.py -v`
Expected: FAIL with `ModuleNotFoundError` (`neuroarena.sim.observation` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/sim/observation.py
"""Assembles the car's Phase 3 observation vector: raycasts + normalised speed + last
action. See the Phase 3 doc for the numeric constants and normalisation rules this encodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroarena.interfaces.spaces import Box
from neuroarena.sim.physics import CarState, PhysicsConstants
from neuroarena.sim.raycast import cast_rays
from neuroarena.sim.track import Segment


@dataclass(frozen=True)
class SensorConfig:
    """Sensor layout — configurable at run creation (Phase 3 doc); the values below are the
    platform-standard default. A non-default `ray_angles_deg` changes `observation_space`'s
    shape, which is what the Phase 0 compatibility check compares."""

    ray_angles_deg: tuple[float, ...] = (-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0)
    range_base: float = 300.0
    range_k: float = 200.0
    max_range: float = 600.0


def observation_space_for(sensor_config: SensorConfig) -> Box:
    size = len(sensor_config.ray_angles_deg) + 3  # rays + speed + 2 last-action components
    return Box(-1.0, 1.0, (size,))


def speed_norm(speed: float, constants: PhysicsConstants) -> float:
    """Signed, single scale by max forward speed (Phase 3 doc): forward reads `[0, 1]`,
    reverse reads down to about `-constants.max_reverse_speed / constants.max_speed`."""
    return speed / constants.max_speed


def build_observation(
    car: CarState,
    boundary: list[Segment],
    last_action: tuple[float, float],
    sensor_config: SensorConfig,
    constants: PhysicsConstants,
) -> np.ndarray:
    norm_speed = speed_norm(car.speed, constants)
    ray_range = max(
        0.0,
        min(
            sensor_config.range_base + sensor_config.range_k * norm_speed,
            sensor_config.max_range,
        ),
    )
    car_radius = constants.car_width / 2
    proximities = cast_rays(
        (car.x, car.y),
        car.heading,
        sensor_config.ray_angles_deg,
        ray_range,
        boundary,
        car_radius,
    )
    values = [*proximities, norm_speed, last_action[0], last_action[1]]
    return np.array(values, dtype=np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_observation.py -v`
Expected: PASS (6 cases)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/neuroarena/sim/observation.py && uv run ruff check src/neuroarena/sim/observation.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/neuroarena/sim/observation.py tests/sim/test_observation.py
git commit -m "feat(phase-3): observation vector assembly and sensor config"
```

---

### Task 4 — `CarEnvironment`: wiring the game to the Phase 0 `Environment` interface

The deliverable the whole phase exists for: a class satisfying `neuroarena.interfaces.protocols.Environment`, composing Phase 1's `Game` unmodified with Tasks 1-3 and the Task 2 helpers.

**Files:**
- Create: `src/neuroarena/sim/car_env.py`
- Test: `tests/sim/test_car_env.py`
- Modify: `tests/sim/test_import_hygiene.py`

**Interfaces:**
- Consumes: `Game`, `TICK_DT` from `neuroarena.sim.game`; `PhysicsConstants` from `neuroarena.sim.physics`; `Track`, `boundary_segments`, `track_loop_length` from `neuroarena.sim.track` (Task 2); `distance_to_boundary` from `neuroarena.sim.collision` (Task 2); `SensorConfig`, `build_observation`, `observation_space_for` from `neuroarena.sim.observation` (Task 3); `Box`, `Space` from `neuroarena.interfaces.spaces`.
- Produces: `CarEnvironmentConfig` (frozen dataclass: `sensor_config: SensorConfig = SensorConfig()`, `physics_constants: PhysicsConstants = PhysicsConstants()`, `max_episode_steps: int = 3000`); `CarEnvironment` (class: `__init__(self, track: Track, track_id: str, config: CarEnvironmentConfig | None = None)`, `observation_space: Space`, `action_space: Space`, `reset(self, *, seed: int | None = None) -> np.ndarray`, `step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_car_env.py
import numpy as np
import pytest

from neuroarena.interfaces.protocols import Environment
from neuroarena.interfaces.spaces import Box
from neuroarena.sim.car_env import CarEnvironment, CarEnvironmentConfig
from neuroarena.sim.track import Facing, GridCell, Track, TileKind, track_loop_length


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


def _track() -> Track:
    return Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)


def test_car_environment_satisfies_the_environment_protocol() -> None:
    assert isinstance(CarEnvironment(_track(), track_id="abc123"), Environment)


def test_observation_and_action_space_descriptors() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    assert env.observation_space == Box(-1.0, 1.0, (10,))
    assert env.action_space == Box(-1.0, 1.0, (2,))


def test_reset_returns_expected_shape_and_zeroed_last_action() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    obs = env.reset()
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert obs[-2] == 0.0
    assert obs[-1] == 0.0


def test_step_returns_environment_protocol_shaped_tuple() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    obs, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert obs.shape == (10,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["track_id"] == "abc123"
    assert info["crashed"] is terminated


def test_terminated_fires_on_boundary_contact() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    terminated = False
    info: dict[str, object] = {}
    for _ in range(600):
        _, terminated, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated
    assert info["crashed"] is True


def test_truncated_fires_at_max_episode_steps() -> None:
    config = CarEnvironmentConfig(max_episode_steps=5)
    env = CarEnvironment(_track(), track_id="abc123", config=config)
    env.reset()
    truncated = False
    for _ in range(5):
        _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not terminated  # stationary car (zero throttle) never crashes
    assert truncated


def test_progress_increases_when_driving_forward() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    _, terminated1, _, info1 = env.step(np.array([0.0, 1.0], dtype=np.float32))
    _, terminated2, _, info2 = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert not terminated1 and not terminated2
    assert info2["progress"] > info1["progress"]


def test_lap_progress_is_progress_over_loop_length() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    _, terminated, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert not terminated
    assert info["lap_progress"] == pytest.approx(info["progress"] / track_loop_length(_track()))


def test_reset_clears_progress_from_a_previous_episode() -> None:
    env = CarEnvironment(_track(), track_id="abc123")
    env.reset()
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    env.reset()
    _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not terminated
    assert info["progress"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/sim/test_car_env.py -v`
Expected: FAIL with `ModuleNotFoundError` (`neuroarena.sim.car_env` doesn't exist yet).

- [ ] **Step 3: Implement**

```python
# src/neuroarena/sim/car_env.py
"""Phase 3 `Environment` for the car game: wires Phase 1's `Game` to the Phase 0
`Environment` Protocol by adding the episode semantics `Game.tick()` deliberately left out
(reset, `terminated`, `truncated`, `info`) — see `game.py`'s module docstring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuroarena.interfaces.spaces import Box, Space
from neuroarena.sim.collision import distance_to_boundary
from neuroarena.sim.game import TICK_DT, Game
from neuroarena.sim.observation import SensorConfig, build_observation, observation_space_for
from neuroarena.sim.physics import PhysicsConstants
from neuroarena.sim.track import Track, boundary_segments, track_loop_length

_CRASH_EPSILON = 1e-6


@dataclass(frozen=True)
class CarEnvironmentConfig:
    """Starting defaults; Phase 5 wires these through `RunConfig`."""

    sensor_config: SensorConfig = SensorConfig()
    physics_constants: PhysicsConstants = PhysicsConstants()
    max_episode_steps: int = 3000


class CarEnvironment:
    """Satisfies `neuroarena.interfaces.protocols.Environment` for the car game. One
    instance is one episode on one `Track`; call `reset()` to start a new episode on the
    same track (a fresh `Track`/`track_id` needs a new `CarEnvironment`)."""

    def __init__(
        self,
        track: Track,
        track_id: str,
        config: CarEnvironmentConfig | None = None,
    ) -> None:
        self._track = track
        self._track_id = track_id
        self._config = config if config is not None else CarEnvironmentConfig()
        self._boundary = boundary_segments(track)
        self._loop_length = track_loop_length(track)
        self._car_radius = self._config.physics_constants.car_width / 2
        self.observation_space: Space = observation_space_for(self._config.sensor_config)
        self.action_space: Space = Box(-1.0, 1.0, (2,))
        self._game = Game(track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._step_count = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        # No randomness in this env today (deterministic track + physics), so `seed` is
        # accepted for Protocol conformance and unused — see the Phase 3 doc.
        self._game = Game(self._track, self._config.physics_constants)
        self._last_action = (0.0, 0.0)
        self._progress = 0.0
        self._step_count = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        steering, throttle = float(action[0]), float(action[1])
        state = self._game.tick(steering, throttle)
        self._progress += state.car.speed * TICK_DT
        self._last_action = (steering, throttle)
        self._step_count += 1

        crashed = distance_to_boundary((state.car.x, state.car.y), self._boundary) <= (
            self._car_radius + _CRASH_EPSILON
        )
        truncated = self._step_count >= self._config.max_episode_steps

        info: dict[str, Any] = {
            "crashed": crashed,
            "track_id": self._track_id,
            "progress": self._progress,
            "lap_progress": self._progress / self._loop_length,
        }
        return self._observation(), crashed, truncated, info

    def _observation(self) -> np.ndarray:
        return build_observation(
            self._game.car,
            self._boundary,
            self._last_action,
            self._config.sensor_config,
            self._config.physics_constants,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/sim/test_car_env.py -v`
Expected: PASS (8 cases)

- [ ] **Step 5: Cover the new modules in the import-hygiene test**

```python
# tests/sim/test_import_hygiene.py — replace the whole file with this
import json
import subprocess
import sys


def test_sim_does_not_import_forbidden_dependencies():
    code = (
        "import sys, json; "
        "import neuroarena.sim.track, neuroarena.sim.track_io, neuroarena.sim.track_generation, "
        "neuroarena.sim.physics, neuroarena.sim.collision, neuroarena.sim.game, "
        "neuroarena.sim.raycast, neuroarena.sim.observation, neuroarena.sim.car_env; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into neuroarena.sim's import graph"
```

- [ ] **Step 6: Run the full test suite, typecheck, and lint**

Run: `uv run pytest -v && uv run mypy src tests && uv run ruff check . && uv run ruff format --check .`
Expected: all clean

- [ ] **Step 7: Commit**

```bash
git add src/neuroarena/sim/car_env.py tests/sim/test_car_env.py tests/sim/test_import_hygiene.py
git commit -m "feat(phase-3): CarEnvironment wiring the game to the Phase 0 Environment interface"
```

---

## Revision history
- 2026-09-12 — Plan written, following Phase 3's FINALIZED requirements. Four tasks: boundary raycasting geometry, two small additions to Phase 1's own `track.py`/`collision.py` (loop length, boundary-distance query), observation-vector assembly, and `CarEnvironment` tying it all together. No changes to `sim/game.py` or `sim/physics.py` — `CarEnvironment` composes `Game` unmodified, exactly as Phase 1's own docstring anticipated. `progress` is implemented as the car's own integrated signed speed (an odometer), not a geometric projection onto the track centerline — simpler, and satisfies the Phase 3 doc's continuous/unwrapped/signed requirement without needing path-projection geometry.
