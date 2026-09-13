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

    `sensor_config` and `physics_constants` are real Phase 5 knobs, not stubs: they're wired
    through `CarEnvironmentConfig.from_run_config` into the `CarEnvironment` a run trains
    against, changing the car's raycast layout and kinematic constants respectively.
    `neat_hyperparameters` is likewise a real knob — a flat `key: value` override dict applied
    on top of `build_neat_config`'s bundled template defaults (see
    `backends.neat.config._apply_hyperparameter_overrides` for exactly which keys are
    accepted).

    `checkpoint_every_n_generations` and `champion_retention_cap` are Phase 6 knobs
    consumed by `neuroarena.persistence.recorder` — the first sets resume-checkpoint
    cadence (every save kept, never overwritten), the second optionally bounds how many
    champion checkpoints (Phase 4's per-generation capture) are retained."""

    master_seed: int = 0
    max_generation_steps: int = 300_000
    population_size: int = 150
    track_id: str | None = None
    headless: bool = True
    sim_speed: float = 1.0
    max_episode_steps: int = 3000
    max_generations: int | None = None
    target_fitness: float | None = None
    checkpoint_every_n_generations: int = 10
    champion_retention_cap: int | None = None
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
    physics_constants: PhysicsConstants = field(default_factory=PhysicsConstants)
    neat_hyperparameters: dict[str, float | int | bool | str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
