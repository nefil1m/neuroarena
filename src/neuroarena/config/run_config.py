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
    neat_hyperparameters: dict[str, float | int | bool | str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
