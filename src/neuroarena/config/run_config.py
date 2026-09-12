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
