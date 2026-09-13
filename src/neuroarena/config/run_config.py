from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1


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
