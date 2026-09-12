"""JSON codec for `Track`, mirroring `neuroarena.config.serialization`'s convention."""

from __future__ import annotations

import json
from typing import Any

from neuroarena.sim.track import CELL_SIZE, Facing, GridCell, TileKind, Track

SCHEMA_VERSION = 1


class UnknownSchemaVersionError(ValueError):
    """Track file declares a schema_version this build cannot read."""


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate a decoded track dict toward SCHEMA_VERSION. Identity today (only v1)."""
    return raw


def dumps(track: Track) -> str:
    raw = {
        "schema_version": SCHEMA_VERSION,
        "cell_size": track.cell_size,
        "start_cell": list(track.start_cell),
        "start_facing": track.start_facing.value,
        "cells": [{"cell": list(cell), "kind": kind.value} for cell, kind in track.cells.items()],
    }
    return json.dumps(raw, indent=2)


def loads(text: str) -> Track:
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

    cells: dict[GridCell, TileKind] = {}
    for entry in raw["cells"]:
        cell = tuple(entry["cell"])
        cells[cell] = TileKind(entry["kind"])

    return Track(
        cells=cells,
        start_cell=tuple(raw["start_cell"]),
        start_facing=Facing(raw["start_facing"]),
        cell_size=raw.get("cell_size", CELL_SIZE),
    )
