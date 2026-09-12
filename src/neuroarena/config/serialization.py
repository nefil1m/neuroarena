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
