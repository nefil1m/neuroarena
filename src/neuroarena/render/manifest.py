"""Per-style asset manifest loader.

The road surface is a single undirected, edge-to-edge fill texture placed uniformly under
every cell; kerbs and lane markings are drawn procedurally from the track's own collision
geometry (`neuroarena.sim.track.boundary_segments`) rather than from per-`TileKind` bitmap
art (see the Phase 1 doc's revision history — the vendored kit's per-shape kerb art carried
non-square canvas bleed that never tiled seamlessly, and driving the kerb rotation off a
manual per-`TileKind` degrees mapping was a second, independent source of bugs; deriving it
from the already-validated boundary geometry instead avoids both). No image library or
`arcade` import here — this stays at the path/metadata level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetManifest:
    base_dir: Path
    surface: Path
    car: Path
    decor: dict[str, Path]
    background: dict[str, Path]

    @classmethod
    def load(cls, path: Path | str) -> AssetManifest:
        path = Path(path)
        base_dir = path.parent
        raw: dict[str, Any] = json.loads(path.read_text())

        return cls(
            base_dir=base_dir,
            surface=_existing(base_dir / raw["surface"]),
            car=_existing(base_dir / raw["car"]),
            decor={name: _existing(base_dir / rel) for name, rel in raw["decor"].items()},
            background={name: _existing(base_dir / rel) for name, rel in raw["background"].items()},
        )


def _existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"manifest references a missing asset file: {path}")
    return path
