"""Per-style asset manifest loader.

Resolves each `TileKind` to a composite-or-layered texture path list plus a rotation, so
the renderer draws uniformly regardless of which art form a style provides (see the Phase 1
doc's "Tile art" requirement). No image library or `arcade` import — this stays at the
path/metadata level; the renderer loads and rotates the actual textures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroarena.sim.track import TileKind


@dataclass(frozen=True)
class ResolvedTile:
    paths: tuple[Path, ...]  # one path if composite, several (bottom-to-top z-order) if layered
    rotate_degrees: int


@dataclass(frozen=True)
class AssetManifest:
    base_dir: Path
    tiles: dict[TileKind, ResolvedTile]
    car: Path
    decor: dict[str, Path]
    background: dict[str, Path]

    def resolve(self, kind: TileKind) -> ResolvedTile:
        return self.tiles[kind]

    @classmethod
    def load(cls, path: Path | str) -> AssetManifest:
        path = Path(path)
        base_dir = path.parent
        raw: dict[str, Any] = json.loads(path.read_text())

        base_tiles = raw["base_tiles"]
        tiles: dict[TileKind, ResolvedTile] = {
            TileKind(name): _resolve_entry(base_dir, entry, rotate_degrees=0)
            for name, entry in base_tiles.items()
        }
        for name, derived in raw["derived_tiles"].items():
            source_entry = base_tiles[derived["from"]]
            tiles[TileKind(name)] = _resolve_entry(
                base_dir, source_entry, rotate_degrees=derived["rotate_degrees"]
            )

        missing = set(TileKind) - set(tiles)
        if missing:
            raise ValueError(
                f"manifest at {path} is missing tile kinds: {sorted(k.value for k in missing)}"
            )

        return cls(
            base_dir=base_dir,
            tiles=tiles,
            car=_existing(base_dir / raw["car"]),
            decor={name: _existing(base_dir / rel) for name, rel in raw["decor"].items()},
            background={name: _existing(base_dir / rel) for name, rel in raw["background"].items()},
        )


def _resolve_entry(base_dir: Path, entry: dict[str, Any], rotate_degrees: int) -> ResolvedTile:
    # Composite is preferred over layers when a manifest entry offers both. The vendored
    # kit's layer exports are cropped to each layer's own content (e.g. the curve tile's
    # "Road_Side_02" is a 96x96 drain-cover icon, not a full-tile canvas) with no offset
    # metadata to re-place them correctly — compositing them here would misalign them. The
    # composite is already correctly flattened, so it's the reliable form until a manifest
    # entry can record per-layer canvas offsets.
    paths: tuple[Path, ...]
    if "composite" in entry:
        paths = (_existing(base_dir / entry["composite"]),)
    elif "layers" in entry:
        paths = tuple(_existing(base_dir / rel) for rel in entry["layers"])
    else:
        raise ValueError(f"tile entry has neither 'composite' nor 'layers': {entry!r}")
    return ResolvedTile(paths=paths, rotate_degrees=rotate_degrees)


def _existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"manifest references a missing asset file: {path}")
    return path
