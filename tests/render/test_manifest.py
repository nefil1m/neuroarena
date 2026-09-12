import json
from pathlib import Path

import pytest

from neuroarena.render.manifest import AssetManifest
from neuroarena.sim.track import TileKind

MANIFEST_PATH = Path(__file__).parents[2] / "src/neuroarena/render/assets/road_01/manifest.json"


def test_road_01_manifest_loads_and_resolves_all_tile_kinds() -> None:
    manifest = AssetManifest.load(MANIFEST_PATH)
    for kind in TileKind:
        resolved = manifest.resolve(kind)
        assert resolved.paths
        for p in resolved.paths:
            assert p.is_file()


def test_base_tiles_have_zero_rotation() -> None:
    manifest = AssetManifest.load(MANIFEST_PATH)
    assert manifest.resolve(TileKind.STRAIGHT_EW).rotate_degrees == 0
    assert manifest.resolve(TileKind.CURVE_SE).rotate_degrees == 0


def test_derived_tiles_have_documented_rotation() -> None:
    manifest = AssetManifest.load(MANIFEST_PATH)
    assert manifest.resolve(TileKind.STRAIGHT_NS).rotate_degrees == 90
    assert manifest.resolve(TileKind.CURVE_SW).rotate_degrees == 90
    assert manifest.resolve(TileKind.CURVE_NW).rotate_degrees == 180
    assert manifest.resolve(TileKind.CURVE_NE).rotate_degrees == 270


def test_composite_preferred_over_layers_when_both_present() -> None:
    # The vendored layer exports are cropped to each layer's own content with no shared
    # canvas/offset metadata (e.g. the curve's Road_Side_02 is a small drain-cover icon,
    # not a full-tile image) — compositing them naively would misplace them, so the
    # pre-flattened composite is used until per-layer offsets are recorded.
    manifest = AssetManifest.load(MANIFEST_PATH)
    resolved = manifest.resolve(TileKind.STRAIGHT_EW)
    assert len(resolved.paths) == 1
    assert "composite" in str(resolved.paths[0])


def test_car_decor_and_background_resolve() -> None:
    manifest = AssetManifest.load(MANIFEST_PATH)
    assert manifest.car.is_file()
    assert manifest.decor["start_finish"].is_file()
    assert manifest.background["grass"].is_file()


def test_missing_referenced_file_raises(tmp_path: Path) -> None:
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    manifest_data = json.loads(MANIFEST_PATH.read_text())
    manifest_data["car"] = "../does_not_exist.png"
    broken_path = broken_dir / "manifest.json"
    broken_path.write_text(json.dumps(manifest_data))
    with pytest.raises(FileNotFoundError):
        AssetManifest.load(broken_path)
