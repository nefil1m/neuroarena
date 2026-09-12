import json
from pathlib import Path

import pytest

from neuroarena.render.manifest import AssetManifest

MANIFEST_PATH = Path(__file__).parents[2] / "src/neuroarena/render/assets/road_01/manifest.json"


def test_road_01_manifest_loads_and_resolves_all_files() -> None:
    manifest = AssetManifest.load(MANIFEST_PATH)
    assert manifest.surface.is_file()
    assert manifest.car.is_file()
    assert manifest.decor["start_finish"].is_file()
    assert manifest.background["grass"].is_file()


def test_missing_referenced_file_raises(tmp_path: Path) -> None:
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    manifest_data = json.loads(MANIFEST_PATH.read_text())
    manifest_data["surface"] = "does_not_exist.png"
    broken_path = broken_dir / "manifest.json"
    broken_path.write_text(json.dumps(manifest_data))
    with pytest.raises(FileNotFoundError):
        AssetManifest.load(broken_path)
