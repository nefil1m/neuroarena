import pytest


def test_the_viewer_and_scene_modules_import_without_opening_a_window() -> None:
    pytest.importorskip("arcade")
    import neuroarena.render.scene as scene
    import neuroarena.render.viewer as viewer

    assert callable(viewer.main)
    assert scene.DEFAULT_MANIFEST_PATH.name == "manifest.json"
