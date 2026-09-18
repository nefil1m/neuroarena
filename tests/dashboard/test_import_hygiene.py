import json
import subprocess
import sys


def test_the_dashboard_does_not_import_arcade_or_pyglet() -> None:
    code = (
        "import sys, json; "
        "import neuroarena.dashboard.app, neuroarena.dashboard.cli, "
        "neuroarena.dashboard.viewer_manager; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("arcade", "pyglet"):
        assert forbidden not in loaded, f"{forbidden} leaked into the dashboard's import graph"
