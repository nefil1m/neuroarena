import json
import subprocess
import sys


def test_core_packages_do_not_import_forbidden_dependencies():
    code = (
        "import sys, json; "
        "import neuroarena.interfaces, neuroarena.config; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into the core import graph"
