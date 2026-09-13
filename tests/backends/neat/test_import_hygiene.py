import json
import subprocess
import sys


def test_backends_neat_does_not_import_forbidden_dependencies() -> None:
    code = (
        "import sys, json; "
        "import neuroarena.backends.neat.config, neuroarena.backends.neat.model, "
        "neuroarena.backends.neat.evaluation, neuroarena.backends.neat.trainer; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, (
            f"{forbidden} leaked into neuroarena.backends.neat's import graph"
        )


def test_backends_neat_does_not_import_sim_or_render() -> None:
    code = (
        "import sys, json; "
        "import neuroarena.backends.neat.config, neuroarena.backends.neat.model, "
        "neuroarena.backends.neat.evaluation, neuroarena.backends.neat.trainer; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in loaded:
        assert not forbidden.startswith("neuroarena.sim"), (
            "neuroarena.backends.neat must consume only the Environment/Model/Objective "
            "interfaces, not neuroarena.sim directly"
        )
        assert not forbidden.startswith("neuroarena.render")
