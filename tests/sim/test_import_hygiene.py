import json
import subprocess
import sys


def test_sim_does_not_import_forbidden_dependencies():
    code = (
        "import sys, json; "
        "import neuroarena.sim.track, neuroarena.sim.track_io, neuroarena.sim.track_generation, "
        "neuroarena.sim.physics, neuroarena.sim.collision, neuroarena.sim.game, "
        "neuroarena.sim.raycast, neuroarena.sim.observation, neuroarena.sim.car_env; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into neuroarena.sim's import graph"
