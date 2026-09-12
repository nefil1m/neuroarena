import sys


def _fresh_import(*modules: str) -> set[str]:
    for name in list(sys.modules):
        if name.startswith("neuroarena"):
            del sys.modules[name]
    for m in modules:
        __import__(m)
    return set(sys.modules)


def test_core_packages_do_not_import_forbidden_dependencies():
    loaded = _fresh_import("neuroarena.interfaces", "neuroarena.config")
    for forbidden in ("gymnasium", "arcade", "pygame", "stable_baselines3", "torch"):
        assert forbidden not in loaded, f"{forbidden} leaked into the core import graph"
