from collections.abc import Mapping

from neuroarena.interfaces.protocols import Visualizable
from tests.interfaces.doubles import DummyEnvironment


def test_an_environment_without_visual_state_is_not_visualizable() -> None:
    assert not isinstance(DummyEnvironment(), Visualizable)


def test_an_object_with_visual_state_is_visualizable() -> None:
    class Drawable:
        def visual_state(self) -> Mapping[str, float]:
            return {"x": 1.0}

    assert isinstance(Drawable(), Visualizable)
