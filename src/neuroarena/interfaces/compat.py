from __future__ import annotations

from neuroarena.interfaces.protocols import Environment, Model


class IncompatibleDescriptorsError(ValueError):
    """A model's trained descriptors do not match an environment's."""


def check_compatibility(model: Model, environment: Environment) -> None:
    """Raise IncompatibleDescriptorsError if model and environment descriptors differ.

    Resume-safety check, run at run creation. Equality is the full field-wise
    comparison defined by the space descriptors.
    """
    if model.observation_space != environment.observation_space:
        raise IncompatibleDescriptorsError(
            f"observation space mismatch: model {model.observation_space!r} "
            f"!= environment {environment.observation_space!r}"
        )
    if model.action_space != environment.action_space:
        raise IncompatibleDescriptorsError(
            f"action space mismatch: model {model.action_space!r} "
            f"!= environment {environment.action_space!r}"
        )
