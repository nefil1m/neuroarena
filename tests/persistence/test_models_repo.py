from pathlib import Path

import pytest

from neuroarena.interfaces.spaces import Box, Discrete
from neuroarena.persistence.db import connect
from neuroarena.persistence.models_repo import (
    UnknownModelError,
    create_model,
    get_model,
    list_models,
)


def test_create_model_round_trips_spaces(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_model(
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (10,)),
        action_space=Box(-1.0, 1.0, (2,)),
    )
    loaded = get_model(conn, record.model_id)
    assert loaded == record
    assert loaded.observation_space == Box(-1.0, 1.0, (10,))


def test_create_model_round_trips_discrete_space(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    record = create_model(
        conn,
        backend="neat",
        observation_space=Box(-1.0, 1.0, (3,)),
        action_space=Discrete(4),
    )
    loaded = get_model(conn, record.model_id)
    assert loaded.action_space == Discrete(4)


def test_get_model_unknown_id_raises(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    with pytest.raises(UnknownModelError):
        get_model(conn, "does-not-exist")


def test_list_models_returns_all_created(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    for _ in range(3):
        create_model(
            conn,
            backend="neat",
            observation_space=Box(-1.0, 1.0, (10,)),
            action_space=Box(-1.0, 1.0, (2,)),
        )
    assert len(list_models(conn)) == 3
