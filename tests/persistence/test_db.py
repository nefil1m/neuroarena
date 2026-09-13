from pathlib import Path

from neuroarena.persistence.db import connect


def test_connect_creates_all_five_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"models", "runs", "generation_stats", "settings_history", "checkpoints"} <= tables


def test_connect_sets_user_version(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_connect_enables_foreign_keys(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connect_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    connect(db_path)
    conn = connect(db_path)  # second connect on the same file must not error or re-apply
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_row_factory_supports_column_access(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO models (model_id, backend, observation_space, action_space, created_at) "
        "VALUES ('m1', 'neat', '{}', '{}', 'now')"
    )
    conn.commit()
    row = conn.execute("SELECT * FROM models WHERE model_id = 'm1'").fetchone()
    assert row["backend"] == "neat"
