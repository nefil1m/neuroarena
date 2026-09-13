"""Migration-driven SQLite connection helper. Schema versioning follows the same
versioned-and-migrated shape used elsewhere in this codebase (`RunConfig.schema_version`,
NEAT checkpoint `schema_version`, `TrackRecord.schema_version`) — see
`../../../docs/phases/phase-6-persistence.md`'s "Schema versioning" section — applied to
the database as a whole via `PRAGMA user_version` plus ordered `.sql` files here."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DEFAULT_DB_PATH = Path("data/neuroarena.db")


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for path in migrations:
        version = int(path.stem.split("_")[0])
        if version <= current_version:
            continue
        conn.executescript(path.read_text())
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
