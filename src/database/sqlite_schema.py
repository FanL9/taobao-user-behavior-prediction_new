"""Create and inspect the empty local SQLite database structure."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "sql" / "ddl" / "001_create_schema.sql"


@dataclass(frozen=True)
class DatabaseSummary:
    """Database path and objects created by :func:`initialize_database`."""

    database_path: Path
    tables: tuple[str, ...]


def _object_names(connection: sqlite3.Connection, object_type: str) -> tuple[str, ...]:
    """Return sorted non-internal SQLite object names for one object type."""

    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = ? AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """,
        (object_type,),
    ).fetchall()
    return tuple(row[0] for row in rows)


def initialize_database(
    database_path: str | Path,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> DatabaseSummary:
    """Initialize the empty SQLite schema in an idempotent transaction.

    Args:
        database_path: SQLite file to create or update.
        schema_path: UTF-8 SQL file containing idempotent DDL statements.

    Returns:
        A :class:`DatabaseSummary` listing the resulting tables. The
        ``user_behavior`` table contains zero rows on a new database.

    Raises:
        FileNotFoundError: If the schema SQL file does not exist.
        ValueError: If ``database_path`` points to a directory.
        sqlite3.Error: If SQLite cannot execute the schema.
    """

    destination = Path(database_path).expanduser().resolve()
    schema = Path(schema_path).expanduser().resolve()
    if not schema.is_file():
        raise FileNotFoundError(f"Schema SQL does not exist: {schema}")
    if destination.exists() and destination.is_dir():
        raise ValueError(f"Database path is a directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    ddl = schema.read_text(encoding="utf-8")

    with sqlite3.connect(destination) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(ddl)
        summary = DatabaseSummary(
            database_path=destination,
            tables=_object_names(connection, "table"),
        )

    return summary
