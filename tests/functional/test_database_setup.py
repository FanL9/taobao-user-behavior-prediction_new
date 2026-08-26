"""Functional checks for the stage-one SQLite schema."""

from __future__ import annotations

import sqlite3

import pytest

from src.database import initialize_database


def test_initialize_database_creates_empty_schema(tmp_path) -> None:
    database_path = tmp_path / "local.db"

    summary = initialize_database(database_path)

    assert summary.tables == ("user_behavior",)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM user_behavior"
        ).fetchone()[0] == 0
        extra_objects = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type IN ('index', 'view') AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        assert extra_objects == []


def test_database_constraints_and_idempotency(tmp_path) -> None:
    database_path = tmp_path / "local.db"
    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO user_behavior VALUES (?, ?, ?, ?, ?)",
            ("2025-11-18 00", 1, 2, 3, 4),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO user_behavior VALUES (?, ?, ?, ?, ?)",
                ("2025-11-18 00", 1, 2, 3, 9),
            )

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM user_behavior"
        ).fetchone()[0] == 1
