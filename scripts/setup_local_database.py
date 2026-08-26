"""Command-line entry point for creating the empty local SQLite schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import initialize_database  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse database and schema paths supplied on the command line."""

    parser = argparse.ArgumentParser(
        description="Create the empty stage-one SQLite table structure."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "database" / "taobao_user_behavior.db",
        help="SQLite output path (default: database/taobao_user_behavior.db).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "sql" / "ddl" / "001_create_schema.sql",
        help="DDL file path (default: sql/ddl/001_create_schema.sql).",
    )
    return parser.parse_args()


def main() -> int:
    """Initialize the database and print a concise object summary."""

    args = parse_args()
    try:
        summary = initialize_database(args.database, args.schema)
    except Exception as error:
        print(f"Database initialization failed: {error}", file=sys.stderr)
        return 1

    print(f"Database: {summary.database_path}")
    print(f"Tables: {', '.join(summary.tables)}")
    print("Initialization complete; no behavior rows were imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
