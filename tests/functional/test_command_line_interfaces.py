"""End-to-end checks for the public stage-one command-line entry points."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_database_and_conversion_cli(tmp_path) -> None:
    database_path = tmp_path / "cli.db"
    database_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "setup_local_database.py"),
            "--database",
            str(database_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert database_run.returncode == 0, database_run.stderr
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM user_behavior"
        ).fetchone()[0] == 0

    source = tmp_path / "cli.csv"
    destination = tmp_path / "cli.parquet"
    pd.DataFrame(
        {
            "time": ["2025-11-18 00"],
            "user_id": [1],
            "item_id": [2],
            "item_category": [3],
            "behavior_type": [1],
        }
    ).to_csv(source, index=False)
    conversion_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "convert_csv_to_parquet.py"),
            "--input",
            str(source),
            "--output",
            str(destination),
            "--chunksize",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert conversion_run.returncode == 0, conversion_run.stderr
    assert pq.read_table(destination).num_rows == 1

    quality_report = tmp_path / "quality.json"
    quality_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_data_quality.py"),
            "--input",
            str(source),
            "--output",
            str(quality_report),
            "--chunksize",
            "1",
            "--duplicate-partitions",
            "2",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert quality_run.returncode == 0, quality_run.stderr
    report = json.loads(quality_report.read_text(encoding="utf-8"))
    assert report["scale"]["row_count"] == 1
    assert report["decisions"]["cleaning_performed"] is False
