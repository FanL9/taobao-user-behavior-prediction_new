"""Functional checks for read-only CSV quality inspection."""

from __future__ import annotations

import json

import pandas as pd

from src.data import check_csv_quality, write_quality_report


def test_quality_check_reports_issues_without_modifying_source(tmp_path) -> None:
    source = tmp_path / "user_behavior_processed.csv"
    frame = pd.DataFrame(
        [
            ["2025-11-18 00", "1", "10", "100", "1"],
            ["2025-11-18 00", "1", "10", "100", "1"],
            ["2025-11-18 00", "1", "10", "101", "1"],
            ["2025-11-18 01", "2", "", "100", "2"],
            ["2025-11-18 02", "-1", "11", "100", "3"],
            ["2025-11-18 03", "3", "12", "100", "9"],
            ["bad-time", "4", "13", "100", "4"],
        ],
        columns=["time", "user_id", "item_id", "item_category", "behavior_type"],
    )
    frame.to_csv(source, index=False)
    original_bytes = source.read_bytes()

    report = check_csv_quality(source, chunksize=2, duplicate_partitions=4)

    assert source.read_bytes() == original_bytes
    assert report["scale"]["row_count"] == 7
    assert report["completeness"]["rows_with_any_missing"] == 1
    assert report["completeness"]["missing_by_column"]["item_id"] == 1
    assert report["validity"]["invalid_by_column"]["user_id"] == 1
    assert report["validity"]["invalid_by_column"]["behavior_type"] == 1
    assert report["validity"]["invalid_by_column"]["time"] == 1
    assert report["duplicates"]["exact_rows"] == {
        "group_count": 1,
        "record_count": 2,
        "excess_count": 1,
    }
    assert report["duplicates"]["suspected_key"]["group_count"] == 1
    assert report["duplicates"]["suspected_key"]["record_count"] == 3
    assert report["duplicates"]["suspected_key"]["excess_count"] == 2
    assert report["decisions"]["source_modified"] is False
    assert report["decisions"]["cleaning_performed"] is False
    assert report["status"] == "REVIEW"

    output = write_quality_report(report, tmp_path / "report.json")
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_quality_check_passes_valid_unique_data(tmp_path) -> None:
    source = tmp_path / "valid.csv"
    pd.DataFrame(
        [
            ["2025-11-18 00", 1, 10, 100, 1],
            ["2025-11-18 01", 2, 11, 101, 4],
        ],
        columns=["time", "user_id", "item_id", "item_category", "behavior_type"],
    ).to_csv(source, index=False)

    report = check_csv_quality(source, duplicate_partitions=2)

    assert report["status"] == "PASS"
    assert report["duplicates"]["exact_rows"]["record_count"] == 0
    assert report["duplicates"]["suspected_key"]["record_count"] == 0
