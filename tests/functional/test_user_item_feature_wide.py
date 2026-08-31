"""Functional tests for the stage-two user-item feature wide table."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features.feature import generate_all_feature_tables
from src.features.user_item_feature_wide import (
    PRIMARY_KEY,
    generate_user_item_feature_wide,
)


def _clean_frame() -> pd.DataFrame:
    """Create events covering all three fixed stage-two history windows.

    Returns:
        A small clean-format behavior DataFrame.
    """

    rows = [
        ("2025-11-18 00", 1, 10, 100, "pv"),
        ("2025-12-07 22", 1, 10, 100, "cart"),
        ("2025-12-07 23", 2, 11, 101, "buy"),
        ("2025-12-09 00", 1, 10, 100, "fav"),
        ("2025-12-14 23", 1, 12, 102, "pv"),
        ("2025-12-16 00", 1, 10, 100, "pv"),
        ("2025-12-17 23", 3, 13, 103, "buy"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "time", "user_id", "item_id", "category_id", "behavior_name"
        ],
    )
    frame["behavior_date"] = pd.to_datetime(
        frame["time"], format="%Y-%m-%d %H"
    ).dt.strftime("%Y-%m-%d")
    return frame


def _write_eight_tables(tmp_path):
    """Generate the eight prerequisite feature Parquets for one test.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Directory containing the eight generated feature tables.
    """

    clean_path = tmp_path / "user_behavior_clean.parquet"
    feature_directory = tmp_path / "features"
    _clean_frame().to_parquet(clean_path, index=False)
    generate_all_feature_tables(clean_path, feature_directory)
    return feature_directory


def test_user_item_wide_merges_all_eight_tables_without_label(tmp_path) -> None:
    """Verify grain, merge routes, field roles, checks, and exclusions.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None. Assertions validate the generated outputs.
    """

    feature_directory = _write_eight_tables(tmp_path)
    output = tmp_path / "user_item_feature_wide.parquet"
    report_path = tmp_path / "wide_quality.json"

    result = generate_user_item_feature_wide(
        feature_directory,
        output,
        report_path,
        batch_size=2,
    )
    wide = pd.read_parquet(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["output_path"] == output.resolve()
    assert wide.shape == (6, 64)
    assert not wide.duplicated(list(PRIMARY_KEY)).any()
    assert "label" not in wide.columns
    assert set(wide["dataset_split"]) == {"train", "validation", "test"}
    assert wide["history_end"].lt(wide["label_date"]).all()
    assert wide["last_behavior_date"].between(
        wide["history_start"], wide["history_end"]
    ).all()

    representative_columns = {
        "user_pv_count",
        "user_activity_level",
        "last_10_behavior_sequence",
        "item_total_count",
        "item_total_count_rank",
        "category_total_count",
        "time_total_count",
        "buy_per_pv",
    }
    assert representative_columns.issubset(wide.columns)

    train_pair = wide.query(
        "dataset_split == 'train' and user_id == 1 and item_id == 10"
    ).iloc[0]
    assert train_pair["category_id"] == 100
    assert train_pair["last_behavior_date"] == pd.Timestamp("2025-12-07")
    assert train_pair["last_behavior_hour"] == 22
    assert train_pair["time_total_count"] == 1
    assert train_pair["last_10_behavior_sequence"] == "pv→cart"

    assert report["status"] == "passed"
    assert report["row_count"] == report["anchor_row_count"] == 6
    assert report["column_count"] == 64
    assert report["duplicate_primary_key_count"] == 0
    assert report["time_window_violation_count"] == 0
    assert report["abnormal_values"] == {}
    assert report["field_roles"]["primary_key_fields"] == list(PRIMARY_KEY)
    assert "label_date" in report["field_roles"][
        "prohibited_model_input_fields"
    ]


def test_user_item_wide_rejects_inconsistent_feature_windows(tmp_path) -> None:
    """Verify a source table with a conflicting history window is rejected.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None. The expected validation error is asserted.
    """

    feature_directory = _write_eight_tables(tmp_path)
    category_path = feature_directory / "category_behavior_features.parquet"
    category = pd.read_parquet(category_path)
    category.loc[category["dataset_split"].eq("train"), "history_end"] = (
        "2025-12-06"
    )
    category.to_parquet(category_path, index=False)

    with pytest.raises(ValueError, match="inconsistent history_end"):
        generate_user_item_feature_wide(
            feature_directory,
            tmp_path / "wide.parquet",
            tmp_path / "quality.json",
        )
