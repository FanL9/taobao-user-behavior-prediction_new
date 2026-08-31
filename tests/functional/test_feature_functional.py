"""Functional tests for the first four stage-two feature tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature import (
    OUTPUT_FILENAMES,
    build_feature_tables,
    generate_feature_tables,
)
from src.features.stage2_intermediate_tables import HistoryWindow


def _clean_frame() -> pd.DataFrame:
    """Create history, label-date, and out-of-window behavior rows."""

    rows = [
        ("2025-11-17 23", 9, 19, 109, "pv"),
        ("2025-11-18 00", 1, 10, 100, "pv"),
        ("2025-12-07 22", 1, 10, 100, "cart"),
        ("2025-12-07 23", 2, 11, 101, "buy"),
        ("2025-12-08 00", 1, 10, 100, "buy"),
        ("2025-12-09 00", 1, 10, 100, "fav"),
        ("2025-12-14 23", 1, 12, 102, "pv"),
        ("2025-12-15 00", 1, 12, 102, "buy"),
        ("2025-12-16 00", 1, 10, 100, "pv"),
        ("2025-12-17 23", 3, 13, 103, "buy"),
        ("2025-12-18 00", 3, 13, 103, "buy"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "time",
            "user_id",
            "item_id",
            "category_id",
            "behavior_name",
        ],
    )
    frame["behavior_date"] = pd.to_datetime(
        frame["time"], format="%Y-%m-%d %H"
    ).dt.strftime("%Y-%m-%d")
    return frame


def test_four_feature_tables_follow_scope_windows_and_grains() -> None:
    """Verify table scope, keys, history counts, and excluded label dates."""

    tables = build_feature_tables(_clean_frame())
    assert set(tables) == {
        "user_basic",
        "user_activity",
        "user_sequence",
        "item_behavior",
    }
    key_columns = {
        "user_basic": ["dataset_split", "user_id"],
        "user_activity": ["dataset_split", "user_id"],
        "user_sequence": ["dataset_split", "user_id", "item_id"],
        "item_behavior": ["dataset_split", "item_id"],
    }
    for name, table in tables.items():
        assert set(table["dataset_split"]) == {"train", "validation", "test"}
        assert not table.duplicated(key_columns[name]).any()
        assert table["history_end"].lt(table["label_date"]).all()
        assert "label" not in table.columns

    expected_events = {"train": 3, "validation": 2, "test": 2}
    user_totals = (
        tables["user_basic"].groupby("dataset_split")["event_count"].sum().to_dict()
    )
    item_totals = (
        tables["item_behavior"]
        .groupby("dataset_split")["item_total_count"]
        .sum()
        .to_dict()
    )
    assert user_totals == expected_events
    assert item_totals == expected_events

    train_pair = tables["user_sequence"].query(
        "dataset_split == 'train' and user_id == 1 and item_id == 10"
    ).iloc[0]
    assert train_pair["last_behavior_type"] == "cart"
    assert train_pair["last_10_behavior_sequence"] == "pv→cart"


def test_feature_table_responsibilities_do_not_overlap_remaining_tables() -> None:
    """Verify heat and conversion-chain fields are not emitted prematurely."""

    tables = build_feature_tables(_clean_frame())
    all_columns = set().union(*(set(table.columns) for table in tables.values()))
    forbidden_columns = {
        "buy_conversion_rate",
        "item_heat_level",
        "item_fav_to_pv_rate",
        "item_cart_to_pv_rate",
        "item_buy_to_pv_rate",
        "pv_to_cart_count",
        "cart_to_buy_count",
        "pv_to_buy_count",
        "fav_to_buy_count",
    }
    assert all_columns.isdisjoint(forbidden_columns)


def test_activity_levels_use_fixed_train_thresholds() -> None:
    """Verify all splits use P25/P75 learned only from train activity."""

    activity = build_feature_tables(_clean_frame())["user_activity"]
    train_values = activity.loc[
        activity["dataset_split"].eq("train"), "avg_daily_event_count"
    ]
    p25 = train_values.quantile(0.25)
    p75 = train_values.quantile(0.75)
    expected = np.select(
        [
            activity["avg_daily_event_count"].le(p25),
            activity["avg_daily_event_count"].le(p75),
        ],
        ["low", "medium"],
        default="high",
    )
    assert np.array_equal(activity["activity_level"].to_numpy(), expected)


def test_history_end_is_enforced_even_when_label_date_has_a_gap() -> None:
    """Verify functions do not treat all pre-label rows as history rows."""

    window = HistoryWindow("train", "2025-11-18", "2025-11-18", "2025-12-08")
    tables = build_feature_tables(_clean_frame(), windows=[window])
    assert tables["user_basic"]["event_count"].sum() == 1
    assert tables["item_behavior"]["item_total_count"].sum() == 1
    assert tables["user_sequence"].iloc[0]["last_behavior_type"] == "pv"


def test_generator_writes_exactly_four_feature_parquet_files(tmp_path) -> None:
    """Verify the public generator writes the four requested output files."""

    input_path = tmp_path / "user_behavior_clean.parquet"
    output_directory = tmp_path / "features"
    _clean_frame().to_parquet(input_path, index=False)

    outputs = generate_feature_tables(input_path, output_directory)

    assert set(outputs) == set(OUTPUT_FILENAMES)
    assert {path.name for path in output_directory.iterdir()} == set(
        OUTPUT_FILENAMES.values()
    )
    assert all(path.is_file() for path in outputs.values())


def test_generation_rejects_inconsistent_behavior_date() -> None:
    """Verify clean time fields are validated before feature construction."""

    frame = _clean_frame()
    frame.loc[0, "behavior_date"] = "2025-11-18"
    with pytest.raises(ValueError, match="behavior_date"):
        build_feature_tables(frame)
