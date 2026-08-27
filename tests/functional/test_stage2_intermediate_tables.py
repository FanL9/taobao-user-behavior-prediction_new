"""Functional tests for stage-two intermediate-table generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.stage2_intermediate_tables import (
    OUTPUT_FILENAMES,
    build_intermediate_tables,
    generate_intermediate_tables,
)


def _clean_frame() -> pd.DataFrame:
    """Create history, label-date, and out-of-window events for all windows."""

    rows = [
        ("2025-11-17 23", 9, 19, 109, 1),
        ("2025-11-18 00", 1, 10, 100, 1),
        ("2025-12-07 23", 1, 10, 100, 4),
        ("2025-12-08 00", 1, 10, 100, 4),
        ("2025-12-09 00", 2, 11, 101, 2),
        ("2025-12-14 23", 2, 11, 101, 3),
        ("2025-12-15 00", 2, 11, 101, 4),
        ("2025-12-16 00", 3, 12, 102, 1),
        ("2025-12-17 23", 3, 12, 102, 4),
        ("2025-12-18 00", 3, 12, 102, 4),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["time", "user_id", "item_id", "category_id", "behavior_type"],
    )
    parsed = pd.to_datetime(frame["time"], format="%Y-%m-%d %H")
    names = {1: "pv", 2: "fav", 3: "cart", 4: "buy"}
    frame["behavior_name"] = frame["behavior_type"].map(names)
    frame["behavior_date"] = parsed.dt.strftime("%Y-%m-%d")
    frame["behavior_hour"] = parsed.dt.hour
    frame["weekday"] = parsed.dt.weekday
    return frame


def test_four_intermediate_tables_follow_windows_and_grains() -> None:
    """Verify table scope, keys, counts, and label-date exclusion."""

    tables = build_intermediate_tables(_clean_frame())

    assert set(tables) == {"user", "item", "category", "time"}
    expected_events = {"train": 2, "validation": 2, "test": 2}
    key_columns = {
        "user": ["dataset_split", "user_id"],
        "item": ["dataset_split", "item_id"],
        "category": ["dataset_split", "category_id"],
        "time": ["dataset_split", "behavior_date", "behavior_hour"],
    }

    for name, table in tables.items():
        assert not table.duplicated(key_columns[name]).any()
        assert "label" not in table.columns
        assert "user_item" not in " ".join(table.columns)
        totals = table.groupby("dataset_split")["event_count"].sum().to_dict()
        assert totals == expected_events
        behavior_total = table[["pv_count", "fav_count", "cart_count", "buy_count"]].sum(axis=1)
        assert behavior_total.equals(table["event_count"])
        assert table["history_end"].lt(table["label_date"]).all()

    train_user = tables["user"].query("dataset_split == 'train'").iloc[0]
    assert train_user["user_id"] == 1
    assert train_user["pv_count"] == 1
    assert train_user["buy_count"] == 1


def test_generator_writes_exactly_four_intermediate_parquet_files(tmp_path) -> None:
    """Verify the generation interface writes only four Parquet outputs."""

    input_path = tmp_path / "user_behavior_clean.parquet"
    output_dir = tmp_path / "stage2"
    _clean_frame().to_parquet(input_path, index=False)

    outputs = generate_intermediate_tables(input_path, output_dir)

    assert set(outputs) == set(OUTPUT_FILENAMES)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_FILENAMES.values())
    for path in outputs.values():
        assert path.is_file()
        assert pd.read_parquet(path)["dataset_split"].nunique() == 3


def test_generation_rejects_inconsistent_clean_fields() -> None:
    """Verify generation rejects inconsistent derived time fields."""

    frame = _clean_frame()
    frame.loc[0, "behavior_hour"] = 8

    with pytest.raises(ValueError, match="behavior_hour"):
        build_intermediate_tables(frame)
