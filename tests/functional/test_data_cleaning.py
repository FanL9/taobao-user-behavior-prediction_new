import pandas as pd

from src.data.cleaning import CLEAN_COLUMNS, clean_chunk


def test_clean_chunk_generates_standard_columns():
    raw = pd.DataFrame(
        {
            "time": ["2025-11-18 09", "2025-11-18 23"],
            "user_id": ["1001", "1002"],
            "item_id": ["2001", "2002"],
            "item_category": ["3001", "3002"],
            "behavior_type": ["1", "4"],
        }
    )

    result = clean_chunk(raw)
    cleaned = result.frame

    assert tuple(cleaned.columns) == CLEAN_COLUMNS
    assert cleaned["category_id"].tolist() == [3001, 3002]
    assert cleaned["behavior_name"].tolist() == ["pv", "buy"]
    assert cleaned["behavior_hour"].tolist() == [9, 23]


def test_clean_chunk_removes_invalid_rows():
    raw = pd.DataFrame(
        {
            "time": [
                "2025-11-18 09",
                "bad-time",
                "2025-11-18 10",
                "2025-11-18 11",
            ],
            "user_id": ["1001", "1002", "-1", "1004"],
            "item_id": ["2001", "2002", "2003", "2004"],
            "item_category": ["3001", "3002", "3003", "3004"],
            "behavior_type": ["1", "2", "3", "9"],
        }
    )

    result = clean_chunk(raw)

    assert len(result.frame) == 1
    assert result.stats.input_rows == 4
    assert result.stats.output_rows == 1
    assert result.stats.removed_invalid_id_rows == 1
    assert result.stats.removed_invalid_behavior_rows == 1
    assert result.stats.removed_invalid_time_rows == 1


def test_clean_chunk_keeps_possible_hour_level_repeat():
    raw = pd.DataFrame(
        {
            "time": ["2025-11-18 09", "2025-11-18 09"],
            "user_id": ["1001", "1001"],
            "item_id": ["2001", "2001"],
            "item_category": ["3001", "3002"],
            "behavior_type": ["1", "1"],
        }
    )

    result = clean_chunk(raw)

    # Same user/item/behavior/hour is only a suspected duplicate.
    # It must not be blindly removed here.
    assert len(result.frame) == 2

from src.data.cleaning_pipeline import clean_user_behavior_file


def test_full_cleaning_pipeline_removes_global_exact_duplicates(
    tmp_path,
):
    raw = pd.DataFrame(
        {
            "time": [
                "2025-11-18 09",
                "2025-11-18 09",
                "2025-11-18 09",
                "2025-11-18 10",
            ],
            "user_id": [
                "1001",
                "1001",
                "1001",
                "1002",
            ],
            "item_id": [
                "2001",
                "2001",
                "2001",
                "2002",
            ],
            "item_category": [
                "3001",
                "3001",
                "3002",
                "3003",
            ],
            "behavior_type": [
                "1",
                "1",
                "1",
                "4",
            ],
        }
    )

    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "clean.csv"
    output_parquet = tmp_path / "clean.parquet"
    report_json = tmp_path / "report.json"

    raw.to_csv(input_csv, index=False)

    report = clean_user_behavior_file(
        input_csv=input_csv,
        output_csv=output_csv,
        output_parquet=output_parquet,
        report_json=report_json,
        chunksize=1,
        partitions=4,
    )

    cleaned = pd.read_parquet(output_parquet)

    # Rows 1 and 2 are exact duplicates even though chunksize=1
    # forces them into separate input chunks.
    assert len(cleaned) == 3

    assert report["removed"]["exact_duplicate_rows"] == 1

    # The row with category 3002 shares the same
    # user/item/behavior/hour key but is not an exact duplicate.
    assert (
        report["suspected_duplicates"][
            "retained_group_count_after_exact_dedup"
        ]
        == 1
    )

    assert output_csv.exists()
    assert output_parquet.exists()
    assert report_json.exists()
