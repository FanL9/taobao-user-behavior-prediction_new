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


def test_full_cleaning_pipeline_keeps_low_frequency_repeats(
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
            "user_id": ["1001", "1001", "1001", "1002"],
            "item_id": ["2001", "2001", "2001", "2002"],
            "item_category": ["3001", "3001", "3002", "3003"],
            "behavior_type": ["1", "1", "1", "4"],
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

    # The same four-field key occurs only three times, so all three
    # records must be kept, even when two rows are exactly identical.
    assert len(cleaned) == 4
    assert report["removed"]["high_frequency_duplicate_rows"] == 0

    duplicate_report = report["duplicate_handling"]
    assert duplicate_report["threshold"] == 60
    assert duplicate_report["normal_frequency_group_count"] == 1
    assert duplicate_report["normal_frequency_record_count"] == 3
    assert duplicate_report["high_frequency_group_count"] == 0

    assert output_csv.exists()
    assert output_parquet.exists()
    assert report_json.exists()


def test_full_cleaning_pipeline_frequency_threshold_boundaries(
    tmp_path,
):
    records = []

    def add_group(
        user_id,
        item_id,
        count,
        first_category,
        other_category,
    ):
        for index in range(count):
            records.append(
                {
                    "time": "2025-11-18 09",
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "item_category": str(
                        first_category
                        if index == 0
                        else other_category
                    ),
                    "behavior_type": "1",
                }
            )

    # 2 and 59 are fully retained.
    add_group(2000, 3000, 2, 5001, 5001)
    add_group(2001, 3001, 59, 6001, 6001)

    # 60 and 61 are high-frequency groups; only first survives.
    add_group(2002, 3002, 60, 7001, 7002)
    add_group(2003, 3003, 61, 8001, 8002)

    raw = pd.DataFrame(records)

    input_csv = tmp_path / "frequency_raw.csv"
    output_csv = tmp_path / "frequency_clean.csv"
    output_parquet = tmp_path / "frequency_clean.parquet"
    report_json = tmp_path / "frequency_report.json"

    raw.to_csv(input_csv, index=False)

    report = clean_user_behavior_file(
        input_csv=input_csv,
        output_csv=output_csv,
        output_parquet=output_parquet,
        report_json=report_json,
        chunksize=17,
        partitions=4,
    )

    cleaned = pd.read_parquet(output_parquet)

    group_2 = cleaned[cleaned["user_id"] == 2000]
    group_59 = cleaned[cleaned["user_id"] == 2001]
    group_60 = cleaned[cleaned["user_id"] == 2002]
    group_61 = cleaned[cleaned["user_id"] == 2003]

    assert len(group_2) == 2
    assert len(group_59) == 59
    assert len(group_60) == 1
    assert len(group_61) == 1

    # "First" means first encountered in the original input stream.
    assert int(group_60.iloc[0]["category_id"]) == 7001
    assert int(group_61.iloc[0]["category_id"]) == 8001

    # 60 group removes 59; 61 group removes 60.
    assert report["removed"]["high_frequency_duplicate_rows"] == 119

    duplicate_report = report["duplicate_handling"]
    assert duplicate_report["threshold"] == 60
    assert duplicate_report["normal_frequency_group_count"] == 2
    assert duplicate_report["normal_frequency_record_count"] == 61
    assert duplicate_report["high_frequency_group_count"] == 2
    assert duplicate_report["high_frequency_record_count"] == 121
    assert duplicate_report["removed_high_frequency_rows"] == 119
    assert duplicate_report["retained_high_frequency_rows"] == 2
