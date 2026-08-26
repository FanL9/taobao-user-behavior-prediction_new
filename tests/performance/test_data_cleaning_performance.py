import time

import pandas as pd

from src.data.cleaning_pipeline import clean_user_behavior_file


def test_data_cleaning_pipeline_performance(tmp_path):
    row_count = 50_000

    raw = pd.DataFrame(
        {
            "time": ["2025-11-18 09"] * row_count,
            "user_id": range(1, row_count + 1),
            "item_id": range(
                100_001,
                100_001 + row_count,
            ),
            "item_category": [3001] * row_count,
            "behavior_type": [1] * row_count,
        }
    )

    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "clean.csv"
    output_parquet = tmp_path / "clean.parquet"
    report_json = tmp_path / "report.json"

    raw.to_csv(
        input_csv,
        index=False,
    )

    started = time.perf_counter()

    report = clean_user_behavior_file(
        input_csv=input_csv,
        output_csv=output_csv,
        output_parquet=output_parquet,
        report_json=report_json,
        chunksize=10_000,
        partitions=8,
    )

    elapsed = time.perf_counter() - started

    assert report["input"]["rows"] == row_count
    assert report["output"]["rows"] == row_count

    assert output_csv.exists()
    assert output_parquet.exists()
    assert report_json.exists()

    # Generous threshold for normal development machines.
    assert elapsed < 15
