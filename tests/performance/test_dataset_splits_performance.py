"""Performance test for time-ordered train/validation/test dataset generation."""

from __future__ import annotations

import json
import time

import pandas as pd

from src.data.dataset_splits import generate_time_ordered_datasets


def test_dataset_split_generation_performance(tmp_path) -> None:
    """Record runtime and process resources for a 50,000-row labeled sample."""

    rows = 50_000
    row_index = pd.RangeIndex(rows)
    splits = pd.Series(["train", "validation", "test"]).take(row_index % 3)
    windows = {
        "train": ("2025-11-18", "2025-12-07", "2025-12-08"),
        "validation": ("2025-12-09", "2025-12-14", "2025-12-15"),
        "test": ("2025-12-16", "2025-12-17", "2025-12-18"),
    }
    labeled = pd.DataFrame(
        {
            "dataset_split": splits.to_numpy(),
            "user_id": row_index.to_numpy() + 1,
            "item_id": row_index.to_numpy() + 100_001,
            "history_start": pd.to_datetime(splits.map(lambda value: windows[value][0])),
            "history_end": pd.to_datetime(splits.map(lambda value: windows[value][1])),
            "label_date": pd.to_datetime(splits.map(lambda value: windows[value][2])),
            "feature_value": row_index.to_numpy(),
            "label": (row_index.to_numpy() % 10 == 0).astype("int8"),
        }
    )
    source = tmp_path / "labeled.parquet"
    labeled.to_parquet(source, index=False)

    started_at = time.perf_counter()
    result = generate_time_ordered_datasets(
        source,
        tmp_path / "datasets",
        tmp_path / "statistics.json",
        batch_size=10_000,
    )
    wall_runtime = time.perf_counter() - started_at
    metrics = result["report"]["performance"]
    metrics["test_wall_runtime_seconds"] = round(wall_runtime, 6)
    (tmp_path / "dataset_split_performance.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))

    overall = result["report"]["statistics"]["overall"]
    assert overall["sample_count"] == rows
    assert overall["positive_count"] == 5_000
    assert 0 < wall_runtime < 15
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
