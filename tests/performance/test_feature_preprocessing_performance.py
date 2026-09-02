"""Performance test for train-fitted feature preprocessing."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.features.preprocessing import preprocess_feature_datasets


def _performance_frame(split: str, rows: int, offset: int) -> pd.DataFrame:
    """Build one valid labeled partition for the preprocessing performance test."""

    windows = {
        "train": ("2025-11-18", "2025-12-07", "2025-12-08"),
        "validation": ("2025-12-09", "2025-12-14", "2025-12-15"),
        "test": ("2025-12-16", "2025-12-17", "2025-12-18"),
    }
    start, end, label_date = windows[split]
    index = np.arange(rows) + offset
    return pd.DataFrame(
        {
            "dataset_split": split,
            "user_id": index + 1,
            "item_id": index + 100_001,
            "category_id": index % 100 + 1,
            "history_start": pd.Timestamp(start),
            "history_end": pd.Timestamp(end),
            "label_date": pd.Timestamp(label_date),
            "last_behavior_date": pd.Timestamp(end),
            "numeric_feature": np.where(index % 17 == 0, np.nan, index % 97),
            "ratio_feature": np.where(index % 13 == 0, np.nan, index / 1000),
            "last_behavior_type": np.array(["pv", "fav", "cart", "buy"])[index % 4],
            "last_10_behavior_sequence": np.array(["pv", "pv→cart", "fav→pv"])[index % 3],
            "user_activity_level": np.array(["low", "medium", "high"])[index % 3],
            "label": (index % 10 == 0).astype("int8"),
        }
    )


def test_feature_preprocessing_performance(tmp_path) -> None:
    """Record runtime and resources for 50,000 rows across three datasets."""

    row_counts = {"train": 30_000, "validation": 12_000, "test": 8_000}
    inputs = {}
    offset = 0
    for split, rows in row_counts.items():
        path = tmp_path / f"{split}.parquet"
        _performance_frame(split, rows, offset).to_parquet(path, index=False)
        inputs[split] = path
        offset += rows

    started_at = time.perf_counter()
    result = preprocess_feature_datasets(
        inputs,
        tmp_path / "processed",
        tmp_path / "rules.json",
        tmp_path / "report.json",
        batch_size=10_000,
    )
    wall_runtime = time.perf_counter() - started_at
    metrics = result["report"]["performance"]
    metrics["test_wall_runtime_seconds"] = round(wall_runtime, 6)
    (tmp_path / "preprocessing_performance.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))

    overall = result["report"]["statistics"]["overall"]
    assert overall["sample_count"] == 50_000
    assert overall["positive_count"] == 5_000
    assert 0 < wall_runtime < 15
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
