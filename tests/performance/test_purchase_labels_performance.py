"""Performance test for future-one-day purchase label generation."""

from __future__ import annotations

import json
import time

import pandas as pd

from src.features.labels import generate_purchase_labels


def test_purchase_label_generation_performance(tmp_path) -> None:
    """Record runtime and process-resource usage for a 50,000-row sample."""

    rows = 50_000
    row_index = pd.RangeIndex(rows)
    split = pd.Series(["train", "validation", "test"]).take(row_index % 3)
    window_values = {
        "train": ("2025-11-18", "2025-12-07", "2025-12-08"),
        "validation": ("2025-12-09", "2025-12-14", "2025-12-15"),
        "test": ("2025-12-16", "2025-12-17", "2025-12-18"),
    }
    wide = pd.DataFrame(
        {
            "dataset_split": split.to_numpy(),
            "user_id": row_index.to_numpy() + 1,
            "item_id": row_index.to_numpy() + 100_001,
            "history_start": pd.to_datetime(split.map(lambda x: window_values[x][0])),
            "history_end": pd.to_datetime(split.map(lambda x: window_values[x][1])),
            "label_date": pd.to_datetime(split.map(lambda x: window_values[x][2])),
            "feature_value": row_index.to_numpy(),
        }
    )
    positives = wide.iloc[::10]
    clean = pd.DataFrame(
        {
            "user_id": positives["user_id"].to_numpy(),
            "item_id": positives["item_id"].to_numpy(),
            "behavior_type": 4,
            "behavior_date": positives["label_date"].dt.strftime("%Y-%m-%d"),
        }
    )
    wide_path = tmp_path / "wide.parquet"
    clean_path = tmp_path / "clean.parquet"
    wide.to_parquet(wide_path, index=False)
    clean.to_parquet(clean_path, index=False)

    started_at = time.perf_counter()
    result = generate_purchase_labels(
        wide_path,
        clean_path,
        tmp_path / "labeled.parquet",
        tmp_path / "statistics.json",
        batch_size=10_000,
    )
    wall_runtime = time.perf_counter() - started_at
    metrics = result["report"]["performance"]
    metrics["test_wall_runtime_seconds"] = round(wall_runtime, 6)
    (tmp_path / "purchase_label_performance.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))

    overall = result["report"]["statistics"]["overall"]
    assert overall["sample_count"] == rows
    assert overall["positive_count"] == len(positives)
    assert 0 < wall_runtime < 15
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
