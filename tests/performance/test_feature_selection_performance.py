"""Performance test for training-only feature selection."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.features.feature_selection import select_model_features


def _performance_frame(rows: int, offset: int) -> pd.DataFrame:
    """Create one synthetic preprocessed partition for performance measurement."""

    index = np.arange(rows) + offset
    base = (index % 997).astype("float32")
    return pd.DataFrame(
        {
            "user_id": index + 1,
            "item_id": index + 100_001,
            "category_id": index % 100 + 1,
            "label": (index % 10 == 0).astype("int8"),
            "feature_a": base,
            "feature_a_duplicate": base * 2,
            "feature_b": (index % 31).astype("float32"),
            "constant_feature": 0.0,
            "feature_c": (index % 17).astype("float32"),
            "outlier_feature": np.where(index % 1_000 == 0, 10.0, 0.0),
        }
    )


def test_feature_selection_performance(tmp_path) -> None:
    """Record runtime and resources for 50,000 rows across three datasets."""

    counts = {"train": 30_000, "validation": 12_000, "test": 8_000}
    inputs = {}
    offset = 0
    for split, rows in counts.items():
        path = tmp_path / f"{split}.parquet"
        _performance_frame(rows, offset).to_parquet(path, index=False)
        inputs[split] = path
        offset += rows

    started_at = time.perf_counter()
    result = select_model_features(
        inputs,
        tmp_path / "selected",
        tmp_path / "features.json",
        tmp_path / "report.json",
        batch_size=10_000,
    )
    wall_runtime = time.perf_counter() - started_at
    metrics = result["report"]["performance"]
    metrics["test_wall_runtime_seconds"] = round(wall_runtime, 6)
    (tmp_path / "feature_selection_performance.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))

    assert result["feature_list"]["model_feature_count"] == 4
    assert result["report"]["dataset_statistics"]["train"]["sample_count"] == 30_000
    assert 0 < wall_runtime < 15
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
