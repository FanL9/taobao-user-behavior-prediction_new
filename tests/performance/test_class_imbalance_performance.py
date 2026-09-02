"""Performance test for class-imbalance strategy preparation."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import time

from src.sampling.class_imbalance import prepare_class_imbalance_strategies


def _frame(rows: int, positives: int, offset: int) -> pd.DataFrame:
    """Build one selected dataset with positive rows positioned at the end."""

    index = np.arange(rows) + offset
    labels = np.zeros(rows, dtype="int8")
    labels[-positives:] = 1
    return pd.DataFrame(
        {
            "user_id": index + 1,
            "item_id": index + 100_001,
            "category_id": index % 100 + 1,
            "label": labels,
            "feature_a": (index % 997).astype("float32"),
            "feature_b": (index % 31).astype("float32"),
            "last_behavior_type_code": (index % 4).astype("int32"),
        }
    )


def test_class_imbalance_performance(tmp_path) -> None:
    """Record runtime and resource usage for 50,000 rows across three splits."""

    specifications = {"train": (30_000, 3_000), "validation": (12_000, 1_200), "test": (8_000, 800)}
    inputs = {}
    offset = 0
    for split, (rows, positives) in specifications.items():
        path = tmp_path / f"{split}.parquet"
        _frame(rows, positives, offset).to_parquet(path, index=False)
        inputs[split] = path
        offset += rows

    started_at = time.perf_counter()
    result = prepare_class_imbalance_strategies(
        inputs,
        tmp_path / "imbalance",
        tmp_path / "weights.json",
        tmp_path / "versions.json",
        tmp_path / "report.json",
        batch_size=10_000,
    )
    wall_runtime = time.perf_counter() - started_at
    metrics = result["report"]["performance"]
    metrics["test_wall_runtime_seconds"] = round(wall_runtime, 6)
    (tmp_path / "class_imbalance_performance.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))

    smote = result["report"]["training_strategy_statistics"]["smote"]
    assert smote["positive_count"] == smote["negative_count"] == 27_000
    assert result["report"]["training_strategy_statistics"]["undersampled"]["sample_count"] == 6_000
    assert 0 < wall_runtime < 15
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
