"""Performance test for a fixed Logistic Regression baseline training run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline_training import train_baseline_models


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Build a numeric selected split suitable for bounded baseline timing."""

    values = np.arange(rows, dtype="float32") + offset
    return pd.DataFrame(
        {
            "user_id": values.astype("int64"), "item_id": values.astype("int64") + 10_000,
            "category_id": values.astype("int64") % 17, "label": (values.astype("int64") % 20 == 0).astype("int8"),
            "feature_a": values / 100, "feature_b": values % 11, "feature_c": values % 5,
        }
    )


def test_baseline_training_performance(tmp_path: Path) -> None:
    """Record Logistic Regression runtime for 20,000 train and 8,000 evaluation rows."""

    paths = {}
    for split, rows, offset in (("train", 12_000, 0), ("validation", 5_000, 20_000), ("test", 3_000, 30_000)):
        path = tmp_path / f"{split}.parquet"
        _frame(rows, offset).to_parquet(path, index=False)
        paths[split] = path
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "feature_c"]}), encoding="utf-8")
    started = time.perf_counter()
    result = train_baseline_models(paths, features, tmp_path / "models", tmp_path / "reports", model_names=("logistic_regression",), training_strategy="baseline")
    runtime = time.perf_counter() - started
    payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    payload["test_wall_runtime_seconds"] = round(runtime, 6)
    (tmp_path / "baseline_training_performance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_seconds": payload["runtime_seconds"], "test_wall_runtime_seconds": payload["test_wall_runtime_seconds"]}))
    assert 0 < runtime < 30
    assert payload["models"]["logistic_regression"]["fit_seconds"] > 0
