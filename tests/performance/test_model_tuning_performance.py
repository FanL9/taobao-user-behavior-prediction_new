"""Performance test for bounded validation-only Issue2 Optuna tuning."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_tuning import tune_all_models


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Build selected numeric data with both labels for repeatable timing."""

    values = np.arange(rows, dtype="float32") + offset
    return pd.DataFrame(
        {
            "user_id": values.astype("int64"), "item_id": values.astype("int64") + 50_000,
            "category_id": values.astype("int64") % 17, "label": (values.astype("int64") % 19 == 0).astype("int8"),
            "feature_a": values / 100, "feature_b": values % 11, "feature_c": values % 5,
        }
    )


def test_model_tuning_performance(tmp_path: Path) -> None:
    """Record a two-trial LightGBM validation-only Optuna search."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _frame(3_000, 0).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet", index=False)
    _frame(1_200, 10_000).to_parquet(split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet", index=False)
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "feature_c"]}), encoding="utf-8")
    weights = tmp_path / "class_weight_config.json"
    weights.write_text(json.dumps({"class_weight": {"0": 0.6, "1": 3.0}}), encoding="utf-8")
    started = time.perf_counter()
    result = tune_all_models(
        split_dir, features, tmp_path / "models", tmp_path / "reports", weights,
        candidates=("baseline__lightgbm",), n_trials=2,
    )
    wall_seconds = time.perf_counter() - started
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    summary["test_wall_runtime_seconds"] = round(wall_seconds, 6)
    (tmp_path / "tuning_performance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert summary["trial_count"] == 2
    assert summary["test_data_read"] is False
    assert 0 < wall_seconds < 60
