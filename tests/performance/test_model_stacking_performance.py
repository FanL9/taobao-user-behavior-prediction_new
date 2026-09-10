"""Performance test for bounded Stage4 Issue2.5 OOF stacking."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_stacking import run_stacking_fusion


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Create deterministic numeric data for fusion timing."""

    values = np.arange(rows, dtype="float32") + offset
    return pd.DataFrame({
        "user_id": values.astype("int64"), "item_id": values.astype("int64") + 10_000,
        "category_id": values.astype("int64") % 9, "label": (values.astype("int64") % 17 == 0).astype("int8"),
        "feature_a": values / 100, "feature_b": values % 11, "last_behavior_type_code": values.astype("int64") % 4,
    })


def test_stacking_performance(tmp_path: Path) -> None:
    """Record a three-fold two-model validation-only stacking run."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _frame(2_000, 0).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet", index=False)
    _frame(120, 2_200).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet", index=False)
    _frame(600, 5_000).to_parquet(split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet", index=False)
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "last_behavior_type_code"]}), encoding="utf-8")
    weights = tmp_path / "class_weight_config.json"
    weights.write_text(json.dumps({"class_weight": {"0": 0.6, "1": 3.0}}), encoding="utf-8")
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({
        "baseline__lightgbm": {"best_parameters": {"n_estimators": 20, "num_leaves": 7, "max_depth": 4, "learning_rate": 0.1, "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0}},
        "undersampled__logistic_regression": {"best_parameters": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 200}},
    }), encoding="utf-8")
    pool = tmp_path / "pool.csv"
    pd.DataFrame([
        {"candidate": "baseline__lightgbm", "auc": 0.8, "average_precision": 0.2, "logloss": 0.3},
        {"candidate": "undersampled__logistic_regression", "auc": 0.7, "average_precision": 0.1, "logloss": 0.4},
    ]).to_csv(pool, index=False)
    started = time.perf_counter()
    result = run_stacking_fusion(split_dir, features, weights, parameters, pool, tmp_path / "models", tmp_path / "reports", base_model_count=2, folds=3)
    wall_seconds = time.perf_counter() - started
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    summary["test_wall_runtime_seconds"] = round(wall_seconds, 6)
    (tmp_path / "stacking_performance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert summary["test_data_read"] is False
    assert 0 < wall_seconds < 60
