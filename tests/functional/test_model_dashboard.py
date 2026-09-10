"""Functional tests for the existing Streamlit model dashboard section."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboards.eda.model_dashboard import REQUIRED_REPORTS, build_feature_figure, build_performance_figure, build_segment_figure, build_threshold_figure, read_model_dashboard_outputs


def _write_outputs(root: Path) -> None:
    """Create compact persisted Issue3/Issue4 output fixtures."""

    frames = {
        "metrics": pd.DataFrame([{"candidate": "baseline__xgboost", "dataset_split": "test", "auc": 0.8, "average_precision": 0.1, "f1": 0.2, "precision": 0.3, "recall": 0.4, "logloss": 0.2}]),
        "threshold": pd.DataFrame([{"candidate": "baseline__xgboost", "threshold": 0.1, "precision": 0.2, "recall": 0.5, "f1": 0.3}]),
        "confusion": pd.DataFrame([{"candidate": "baseline__xgboost", "tn": 10, "fp": 2, "fn": 3, "tp": 4}]),
        "segments": pd.DataFrame([{"segment": "all_test_samples", "precision": 0.2, "recall": 0.5, "f1": 0.3}, {"segment": "long_tail_item", "precision": 0.1, "recall": 0.4, "f1": 0.2}]),
        "importance": pd.DataFrame([{"feature": "item_cart_count", "mean_abs_shap": 0.4, "mean_shap": 0.1}]),
        "direction": pd.DataFrame([{"feature": "item_cart_count", "mean_shap_on_actual_positive": 0.1, "mean_shap_on_actual_negative": -0.1, "positive_prediction_direction": "positive"}]),
        "errors": pd.DataFrame([{"tn": 10, "fp": 2, "fn": 3, "tp": 4}]),
        "false_positive": pd.DataFrame([{"user_id": 1, "prediction_score": 0.8}]),
        "false_negative": pd.DataFrame([{"user_id": 2, "prediction_score": 0.2}]),
    }
    for name, (folder, filename) in REQUIRED_REPORTS.items():
        path = root / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(path, index=False)


def test_model_dashboard_loads_persisted_outputs_and_builds_charts(tmp_path: Path) -> None:
    """Dashboard charts use persisted Stage4 files without model execution."""

    _write_outputs(tmp_path)
    data = read_model_dashboard_outputs(tmp_path)
    assert build_performance_figure(data["metrics"], "auc", "test").data
    assert build_threshold_figure(data["threshold"], "baseline__xgboost").data
    assert build_segment_figure(data["segments"]).data
    assert build_feature_figure(data["importance"]).data
