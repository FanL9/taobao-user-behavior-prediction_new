"""Functional checks for Stage4 Issue4 interpretation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.model_interpretability import analyze_model_interpretability


def test_interpretability_writes_feature_and_error_outputs(tmp_path: Path) -> None:
    """Issue4 loads a fixed model and writes explanation/error artifacts only."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    values = np.arange(120, dtype="float32")
    frame = pd.DataFrame({"user_id": values.astype("int64"), "item_id": values.astype("int64") + 100, "category_id": values.astype("int64") % 4, "label": (values.astype("int64") % 9 == 0).astype("int8"), "user_event_count": values % 3, "item_total_count": values % 5, "feature_c": values / 10})
    frame.to_parquet(split_dir / "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet", index=False)
    features = tmp_path / "features.json"
    feature_names = ["user_event_count", "item_total_count", "feature_c"]
    features.write_text(json.dumps({"selected_model_features": feature_names}), encoding="utf-8")
    preprocessing = tmp_path / "preprocessing_rules.json"
    preprocessing.write_text(json.dumps({"numeric_rules": {"user_event_count": {"mean": 0.0, "scale": 1.0}, "item_total_count": {"mean": 0.0, "scale": 1.0}}}), encoding="utf-8")
    model = XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.2, eval_metric="logloss", random_state=7)
    model.fit(frame[feature_names], frame["label"])
    model_root = tmp_path / "models" / "stacking" / "models"
    model_root.mkdir(parents=True)
    joblib.dump({"baseline__xgboost": model}, model_root / "base_models.joblib")
    prediction_path = tmp_path / "predictions.parquet"
    scores = model.predict_proba(frame[feature_names])[:, 1]
    pd.DataFrame({"user_id": frame["user_id"], "item_id": frame["item_id"], "category_id": frame["category_id"], "label": frame["label"], "score_baseline__xgboost": scores}).to_parquet(prediction_path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"validation_leader": "baseline__xgboost"}), encoding="utf-8")
    issue3 = tmp_path / "issue3"
    issue3.mkdir()
    pd.DataFrame([{"candidate": "baseline__xgboost", "threshold": 0.05, "f1": 0.2}]).to_csv(issue3 / "test_threshold_sensitivity.csv", index=False)
    result = analyze_model_interpretability(split_dir, features, manifest, prediction_path, tmp_path / "models", issue3, tmp_path / "reports", shap_sample_size=80, error_sample_size=10, random_state=7, preprocessing_rules_path=preprocessing)
    assert result["model_importance_path"].is_file()
    assert result["shap_importance_path"].is_file()
    assert result["false_positive_path"].is_file()
    assert result["false_negative_path"].is_file()
    assert result["segments_path"].is_file()
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert summary["model_training_or_tuning_performed"] is False
