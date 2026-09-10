"""Functional tests for Stage4 Issue3 final model evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_evaluation import evaluate_final_candidates
from src.models.model_stacking import run_stacking_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Create compact selected data with a binary target for final evaluation."""

    value = np.arange(rows, dtype="float32") + offset
    return pd.DataFrame({
        "user_id": value.astype("int64") + 1, "item_id": value.astype("int64") + 100,
        "category_id": value.astype("int64") % 5, "label": (value.astype("int64") % 7 == 0).astype("int8"),
        "feature_a": value / 20, "feature_b": value % 9, "last_behavior_type_code": value.astype("int64") % 4,
    })


def _stacking_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build compact Issue2.5 artifacts that Issue3 must load rather than retrain."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _frame(180, 0).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet", index=False)
    _frame(80, 300).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet", index=False)
    _frame(90, 1000).to_parquet(split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet", index=False)
    _frame(70, 2000).to_parquet(split_dir / "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet", index=False)
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
    models = tmp_path / "models"
    stacking_reports = tmp_path / "stacking_reports"
    stack = run_stacking_fusion(split_dir, features, weights, parameters, pool, models, stacking_reports, base_model_count=2, folds=3, random_state=7)
    return split_dir, features, models, stack["manifest_path"]


def test_final_evaluation_writes_metrics_plots_and_test_predictions(tmp_path: Path) -> None:
    """Issue3 loads fixed Issue2.5 models and evaluates the test set once."""

    split_dir, features, models, manifest = _stacking_artifacts(tmp_path)
    result = evaluate_final_candidates(split_dir, features, manifest, models, tmp_path / "evaluation_reports")
    metrics = pd.read_csv(result["metrics_path"])
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    predictions = pd.read_parquet(result["test_prediction_path"])
    assert set(metrics["dataset_split"]) == {"validation", "test"}
    assert {"auc", "accuracy", "precision", "recall", "f1", "logloss"}.issubset(metrics.columns)
    assert summary["test_evaluation_count"] == 1
    assert summary["test_used_for_tuning_or_selection"] is False
    assert predictions.shape[0] == 70
    report_root = tmp_path / "evaluation_reports"
    assert (report_root / "test_roc_curves.png").is_file()
    assert (report_root / "test_pr_curves.png").is_file()
    assert (report_root / "test_confusion_matrices.png").is_file()
    assert (report_root / "test_threshold_sensitivity.csv").is_file()
    assert result["report_path"].is_file()


def test_final_evaluation_cli_runs(tmp_path: Path) -> None:
    """The Issue3 command-line entry point loads and evaluates existing artifacts."""

    split_dir, features, models, manifest = _stacking_artifacts(tmp_path)
    run = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_final_models.py"),
            "--split-dir", str(split_dir), "--feature-list", str(features),
            "--issue2-5-manifest", str(manifest), "--models-dir", str(models),
            "--reports-dir", str(tmp_path / "evaluation_reports"),
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "status: passed" in run.stdout
