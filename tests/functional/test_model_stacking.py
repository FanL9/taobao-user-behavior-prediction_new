"""Functional tests for leakage-safe Stage4 Issue2.5 stacking."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_stacking import run_stacking_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Create compact data with both labels and numeric selected features."""

    value = np.arange(rows, dtype="float32") + offset
    return pd.DataFrame(
        {
            "user_id": value.astype("int64") + 1,
            "item_id": value.astype("int64") + 100,
            "category_id": value.astype("int64") % 5,
            "label": (value.astype("int64") % 7 == 0).astype("int8"),
            "feature_a": value / 20,
            "feature_b": value % 9,
            "last_behavior_type_code": value.astype("int64") % 4,
        }
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Create Issue2-compatible artifacts and the datasets used by selected models."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _frame(180, 0).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet", index=False)
    _frame(80, 300).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet", index=False)
    _frame(90, 1000).to_parquet(split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet", index=False)
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "last_behavior_type_code"]}), encoding="utf-8")
    weights = tmp_path / "class_weight_config.json"
    weights.write_text(json.dumps({"class_weight": {"0": 0.6, "1": 3.0}}), encoding="utf-8")
    parameters = tmp_path / "best_tuned_model_parameters.json"
    parameters.write_text(json.dumps({
        "baseline__lightgbm": {"best_parameters": {"n_estimators": 20, "num_leaves": 7, "max_depth": 4, "learning_rate": 0.1, "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0}},
        "undersampled__logistic_regression": {"best_parameters": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 200}},
    }), encoding="utf-8")
    pool = tmp_path / "model_fusion_candidate_pool.csv"
    pd.DataFrame([
        {"candidate": "baseline__lightgbm", "auc": 0.8, "average_precision": 0.2, "logloss": 0.3, "issue2_5_eligible": True},
        {"candidate": "undersampled__logistic_regression", "auc": 0.75, "average_precision": 0.1, "logloss": 0.4, "issue2_5_eligible": True},
    ]).to_csv(pool, index=False)
    return split_dir, features, weights, parameters, pool


def test_stacking_generates_oof_and_validation_only_artifacts(tmp_path: Path) -> None:
    """OOF base predictions train the meta model without accessing a test split."""

    split_dir, features, weights, parameters, pool = _inputs(tmp_path)
    result = run_stacking_fusion(
        split_dir, features, weights, parameters, pool, tmp_path / "models", tmp_path / "reports",
        base_model_count=2, folds=3, random_state=7,
    )
    comparison = pd.read_csv(result["comparison_path"])
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    oof = pd.read_parquet(result["oof_path"])
    validation = pd.read_parquet(result["validation_path"])
    assert set(comparison["variant"]) == {"single_model", "stacking"}
    assert manifest["test_data_read"] is False
    assert len(manifest["base_candidates"]) == 2
    assert oof.shape[0] == 180
    assert validation.shape[0] == 90
    assert "stacking_prediction_score" in validation
    assert oof.filter(like="score_").notna().all().all()


def test_stacking_cli_runs(tmp_path: Path) -> None:
    """The command-line entry point runs a two-model, three-fold fusion."""

    split_dir, features, weights, parameters, pool = _inputs(tmp_path)
    run = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_stacking_fusion.py"),
            "--split-dir", str(split_dir), "--feature-list", str(features),
            "--class-weights", str(weights), "--issue2-parameters", str(parameters),
            "--issue2-candidate-pool", str(pool), "--models-dir", str(tmp_path / "models"),
            "--reports-dir", str(tmp_path / "reports"), "--base-model-count", "2", "--folds", "3",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "status: passed" in run.stdout
