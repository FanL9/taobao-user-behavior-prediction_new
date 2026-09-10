"""Functional tests for validation-only Stage4 Issue2 Optuna tuning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_tuning import TUNING_CANDIDATES, tune_all_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(rows: int, offset: int) -> pd.DataFrame:
    """Build a compact numeric split containing both binary label classes."""

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


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the exact Issue2 train and validation input versions."""

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _frame(180, 0).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet", index=False)
    _frame(180, 200).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_smote.parquet", index=False)
    _frame(80, 500).to_parquet(split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet", index=False)
    _frame(90, 1000).to_parquet(split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet", index=False)
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "last_behavior_type_code"]}), encoding="utf-8")
    class_weight_path = tmp_path / "class_weight_config.json"
    class_weight_path.write_text(json.dumps({"class_weight": {"0": 0.6, "1": 3.0}}), encoding="utf-8")
    return split_dir, feature_path, class_weight_path


def test_tuning_all_combinations_uses_validation_only(tmp_path: Path) -> None:
    """Run one uniform Optuna trial for every Issue2 model-strategy combination."""

    split_dir, features, weights = _inputs(tmp_path)
    result = tune_all_models(
        split_dir, features, tmp_path / "models", tmp_path / "reports", weights,
        n_trials=1, random_state=7,
    )
    search = pd.read_csv(result["search_path"])
    comparison = pd.read_csv(result["comparison_path"])
    parameters = json.loads(result["parameters_path"].read_text(encoding="utf-8"))
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    pool = pd.read_csv(result["candidate_pool_path"])

    assert set(search["candidate"]) == set(TUNING_CANDIDATES)
    assert search.shape[0] == len(TUNING_CANDIDATES)
    assert set(comparison["dataset_split"]) == {"validation"}
    assert set(parameters) == set(TUNING_CANDIDATES)
    assert summary["test_data_read"] is False
    assert summary["trial_budget_per_candidate"] == 1
    assert pool.shape[0] == len(TUNING_CANDIDATES)
    assert pool["issue2_5_eligible"].all()
    first = next(iter(parameters.values()))
    assert first["selection_split"] == "validation"
    assert first["test_data_read"] is False


def test_tuning_cli_runs_one_candidate(tmp_path: Path) -> None:
    """The CLI accepts a bounded trial budget and produces no test prediction."""

    split_dir, features, weights = _inputs(tmp_path)
    run = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "tune_primary_models.py"),
            "--split-dir", str(split_dir), "--feature-list", str(features),
            "--class-weights", str(weights), "--models-dir", str(tmp_path / "models"),
            "--reports-dir", str(tmp_path / "reports"), "--candidates", "baseline__lightgbm",
            "--n-trials", "1",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "status: passed" in run.stdout
