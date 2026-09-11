"""Functional tests for Stage 5 frozen-artifact batch inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.inference.batch_inference import (
    load_inference_contract,
    predict_purchase_probabilities,
    run_batch_inference,
    transform_inference_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISSING = "\u0000__MISSING__"


def _write_inference_fixture(tmp_path: Path, rows: int = 24) -> dict[str, Path]:
    """Create compact raw features and frozen Stage 3/4 artifacts."""

    reports3 = tmp_path / "reports" / "stage3"
    selection_dir = tmp_path / "reports" / "stage4" / "stacking_fusion"
    evaluation_dir = tmp_path / "reports" / "stage4" / "model_evaluation"
    models_dir = tmp_path / "models" / "traditional_ml" / "stacking" / "models"
    configs = tmp_path / "configs"
    for directory in (reports3, selection_dir, evaluation_dir, models_dir, configs):
        directory.mkdir(parents=True, exist_ok=True)

    feature_list = reports3 / "final_model_features.json"
    feature_list.write_text(
        json.dumps(
            {
                "tracking_columns": ["user_id", "item_id", "category_id"],
                "model_feature_columns": ["numeric_feature", "behavior_type_code"],
                "model_feature_count": 2,
            }
        ),
        encoding="utf-8",
    )
    rules = reports3 / "preprocessing_rules.json"
    rules.write_text(
        json.dumps(
            {
                "tracking_columns": ["user_id", "item_id", "category_id"],
                "numeric_rules": {
                    "numeric_feature": {"fill_value": 10.0, "mean": 10.0, "scale": 2.0}
                },
                "categorical_rules": {
                    "behavior_type": {MISSING: 0, "buy": 1, "cart": 2, "pv": 3}
                },
                "categorical_encoding_policy": {
                    "method": "training-fitted ordinal encoding",
                    "unknown_category_code": -1,
                },
            }
        ),
        encoding="utf-8",
    )
    selection = selection_dir / "issue3_candidate_manifest.json"
    selection.write_text(
        json.dumps(
            {
                "validation_leader": "baseline__xgboost",
                "test_data_read": False,
                "test_used_for_training_or_selection": False,
            }
        ),
        encoding="utf-8",
    )
    evaluation = evaluation_dir / "model_evaluation_summary.json"
    evaluation.write_text(
        json.dumps(
            {
                "candidates": ["baseline__xgboost", "stacking_logistic_regression"],
                "test_evaluation_count": 1,
                "test_used_for_tuning_or_selection": False,
            }
        ),
        encoding="utf-8",
    )

    train_x = pd.DataFrame(
        {
            "numeric_feature": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "behavior_type_code": [3, 3, 2, 2, 1, 1],
        }
    )
    train_y = np.array([0, 0, 0, 1, 1, 1], dtype="int8")
    estimator = LogisticRegression(random_state=42).fit(train_x, train_y)
    model_path = models_dir / "base_models.joblib"
    joblib.dump({"baseline__xgboost": estimator}, model_path)

    config = configs / "stage5_inference.json"
    config.write_text(
        json.dumps(
            {
                "final_candidate": "baseline__xgboost",
                "model_artifact": str(model_path),
                "model_bundle_key": "baseline__xgboost",
                "preprocessing_rules": str(rules),
                "feature_list": str(feature_list),
                "selection_manifest": str(selection),
                "evaluation_summary": str(evaluation),
                "prediction_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    index = np.arange(rows)
    input_frame = pd.DataFrame(
        {
            "user_id": index + 1,
            "item_id": index + 1001,
            "category_id": index % 5,
            "numeric_feature": np.where(index % 7 == 0, np.nan, 8.0 + index % 6),
            "behavior_type": np.where(index % 5 == 0, "unknown", np.where(index % 2 == 0, "pv", "cart")),
            "unused_column": "ignored",
        }
    )
    input_path = tmp_path / "inference_input.csv"
    input_frame.to_csv(input_path, index=False)
    return {
        "config": config,
        "input": input_path,
        "model": model_path,
    }


def test_contract_uses_frozen_stage3_stage4_artifacts(tmp_path: Path) -> None:
    fixture = _write_inference_fixture(tmp_path)
    contract = load_inference_contract(
        fixture["config"], project_root=tmp_path, require_model=True
    )
    assert contract.final_candidate == "baseline__xgboost"
    assert contract.model_bundle_key == "baseline__xgboost"
    assert contract.model_features == ("numeric_feature", "behavior_type_code")
    assert contract.raw_feature_columns == ("numeric_feature", "behavior_type")


def test_transform_reuses_fill_scale_and_unknown_category_code(tmp_path: Path) -> None:
    fixture = _write_inference_fixture(tmp_path)
    contract = load_inference_contract(fixture["config"], project_root=tmp_path)
    raw = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [11, 12],
            "category_id": [21, 22],
            "numeric_feature": [12.0, np.nan],
            "behavior_type": ["pv", "never_seen_in_training"],
        }
    )
    transformed = transform_inference_features(raw, contract)
    assert transformed.columns.tolist() == ["numeric_feature", "behavior_type_code"]
    assert transformed["numeric_feature"].tolist() == [1.0, 0.0]
    assert transformed["behavior_type_code"].tolist() == [3, -1]


def test_batch_inference_outputs_user_item_purchase_probabilities(tmp_path: Path) -> None:
    fixture = _write_inference_fixture(tmp_path)
    output = tmp_path / "predictions.csv"
    summary_path = tmp_path / "summary.json"
    summary = run_batch_inference(
        fixture["input"],
        output,
        fixture["config"],
        summary_path,
        batch_size=7,
        project_root=tmp_path,
    )
    predictions = pd.read_csv(output)
    assert len(predictions) == 24
    assert predictions.columns.tolist() == [
        "user_id", "item_id", "category_id", "purchase_probability", "predicted_purchase"
    ]
    assert predictions["purchase_probability"].between(0, 1).all()
    assert predictions["predicted_purchase"].isin([0, 1]).all()
    assert summary["model_training_performed"] is False
    assert summary["feature_selection_performed"] is False
    assert summary["preprocessing_refit_performed"] is False
    assert json.loads(summary_path.read_text(encoding="utf-8"))["row_count"] == 24


def test_predict_frame_preserves_tracking_columns(tmp_path: Path) -> None:
    fixture = _write_inference_fixture(tmp_path, rows=4)
    contract = load_inference_contract(fixture["config"], project_root=tmp_path)
    model_bundle = joblib.load(fixture["model"])
    raw = pd.read_csv(fixture["input"])
    output = predict_purchase_probabilities(raw, model_bundle["baseline__xgboost"], contract)
    assert output[["user_id", "item_id", "category_id"]].equals(
        raw[["user_id", "item_id", "category_id"]]
    )


def test_batch_inference_cli_runs(tmp_path: Path) -> None:
    fixture = _write_inference_fixture(tmp_path)
    output = tmp_path / "cli_predictions.csv"
    summary = tmp_path / "cli_summary.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_batch_inference.py"),
            "--input", str(fixture["input"]),
            "--output", str(output),
            "--config", str(fixture["config"]),
            "--summary-output", str(summary),
            "--batch-size", "5",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "status: passed" in run.stdout
    assert output.is_file()
    assert summary.is_file()
