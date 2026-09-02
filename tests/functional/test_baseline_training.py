"""Functional tests for fixed-parameter traditional-model baseline training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline_training import MODEL_NAMES, train_baseline_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(rows: int, positive_every: int, offset: int) -> pd.DataFrame:
    """Create a compact numeric selected split with both label classes."""

    values = np.arange(rows, dtype="float32") + offset
    labels = ((values.astype("int64") % positive_every) == 0).astype("int8")
    return pd.DataFrame(
        {
            "user_id": values.astype("int64") + 1,
            "item_id": values.astype("int64") + 1000,
            "category_id": (values.astype("int64") % 7) + 1,
            "label": labels,
            "feature_a": values / 10,
            "feature_b": (values % 5).astype("float32"),
            "last_behavior_type_code": (values.astype("int64") % 4).astype("int32"),
            "is_synthetic": False,
        }
    )


def _inputs(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    """Write selected split inputs plus Issue4/Issue5 configuration files."""

    paths = {}
    for split, rows, offset in (("train", 160, 0), ("validation", 80, 1000), ("test", 70, 2000)):
        path = tmp_path / f"{split}.parquet"
        _frame(rows, 8, offset).to_parquet(path, index=False)
        paths[split] = path
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"selected_model_features": ["feature_a", "feature_b", "last_behavior_type_code"]}), encoding="utf-8")
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"class_weight": {"0": 0.6, "1": 3.0}}), encoding="utf-8")
    return paths, features, weights


def test_train_all_baseline_models_writes_required_artifacts(tmp_path) -> None:
    """Train all four baselines and preserve validation/test prediction traceability."""

    paths, features, weights = _inputs(tmp_path)
    result = train_baseline_models(
        paths, features, tmp_path / "models" / "baselines", tmp_path / "reports",
        class_weight_path=weights, training_strategy="class_weight", random_state=7,
    )
    comparison = pd.read_csv(result["comparison_path"])
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert set(comparison["model_name"]) == set(MODEL_NAMES)
    assert comparison.shape[0] == 8
    assert set(comparison["dataset_split"]) == {"validation", "test"}
    assert summary["test_used_for_tuning_or_selection"] is False
    assert summary["model_selection_decision"] is None
    for name in MODEL_NAMES:
        assert (tmp_path / "models" / "baselines" / "artifacts" / f"{name}.joblib").is_file()
        assert (tmp_path / "models" / "baselines" / "logs" / f"{name}_run.json").is_file()
        for split, expected_rows in (("validation", 80), ("test", 70)):
            prediction = pd.read_parquet(tmp_path / "models" / "baselines" / "predictions" / f"{name}_{split}_predictions.parquet")
            assert prediction.shape[0] == expected_rows
            assert list(prediction.columns) == ["user_id", "item_id", "category_id", "label", "prediction_score", "prediction_label", "model_name", "dataset_split"]


def test_baseline_training_cli_supports_one_requested_model(tmp_path) -> None:
    """Run the command-line interface without requiring a test-data selection step."""

    paths, features, _ = _inputs(tmp_path)
    run = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train_baseline_models.py"),
            "--train-input", str(paths["train"]), "--validation-input", str(paths["validation"]),
            "--test-input", str(paths["test"]), "--feature-list", str(features),
            "--models-dir", str(tmp_path / "models"), "--reports-dir", str(tmp_path / "reports"),
            "--training-strategy", "baseline", "--models", "logistic_regression",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "status: passed" in run.stdout
