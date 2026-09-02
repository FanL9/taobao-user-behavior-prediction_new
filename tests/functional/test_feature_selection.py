"""Functional tests for training-only feature selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection import (
    SELECTED_FILENAMES,
    select_model_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(split: str) -> pd.DataFrame:
    """Create a compact compatible preprocessed input with screening cases."""

    offsets = {"train": 0, "validation": 10, "test": 20}
    offset = offsets[split]
    return pd.DataFrame(
        {
            "user_id": np.arange(1, 5) + offset,
            "item_id": np.arange(101, 105) + offset,
            "category_id": np.arange(201, 205) + offset,
            "label": [1, 0, 1, 0],
            "strong_feature": [0.0, 1.0, 2.0, 3.0],
            "duplicate_feature": [0.0, 2.0, 4.0, 6.0],
            "constant_feature": [1.0, 1.0, 1.0, 1.0],
            "future_signal": [3.0, 2.0, 1.0, 0.0],
            "nonfinite_feature": [0.0, np.nan, 1.0, 2.0] if split == "train" else [0.0, 1.0, 2.0, 3.0],
            "outlier_feature": [0.0, 0.0, 0.0, 100.0],
        }
    )


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write train, validation, and test feature-selection inputs."""

    paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.parquet"
        _frame(split).to_parquet(path, index=False)
        paths[split] = path
    return paths


def test_select_model_features_uses_training_rules_only(tmp_path) -> None:
    """Screening removes invalid/redundant/leakage fields and reuses one list."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "selected"
    list_path = tmp_path / "features.json"
    report_path = tmp_path / "report.json"
    result = select_model_features(
        inputs,
        output_directory,
        list_path,
        report_path,
        batch_size=1,
    )

    expected_features = ["strong_feature", "outlier_feature"]
    assert result["feature_list"]["model_feature_columns"] == expected_features
    for split, filename in SELECTED_FILENAMES.items():
        actual = pd.read_parquet(output_directory / filename)
        assert actual.columns.tolist() == [
            "user_id",
            "item_id",
            "category_id",
            "label",
            *expected_features,
        ]
        pd.testing.assert_frame_equal(
            actual[["user_id", "item_id", "category_id", "label"]],
            _frame(split)[["user_id", "item_id", "category_id", "label"]],
        )

    feature_list = json.loads(list_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["feature_list"] == feature_list
    assert result["report"] == report
    removed = report["removed_features"]
    assert "low_variance" in removed["constant_feature"]
    assert any(reason.startswith("high_correlation_with:strong_feature") for reason in removed["duplicate_feature"])
    assert "nonfinite_values_in_training" in removed["nonfinite_feature"]
    assert any(reason.startswith("suspected_future_information") for reason in removed["future_signal"])
    assert report["data_leakage_check"]["rules_fitted_on_train_only"] is True
    assert report["data_leakage_check"]["validation_and_test_refit_rules"] is False
    assert report["selection_thresholds"]["outlier_rate_removes_feature"] is False


def test_select_model_features_rejects_invalid_label(tmp_path) -> None:
    """A non-binary label prevents selected outputs from being produced."""

    inputs = _write_inputs(tmp_path)
    test = _frame("test")
    test.loc[0, "label"] = 2
    test.to_parquet(inputs["test"], index=False)

    with pytest.raises(ValueError, match="label values outside 0 and 1"):
        select_model_features(
            inputs,
            tmp_path / "selected",
            tmp_path / "features.json",
            tmp_path / "report.json",
        )


def test_feature_selection_cli(tmp_path) -> None:
    """The command line writes all selected datasets and JSON artifacts."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "selected"
    list_path = tmp_path / "features.json"
    report_path = tmp_path / "report.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "select_features.py"),
            "--train-input",
            str(inputs["train"]),
            "--validation-input",
            str(inputs["validation"]),
            "--test-input",
            str(inputs["test"]),
            "--output-dir",
            str(output_directory),
            "--feature-list",
            str(list_path),
            "--report",
            str(report_path),
            "--batch-size",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert all(
        (output_directory / filename).is_file()
        for filename in SELECTED_FILENAMES.values()
    )
    assert list_path.is_file()
    assert report_path.is_file()
    assert "selected_features: 2" in run.stdout
    assert "status: passed" in run.stdout
