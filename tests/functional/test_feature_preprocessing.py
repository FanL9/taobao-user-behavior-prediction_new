"""Functional tests for train-fitted feature preprocessing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    PREPROCESSED_FILENAMES,
    preprocess_feature_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame_for_split(split: str) -> pd.DataFrame:
    """Return a small labeled input frame with split-specific window metadata."""

    windows = {
        "train": ("2025-11-18", "2025-12-07", "2025-12-08"),
        "validation": ("2025-12-09", "2025-12-14", "2025-12-15"),
        "test": ("2025-12-16", "2025-12-17", "2025-12-18"),
    }
    start, end, label_date = windows[split]
    values = {
        "train": {
            "numeric_feature": [1.0, np.nan, 3.0],
            "constant_feature": [5.0, 5.0, 5.0],
            "last_behavior_type": ["pv", "cart", "pv"],
            "last_10_behavior_sequence": ["pv", "pv→cart", "pv"],
            "user_activity_level": ["low", "high", "low"],
            "label": [1, 0, 1],
        },
        "validation": {
            "numeric_feature": [np.nan, 5.0],
            "constant_feature": [5.0, 5.0],
            "last_behavior_type": ["fav", "pv"],
            "last_10_behavior_sequence": ["fav", "pv→cart"],
            "user_activity_level": ["medium", "low"],
            "label": [0, 1],
        },
        "test": {
            "numeric_feature": [7.0, np.nan],
            "constant_feature": [5.0, 5.0],
            "last_behavior_type": ["buy", "pv"],
            "last_10_behavior_sequence": ["buy", "pv"],
            "user_activity_level": ["high", "new_level"],
            "label": [1, 0],
        },
    }[split]
    row_count = len(values["label"])
    return pd.DataFrame(
        {
            "dataset_split": [split] * row_count,
            "user_id": range(1, row_count + 1),
            "item_id": range(101, 101 + row_count),
            "category_id": range(201, 201 + row_count),
            "history_start": pd.to_datetime([start] * row_count),
            "history_end": pd.to_datetime([end] * row_count),
            "label_date": pd.to_datetime([label_date] * row_count),
            "last_behavior_date": pd.to_datetime([end] * row_count),
            "category_first_event_time": pd.to_datetime([start] * row_count),
            "category_last_event_time": pd.to_datetime([end] * row_count),
            **values,
        }
    )


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write compatible train/validation/test preprocessing inputs."""

    paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.parquet"
        _frame_for_split(split).to_parquet(path, index=False)
        paths[split] = path
    return paths


def test_preprocess_feature_datasets_uses_train_only_rules(tmp_path) -> None:
    """Fill, scale, and category codes are fitted only on training rows."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "processed"
    rules_path = tmp_path / "rules.json"
    report_path = tmp_path / "report.json"
    result = preprocess_feature_datasets(
        inputs,
        output_directory,
        rules_path,
        report_path,
        batch_size=1,
    )

    train = pd.read_parquet(output_directory / PREPROCESSED_FILENAMES["train"])
    validation = pd.read_parquet(
        output_directory / PREPROCESSED_FILENAMES["validation"]
    )
    test = pd.read_parquet(output_directory / PREPROCESSED_FILENAMES["test"])
    expected_columns = [
        "user_id",
        "item_id",
        "category_id",
        "label",
        "numeric_feature",
        "constant_feature",
        "last_behavior_type_code",
        "last_10_behavior_sequence_code",
        "user_activity_level_code",
    ]
    assert train.columns.tolist() == expected_columns
    assert set(["dataset_split", "history_start", "label_date", "last_behavior_date"]).isdisjoint(train.columns)
    assert train[["user_id", "item_id", "category_id", "label"]].equals(
        _frame_for_split("train")[["user_id", "item_id", "category_id", "label"]]
    )
    assert abs(float(train["numeric_feature"].mean())) < 1e-6
    assert train.loc[1, "numeric_feature"] == pytest.approx(0.0)
    assert train["constant_feature"].eq(0.0).all()
    assert validation.loc[0, "numeric_feature"] == pytest.approx(0.0)
    assert test.loc[0, "last_behavior_type_code"] == -1
    assert test.loc[1, "user_activity_level_code"] == -1

    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["rules"] == rules
    assert result["report"] == report
    assert rules["fitted_on"] == "train"
    assert rules["excluded_from_model_input"] == [
        "user_id",
        "item_id",
        "category_id",
        "label",
    ]
    assert report["checks"]["rules_fitted_on_train_only"] is True
    assert report["checks"]["validation_and_test_refit_rules"] is False
    assert report["statistics"]["by_split"]["test"]["unknown_category_count"]["last_behavior_type"] == 1


def test_preprocess_feature_datasets_rejects_bad_label(tmp_path) -> None:
    """Inputs with labels outside the binary contract are rejected."""

    inputs = _write_inputs(tmp_path)
    train = _frame_for_split("train")
    train.loc[0, "label"] = 2
    train.to_parquet(inputs["train"], index=False)

    with pytest.raises(ValueError, match="label values outside 0 and 1"):
        preprocess_feature_datasets(
            inputs,
            tmp_path / "processed",
            tmp_path / "rules.json",
            tmp_path / "report.json",
        )


def test_feature_preprocessing_cli(tmp_path) -> None:
    """The command line writes all processed datasets, rules, and report."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "processed"
    rules_path = tmp_path / "rules.json"
    report_path = tmp_path / "report.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "preprocess_features.py"),
            "--train-input",
            str(inputs["train"]),
            "--validation-input",
            str(inputs["validation"]),
            "--test-input",
            str(inputs["test"]),
            "--output-dir",
            str(output_directory),
            "--rules",
            str(rules_path),
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
        for filename in PREPROCESSED_FILENAMES.values()
    )
    assert rules_path.is_file()
    assert report_path.is_file()
    assert "status: passed" in run.stdout
