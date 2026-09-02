"""Functional tests for train-only class-imbalance preparation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.sampling.class_imbalance import (
    OUTPUT_FILENAMES,
    prepare_class_imbalance_strategies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(split: str) -> pd.DataFrame:
    """Create a small selected dataset with a positive minority in train."""

    labels = {
        "train": [0, 0, 0, 0, 0, 0, 1, 1],
        "validation": [0, 0, 0, 1],
        "test": [0, 0, 1, 0],
    }[split]
    count = len(labels)
    return pd.DataFrame(
        {
            "user_id": range(1, count + 1),
            "item_id": range(101, 101 + count),
            "category_id": range(201, 201 + count),
            "label": labels,
            "continuous_feature": [float(index) for index in range(count)],
            "other_feature": [float(index % 3) for index in range(count)],
            "last_behavior_type_code": [index % 2 for index in range(count)],
        }
    )


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write compatible selected train, validation, and test datasets."""

    paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.parquet"
        _frame(split).to_parquet(path, index=False)
        paths[split] = path
    return paths


def test_prepare_class_imbalance_strategies_train_only(tmp_path) -> None:
    """Create baseline, SMOTE, undersampling, and weight-only alternatives."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "imbalance"
    weight_path = tmp_path / "weights.json"
    versions_path = tmp_path / "versions.json"
    report_path = tmp_path / "report.json"
    result = prepare_class_imbalance_strategies(
        inputs,
        output_directory,
        weight_path,
        versions_path,
        report_path,
        batch_size=2,
        random_state=7,
        smote_k_neighbors=5,
    )

    baseline = pd.read_parquet(output_directory / OUTPUT_FILENAMES["train_baseline"])
    smote = pd.read_parquet(output_directory / OUTPUT_FILENAMES["train_smote"])
    undersampled = pd.read_parquet(output_directory / OUTPUT_FILENAMES["train_undersampled"])
    validation = pd.read_parquet(output_directory / OUTPUT_FILENAMES["validation_original"])
    test = pd.read_parquet(output_directory / OUTPUT_FILENAMES["test_original"])
    assert baseline.shape[0] == 8
    assert baseline["is_synthetic"].eq(False).all()
    assert smote["label"].value_counts().to_dict() == {0: 6, 1: 6}
    assert smote["is_synthetic"].sum() == 4
    assert smote.loc[smote["is_synthetic"], ["user_id", "item_id", "category_id"]].eq(-1).all().all()
    assert undersampled["label"].value_counts().to_dict() == {1: 2, 0: 2}
    assert undersampled["is_synthetic"].eq(False).all()
    pd.testing.assert_frame_equal(
        validation.drop(columns="is_synthetic"), _frame("validation")
    )
    pd.testing.assert_frame_equal(test.drop(columns="is_synthetic"), _frame("test"))

    weights = json.loads(weight_path.read_text(encoding="utf-8"))
    versions = json.loads(versions_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["class_weights"] == weights
    assert result["versions"] == versions
    assert result["report"] == report
    assert weights["class_weight"] == {"0": pytest.approx(2 / 3), "1": pytest.approx(2.0)}
    assert report["checks"]["validation_sampled"] is False
    assert report["checks"]["test_sampled"] is False
    assert report["checks"]["baseline_preserved"] is True


def test_class_imbalance_rejects_single_positive_train_sample(tmp_path) -> None:
    """SMOTE is rejected when training has fewer than two positive rows."""

    inputs = _write_inputs(tmp_path)
    train = _frame("train").iloc[:-1].copy()
    train.loc[:, "label"] = [0, 0, 0, 0, 0, 0, 1]
    train.to_parquet(inputs["train"], index=False)
    with pytest.raises(ValueError, match="at least two positive"):
        prepare_class_imbalance_strategies(
            inputs,
            tmp_path / "imbalance",
            tmp_path / "weights.json",
            tmp_path / "versions.json",
            tmp_path / "report.json",
        )


def test_class_imbalance_cli(tmp_path) -> None:
    """The CLI writes every required training strategy and JSON artifact."""

    inputs = _write_inputs(tmp_path)
    output_directory = tmp_path / "imbalance"
    weight_path = tmp_path / "weights.json"
    versions_path = tmp_path / "versions.json"
    report_path = tmp_path / "report.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_class_imbalance.py"),
            "--train-input", str(inputs["train"]),
            "--validation-input", str(inputs["validation"]),
            "--test-input", str(inputs["test"]),
            "--output-dir", str(output_directory),
            "--class-weights", str(weight_path),
            "--versions", str(versions_path),
            "--report", str(report_path),
            "--batch-size", "2",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert all((output_directory / filename).is_file() for filename in OUTPUT_FILENAMES.values())
    assert weight_path.is_file() and versions_path.is_file() and report_path.is_file()
    assert "status: passed" in run.stdout
