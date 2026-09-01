"""Functional tests for future-one-day purchase label generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.features.labels import generate_purchase_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    """Write a small three-window wide table and clean behavior source."""

    wide = pd.DataFrame(
        {
            "dataset_split": ["train", "train", "validation", "test"],
            "user_id": [1, 2, 1, 3],
            "item_id": [10, 20, 10, 30],
            "history_start": pd.to_datetime(
                ["2025-11-18", "2025-11-18", "2025-12-09", "2025-12-16"]
            ),
            "history_end": pd.to_datetime(
                ["2025-12-07", "2025-12-07", "2025-12-14", "2025-12-17"]
            ),
            "label_date": pd.to_datetime(
                ["2025-12-08", "2025-12-08", "2025-12-15", "2025-12-18"]
            ),
            "feature_value": [11, 22, 33, 44],
        }
    )
    clean = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3, 3, 999],
            "item_id": [10, 20, 10, 30, 30, 999],
            "behavior_type": [4, 4, 4, 4, 1, 4],
            "behavior_date": [
                "2025-12-08",
                "2025-12-07",
                "2025-12-15",
                "2025-12-19",
                "2025-12-18",
                "2025-12-18",
            ],
        }
    )
    wide_path = tmp_path / "wide.parquet"
    clean_path = tmp_path / "clean.parquet"
    wide.to_parquet(wide_path, index=False)
    clean.to_parquet(clean_path, index=False)
    return wide_path, clean_path, wide


def test_generate_purchase_labels_exact_dates_and_statistics(tmp_path) -> None:
    """Only an exact label-day purchase creates a positive label."""

    wide_path, clean_path, original = _write_inputs(tmp_path)
    output_path = tmp_path / "labeled.parquet"
    report_path = tmp_path / "statistics.json"

    result = generate_purchase_labels(
        wide_path,
        clean_path,
        output_path,
        report_path,
        batch_size=2,
    )

    labeled = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(
        labeled.drop(columns="label"), original, check_dtype=True
    )
    assert labeled["label"].tolist() == [1, 0, 1, 0]
    assert str(labeled["label"].dtype) == "int8"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["report"] == report
    assert report["statistics"]["overall"] == {
        "sample_count": 4,
        "positive_count": 2,
        "negative_count": 2,
        "positive_ratio": 0.5,
    }
    assert report["statistics"]["by_split"]["train"]["positive_count"] == 1
    assert report["statistics"]["by_split"]["validation"]["positive_count"] == 1
    assert report["statistics"]["by_split"]["test"]["positive_count"] == 0
    leakage = report["data_leakage_checks"]
    assert leakage["feature_and_label_window_overlap_rows"] == 0
    assert leakage["label_window_used_for_feature_calculation"] is False
    assert leakage["feature_columns_modified"] == []


def test_generate_purchase_labels_rejects_window_metadata_mismatch(tmp_path) -> None:
    """A feature row whose label date differs from its split contract fails."""

    wide_path, clean_path, wide = _write_inputs(tmp_path)
    wide.loc[wide["dataset_split"].eq("train"), "label_date"] = pd.Timestamp(
        "2025-12-09"
    )
    wide.to_parquet(wide_path, index=False)

    with pytest.raises(ValueError, match="inconsistent label_date"):
        generate_purchase_labels(
            wide_path,
            clean_path,
            tmp_path / "labeled.parquet",
            tmp_path / "statistics.json",
        )


def test_generate_purchase_labels_cli(tmp_path) -> None:
    """The command-line interface writes the requested labeled artifacts."""

    wide_path, clean_path, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "cli_labeled.parquet"
    report_path = tmp_path / "cli_report.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_purchase_labels.py"),
            "--wide-table",
            str(wide_path),
            "--clean-data",
            str(clean_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--batch-size",
            "2",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert output_path.is_file()
    assert report_path.is_file()
    assert "status: passed" in run.stdout
