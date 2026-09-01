"""Functional tests for fixed time-ordered dataset partitioning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset_splits import (
    DATASET_FILENAMES,
    generate_time_ordered_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_labeled_input(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Write a minimal correctly labeled table covering all three windows."""

    frame = pd.DataFrame(
        {
            "dataset_split": [
                "train",
                "validation",
                "test",
                "train",
                "validation",
                "test",
            ],
            "user_id": [1, 2, 3, 4, 5, 6],
            "item_id": [11, 12, 13, 14, 15, 16],
            "history_start": pd.to_datetime(
                [
                    "2025-11-18",
                    "2025-12-09",
                    "2025-12-16",
                    "2025-11-18",
                    "2025-12-09",
                    "2025-12-16",
                ]
            ),
            "history_end": pd.to_datetime(
                [
                    "2025-12-07",
                    "2025-12-14",
                    "2025-12-17",
                    "2025-12-07",
                    "2025-12-14",
                    "2025-12-17",
                ]
            ),
            "label_date": pd.to_datetime(
                [
                    "2025-12-08",
                    "2025-12-15",
                    "2025-12-18",
                    "2025-12-08",
                    "2025-12-15",
                    "2025-12-18",
                ]
            ),
            "feature_value": [10, 20, 30, 40, 50, 60],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "labeled.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def test_generate_time_ordered_datasets_preserves_rows_and_windows(tmp_path) -> None:
    """Each output contains only its fixed time window and unchanged rows."""

    source, original = _write_labeled_input(tmp_path)
    output_directory = tmp_path / "datasets"
    report_path = tmp_path / "statistics.json"
    result = generate_time_ordered_datasets(
        source,
        output_directory,
        report_path,
        batch_size=1,
    )

    expected_dates = {
        "train": ("2025-11-18", "2025-12-07", "2025-12-08"),
        "validation": ("2025-12-09", "2025-12-14", "2025-12-15"),
        "test": ("2025-12-16", "2025-12-17", "2025-12-18"),
    }
    for split, filename in DATASET_FILENAMES.items():
        output = output_directory / filename
        assert result["output_paths"][split] == output
        actual = pd.read_parquet(output).reset_index(drop=True)
        expected = original.loc[
            original["dataset_split"].eq(split)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
        assert actual["dataset_split"].eq(split).all()
        assert tuple(
            actual[["history_start", "history_end", "label_date"]]
            .iloc[0]
            .dt.strftime("%Y-%m-%d")
        ) == expected_dates[split]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["report"] == report
    assert report["statistics"]["overall"] == {
        "sample_count": 6,
        "positive_count": 3,
        "negative_count": 3,
        "positive_ratio": 0.5,
    }
    assert report["time_window_checks"]["random_split_used"] is False
    assert report["time_window_checks"]["dataset_order"] == [
        "train",
        "validation",
        "test",
    ]


def test_generate_time_ordered_datasets_rejects_window_mismatch(tmp_path) -> None:
    """A row outside the configured split window is rejected."""

    source, frame = _write_labeled_input(tmp_path)
    frame.loc[frame["dataset_split"].eq("test"), "history_end"] = pd.Timestamp(
        "2025-12-18"
    )
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="inconsistent history_end"):
        generate_time_ordered_datasets(
            source,
            tmp_path / "datasets",
            tmp_path / "statistics.json",
        )


def test_dataset_split_cli(tmp_path) -> None:
    """The command-line interface writes all three labeled datasets."""

    source, _ = _write_labeled_input(tmp_path)
    output_directory = tmp_path / "datasets"
    report_path = tmp_path / "statistics.json"
    run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "split_labeled_datasets.py"),
            "--input",
            str(source),
            "--output-dir",
            str(output_directory),
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
    assert all((output_directory / filename).is_file() for filename in DATASET_FILENAMES.values())
    assert report_path.is_file()
    assert "status: passed" in run.stdout
