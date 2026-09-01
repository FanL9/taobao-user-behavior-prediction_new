"""Create leakage-safe train, validation, and test datasets from labels.

The labeled wide table already carries the fixed stage-two time-window metadata.
This module only partitions its rows by ``dataset_split``; it does not alter
features, labels, sampling, or model inputs.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.features.stage2_intermediate_tables import HISTORY_WINDOWS


DATASET_FILENAMES = {
    "train": "user_item_feature_wide_labeled_train.parquet",
    "validation": "user_item_feature_wide_labeled_validation.parquet",
    "test": "user_item_feature_wide_labeled_test.parquet",
}
SPLIT_REPORT_FILENAME = "dataset_split_statistics.json"
PRIMARY_KEY = ("dataset_split", "user_id", "item_id")
REQUIRED_COLUMNS = {
    *PRIMARY_KEY,
    "history_start",
    "history_end",
    "label_date",
    "label",
}
WINDOWS_BY_SPLIT = {
    window.dataset_split: {
        "history_start": pd.Timestamp(window.history_start),
        "history_end": pd.Timestamp(window.history_end),
        "label_date": pd.Timestamp(window.label_date),
    }
    for window in HISTORY_WINDOWS
}


def _summary(positive_count: int, sample_count: int) -> dict[str, int | float]:
    """Return sample, positive, negative, and positive-ratio statistics."""

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": sample_count - positive_count,
        "positive_ratio": positive_count / sample_count if sample_count else 0.0,
    }


def _validate_source(
    labeled_table: str | Path,
    output_directory: str | Path,
) -> tuple[Path, Path, int]:
    """Resolve and validate the labeled wide-table input contract.

    Args:
        labeled_table: Parquet produced by purchase-label generation.
        output_directory: Directory receiving the three partitioned datasets.

    Returns:
        Resolved input/output paths and the source row count.
    """

    source = Path(labeled_table).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Labeled sample Parquet does not exist: {source}")

    schema = pq.ParquetFile(source).schema_arrow
    missing_columns = sorted(REQUIRED_COLUMNS - set(schema.names))
    if missing_columns:
        raise ValueError(f"Labeled sample is missing columns: {missing_columns}")
    return source, destination, pq.ParquetFile(source).metadata.num_rows


def _validate_batch(batch, split: str) -> tuple[int, int]:
    """Validate one output batch's time-window and label contract.

    Args:
        batch: Arrow record batch already filtered to one dataset split.
        split: Expected ``dataset_split`` value for the batch.

    Returns:
        The batch row count and positive-label count.
    """

    fields = [*PRIMARY_KEY, "history_start", "history_end", "label_date", "label"]
    frame = batch.select(fields).to_pandas()
    if not frame["dataset_split"].eq(split).all():
        raise ValueError(f"Output batch contains rows outside dataset_split={split}.")
    if frame[list(PRIMARY_KEY)].isna().any(axis=None):
        raise ValueError("Labeled sample contains a missing primary-key value.")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labeled sample contains label values outside 0 and 1.")

    expected = WINDOWS_BY_SPLIT[split]
    for column, expected_value in expected.items():
        actual = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        if actual.isna().any() or not actual.eq(expected_value).all():
            raise ValueError(
                f"Labeled sample has inconsistent {column} for "
                f"dataset_split={split}."
            )
    history_end = pd.to_datetime(frame["history_end"])
    label_date = pd.to_datetime(frame["label_date"])
    if history_end.ge(label_date).any():
        raise ValueError("Feature and label windows overlap.")
    return len(frame), int(labels.sum())


def _source_split_values(source: Path, batch_size: int) -> set[str]:
    """Read the partition column alone and return all observed split values."""

    values: set[str] = set()
    scanner = ds.dataset(source, format="parquet").scanner(
        columns=["dataset_split"],
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        values.update(batch.column(0).to_pylist())
    return values


def generate_time_ordered_datasets(
    labeled_table: str | Path,
    output_directory: str | Path,
    report_path: str | Path,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Write train, validation, and test Parquets in fixed time order.

    Rows are selected solely by their existing ``dataset_split`` value. The
    function validates the fixed history/label dates for every output batch and
    writes the original Arrow batches unchanged, so no feature preprocessing,
    feature selection, sampling, training, or evaluation takes place.

    Args:
        labeled_table: Labeled user-item feature-wide Parquet.
        output_directory: Destination directory for the three dataset Parquets.
        report_path: JSON path for sample statistics and time-window checks.
        batch_size: Maximum source rows processed per Arrow batch.

    Returns:
        Output paths and the generated statistics/check report.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    source, destination, source_row_count = _validate_source(
        labeled_table, output_directory
    )
    report_output = Path(report_path).expanduser().resolve()
    if report_output == source:
        raise ValueError("Report path must be different from the labeled input.")

    started_at = time.perf_counter()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    rss_samples = [process.memory_info().rss]
    stop_event = threading.Event()

    def sample_memory() -> None:
        while not stop_event.wait(0.05):
            try:
                rss_samples.append(process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    temporary_paths: list[Path] = []
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    output_paths = {
        split: destination / filename
        for split, filename in DATASET_FILENAMES.items()
    }
    counts = {split: {"total": 0, "positive": 0} for split in DATASET_FILENAMES}

    try:
        observed_splits = _source_split_values(source, batch_size)
        expected_splits = set(DATASET_FILENAMES)
        if observed_splits != expected_splits:
            raise ValueError(
                "Labeled sample must contain exactly train, validation, and "
                f"test rows; observed: {sorted(observed_splits)}"
            )

        destination.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        schema = pq.ParquetFile(source).schema_arrow
        for split in DATASET_FILENAMES:
            target = output_paths[split]
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary_paths.append(temporary)
            writer = pq.ParquetWriter(temporary, schema, compression="snappy")
            try:
                scanner = ds.dataset(source, format="parquet").scanner(
                    filter=ds.field("dataset_split") == split,
                    batch_size=batch_size,
                    batch_readahead=0,
                    fragment_readahead=1,
                )
                for batch in scanner.to_batches():
                    row_count, positive_count = _validate_batch(batch, split)
                    counts[split]["total"] += row_count
                    counts[split]["positive"] += positive_count
                    writer.write_batch(batch)
            finally:
                writer.close()

            if counts[split]["total"] == 0:
                raise ValueError(f"Labeled sample has no rows for {split}.")

        written_row_count = sum(values["total"] for values in counts.values())
        if written_row_count != source_row_count:
            raise ValueError(
                "Output row count does not match the labeled input row count."
            )
        for temporary, target in zip(temporary_paths, output_paths.values()):
            temporary.replace(target)

        cpu_after = process.cpu_times()
        rss_samples.append(process.memory_info().rss)
        by_split = {
            split: _summary(values["positive"], values["total"])
            for split, values in counts.items()
        }
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "input_labeled_parquet": str(source),
            "output_datasets": {
                split: str(path) for split, path in output_paths.items()
            },
            "statistics": {
                "overall": _summary(
                    sum(values["positive"] for values in counts.values()),
                    written_row_count,
                ),
                "by_split": by_split,
            },
            "time_window_checks": {
                "status": "passed",
                "partition_column": "dataset_split",
                "random_split_used": False,
                "dataset_order": ["train", "validation", "test"],
                "feature_and_label_window_overlap_rows": 0,
                "window_metadata_mismatch_rows": 0,
                "windows": {
                    split: {
                        key: value.strftime("%Y-%m-%d")
                        for key, value in window.items()
                    }
                    for split, window in WINDOWS_BY_SPLIT.items()
                },
            },
            "performance": {
                "runtime_seconds": round(time.perf_counter() - started_at, 6),
                "process_cpu_time_seconds": round(
                    cpu_after.user
                    + cpu_after.system
                    - cpu_before.user
                    - cpu_before.system,
                    6,
                ),
                "peak_process_rss_bytes": max(rss_samples),
                "gpu_used": False,
                "output_file_size_bytes": {
                    split: output_paths[split].stat().st_size
                    for split in DATASET_FILENAMES
                },
            },
        }
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_output)
        return {
            "output_paths": output_paths,
            "report_path": report_output,
            "report": report,
        }
    finally:
        stop_event.set()
        sampler.join()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
