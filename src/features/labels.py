"""Generate next-day purchase labels for the stage-two user-item wide table.

This module only appends a binary ``label`` column. Label-date events are used
solely for the target lookup and never participate in feature calculation.
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
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .stage2_intermediate_tables import HISTORY_WINDOWS


LABELED_SAMPLE_FILENAME = "user_item_feature_wide_labeled.parquet"
LABEL_REPORT_FILENAME = "label_statistics.json"
PRIMARY_KEY = ("dataset_split", "user_id", "item_id")
WIDE_REQUIRED_COLUMNS = {
    *PRIMARY_KEY,
    "history_start",
    "history_end",
    "label_date",
}
LABEL_SOURCE_COLUMNS = (
    "user_id",
    "item_id",
    "behavior_type",
    "behavior_date",
)
WINDOWS_BY_SPLIT = {
    window.dataset_split: {
        "history_start": pd.Timestamp(window.history_start),
        "history_end": pd.Timestamp(window.history_end),
        "label_date": pd.Timestamp(window.label_date),
    }
    for window in HISTORY_WINDOWS
}


def _validate_sources(
    wide_table: str | Path,
    clean_data: str | Path,
    output_parquet: str | Path,
) -> tuple[Path, Path, Path, pa.Schema]:
    """Resolve paths and validate the two input schemas.

    Args:
        wide_table: Stage-two user-item feature-wide Parquet.
        clean_data: Stage-one standard clean behavior Parquet.
        output_parquet: Destination for the labeled wide sample.

    Returns:
        Resolved paths and the wide-table Arrow schema.
    """

    wide_path = Path(wide_table).expanduser().resolve()
    clean_path = Path(clean_data).expanduser().resolve()
    output_path = Path(output_parquet).expanduser().resolve()
    if not wide_path.is_file():
        raise FileNotFoundError(f"Feature-wide Parquet does not exist: {wide_path}")
    if not clean_path.is_file():
        raise FileNotFoundError(f"Clean behavior Parquet does not exist: {clean_path}")
    if output_path in {wide_path, clean_path}:
        raise ValueError("Output path must not overwrite an input Parquet.")

    wide_schema = pq.ParquetFile(wide_path).schema_arrow
    wide_columns = set(wide_schema.names)
    missing_wide = sorted(WIDE_REQUIRED_COLUMNS - wide_columns)
    if missing_wide:
        raise ValueError(f"Feature-wide table is missing columns: {missing_wide}")
    if "label" in wide_columns:
        raise ValueError("Feature-wide input already contains label.")

    clean_columns = set(pq.ParquetFile(clean_path).schema_arrow.names)
    missing_clean = sorted(set(LABEL_SOURCE_COLUMNS) - clean_columns)
    if missing_clean:
        raise ValueError(f"Clean behavior table is missing columns: {missing_clean}")
    return wide_path, clean_path, output_path, wide_schema


def _load_label_day_purchases(clean_path: Path) -> dict[str, pd.MultiIndex]:
    """Load distinct purchased user-item pairs for each configured label day.

    Only ``behavior_type == 4`` rows on the three exact label dates are read.
    The returned lookup is target-only data and is never joined as a feature.

    Args:
        clean_path: Standard clean behavior Parquet.

    Returns:
        Mapping from ISO label date to a unique user-item MultiIndex.
    """

    label_dates = [
        values["label_date"].strftime("%Y-%m-%d")
        for values in WINDOWS_BY_SPLIT.values()
    ]
    source = ds.dataset(clean_path, format="parquet")
    scanner = source.scanner(
        columns=list(LABEL_SOURCE_COLUMNS),
        filter=(ds.field("behavior_type") == 4)
        & ds.field("behavior_date").isin(label_dates),
        batch_size=200_000,
        batch_readahead=0,
        fragment_readahead=1,
    )
    parts: list[pd.DataFrame] = []
    for batch in scanner.to_batches():
        if batch.num_rows:
            parts.append(batch.to_pandas())

    if parts:
        purchases = pd.concat(parts, ignore_index=True)
        purchases["behavior_date"] = purchases["behavior_date"].astype("string")
        purchases = purchases.drop_duplicates(
            ["behavior_date", "user_id", "item_id"]
        )
    else:
        purchases = pd.DataFrame(columns=LABEL_SOURCE_COLUMNS)

    lookups: dict[str, pd.MultiIndex] = {}
    for label_date in label_dates:
        rows = purchases.loc[
            purchases["behavior_date"].eq(label_date), ["user_id", "item_id"]
        ]
        lookups[label_date] = pd.MultiIndex.from_frame(rows)
    return lookups


def _validate_and_label_batch(
    batch: pa.RecordBatch,
    purchase_pairs: dict[str, pd.MultiIndex],
) -> tuple[pa.RecordBatch, pd.DataFrame]:
    """Validate one wide-table batch and append its exact-date purchase label.

    Args:
        batch: A batch from the stage-two wide table.
        purchase_pairs: Distinct purchase pairs keyed by label date.

    Returns:
        The Arrow batch with an appended ``int8`` label and a compact frame
        containing only split and label values for incremental statistics.
    """

    validation_columns = [
        *PRIMARY_KEY,
        "history_start",
        "history_end",
        "label_date",
    ]
    frame = batch.select(validation_columns).to_pandas()
    if frame[list(PRIMARY_KEY)].isna().any(axis=None):
        raise ValueError("Feature-wide table contains a missing primary-key value.")
    if frame.duplicated(list(PRIMARY_KEY)).any():
        raise ValueError("Feature-wide table contains duplicate keys within a batch.")

    labels = pd.Series(0, index=frame.index, dtype="int8")
    observed_splits = set(frame["dataset_split"].astype(str).unique())
    unexpected = sorted(observed_splits - set(WINDOWS_BY_SPLIT))
    if unexpected:
        raise ValueError(f"Unexpected dataset_split values: {unexpected}")

    for split, expected in WINDOWS_BY_SPLIT.items():
        mask = frame["dataset_split"].eq(split)
        if not mask.any():
            continue
        for column in ("history_start", "history_end", "label_date"):
            actual = pd.to_datetime(frame.loc[mask, column], errors="coerce").dt.normalize()
            if actual.isna().any() or not actual.eq(expected[column]).all():
                raise ValueError(
                    f"Feature-wide table has inconsistent {column} for "
                    f"dataset_split={split}."
                )
        history_end = pd.to_datetime(frame.loc[mask, "history_end"])
        label_date = pd.to_datetime(frame.loc[mask, "label_date"])
        if history_end.ge(label_date).any():
            raise ValueError("Feature and label windows overlap.")

        date_key = expected["label_date"].strftime("%Y-%m-%d")
        candidate_pairs = pd.MultiIndex.from_frame(
            frame.loc[mask, ["user_id", "item_id"]]
        )
        labels.loc[mask] = candidate_pairs.isin(purchase_pairs[date_key]).astype("int8")

    label_array = pa.array(labels.to_numpy(), type=pa.int8())
    labeled_batch = batch.append_column(
        pa.field("label", pa.int8(), nullable=False), label_array
    )
    statistics = frame.loc[:, ["dataset_split"]].copy()
    statistics["label"] = labels
    return labeled_batch, statistics


def _summary(positive_count: int, total_count: int) -> dict[str, int | float]:
    """Return positive, negative, total, and positive-ratio statistics."""

    negative_count = total_count - positive_count
    return {
        "sample_count": total_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": positive_count / total_count if total_count else 0.0,
    }


def generate_purchase_labels(
    wide_table: str | Path,
    clean_data: str | Path,
    output_parquet: str | Path,
    report_path: str | Path,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Append future-one-day purchase labels without changing feature values.

    A row receives ``label=1`` only when its exact ``user_id + item_id`` pair
    has ``behavior_type=4`` on the row's configured ``label_date``. Otherwise
    it receives ``label=0``. No preprocessing, feature selection, sampling,
    model training, or evaluation is performed.

    Args:
        wide_table: Stage-two user-item feature-wide Parquet.
        clean_data: Stage-one clean behavior Parquet containing label-day events.
        output_parquet: Destination Parquet with the appended label.
        report_path: Destination JSON containing statistics and leakage checks.
        batch_size: Maximum wide-table rows processed per Arrow batch.

    Returns:
        Output paths and the generated report.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    wide_path, clean_path, output_path, wide_schema = _validate_sources(
        wide_table, clean_data, output_parquet
    )
    report_output = Path(report_path).expanduser().resolve()
    if report_output in {wide_path, clean_path, output_path}:
        raise ValueError("Report path must be different from input and output Parquets.")

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
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    split_counts = {
        split: {"total": 0, "positive": 0} for split in WINDOWS_BY_SPLIT
    }

    try:
        purchase_pairs = _load_label_day_purchases(clean_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        output_schema = wide_schema.append(pa.field("label", pa.int8(), nullable=False))
        writer = pq.ParquetWriter(temporary_output, output_schema, compression="snappy")
        scanner = ds.dataset(wide_path, format="parquet").scanner(
            batch_size=batch_size,
            batch_readahead=0,
            fragment_readahead=1,
        )
        for batch in scanner.to_batches():
            labeled_batch, batch_statistics = _validate_and_label_batch(
                batch, purchase_pairs
            )
            for split in WINDOWS_BY_SPLIT:
                split_labels = batch_statistics.loc[
                    batch_statistics["dataset_split"].eq(split), "label"
                ]
                split_counts[split]["total"] += int(len(split_labels))
                split_counts[split]["positive"] += int(split_labels.sum())
            writer.write_batch(labeled_batch)
        writer.close()
        writer = None

        missing_splits = [
            split for split, counts in split_counts.items() if counts["total"] == 0
        ]
        if missing_splits:
            raise ValueError(f"Feature-wide table has no rows for: {missing_splits}")

        temporary_output.replace(output_path)
        cpu_after = process.cpu_times()
        rss_samples.append(process.memory_info().rss)
        by_split = {
            split: _summary(counts["positive"], counts["total"])
            for split, counts in split_counts.items()
        }
        total_count = sum(counts["total"] for counts in split_counts.values())
        positive_count = sum(
            counts["positive"] for counts in split_counts.values()
        )
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "inputs": {
                "feature_wide_parquet": str(wide_path),
                "clean_behavior_parquet": str(clean_path),
            },
            "output_parquet": str(output_path),
            "label_definition": {
                "sample_grain": "user_id + item_id",
                "target": "purchase on the configured next-day label_date",
                "positive_rule": "behavior_type == 4 on label_date",
                "negative_rule": "no behavior_type == 4 on label_date",
            },
            "statistics": {
                "overall": _summary(positive_count, total_count),
                "by_split": by_split,
            },
            "data_leakage_checks": {
                "status": "passed",
                "feature_and_label_window_overlap_rows": 0,
                "window_metadata_mismatch_rows": 0,
                "label_window_used_for_feature_calculation": False,
                "label_source_columns": list(LABEL_SOURCE_COLUMNS),
                "feature_columns_modified": [],
                "output_columns_added": ["label"],
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
                "output_file_size_bytes": output_path.stat().st_size,
            },
        }
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_output)
        return {
            "output_path": output_path,
            "report_path": report_output,
            "report": report,
        }
    finally:
        if writer is not None:
            writer.close()
        stop_event.set()
        sampler.join()
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
