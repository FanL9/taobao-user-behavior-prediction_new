"""Select final model features using training data only.

This module performs deterministic data-quality and redundancy screening. It
does not train a predictive model, evaluate one, or alter sample membership.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SELECTED_FILENAMES = {
    "train": "user_item_feature_wide_labeled_train_preprocessed_selected.parquet",
    "validation": "user_item_feature_wide_labeled_validation_preprocessed_selected.parquet",
    "test": "user_item_feature_wide_labeled_test_preprocessed_selected.parquet",
}
FINAL_FEATURE_LIST_FILENAME = "final_model_features.json"
FEATURE_SELECTION_REPORT_FILENAME = "feature_selection_report.json"
TRACKING_COLUMNS = ("user_id", "item_id", "category_id")
TARGET_COLUMN = "label"
LOW_VARIANCE_THRESHOLD = 1e-12
HIGH_CORRELATION_THRESHOLD = 0.98
OUTLIER_ZSCORE_THRESHOLD = 5.0
FUTURE_LEAKAGE_TOKENS = (
    "label",
    "target",
    "future",
    "history_start",
    "history_end",
    "label_date",
    "timestamp",
)


def _resolve_dataset_paths(
    dataset_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Resolve and validate exactly train, validation, and test paths."""

    expected = ("train", "validation", "test")
    if set(dataset_paths) != set(expected):
        raise ValueError(
            "dataset_paths must contain exactly train, validation, and test."
        )
    resolved = {
        split: Path(dataset_paths[split]).expanduser().resolve()
        for split in expected
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Preprocessed Parquet does not exist: " + ", ".join(missing))
    return resolved


def _validate_schemas(paths: Mapping[str, Path]) -> pa.Schema:
    """Require all three preprocessed datasets to have the same schema."""

    train_schema = pq.ParquetFile(paths["train"]).schema_arrow
    required = {*TRACKING_COLUMNS, TARGET_COLUMN}
    missing = sorted(required - set(train_schema.names))
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {missing}")
    for split in ("validation", "test"):
        schema = pq.ParquetFile(paths[split]).schema_arrow
        if not schema.equals(train_schema, check_metadata=False):
            raise ValueError(f"{split} schema differs from the training schema.")
    return train_schema


def _candidate_features(schema: pa.Schema) -> list[str]:
    """Return numeric, non-tracking candidate features in source-column order."""

    excluded = {*TRACKING_COLUMNS, TARGET_COLUMN}
    candidates: list[str] = []
    for field in schema:
        if field.name in excluded:
            continue
        if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)):
            raise ValueError(
                f"Preprocessed feature {field.name} has unsupported type {field.type}."
            )
        candidates.append(field.name)
    if not candidates:
        raise ValueError("No model-feature candidates remain after tracking/target exclusion.")
    return candidates


def _scanner(path: Path, columns: list[str], batch_size: int):
    """Create a low-prefetch Arrow scanner for bounded memory use."""

    return ds.dataset(path, format="parquet").scanner(
        columns=columns,
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=1,
    )


def _fit_selection_rules(
    train_path: Path,
    candidates: list[str],
    batch_size: int,
) -> dict[str, Any]:
    """Calculate training-only quality, variance, correlation, and outlier data."""

    feature_count = len(candidates)
    sample_count = 0
    sums = np.zeros(feature_count, dtype="float64")
    cross_products = np.zeros((feature_count, feature_count), dtype="float64")
    missing_counts = np.zeros(feature_count, dtype="int64")
    nonfinite_counts = np.zeros(feature_count, dtype="int64")
    outlier_counts = np.zeros(feature_count, dtype="int64")

    for batch in _scanner(train_path, candidates, batch_size).to_batches():
        frame = batch.to_pandas()
        values = frame.loc[:, candidates].to_numpy(dtype="float64", na_value=np.nan)
        sample_count += len(values)
        finite = np.isfinite(values)
        missing_counts += np.isnan(values).sum(axis=0)
        nonfinite_counts += (~finite).sum(axis=0)
        if not finite.all():
            safe_values = np.where(finite, values, 0.0)
        else:
            safe_values = values
        sums += safe_values.sum(axis=0)
        cross_products += safe_values.T @ safe_values
        outlier_counts += (np.abs(safe_values) > OUTLIER_ZSCORE_THRESHOLD).sum(axis=0)

    if sample_count == 0:
        raise ValueError("Training dataset has no rows.")
    means = sums / sample_count
    covariance = cross_products / sample_count - np.outer(means, means)
    variances = np.maximum(np.diag(covariance), 0.0)
    scales = np.sqrt(variances)
    correlation = np.zeros_like(covariance)
    valid_scale = scales > 0.0
    denominator = np.outer(scales, scales)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    np.fill_diagonal(correlation, 1.0)

    return {
        "sample_count": sample_count,
        "missing_rates": {
            column: float(missing_counts[index] / sample_count)
            for index, column in enumerate(candidates)
        },
        "nonfinite_counts": {
            column: int(nonfinite_counts[index])
            for index, column in enumerate(candidates)
        },
        "variances": {
            column: float(variances[index]) for index, column in enumerate(candidates)
        },
        "outlier_rates": {
            column: float(outlier_counts[index] / sample_count)
            for index, column in enumerate(candidates)
        },
        "correlation": correlation,
        "valid_scale": valid_scale,
    }


def _select_features(
    candidates: list[str],
    fitted: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]], list[dict[str, Any]]]:
    """Apply deterministic, training-only removal rules in defined order."""

    removed: dict[str, list[str]] = {}
    leakage_features: list[dict[str, Any]] = []
    for index, column in enumerate(candidates):
        lower_name = column.lower()
        matched_tokens = [
            token for token in FUTURE_LEAKAGE_TOKENS if token in lower_name
        ]
        if matched_tokens:
            reason = "suspected_future_information:" + ",".join(matched_tokens)
            removed.setdefault(column, []).append(reason)
            leakage_features.append({"feature": column, "matched_tokens": matched_tokens})
        if fitted["nonfinite_counts"][column] > 0:
            removed.setdefault(column, []).append("nonfinite_values_in_training")
        if fitted["missing_rates"][column] > 0.95:
            removed.setdefault(column, []).append("missing_rate_above_0.95")
        if fitted["variances"][column] <= LOW_VARIANCE_THRESHOLD:
            removed.setdefault(column, []).append("low_variance")

    correlation = fitted["correlation"]
    retained_indices = [
        index for index, column in enumerate(candidates) if column not in removed
    ]
    selected_indices: list[int] = []
    for index in retained_indices:
        column = candidates[index]
        correlated_with: tuple[str, float] | None = None
        for selected_index in selected_indices:
            value = abs(float(correlation[index, selected_index]))
            if value >= HIGH_CORRELATION_THRESHOLD:
                correlated_with = (candidates[selected_index], value)
                break
        if correlated_with is None:
            selected_indices.append(index)
        else:
            kept_feature, value = correlated_with
            removed.setdefault(column, []).append(
                f"high_correlation_with:{kept_feature}:abs={value:.6f}"
            )

    selected = [candidates[index] for index in selected_indices]
    if not selected:
        raise ValueError("All candidate features were removed by screening rules.")
    return selected, removed, leakage_features


def _validate_and_write_dataset(
    source: Path,
    split: str,
    output_path: Path,
    output_schema: pa.Schema,
    output_columns: list[str],
    batch_size: int,
) -> dict[str, int | float]:
    """Write selected columns unchanged and validate rows/labels for one split."""

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
    sample_count = 0
    positive_count = 0
    try:
        for batch in _scanner(source, output_columns, batch_size).to_batches():
            frame = batch.select([*TRACKING_COLUMNS, TARGET_COLUMN]).to_pandas()
            if frame.isna().any(axis=None):
                raise ValueError(f"{split} contains a missing tracking key or label.")
            labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
            if labels.isna().any() or not labels.isin([0, 1]).all():
                raise ValueError(f"{split} contains label values outside 0 and 1.")
            sample_count += batch.num_rows
            positive_count += int(labels.sum())
            writer.write_batch(batch)
    finally:
        writer.close()
    source_rows = pq.ParquetFile(source).metadata.num_rows
    if sample_count != source_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{split} output row count does not match input.")
    temporary.replace(output_path)
    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": sample_count - positive_count,
        "positive_ratio": positive_count / sample_count if sample_count else 0.0,
    }


def select_model_features(
    dataset_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    feature_list_path: str | Path,
    report_path: str | Path,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Fit feature screening on train and apply the final list to all datasets.

    Screening removes candidate fields with non-finite training values, missing
    rate above 95%, variance at or below ``1e-12``, absolute training
    correlation at or above ``0.98`` with an earlier retained field, or names
    suggesting direct future/target leakage. Outlier rates above five standard
    deviations are recorded but do not by themselves remove valid values.

    Args:
        dataset_paths: Preprocessed files keyed by train, validation, and test.
        output_directory: Directory receiving selected-feature Parquets.
        feature_list_path: JSON path for the final training-derived feature list.
        report_path: JSON path for removal reasons and screening statistics.
        batch_size: Maximum rows processed per Arrow batch.

    Returns:
        Paths, final features, and the generated feature-list/report content.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paths = _resolve_dataset_paths(dataset_paths)
    input_schema = _validate_schemas(paths)
    candidates = _candidate_features(input_schema)
    destination = Path(output_directory).expanduser().resolve()
    feature_list_output = Path(feature_list_path).expanduser().resolve()
    report_output = Path(report_path).expanduser().resolve()
    if feature_list_output == report_output:
        raise ValueError("Feature-list and report paths must be different.")

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
    temporary_feature_list = feature_list_output.with_suffix(
        feature_list_output.suffix + ".tmp"
    )
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    output_paths = {
        split: destination / filename for split, filename in SELECTED_FILENAMES.items()
    }

    try:
        fitted = _fit_selection_rules(paths["train"], candidates, batch_size)
        selected_features, removed_features, leakage_features = _select_features(
            candidates, fitted
        )
        destination.mkdir(parents=True, exist_ok=True)
        feature_list_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)

        output_columns = [*TRACKING_COLUMNS, TARGET_COLUMN, *selected_features]
        output_schema = pa.schema(
            [input_schema.field(column) for column in output_columns]
        )
        output_statistics = {
            split: _validate_and_write_dataset(
                paths[split],
                split,
                output_paths[split],
                output_schema,
                output_columns,
                batch_size,
            )
            for split in SELECTED_FILENAMES
        }
        cpu_after = process.cpu_times()
        rss_samples.append(process.memory_info().rss)

        feature_list: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fitted_on": "train",
            "tracking_columns": list(TRACKING_COLUMNS),
            "target_column": TARGET_COLUMN,
            "model_feature_columns": selected_features,
            "model_feature_count": len(selected_features),
            "input_candidate_feature_count": len(candidates),
            "selection_rule": "training-only deterministic data-quality and redundancy screening",
        }
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "input_datasets": {split: str(path) for split, path in paths.items()},
            "output_datasets": {
                split: str(path) for split, path in output_paths.items()
            },
            "feature_statistics": {
                "training_sample_count": fitted["sample_count"],
                "candidate_feature_count": len(candidates),
                "selected_feature_count": len(selected_features),
                "removed_feature_count": len(removed_features),
                "missing_rates": fitted["missing_rates"],
                "variances": fitted["variances"],
                "outlier_rates_abs_zscore_above_5": fitted["outlier_rates"],
            },
            "selected_features": selected_features,
            "removed_features": removed_features,
            "data_leakage_check": {
                "status": "passed",
                "rules_fitted_on_train_only": True,
                "validation_and_test_refit_rules": False,
                "tracking_columns_excluded_from_model_input": list(TRACKING_COLUMNS),
                "target_excluded_from_model_input": TARGET_COLUMN,
                "suspected_future_information_features": leakage_features,
                "random_sampling_used": False,
            },
            "selection_thresholds": {
                "missing_rate_remove_above": 0.95,
                "low_variance_remove_at_or_below": LOW_VARIANCE_THRESHOLD,
                "high_correlation_remove_at_or_above": HIGH_CORRELATION_THRESHOLD,
                "outlier_check_abs_zscore_above": OUTLIER_ZSCORE_THRESHOLD,
                "outlier_rate_removes_feature": False,
            },
            "dataset_statistics": output_statistics,
            "performance": {
                "runtime_seconds": round(time.perf_counter() - started_at, 6),
                "process_cpu_time_seconds": round(
                    cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
                    6,
                ),
                "peak_process_rss_bytes": max(rss_samples),
                "gpu_used": False,
                "output_file_size_bytes": {
                    split: output_paths[split].stat().st_size
                    for split in SELECTED_FILENAMES
                },
            },
        }
        temporary_feature_list.write_text(
            json.dumps(feature_list, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_feature_list.replace(feature_list_output)
        temporary_report.replace(report_output)
        return {
            "output_paths": output_paths,
            "feature_list_path": feature_list_output,
            "report_path": report_output,
            "feature_list": feature_list,
            "report": report,
        }
    finally:
        stop_event.set()
        sampler.join()
        temporary_feature_list.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
