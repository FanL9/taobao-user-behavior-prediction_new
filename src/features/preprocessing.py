"""Fit train-only preprocessing rules and apply them to three fixed datasets.

The module prepares feature values only. It preserves identifiers for tracking
and ``label`` for supervision, but neither becomes a model feature.
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

from .stage2_intermediate_tables import HISTORY_WINDOWS


PREPROCESSED_FILENAMES = {
    "train": "user_item_feature_wide_labeled_train_preprocessed.parquet",
    "validation": "user_item_feature_wide_labeled_validation_preprocessed.parquet",
    "test": "user_item_feature_wide_labeled_test_preprocessed.parquet",
}
PREPROCESSING_RULES_FILENAME = "preprocessing_rules.json"
PREPROCESSING_REPORT_FILENAME = "preprocessing_statistics.json"
TRACKING_COLUMNS = ("user_id", "item_id", "category_id")
TARGET_COLUMN = "label"
PARTITION_COLUMN = "dataset_split"
WINDOW_COLUMNS = ("history_start", "history_end", "label_date")
MISSING_CATEGORY_VALUE = "\u0000__MISSING__"
UNKNOWN_CATEGORY_CODE = -1

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


def _resolve_dataset_paths(
    dataset_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Resolve and validate the train, validation, and test input paths."""

    expected_splits = tuple(WINDOWS_BY_SPLIT)
    if set(dataset_paths) != set(expected_splits):
        raise ValueError(
            "dataset_paths must contain exactly train, validation, and test."
        )
    resolved = {
        split: Path(dataset_paths[split]).expanduser().resolve()
        for split in expected_splits
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Dataset Parquet does not exist: " + ", ".join(missing))
    return resolved


def _feature_roles(schema: pa.Schema) -> tuple[list[str], list[str], list[str]]:
    """Classify fields into dropped, numeric, and categorical feature roles.

    Identifiers, target, fixed partition/window metadata, and every timestamp
    field are excluded from model features. Unsupported Arrow types fail fast
    instead of being silently transformed.
    """

    required = {
        PARTITION_COLUMN,
        *TRACKING_COLUMNS,
        *WINDOW_COLUMNS,
        TARGET_COLUMN,
    }
    missing = sorted(required - set(schema.names))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dropped = [PARTITION_COLUMN, *WINDOW_COLUMNS]
    numeric: list[str] = []
    categorical: list[str] = []
    excluded = {*TRACKING_COLUMNS, TARGET_COLUMN, *dropped}
    for field in schema:
        if field.name in excluded:
            continue
        if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
            dropped.append(field.name)
        elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric.append(field.name)
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            categorical.append(field.name)
        else:
            raise ValueError(
                f"Unsupported feature type for {field.name}: {field.type}"
            )
    return dropped, numeric, categorical


def _validate_schema_compatibility(paths: Mapping[str, Path]) -> pa.Schema:
    """Require the three datasets to have the same input field names/types."""

    train_schema = pq.ParquetFile(paths["train"]).schema_arrow
    for split in ("validation", "test"):
        schema = pq.ParquetFile(paths[split]).schema_arrow
        if not schema.equals(train_schema, check_metadata=False):
            raise ValueError(f"{split} schema differs from the training schema.")
    return train_schema


def _scanner(path: Path, columns: list[str], batch_size: int):
    """Create a low-prefetch scanner to keep preprocessing memory bounded."""

    return ds.dataset(path, format="parquet").scanner(
        columns=columns,
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=1,
    )


def _category_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return categorical values with a stable train-fitted missing sentinel."""

    return frame[column].astype("string").fillna(MISSING_CATEGORY_VALUE)


def _fit_rules(
    train_path: Path,
    numeric_columns: list[str],
    categorical_columns: list[str],
    batch_size: int,
) -> dict[str, Any]:
    """Fit fill, scale, and encoding rules using training rows only.

    Numerical missing values use the training mean. Standard deviations are
    population deviations after that fill rule; zero-variance fields use a
    scale of one. Category codes are fitted from training values only.
    """

    fit_columns = [*numeric_columns, *categorical_columns]
    sums = np.zeros(len(numeric_columns), dtype="float64")
    sum_squares = np.zeros(len(numeric_columns), dtype="float64")
    valid_counts = np.zeros(len(numeric_columns), dtype="int64")
    row_count = 0
    categories = {column: set() for column in categorical_columns}

    for batch in _scanner(train_path, fit_columns, batch_size).to_batches():
        frame = batch.to_pandas()
        row_count += len(frame)
        for index, column in enumerate(numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan
            )
            valid = np.isfinite(values)
            valid_values = values[valid]
            valid_counts[index] += len(valid_values)
            sums[index] += valid_values.sum()
            sum_squares[index] += np.square(valid_values).sum()
        for column in categorical_columns:
            categories[column].update(_category_values(frame, column).tolist())

    if row_count == 0:
        raise ValueError("Training dataset has no rows.")

    numeric_rules: dict[str, dict[str, float]] = {}
    for index, column in enumerate(numeric_columns):
        if valid_counts[index] == 0:
            fill_value = 0.0
            mean = 0.0
            scale = 1.0
        else:
            fill_value = sums[index] / valid_counts[index]
            mean = fill_value
            nonmissing_variance = max(
                sum_squares[index] / valid_counts[index] - mean**2,
                0.0,
            )
            scale = float(
                np.sqrt(nonmissing_variance * valid_counts[index] / row_count)
            )
            if scale == 0.0 or not np.isfinite(scale):
                scale = 1.0
        numeric_rules[column] = {
            "fill_value": float(fill_value),
            "mean": float(mean),
            "scale": float(scale),
        }

    categorical_rules = {
        column: {
            value: code
            for code, value in enumerate(sorted(values))
        }
        for column, values in categories.items()
    }
    return {
        "numeric_rules": numeric_rules,
        "categorical_rules": categorical_rules,
    }


def _validate_batch(frame: pd.DataFrame, split: str) -> None:
    """Require one input batch to obey its expected time and label contract."""

    if not frame[PARTITION_COLUMN].eq(split).all():
        raise ValueError(f"Input contains rows outside dataset_split={split}.")
    if frame[[*TRACKING_COLUMNS, TARGET_COLUMN]].isna().any(axis=None):
        raise ValueError("Input contains a missing tracking key or label.")
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Input contains label values outside 0 and 1.")

    for column, expected in WINDOWS_BY_SPLIT[split].items():
        actual = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        if actual.isna().any() or not actual.eq(expected).all():
            raise ValueError(
                f"Input has inconsistent {column} for dataset_split={split}."
            )
    history_end = pd.to_datetime(frame["history_end"])
    label_date = pd.to_datetime(frame["label_date"])
    if history_end.ge(label_date).any():
        raise ValueError("Feature and label windows overlap.")


def _output_schema(
    input_schema: pa.Schema,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pa.Schema:
    """Build the tracking/target plus processed-model-feature output schema."""

    fields = [input_schema.field(column) for column in TRACKING_COLUMNS]
    fields.append(input_schema.field(TARGET_COLUMN))
    fields.extend(pa.field(column, pa.float32(), nullable=False) for column in numeric_columns)
    fields.extend(
        pa.field(f"{column}_code", pa.int32(), nullable=False)
        for column in categorical_columns
    )
    return pa.schema(fields)


def _transform_batch(
    frame: pd.DataFrame,
    numeric_rules: Mapping[str, Mapping[str, float]],
    categorical_rules: Mapping[str, Mapping[str, int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply train-fitted fill, scaling, and category codes to one batch."""

    output = frame.loc[:, [*TRACKING_COLUMNS, TARGET_COLUMN]].copy()
    missing_counts: dict[str, int] = {}
    for column, rule in numeric_rules.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        invalid = ~np.isfinite(values)
        missing_counts[column] = int(invalid.sum())
        values[invalid] = rule["fill_value"]
        output[column] = ((values - rule["mean"]) / rule["scale"]).astype(
            "float32"
        )
    for column, mapping in categorical_rules.items():
        values = _category_values(frame, column)
        codes = values.map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype("int32")
        output[f"{column}_code"] = codes
        missing_counts[column] = int(values.eq(MISSING_CATEGORY_VALUE).sum())
    return output, missing_counts


def preprocess_feature_datasets(
    dataset_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    rules_path: str | Path,
    report_path: str | Path,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Fit preprocessing on train and apply the same rules to all three sets.

    Tracking fields ``user_id``, ``item_id``, and ``category_id`` plus target
    ``label`` are retained in output but excluded from ``model_feature_columns``.
    Partition/window metadata and direct timestamp fields are dropped. Numeric
    values use training-mean fill plus training-fitted standardization; string
    categories use training-fitted integer codes with ``-1`` for unseen values.

    Args:
        dataset_paths: Paths keyed by ``train``, ``validation``, and ``test``.
        output_directory: Directory receiving the three processed Parquets.
        rules_path: JSON path receiving the fitted train-only rules.
        report_path: JSON path receiving output statistics and checks.
        batch_size: Maximum rows processed per Arrow batch.

    Returns:
        Paths and in-memory copies of the fitted rules and processing report.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paths = _resolve_dataset_paths(dataset_paths)
    input_schema = _validate_schema_compatibility(paths)
    dropped_columns, numeric_columns, categorical_columns = _feature_roles(input_schema)
    destination = Path(output_directory).expanduser().resolve()
    rules_output = Path(rules_path).expanduser().resolve()
    report_output = Path(report_path).expanduser().resolve()
    if rules_output == report_output:
        raise ValueError("Rules and report paths must be different.")

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
    temporary_rules = rules_output.with_suffix(rules_output.suffix + ".tmp")
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    output_paths = {
        split: destination / filename
        for split, filename in PREPROCESSED_FILENAMES.items()
    }
    input_columns = [
        PARTITION_COLUMN,
        *TRACKING_COLUMNS,
        *WINDOW_COLUMNS,
        TARGET_COLUMN,
        *numeric_columns,
        *categorical_columns,
    ]

    try:
        fitted = _fit_rules(
            paths["train"], numeric_columns, categorical_columns, batch_size
        )
        destination.mkdir(parents=True, exist_ok=True)
        rules_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        output_schema = _output_schema(
            input_schema, numeric_columns, categorical_columns
        )
        dataset_statistics: dict[str, dict[str, Any]] = {}

        for split in PREPROCESSED_FILENAMES:
            target = output_paths[split]
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary_paths.append(temporary)
            writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
            sample_count = 0
            positive_count = 0
            filled_counts = {column: 0 for column in [*numeric_columns, *categorical_columns]}
            unknown_counts = {column: 0 for column in categorical_columns}
            try:
                for batch in _scanner(paths[split], input_columns, batch_size).to_batches():
                    frame = batch.to_pandas()
                    _validate_batch(frame, split)
                    transformed, missing_counts = _transform_batch(
                        frame,
                        fitted["numeric_rules"],
                        fitted["categorical_rules"],
                    )
                    for column, count in missing_counts.items():
                        filled_counts[column] += count
                    for column in categorical_columns:
                        values = _category_values(frame, column)
                        unknown_counts[column] += int(
                            (~values.isin(fitted["categorical_rules"][column])).sum()
                        )
                    sample_count += len(transformed)
                    positive_count += int(transformed[TARGET_COLUMN].sum())
                    writer.write_table(
                        pa.Table.from_pandas(
                            transformed,
                            schema=output_schema,
                            preserve_index=False,
                        )
                    )
            finally:
                writer.close()
            if sample_count == 0:
                raise ValueError(f"{split} dataset has no rows.")
            source_rows = pq.ParquetFile(paths[split]).metadata.num_rows
            if sample_count != source_rows:
                raise ValueError(f"{split} output row count does not match input.")
            dataset_statistics[split] = {
                **_summary(positive_count, sample_count),
                "filled_value_count": filled_counts,
                "unknown_category_count": unknown_counts,
            }

        for temporary, target in zip(temporary_paths, output_paths.values()):
            temporary.replace(target)
        cpu_after = process.cpu_times()
        rss_samples.append(process.memory_info().rss)

        model_feature_columns = [
            *numeric_columns,
            *[f"{column}_code" for column in categorical_columns],
        ]
        rules: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fitted_on": "train",
            "input_datasets": {split: str(path) for split, path in paths.items()},
            "output_datasets": {split: str(path) for split, path in output_paths.items()},
            "tracking_columns": list(TRACKING_COLUMNS),
            "target_column": TARGET_COLUMN,
            "excluded_from_model_input": [*TRACKING_COLUMNS, TARGET_COLUMN],
            "dropped_columns": dropped_columns,
            "model_feature_columns": model_feature_columns,
            "numeric_rules": fitted["numeric_rules"],
            "categorical_rules": fitted["categorical_rules"],
            "missing_value_policy": {
                "numeric": "fill with training-set mean",
                "categorical": "encode missing as the train-fitted missing category",
            },
            "standardization_policy": "(value - training mean) / training population standard deviation",
            "categorical_encoding_policy": {
                "method": "training-fitted ordinal encoding",
                "unknown_category_code": UNKNOWN_CATEGORY_CODE,
            },
        }
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "statistics": {
                "overall": _summary(
                    sum(values["positive_count"] for values in dataset_statistics.values()),
                    sum(values["sample_count"] for values in dataset_statistics.values()),
                ),
                "by_split": dataset_statistics,
            },
            "checks": {
                "status": "passed",
                "rules_fitted_on_train_only": True,
                "validation_and_test_refit_rules": False,
                "random_sampling_used": False,
                "tracking_columns_excluded_from_model_input": list(TRACKING_COLUMNS),
                "target_excluded_from_model_input": TARGET_COLUMN,
                "dropped_columns": dropped_columns,
                "model_feature_count": len(model_feature_columns),
                "feature_and_label_window_overlap_rows": 0,
                "window_metadata_mismatch_rows": 0,
            },
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
                    for split in PREPROCESSED_FILENAMES
                },
            },
        }
        temporary_rules.write_text(
            json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_rules.replace(rules_output)
        temporary_report.replace(report_output)
        return {
            "output_paths": output_paths,
            "rules_path": rules_output,
            "report_path": report_output,
            "rules": rules,
            "report": report,
        }
    finally:
        stop_event.set()
        sampler.join()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        temporary_rules.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
