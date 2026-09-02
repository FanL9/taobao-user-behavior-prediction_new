"""Prepare train-only class-imbalance strategies without fitting a model."""

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


OUTPUT_FILENAMES = {
    "train_baseline": "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet",
    "train_smote": "user_item_feature_wide_labeled_train_preprocessed_selected_smote.parquet",
    "train_undersampled": "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet",
    "validation_original": "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet",
    "test_original": "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet",
}
CLASS_WEIGHT_CONFIG_FILENAME = "class_weight_config.json"
DATASET_VERSIONS_FILENAME = "training_dataset_versions.json"
IMBALANCE_REPORT_FILENAME = "class_imbalance_report.json"
TRACKING_COLUMNS = ("user_id", "item_id", "category_id")
TARGET_COLUMN = "label"
SYNTHETIC_FLAG_COLUMN = "is_synthetic"
RANDOM_STATE = 42
DEFAULT_SMOTE_K_NEIGHBORS = 5


def _resolve_paths(dataset_paths: Mapping[str, str | Path]) -> dict[str, Path]:
    """Resolve and validate exactly train, validation, and test input paths."""

    expected = ("train", "validation", "test")
    if set(dataset_paths) != set(expected):
        raise ValueError("dataset_paths must contain exactly train, validation, and test.")
    resolved = {split: Path(dataset_paths[split]).expanduser().resolve() for split in expected}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Selected dataset does not exist: " + ", ".join(missing))
    return resolved


def _validate_schemas(paths: Mapping[str, Path]) -> pa.Schema:
    """Require all selected datasets to have the same tracking/target schema."""

    train_schema = pq.ParquetFile(paths["train"]).schema_arrow
    required = {*TRACKING_COLUMNS, TARGET_COLUMN}
    missing = sorted(required - set(train_schema.names))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    for split in ("validation", "test"):
        schema = pq.ParquetFile(paths[split]).schema_arrow
        if not schema.equals(train_schema, check_metadata=False):
            raise ValueError(f"{split} schema differs from the training schema.")
    return train_schema


def _feature_columns(schema: pa.Schema) -> list[str]:
    """Return selected model inputs, excluding tracking fields and label."""

    features = [
        field.name
        for field in schema
        if field.name not in {*TRACKING_COLUMNS, TARGET_COLUMN}
    ]
    if not features:
        raise ValueError("No selected model features are available.")
    for field in schema:
        if field.name in features and not (
            pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        ):
            raise ValueError(f"Unsupported model feature type: {field.name}={field.type}")
    return features


def _scanner(path: Path, columns: list[str], batch_size: int, filter_expression=None):
    """Create a bounded-memory scanner, optionally filtered by label."""

    return ds.dataset(path, format="parquet").scanner(
        columns=columns,
        filter=filter_expression,
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=1,
    )


def _class_statistics(path: Path, batch_size: int) -> dict[str, int | float]:
    """Count binary labels without altering the source dataset."""

    counts = {0: 0, 1: 0}
    for batch in _scanner(path, [TARGET_COLUMN], batch_size).to_batches():
        labels = pd.to_numeric(batch.column(0).to_pandas(), errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{path.name} contains label values outside 0 and 1.")
        counts[0] += int(labels.eq(0).sum())
        counts[1] += int(labels.eq(1).sum())
    total = counts[0] + counts[1]
    if total == 0:
        raise ValueError(f"{path.name} has no rows.")
    return {
        "sample_count": total,
        "negative_count": counts[0],
        "positive_count": counts[1],
        "positive_ratio": counts[1] / total,
    }


def _output_schema(source_schema: pa.Schema) -> pa.Schema:
    """Add an explicit non-model flag identifying synthetic SMOTE rows."""

    return source_schema.append(pa.field(SYNTHETIC_FLAG_COLUMN, pa.bool_(), nullable=False))


def _with_origin_flag(batch: pa.RecordBatch, value: bool) -> pa.RecordBatch:
    """Append a constant synthetic-origin flag to an existing source batch."""

    return batch.append_column(
        pa.field(SYNTHETIC_FLAG_COLUMN, pa.bool_(), nullable=False),
        pa.array(np.full(batch.num_rows, value, dtype=bool)),
    )


def _copy_with_origin_flag(
    source: Path,
    target: Path,
    output_schema: pa.Schema,
    batch_size: int,
) -> dict[str, int | float]:
    """Write an unchanged dataset with ``is_synthetic=False`` metadata."""

    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
    try:
        for batch in _scanner(source, list(output_schema.names[:-1]), batch_size).to_batches():
            writer.write_batch(_with_origin_flag(batch, False))
    finally:
        writer.close()
    temporary.replace(target)
    return _class_statistics(target, batch_size)


def _positive_feature_matrix(
    train_path: Path,
    features: list[str],
    batch_size: int,
) -> np.ndarray:
    """Load the typically small positive class for SMOTE neighbor generation."""

    parts: list[np.ndarray] = []
    expression = ds.field(TARGET_COLUMN) == 1
    for batch in _scanner(train_path, features, batch_size, expression).to_batches():
        values = batch.to_pandas().to_numpy(dtype="float64", na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("SMOTE input contains missing or non-finite feature values.")
        parts.append(values)
    if not parts:
        raise ValueError("Training data has no positive samples for SMOTE.")
    return np.concatenate(parts, axis=0)


def _nearest_positive_neighbors(
    positive_values: np.ndarray,
    continuous_indices: list[int],
    k_neighbors: int,
) -> np.ndarray:
    """Compute deterministic nearest minority neighbors for mixed-type SMOTE."""

    if len(positive_values) < 2:
        raise ValueError("SMOTE requires at least two positive training samples.")
    k = min(k_neighbors, len(positive_values) - 1)
    if k <= 0:
        raise ValueError("SMOTE requires a positive k_neighbors value.")
    values = positive_values[:, continuous_indices] if continuous_indices else positive_values
    squared_norm = np.einsum("ij,ij->i", values, values)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (values @ values.T)
    np.fill_diagonal(distances, np.inf)
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


def _synthetic_batch(
    source_schema: pa.Schema,
    features: list[str],
    positive_values: np.ndarray,
    neighbors: np.ndarray,
    continuous_indices: list[int],
    categorical_indices: list[int],
    start: int,
    count: int,
    rng: np.random.Generator,
) -> pa.RecordBatch:
    """Create one mixed-type SMOTE batch with synthetic tracking metadata."""

    base_indices = np.arange(start, start + count) % len(positive_values)
    neighbor_positions = rng.integers(0, neighbors.shape[1], size=count)
    neighbor_indices = neighbors[base_indices, neighbor_positions]
    base = positive_values[base_indices]
    neighbor = positive_values[neighbor_indices]
    values = base.copy()
    if continuous_indices:
        interpolation = rng.random((count, len(continuous_indices)))
        values[:, continuous_indices] = (
            base[:, continuous_indices]
            + interpolation * (neighbor[:, continuous_indices] - base[:, continuous_indices])
        )
    for index in categorical_indices:
        inherit_neighbor = rng.integers(0, 2, size=count).astype(bool)
        values[:, index] = np.where(
            inherit_neighbor, neighbor[:, index], base[:, index]
        )

    arrays = []
    for field in source_schema:
        if field.name in TRACKING_COLUMNS:
            arrays.append(pa.array(np.full(count, -1), type=field.type))
        elif field.name == TARGET_COLUMN:
            arrays.append(pa.array(np.ones(count, dtype="int8"), type=field.type))
        else:
            index = features.index(field.name)
            arrays.append(pa.array(values[:, index], type=field.type))
    arrays.append(pa.array(np.ones(count, dtype=bool), type=pa.bool_()))
    return pa.RecordBatch.from_arrays(arrays, schema=_output_schema(source_schema))


def _write_smote_dataset(
    train_path: Path,
    target: Path,
    source_schema: pa.Schema,
    features: list[str],
    class_stats: Mapping[str, int | float],
    batch_size: int,
    random_state: int,
    k_neighbors: int,
) -> dict[str, int | float]:
    """Write original train rows plus SMOTE positives until classes balance."""

    synthetic_count = int(class_stats["negative_count"] - class_stats["positive_count"])
    if synthetic_count < 0:
        raise ValueError("SMOTE expects positive class to be the minority class.")
    output_schema = _output_schema(source_schema)
    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
    try:
        for batch in _scanner(train_path, list(source_schema.names), batch_size).to_batches():
            writer.write_batch(_with_origin_flag(batch, False))
        if synthetic_count:
            positives = _positive_feature_matrix(train_path, features, batch_size)
            categorical_indices = [
                index for index, column in enumerate(features) if column.endswith("_code")
            ]
            continuous_indices = [
                index for index in range(len(features)) if index not in categorical_indices
            ]
            neighbors = _nearest_positive_neighbors(
                positives, continuous_indices, k_neighbors
            )
            rng = np.random.default_rng(random_state)
            written = 0
            while written < synthetic_count:
                current = min(batch_size, synthetic_count - written)
                writer.write_batch(
                    _synthetic_batch(
                        source_schema,
                        features,
                        positives,
                        neighbors,
                        continuous_indices,
                        categorical_indices,
                        written,
                        current,
                        rng,
                    )
                )
                written += current
    finally:
        writer.close()
    temporary.replace(target)
    stats = _class_statistics(target, batch_size)
    stats["synthetic_count"] = synthetic_count
    return stats


def _write_undersampled_dataset(
    train_path: Path,
    target: Path,
    source_schema: pa.Schema,
    class_stats: Mapping[str, int | float],
    batch_size: int,
    random_state: int,
) -> dict[str, int | float]:
    """Randomly retain as many negatives as positives, using a fixed seed."""

    positive_count = int(class_stats["positive_count"])
    negative_count = int(class_stats["negative_count"])
    rng = np.random.default_rng(random_state)
    selected_negative_positions = np.sort(
        rng.choice(negative_count, size=positive_count, replace=False)
    )
    output_schema = _output_schema(source_schema)
    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
    negative_seen = 0
    try:
        for batch in _scanner(train_path, list(source_schema.names), batch_size).to_batches():
            labels = batch.column(batch.schema.get_field_index(TARGET_COLUMN)).to_numpy(
                zero_copy_only=False
            )
            negative_mask = labels == 0
            positions = np.arange(negative_seen, negative_seen + int(negative_mask.sum()))
            chosen = np.isin(positions, selected_negative_positions)
            take_mask = labels == 1
            take_mask[negative_mask] = chosen
            negative_seen += int(negative_mask.sum())
            if take_mask.any():
                subset = batch.filter(pa.array(take_mask))
                writer.write_batch(_with_origin_flag(subset, False))
    finally:
        writer.close()
    if negative_seen != negative_count:
        temporary.unlink(missing_ok=True)
        raise ValueError("Undersampling negative-row count differs from training statistics.")
    temporary.replace(target)
    return _class_statistics(target, batch_size)


def prepare_class_imbalance_strategies(
    dataset_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    class_weight_path: str | Path,
    versions_path: str | Path,
    report_path: str | Path,
    batch_size: int = 50_000,
    random_state: int = RANDOM_STATE,
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS,
) -> dict[str, Any]:
    """Create baseline, SMOTE, undersampling, and class-weight training options.

    SMOTE and undersampling modify training rows only. Validation/test outputs are
    unchanged copies with their original class distributions. SMOTE interpolates
    continuous features and inherits each ``*_code`` categorical value from one
    of the two selected minority parents; synthetic tracking IDs are ``-1`` and
    are explicitly excluded from model input.
    """

    if batch_size <= 0 or smote_k_neighbors <= 0:
        raise ValueError("batch_size and smote_k_neighbors must be positive.")
    paths = _resolve_paths(dataset_paths)
    source_schema = _validate_schemas(paths)
    features = _feature_columns(source_schema)
    destination = Path(output_directory).expanduser().resolve()
    weight_output = Path(class_weight_path).expanduser().resolve()
    versions_output = Path(versions_path).expanduser().resolve()
    report_output = Path(report_path).expanduser().resolve()
    if len({weight_output, versions_output, report_output}) != 3:
        raise ValueError("Class-weight, versions, and report paths must differ.")

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
    temporary_jsons = [
        weight_output.with_suffix(weight_output.suffix + ".tmp"),
        versions_output.with_suffix(versions_output.suffix + ".tmp"),
        report_output.with_suffix(report_output.suffix + ".tmp"),
    ]
    output_paths = {key: destination / filename for key, filename in OUTPUT_FILENAMES.items()}

    try:
        original_statistics = {
            split: _class_statistics(paths[split], batch_size)
            for split in ("train", "validation", "test")
        }
        if int(original_statistics["train"]["positive_count"]) < 2:
            raise ValueError("SMOTE requires at least two positive training samples.")
        if int(original_statistics["train"]["positive_count"]) > int(original_statistics["train"]["negative_count"]):
            raise ValueError("This workflow expects the purchase label to be the minority class.")

        destination.mkdir(parents=True, exist_ok=True)
        weight_output.parent.mkdir(parents=True, exist_ok=True)
        versions_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        output_schema = _output_schema(source_schema)
        baseline = _copy_with_origin_flag(
            paths["train"], output_paths["train_baseline"], output_schema, batch_size
        )
        validation = _copy_with_origin_flag(
            paths["validation"], output_paths["validation_original"], output_schema, batch_size
        )
        test = _copy_with_origin_flag(
            paths["test"], output_paths["test_original"], output_schema, batch_size
        )
        smote = _write_smote_dataset(
            paths["train"],
            output_paths["train_smote"],
            source_schema,
            features,
            original_statistics["train"],
            batch_size,
            random_state,
            smote_k_neighbors,
        )
        undersampled = _write_undersampled_dataset(
            paths["train"],
            output_paths["train_undersampled"],
            source_schema,
            original_statistics["train"],
            batch_size,
            random_state,
        )
        cpu_after = process.cpu_times()
        rss_samples.append(process.memory_info().rss)

        train_total = int(original_statistics["train"]["sample_count"])
        negative = int(original_statistics["train"]["negative_count"])
        positive = int(original_statistics["train"]["positive_count"])
        class_weights = {
            "class_weight": {
                "0": train_total / (2 * negative),
                "1": train_total / (2 * positive),
            },
            "formula": "n_samples / (n_classes * class_count)",
            "fitted_on": "train",
            "changes_sample_count": False,
            "model_feature_columns": features,
            "excluded_from_model_input": [*TRACKING_COLUMNS, TARGET_COLUMN, SYNTHETIC_FLAG_COLUMN],
        }
        versions = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_feature_columns": features,
            "tracking_columns": list(TRACKING_COLUMNS),
            "target_column": TARGET_COLUMN,
            "versions": {
                "baseline": {
                    "path": str(output_paths["train_baseline"]),
                    "description": "Original unsampled training set for baseline comparison.",
                    "statistics": baseline,
                },
                "smote": {
                    "path": str(output_paths["train_smote"]),
                    "description": "Training-only mixed-type SMOTE balanced training set.",
                    "statistics": smote,
                },
                "undersampled": {
                    "path": str(output_paths["train_undersampled"]),
                    "description": "Training-only seeded random undersampling balanced training set.",
                    "statistics": undersampled,
                },
                "class_weight": {
                    "config_path": str(weight_output),
                    "description": "Balanced class weights; source rows remain unchanged.",
                    "statistics": baseline,
                },
                "validation_original": {
                    "path": str(output_paths["validation_original"]),
                    "description": "Unchanged validation distribution; never sampled.",
                    "statistics": validation,
                },
                "test_original": {
                    "path": str(output_paths["test_original"]),
                    "description": "Unchanged test distribution; never sampled.",
                    "statistics": test,
                },
            },
        }
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "original_statistics": original_statistics,
            "training_strategy_statistics": {
                "baseline": baseline,
                "smote": smote,
                "undersampled": undersampled,
                "class_weight": baseline,
            },
            "validation_test_statistics": {
                "validation": validation,
                "test": test,
            },
            "checks": {
                "status": "passed",
                "sampling_fitted_on_train_only": True,
                "validation_sampled": False,
                "test_sampled": False,
                "baseline_preserved": True,
                "random_state": random_state,
                "smote_k_neighbors": min(smote_k_neighbors, positive - 1),
                "synthetic_tracking_value": -1,
                "tracking_target_and_origin_excluded_from_model_input": [
                    *TRACKING_COLUMNS,
                    TARGET_COLUMN,
                    SYNTHETIC_FLAG_COLUMN,
                ],
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
                    key: output_paths[key].stat().st_size for key in OUTPUT_FILENAMES
                },
            },
        }
        for temporary, payload in zip(temporary_jsons, (class_weights, versions, report)):
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for temporary, target in zip(temporary_jsons, (weight_output, versions_output, report_output)):
            temporary.replace(target)
        return {
            "output_paths": output_paths,
            "class_weight_path": weight_output,
            "versions_path": versions_output,
            "report_path": report_output,
            "class_weights": class_weights,
            "versions": versions,
            "report": report,
        }
    finally:
        stop_event.set()
        sampler.join()
        for temporary in temporary_jsons:
            temporary.unlink(missing_ok=True)
