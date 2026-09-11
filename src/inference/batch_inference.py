"""Stage 5 local batch inference using frozen Stage 3/4 artifacts.

The module intentionally performs no fitting, feature selection, or model
training. Raw user-item feature rows are transformed only with the Stage 3
train-fitted preprocessing rules, restricted to the Stage 3 final feature
list, and scored with the Stage 4 validation-selected final model artifact.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage5_inference.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions" / "purchase_probability_predictions.parquet"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "performance" / "runtime" / "stage5_batch_inference_summary.json"
DEFAULT_BATCH_SIZE = 100_000
MISSING_CATEGORY_VALUE = "\u0000__MISSING__"
DEFAULT_UNKNOWN_CATEGORY_CODE = -1


@dataclass(frozen=True)
class InferenceContract:
    """Resolved, validated artifacts required for Stage 5 inference."""

    final_candidate: str
    model_artifact: Path
    model_bundle_key: str | None
    preprocessing_rules_path: Path
    feature_list_path: Path
    selection_manifest_path: Path
    evaluation_summary_path: Path
    prediction_threshold: float
    tracking_columns: tuple[str, ...]
    model_features: tuple[str, ...]
    raw_feature_columns: tuple[str, ...]
    preprocessing_rules: Mapping[str, Any]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Load a required JSON object with a useful failure message."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_from_project(value: str | Path, project_root: Path) -> Path:
    """Resolve a configuration path relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _selected_features(payload: Mapping[str, Any]) -> list[str]:
    """Read the frozen Stage 3 model-feature list."""

    features = payload.get("selected_model_features", payload.get("model_feature_columns"))
    if not isinstance(features, list) or not features or not all(isinstance(name, str) for name in features):
        raise ValueError(
            "Feature-list JSON must contain a non-empty model_feature_columns "
            "or selected_model_features list."
        )
    if len(features) != len(set(features)):
        raise ValueError("Final model feature list contains duplicate columns.")
    return features


def _raw_feature_for_model_feature(
    model_feature: str,
    numeric_rules: Mapping[str, Any],
    categorical_rules: Mapping[str, Any],
) -> str:
    """Map one processed model feature back to its raw Stage 2 input field."""

    if model_feature in numeric_rules:
        return model_feature
    if model_feature.endswith("_code"):
        raw_name = model_feature[: -len("_code")]
        if raw_name in categorical_rules:
            return raw_name
    raise ValueError(
        f"Final model feature {model_feature!r} is not defined by the frozen "
        "Stage 3 preprocessing rules."
    )


def load_inference_contract(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = PROJECT_ROOT,
    require_model: bool = True,
) -> InferenceContract:
    """Load and cross-check the frozen Stage 3/4 inference contract.

    The configured candidate must still match the Stage 4 validation leader,
    and it must appear in the Stage 4 final evaluation candidate list. This
    protects local inference from silently switching models or feature rules.
    """

    root = Path(project_root).expanduser().resolve()
    config = _load_json(Path(config_path).expanduser().resolve(), "Inference configuration")

    required_keys = {
        "final_candidate",
        "model_artifact",
        "preprocessing_rules",
        "feature_list",
        "selection_manifest",
        "evaluation_summary",
        "prediction_threshold",
    }
    missing_config = sorted(required_keys - set(config))
    if missing_config:
        raise ValueError(f"Inference configuration is missing keys: {missing_config}")

    candidate = str(config["final_candidate"])
    model_path = _resolve_from_project(config["model_artifact"], root)
    rules_path = _resolve_from_project(config["preprocessing_rules"], root)
    feature_path = _resolve_from_project(config["feature_list"], root)
    selection_path = _resolve_from_project(config["selection_manifest"], root)
    evaluation_path = _resolve_from_project(config["evaluation_summary"], root)
    threshold = float(config["prediction_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("prediction_threshold must be between 0 and 1.")

    selection = _load_json(selection_path, "Stage 4 selection manifest")
    leader = selection.get("validation_leader")
    if leader != candidate:
        raise ValueError(
            f"Configured final candidate {candidate!r} does not match the "
            f"Stage 4 validation leader {leader!r}."
        )
    if selection.get("test_data_read") is not False or selection.get("test_used_for_training_or_selection") is not False:
        raise ValueError("Stage 4 selection manifest does not prove test-set isolation.")

    evaluation = _load_json(evaluation_path, "Stage 4 evaluation summary")
    evaluated_candidates = evaluation.get("candidates")
    if not isinstance(evaluated_candidates, list) or candidate not in evaluated_candidates:
        raise ValueError(f"Final candidate {candidate!r} is absent from Stage 4 final evaluation.")
    if evaluation.get("test_evaluation_count") != 1:
        raise ValueError("Stage 4 evaluation summary must record exactly one test evaluation.")
    if evaluation.get("test_used_for_tuning_or_selection") is not False:
        raise ValueError("Stage 4 evaluation summary indicates test data affected model selection.")

    rules = _load_json(rules_path, "Stage 3 preprocessing rules")
    feature_payload = _load_json(feature_path, "Stage 3 final feature list")
    features = _selected_features(feature_payload)
    tracking = feature_payload.get("tracking_columns", rules.get("tracking_columns", ["user_id", "item_id", "category_id"]))
    if not isinstance(tracking, list) or not tracking or not all(isinstance(name, str) for name in tracking):
        raise ValueError("Tracking-column contract is invalid.")

    numeric_rules = rules.get("numeric_rules")
    categorical_rules = rules.get("categorical_rules")
    if not isinstance(numeric_rules, dict) or not isinstance(categorical_rules, dict):
        raise ValueError("Stage 3 preprocessing rules must define numeric_rules and categorical_rules.")

    raw_features = [
        _raw_feature_for_model_feature(feature, numeric_rules, categorical_rules)
        for feature in features
    ]
    raw_features = list(dict.fromkeys(raw_features))

    if require_model and not model_path.is_file():
        raise FileNotFoundError(
            "Final Stage 4 model artifact is not present locally. Copy the exact "
            f"evaluated artifact to: {model_path}"
        )

    bundle_key = config.get("model_bundle_key")
    if bundle_key is not None and str(bundle_key) != candidate:
        raise ValueError("model_bundle_key must match final_candidate for the Stage 4 bundle.")

    return InferenceContract(
        final_candidate=candidate,
        model_artifact=model_path,
        model_bundle_key=str(bundle_key) if bundle_key is not None else None,
        preprocessing_rules_path=rules_path,
        feature_list_path=feature_path,
        selection_manifest_path=selection_path,
        evaluation_summary_path=evaluation_path,
        prediction_threshold=threshold,
        tracking_columns=tuple(tracking),
        model_features=tuple(features),
        raw_feature_columns=tuple(raw_features),
        preprocessing_rules=rules,
    )


def load_final_model(contract: InferenceContract) -> Any:
    """Load the already-evaluated final estimator without fitting or tuning it."""

    artifact = joblib.load(contract.model_artifact)
    if isinstance(artifact, dict):
        if contract.model_bundle_key is None:
            raise ValueError("Model artifact is a bundle but model_bundle_key is not configured.")
        if contract.model_bundle_key not in artifact:
            raise ValueError(
                f"Model bundle does not contain {contract.model_bundle_key!r}. "
                f"Available keys: {sorted(map(str, artifact))}"
            )
        model = artifact[contract.model_bundle_key]
    else:
        model = artifact
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Final model must expose predict_proba().")
    return model


def transform_inference_features(
    frame: pd.DataFrame,
    contract: InferenceContract,
) -> pd.DataFrame:
    """Apply frozen Stage 3 preprocessing rules to raw Stage 2 feature rows.

    No statistics are fitted here. Numeric values use the exact train-fitted
    fill/mean/scale values from ``preprocessing_rules.json``. Categorical
    values use the exact train-fitted ordinal mapping; unknown values receive
    the frozen unknown-category code (normally ``-1``).
    """

    required = {*contract.tracking_columns, *contract.raw_feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Inference input is missing required columns: {missing}")
    if frame.loc[:, contract.tracking_columns].isna().any(axis=None):
        raise ValueError("Inference input contains a missing tracking identifier.")

    rules = contract.preprocessing_rules
    numeric_rules: Mapping[str, Mapping[str, float]] = rules["numeric_rules"]
    categorical_rules: Mapping[str, Mapping[str, int]] = rules["categorical_rules"]
    unknown_code = int(
        rules.get("categorical_encoding_policy", {}).get(
            "unknown_category_code", DEFAULT_UNKNOWN_CATEGORY_CODE
        )
    )

    transformed = pd.DataFrame(index=frame.index)
    for feature in contract.model_features:
        if feature in numeric_rules:
            rule = numeric_rules[feature]
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan
            )
            invalid = ~np.isfinite(values)
            values[invalid] = float(rule["fill_value"])
            scale = float(rule["scale"])
            if not np.isfinite(scale) or scale == 0:
                raise ValueError(f"Invalid frozen scale for numeric feature {feature!r}.")
            transformed[feature] = (
                (values - float(rule["mean"])) / scale
            ).astype("float32")
            continue

        raw_name = feature[: -len("_code")]
        mapping = categorical_rules[raw_name]
        values = frame[raw_name].astype("string").fillna(MISSING_CATEGORY_VALUE)
        transformed[feature] = values.map(mapping).fillna(unknown_code).astype("int32")

    matrix = transformed.to_numpy(dtype="float32")
    if not np.isfinite(matrix).all():
        raise ValueError("Frozen preprocessing produced NaN or infinite model inputs.")
    return transformed.loc[:, contract.model_features]


def predict_purchase_probabilities(
    frame: pd.DataFrame,
    model: Any,
    contract: InferenceContract,
) -> pd.DataFrame:
    """Score one batch and return user-item purchase probabilities."""

    features = transform_inference_features(frame, contract)
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(frame) or probabilities.shape[1] < 2:
        raise ValueError(
            "predict_proba() must return an n_rows x n_classes array with a positive class column."
        )
    scores = probabilities[:, 1].astype("float64", copy=False)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model returned an invalid purchase probability.")

    output = frame.loc[:, contract.tracking_columns].copy()
    output["purchase_probability"] = scores
    output["predicted_purchase"] = (scores >= contract.prediction_threshold).astype("int8")
    return output


def _input_batches(path: Path, columns: list[str], batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield projected CSV or Parquet input batches."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, usecols=columns, chunksize=batch_size)
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required for Parquet batch inference.") from exc
        parquet = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Inference input is missing required columns: {missing}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()
        return
    raise ValueError("Inference input must be a .parquet or .csv file.")


def _write_csv_batches(batches: Iterable[pd.DataFrame], target: Path) -> int:
    """Atomically write prediction batches to CSV and return the row count."""

    temporary = target.with_suffix(target.suffix + ".tmp")
    rows = 0
    first = True
    try:
        for frame in batches:
            frame.to_csv(temporary, mode="w" if first else "a", header=first, index=False)
            first = False
            rows += len(frame)
        if first:
            raise ValueError("Inference input has no rows.")
        temporary.replace(target)
        return rows
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_batches(batches: Iterable[pd.DataFrame], target: Path) -> int:
    """Atomically write prediction batches to Parquet and return the row count."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for Parquet prediction output.") from exc

    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = None
    rows = 0
    try:
        for frame in batches:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
            rows += len(frame)
        if writer is None:
            raise ValueError("Inference input has no rows.")
        writer.close()
        writer = None
        temporary.replace(target)
        return rows
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def run_batch_inference(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    summary_path: str | Path | None = DEFAULT_SUMMARY_PATH,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run local batch purchase-probability inference with frozen artifacts.

    Args:
        input_path: Raw Stage 2 user-item feature CSV/Parquet. It may contain
            extra columns, including labels, but only tracking IDs and raw
            features required by the frozen final feature list are read.
        output_path: CSV/Parquet receiving user/item identifiers and scores.
        config_path: Locked Stage 5 contract pointing to Stage 3/4 artifacts.
        summary_path: Optional JSON runtime/resource record. Pass ``None`` to
            suppress it.
        batch_size: Maximum rows scored per batch.
        project_root: Repository root used to resolve relative config paths.

    Returns:
        JSON-serializable run metadata including row count, runtime, throughput,
        memory usage, model candidate, and frozen artifact paths.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Inference input does not exist: {source}")
    if destination.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Prediction output must be a .csv or .parquet file.")
    if source == destination:
        raise ValueError("Inference input and output paths must be different.")

    contract = load_inference_contract(
        config_path,
        project_root=project_root,
        require_model=True,
    )
    model = load_final_model(contract)
    projected_columns = list(dict.fromkeys([*contract.tracking_columns, *contract.raw_feature_columns]))

    process = psutil.Process(os.getpid())
    memory_samples = [process.memory_info().rss]
    stop_event = threading.Event()

    def sample_memory() -> None:
        while not stop_event.wait(0.05):
            try:
                memory_samples.append(process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, daemon=True)
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    sampler.start()

    try:
        input_batches = _input_batches(source, projected_columns, batch_size)
        prediction_batches = (
            predict_purchase_probabilities(frame, model, contract)
            for frame in input_batches
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".csv":
            row_count = _write_csv_batches(prediction_batches, destination)
        else:
            row_count = _write_parquet_batches(prediction_batches, destination)
    finally:
        stop_event.set()
        sampler.join()

    wall_seconds = time.perf_counter() - started
    cpu_after = process.cpu_times()
    memory_samples.append(process.memory_info().rss)
    cpu_seconds = cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
    summary: dict[str, Any] = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_candidate": contract.final_candidate,
        "model_artifact": str(contract.model_artifact),
        "model_bundle_key": contract.model_bundle_key,
        "preprocessing_rules": str(contract.preprocessing_rules_path),
        "feature_list": str(contract.feature_list_path),
        "model_feature_count": len(contract.model_features),
        "prediction_threshold": contract.prediction_threshold,
        "input_path": str(source),
        "output_path": str(destination),
        "output_columns": [*contract.tracking_columns, "purchase_probability", "predicted_purchase"],
        "row_count": row_count,
        "batch_size": batch_size,
        "runtime_seconds": round(wall_seconds, 6),
        "process_cpu_seconds": round(cpu_seconds, 6),
        "throughput_rows_per_second": round(row_count / wall_seconds, 2) if wall_seconds else None,
        "rss_start_bytes": memory_samples[0],
        "rss_end_bytes": memory_samples[-1],
        "peak_process_rss_bytes": max(memory_samples),
        "gpu_used_by_inference_wrapper": False,
        "preprocessing_refit_performed": False,
        "feature_selection_performed": False,
        "model_training_performed": False,
    }

    if summary_path is not None:
        summary_output = Path(summary_path).expanduser().resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_output)
    return summary
