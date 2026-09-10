"""Leakage-safe Stage4 Issue2.5 stacking on training OOF predictions only."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .model_tuning import TRAIN_FILENAMES, _build_estimator, _candidate_parts
from .training_matrix import (
    TRACKING_COLUMNS,
    _class_weights,
    _json_safe,
    _load_features,
    _load_split,
    _metrics,
)


DEFAULT_FOLDS = 5
DEFAULT_BASE_MODEL_COUNT = 3
VALIDATION_FILENAME = "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet"
BASELINE_TRAIN_FILENAME = "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet"


def _load_issue2_parameters(path: Path) -> dict[str, dict[str, Any]]:
    """Load Issue2 validation-selected parameters and reject incomplete records."""

    if not path.is_file():
        raise FileNotFoundError(f"Issue2 parameter record does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Issue2 parameter record must be a non-empty JSON object.")
    for candidate, record in payload.items():
        _candidate_parts(candidate)
        if not isinstance(record, dict) or not isinstance(record.get("best_parameters"), dict):
            raise ValueError(f"Issue2 parameter record is missing best_parameters for {candidate}.")
    return payload


def _select_base_candidates(pool_path: Path, parameters: Mapping[str, Mapping[str, Any]], count: int) -> list[str]:
    """Choose top validation candidates while requiring different model families."""

    if count < 2:
        raise ValueError("base_model_count must be at least two.")
    if not pool_path.is_file():
        raise FileNotFoundError(f"Issue2 candidate-pool file does not exist: {pool_path}")
    pool = pd.read_csv(pool_path)
    required = {"candidate", "auc", "average_precision", "logloss"}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"Issue2 candidate pool is missing columns: {missing}")
    pool = pool.loc[pool["candidate"].isin(parameters)].copy()
    if "issue2_5_eligible" in pool:
        pool = pool.loc[pool["issue2_5_eligible"].astype(bool)]
    pool = pool.sort_values(["auc", "average_precision", "logloss"], ascending=[False, False, True])
    selected: list[str] = []
    selected_models: set[str] = set()
    for candidate in pool["candidate"]:
        _, model_name = _candidate_parts(candidate)
        if model_name in selected_models:
            continue
        selected.append(candidate)
        selected_models.add(model_name)
        if len(selected) == count:
            break
    if len(selected) < 2:
        raise ValueError("Issue2 candidate pool must contain at least two different model types.")
    return selected


def _fold_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Fit balanced class weights from one cross-validation training fold only."""

    negative = int((labels == 0).sum())
    positive = int((labels == 1).sum())
    if not negative or not positive:
        raise ValueError("Every OOF training fold must contain both label classes.")
    total = len(labels)
    return {0: total / (2 * negative), 1: total / (2 * positive)}


def _undersample(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply train-fold-only random undersampling while retaining all positives."""

    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) == 0 or len(negative) < len(positive):
        raise ValueError("Undersampling requires minority positives and enough negatives.")
    rng = np.random.default_rng(random_state)
    selected = np.concatenate([positive, rng.choice(negative, size=len(positive), replace=False)])
    rng.shuffle(selected)
    return x[selected], y[selected]


def _smote(x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Create train-fold-only mixed-type SMOTE samples without exposing held-out rows."""

    positive = x[y == 1]
    negative_count = int((y == 0).sum())
    if len(positive) < 2 or negative_count < len(positive):
        raise ValueError("SMOTE requires at least two minority positives and enough negatives.")
    synthetic_count = negative_count - len(positive)
    if synthetic_count == 0:
        return x, y
    categorical = [index for index, name in enumerate(feature_names) if name.endswith("_code")]
    continuous = [index for index in range(x.shape[1]) if index not in categorical]
    neighbor_values = positive[:, continuous] if continuous else positive
    neighbors = NearestNeighbors(n_neighbors=min(6, len(positive))).fit(neighbor_values)
    neighbor_indexes = neighbors.kneighbors(neighbor_values, return_distance=False)[:, 1:]
    rng = np.random.default_rng(random_state)
    source_indexes = np.arange(synthetic_count) % len(positive)
    neighbor_choice = rng.integers(0, neighbor_indexes.shape[1], size=synthetic_count)
    matched_indexes = neighbor_indexes[source_indexes, neighbor_choice]
    source = positive[source_indexes]
    matched = positive[matched_indexes]
    synthetic = source.copy()
    if continuous:
        fractions = rng.random((synthetic_count, len(continuous)), dtype="float32")
        synthetic[:, continuous] = source[:, continuous] + fractions * (matched[:, continuous] - source[:, continuous])
    for index in categorical:
        synthetic[:, index] = np.where(rng.integers(0, 2, size=synthetic_count).astype(bool), matched[:, index], source[:, index])
    return np.concatenate([x, synthetic]), np.concatenate([y, np.ones(synthetic_count, dtype=y.dtype)])


def _fold_training_data(
    strategy: str,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, float] | None]:
    """Apply the candidate's strategy inside one OOF training fold."""

    if strategy == "baseline":
        return x, y, None
    if strategy == "undersampled":
        sampled_x, sampled_y = _undersample(x, y, random_state)
        return sampled_x, sampled_y, None
    if strategy == "smote":
        sampled_x, sampled_y = _smote(x, y, feature_names, random_state)
        return sampled_x, sampled_y, None
    if strategy == "class_weight":
        return x, y, _fold_class_weights(y)
    raise ValueError(f"Unsupported training strategy: {strategy}")


def _prediction_frame(
    tracking: pd.DataFrame,
    labels: np.ndarray,
    score_columns: Mapping[str, np.ndarray],
    split: str,
) -> pd.DataFrame:
    """Return tracked OOF or validation probabilities in one reusable schema."""

    output = tracking.copy()
    output["label"] = labels.astype("int8")
    output["dataset_split"] = split
    for name, values in score_columns.items():
        output[name] = values.astype("float64")
    return output


def run_stacking_fusion(
    split_directory: str | Path,
    feature_list_path: str | Path,
    class_weight_path: str | Path,
    issue2_parameters_path: str | Path,
    issue2_candidate_pool_path: str | Path,
    models_directory: str | Path,
    reports_directory: str | Path,
    base_model_count: int = DEFAULT_BASE_MODEL_COUNT,
    folds: int = DEFAULT_FOLDS,
    random_state: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train Issue2.5 stacking using OOF probabilities and validation comparison only.

    The test set is intentionally not resolved, read, or written by this function.
    """

    if folds < 2:
        raise ValueError("folds must be at least two.")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu.")
    split_root = Path(split_directory).expanduser().resolve()
    parameters = _load_issue2_parameters(Path(issue2_parameters_path).expanduser().resolve())
    selected = _select_base_candidates(Path(issue2_candidate_pool_path).expanduser().resolve(), parameters, base_model_count)
    feature_names = _load_features(Path(feature_list_path).expanduser().resolve())
    fixed_class_weights = _class_weights(Path(class_weight_path).expanduser().resolve())
    train_x_frame, train_y, train_tracking = _load_split(split_root / BASELINE_TRAIN_FILENAME, feature_names)
    validation_x_frame, validation_y, validation_tracking = _load_split(split_root / VALIDATION_FILENAME, feature_names)
    train_x = train_x_frame.to_numpy(dtype="float32")
    validation_x = validation_x_frame.to_numpy(dtype="float32")
    if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
        raise ValueError("Training and validation data must each contain both label classes.")

    model_root = Path(models_directory).expanduser().resolve()
    report_root = Path(reports_directory).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = psutil.Process()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_scores = np.zeros((len(train_y), len(selected)), dtype="float64")
    fold_records: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(selected):
        strategy, model_name = _candidate_parts(candidate)
        best_parameters = parameters[candidate]["best_parameters"]
        for fold_number, (fit_index, holdout_index) in enumerate(splitter.split(train_x, train_y), start=1):
            fold_x, fold_y, fold_weights = _fold_training_data(
                strategy, train_x[fit_index], train_y[fit_index], feature_names,
                random_state + candidate_index * 100 + fold_number,
            )
            estimator = _build_estimator(model_name, best_parameters, fold_weights, random_state + fold_number, device=device)
            fold_started = time.perf_counter()
            rss_before = process.memory_info().rss
            estimator.fit(fold_x, fold_y)
            oof_scores[holdout_index, candidate_index] = estimator.predict_proba(train_x[holdout_index])[:, 1]
            fold_records.append({
                "candidate": candidate,
                "fold": fold_number,
                "training_strategy": strategy,
                "model_name": model_name,
                "fit_sample_count": int(len(fold_y)),
                "holdout_sample_count": int(len(holdout_index)),
                "fit_seconds": round(time.perf_counter() - fold_started, 6),
                "rss_delta_bytes": int(process.memory_info().rss - rss_before),
            })

    meta_model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=random_state)
    meta_started = time.perf_counter()
    meta_model.fit(oof_scores, train_y)

    validation_scores: dict[str, np.ndarray] = {}
    full_models: dict[str, Any] = {}
    for candidate in selected:
        strategy, model_name = _candidate_parts(candidate)
        final_train_x, final_train_y, _ = _load_split(split_root / TRAIN_FILENAMES[strategy], feature_names)
        weights = fixed_class_weights if strategy == "class_weight" else None
        estimator = _build_estimator(model_name, parameters[candidate]["best_parameters"], weights, random_state, device=device)
        estimator.fit(final_train_x, final_train_y)
        validation_scores[candidate] = estimator.predict_proba(validation_x)[:, 1]
        full_models[candidate] = estimator
    validation_matrix = np.column_stack([validation_scores[candidate] for candidate in selected])
    stacking_scores = meta_model.predict_proba(validation_matrix)[:, 1]

    artifact_root = model_root / "stacking"
    model_path = artifact_root / "models" / "stacking_meta_model.joblib"
    base_model_path = artifact_root / "models" / "base_models.joblib"
    oof_path = artifact_root / "oof_predictions" / "training_oof_predictions.parquet"
    validation_path = artifact_root / "validation_predictions" / "stacking_validation_predictions.parquet"
    log_path = artifact_root / "run_logs" / "stacking_run.json"
    for directory in (model_path.parent, oof_path.parent, validation_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, model_path)
    joblib.dump(full_models, base_model_path)
    score_names = {f"score_{candidate}": oof_scores[:, index] for index, candidate in enumerate(selected)}
    oof_frame = _prediction_frame(train_tracking, train_y, score_names, "training_oof")
    oof_frame.to_parquet(oof_path, index=False)
    validation_score_names = {f"score_{candidate}": validation_scores[candidate] for candidate in selected}
    validation_score_names["stacking_prediction_score"] = stacking_scores
    validation_frame = _prediction_frame(validation_tracking, validation_y, validation_score_names, "validation")
    validation_frame["stacking_prediction_label"] = (stacking_scores >= 0.5).astype("int8")
    validation_frame.to_parquet(validation_path, index=False)

    comparison_rows = []
    for candidate in selected:
        strategy, model_name = _candidate_parts(candidate)
        comparison_rows.append({
            "candidate": candidate,
            "model_type": model_name,
            "training_strategy": strategy,
            "variant": "single_model",
            **_metrics(validation_y, validation_scores[candidate]),
        })
    comparison_rows.append({
        "candidate": "stacking_logistic_regression",
        "model_type": "logistic_regression",
        "training_strategy": "stacking",
        "variant": "stacking",
        **_metrics(validation_y, stacking_scores),
    })
    comparison = pd.DataFrame(comparison_rows).sort_values(["auc", "average_precision", "logloss"], ascending=[False, False, True])
    comparison_path = report_root / "stacking_validation_performance_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    chosen_row = comparison.iloc[0]
    candidate_manifest = {
        "selection_split": "validation",
        "selection_metric": "auc",
        "base_candidates": selected,
        "base_model_parameters": {candidate: parameters[candidate]["best_parameters"] for candidate in selected},
        "meta_model": {"model_name": "logistic_regression", "parameters": _json_safe(meta_model.get_params())},
        "issue3_candidates": comparison[["candidate", "variant", "auc", "average_precision", "logloss"]].to_dict(orient="records"),
        "validation_leader": str(chosen_row["candidate"]),
        "test_data_read": False,
        "test_used_for_training_or_selection": False,
    }
    manifest_path = report_root / "issue3_candidate_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(candidate_manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fold_path = report_root / "stacking_oof_fold_records.csv"
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    run_log = {
        **candidate_manifest,
        "folds": folds,
        "base_model_count": len(selected),
        "device": device,
        "oof_prediction_path": str(oof_path),
        "validation_prediction_path": str(validation_path),
        "meta_model_path": str(model_path),
        "base_models_path": str(base_model_path),
        "comparison_path": str(comparison_path),
        "fold_records_path": str(fold_path),
        "meta_fit_seconds": round(time.perf_counter() - meta_started, 6),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_process_rss_bytes": process.memory_info().rss,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    log_path.write_text(json.dumps(_json_safe(run_log), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = report_root / "stacking_fusion_summary.json"
    summary_path.write_text(json.dumps(_json_safe(run_log), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "comparison_path": comparison_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "oof_path": oof_path,
        "validation_path": validation_path,
        "selected_candidates": selected,
    }
