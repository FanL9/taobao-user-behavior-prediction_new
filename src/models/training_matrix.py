"""Train fixed-parameter traditional-model baselines on selected features."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


TRACKING_COLUMNS = ("user_id", "item_id", "category_id")
TARGET_COLUMN = "label"
SYNTHETIC_FLAG_COLUMN = "is_synthetic"
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
TRAINING_STRATEGIES = ("baseline", "smote", "undersampled", "class_weight")
PREDICTION_THRESHOLD = 0.5


def _json_safe(value: Any) -> Any:
    """Convert estimator parameters and NumPy values into JSON-compatible data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON configuration object."""

    if not path.is_file():
        raise FileNotFoundError(f"Required configuration does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return payload


def _load_features(feature_list_path: Path) -> list[str]:
    """Read the Issue4 final feature list and reject protected columns."""

    payload = _read_json(feature_list_path)
    features = payload.get("selected_model_features", payload.get("model_feature_columns"))
    if not isinstance(features, list) or not features or not all(isinstance(name, str) for name in features):
        raise ValueError("Feature-list JSON must contain a non-empty selected_model_features list.")
    protected = {*TRACKING_COLUMNS, TARGET_COLUMN, SYNTHETIC_FLAG_COLUMN}
    invalid = sorted(set(features) & protected)
    if invalid:
        raise ValueError(f"Feature list contains non-model columns: {invalid}")
    if len(features) != len(set(features)):
        raise ValueError("Feature list contains duplicate columns.")
    return features


def _load_split(path: Path, features: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load one selected split with tracked IDs, numeric inputs, and binary labels."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    columns = [*TRACKING_COLUMNS, TARGET_COLUMN, *features]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError(f"{path.name} has no rows.")
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{path.name} contains labels outside 0 and 1.")
    inputs = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if inputs.isna().any().any() or not np.isfinite(inputs.to_numpy(dtype="float32")).all():
        raise ValueError(f"{path.name} contains missing or non-finite selected features.")
    return inputs.astype("float32"), labels.to_numpy(dtype="int8"), frame.loc[:, TRACKING_COLUMNS].copy()


def _class_weights(path: Path | None) -> dict[int, float] | None:
    """Load Issue5 balanced class weights when the class-weight strategy is selected."""

    if path is None:
        return None
    payload = _read_json(path)
    raw = payload.get("class_weight")
    if not isinstance(raw, Mapping) or set(raw) != {"0", "1"}:
        raise ValueError("Class-weight JSON must contain numeric string keys '0' and '1'.")
    weights = {int(key): float(value) for key, value in raw.items()}
    if not all(np.isfinite(value) and value > 0 for value in weights.values()):
        raise ValueError("Class weights must be finite positive values.")
    return weights


def _build_estimator(name: str, class_weights: dict[int, float] | None, random_state: int):
    """Build one fixed baseline estimator; no hyperparameter search is performed."""

    if name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs", max_iter=200, C=1.0, class_weight=class_weights,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10,
            class_weight=class_weights, n_jobs=-1, random_state=random_state,
        )
    scale_pos_weight = None if class_weights is None else class_weights[1] / class_weights[0]
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100, num_leaves=31, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            n_jobs=-1, random_state=random_state, verbosity=-1,
        )
    raise ValueError(f"Unsupported model name: {name}")


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Calculate fixed-threshold classification metrics plus ranking and loss metrics."""

    clipped = np.clip(scores.astype("float64"), 1e-15, 1 - 1e-15)
    predicted = (clipped >= PREDICTION_THRESHOLD).astype("int8")
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "average_precision": float(average_precision_score(labels, clipped)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "logloss": float(log_loss(labels, clipped, labels=[0, 1])),
        "positive_prediction_ratio": float(predicted.mean()),
    }


def _write_predictions(
    tracking: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, model_name: str, split: str, target: Path
) -> None:
    """Write tracked probability predictions for validation or untouched test data."""

    target.parent.mkdir(parents=True, exist_ok=True)
    output = tracking.copy()
    output[TARGET_COLUMN] = labels.astype("int8")
    output["prediction_score"] = scores.astype("float64")
    output["prediction_label"] = (scores >= PREDICTION_THRESHOLD).astype("int8")
    output["model_name"] = model_name
    output["dataset_split"] = split
    output.to_parquet(target, index=False)


def _write_aggregate_comparison(report_root: Path) -> Path:
    """Combine completed strategy comparison tables without selecting a winner."""

    tables = []
    strategy_tables = sorted(report_root.glob("baseline_model_comparison_*.csv"))
    legacy = report_root / "baseline_model_comparison.csv"
    # The old unsuffixed table is class_weight only; retain it solely until a
    # matrix-generated class_weight table exists, preventing duplicate rows.
    candidates = strategy_tables if any(path.name.endswith("_class_weight.csv") for path in strategy_tables) else [legacy, *strategy_tables]
    for path in candidates:
        if path.is_file():
            tables.append(pd.read_csv(path))
    if not tables:
        raise FileNotFoundError("No baseline comparison table is available to aggregate.")
    combined = pd.concat(tables, ignore_index=True).sort_values(
        ["dataset_split", "training_strategy", "auc", "model_name"],
        ascending=[True, True, False, True],
    )
    target = report_root / "baseline_model_performance_comparison.csv"
    combined.to_csv(target, index=False)
    return target


def _train_one_strategy(
    dataset_paths: Mapping[str, str | Path],
    feature_list_path: str | Path,
    models_directory: str | Path,
    reports_directory: str | Path,
    class_weight_path: str | Path | None = None,
    model_names: Iterable[str] = MODEL_NAMES,
    training_strategy: str = "class_weight",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train four baseline models and write predictions, metrics, parameters, and logs.

    The validation data is used only to compare recorded baseline results. The test
    data is scored after fitting and is never used to tune parameters or choose a
    model. ``training_strategy='class_weight'`` keeps the original unsampled rows
    and applies Issue5 weights; passing ``None`` for ``class_weight_path`` creates
    an unweighted original-data baseline.
    """

    required_splits = {"train", "validation", "test"}
    if set(dataset_paths) != required_splits:
        raise ValueError("dataset_paths must contain exactly train, validation, and test.")
    names = tuple(model_names)
    if not names or len(names) != len(set(names)) or set(names) - set(MODEL_NAMES):
        raise ValueError(f"model_names must be a unique non-empty subset of {MODEL_NAMES}.")
    if training_strategy not in TRAINING_STRATEGIES:
        raise ValueError("training_strategy must be baseline, class_weight, smote, or undersampled.")
    if training_strategy == "class_weight" and class_weight_path is None:
        raise ValueError("class_weight_path is required for the class_weight strategy.")

    paths = {name: Path(value).expanduser().resolve() for name, value in dataset_paths.items()}
    features = _load_features(Path(feature_list_path).expanduser().resolve())
    class_weights = _class_weights(Path(class_weight_path).expanduser().resolve()) if class_weight_path else None
    model_root = Path(models_directory).expanduser().resolve() / training_strategy
    report_root = Path(reports_directory).expanduser().resolve()
    artifact_dir = model_root / "models"
    validation_prediction_dir = model_root / "validation_predictions"
    test_prediction_dir = model_root / "test_predictions"
    log_dir = model_root / "run_logs"
    for directory in (artifact_dir, validation_prediction_dir, test_prediction_dir, log_dir, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    train_x, train_y, _ = _load_split(paths["train"], features)
    validation_x, validation_y, validation_tracking = _load_split(paths["validation"], features)
    test_x, test_y, test_tracking = _load_split(paths["test"], features)
    if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2 or len(np.unique(test_y)) != 2:
        raise ValueError("Each split must contain both binary label classes for baseline metrics.")

    process = psutil.Process()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for name in names:
        estimator = _build_estimator(name, class_weights, random_state)
        rss_before = process.memory_info().rss
        model_started = time.perf_counter()
        estimator.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - model_started
        validation_scores = estimator.predict_proba(validation_x)[:, 1]
        test_scores = estimator.predict_proba(test_x)[:, 1]
        validation_path = validation_prediction_dir / f"{name}.parquet"
        test_path = test_prediction_dir / f"{name}.parquet"
        _write_predictions(validation_tracking, validation_y, validation_scores, name, "validation", validation_path)
        _write_predictions(test_tracking, test_y, test_scores, name, "test", test_path)
        artifact_path = artifact_dir / f"{name}.joblib"
        joblib.dump(estimator, artifact_path)
        validation_metrics = _metrics(validation_y, validation_scores)
        test_metrics = _metrics(test_y, test_scores)
        for split, metrics in (("validation", validation_metrics), ("test", test_metrics)):
            rows.append({"model_name": name, "training_strategy": training_strategy, "dataset_split": split, **metrics})
        run = {
            "model_name": name,
            "training_strategy": training_strategy,
            "random_state": random_state,
            "feature_count": len(features),
            "feature_list": features,
            "training_sample_count": int(len(train_y)),
            "validation_sample_count": int(len(validation_y)),
            "test_sample_count": int(len(test_y)),
            "class_weights": class_weights,
            "parameters": _json_safe(estimator.get_params()),
            "fit_seconds": round(fit_seconds, 6),
            "rss_delta_bytes": int(process.memory_info().rss - rss_before),
            "artifact_path": str(artifact_path),
            "validation_prediction_path": str(validation_path),
            "test_prediction_path": str(test_path),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_used_for_tuning_or_selection": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        log_path = log_dir / f"{name}_run.json"
        log_path.write_text(json.dumps(_json_safe(run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        runs[name] = run

    comparison = pd.DataFrame(rows).sort_values(["dataset_split", "auc", "model_name"], ascending=[True, False, True])
    comparison_path = report_root / f"baseline_model_comparison_{training_strategy}.csv"
    comparison.to_csv(comparison_path, index=False)
    aggregate_comparison_path = _write_aggregate_comparison(report_root)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "training_strategy": training_strategy,
        "data_paths": {key: str(value) for key, value in paths.items()},
        "feature_list_path": str(Path(feature_list_path).expanduser().resolve()),
        "class_weight_path": str(Path(class_weight_path).expanduser().resolve()) if class_weight_path else None,
        "validation_used_for_comparison": True,
        "test_used_for_tuning_or_selection": False,
        "model_selection_decision": None,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "models": runs,
        "comparison_path": str(comparison_path),
        "aggregate_comparison_path": str(aggregate_comparison_path),
    }
    summary_path = report_root / f"baseline_training_summary_{training_strategy}.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"comparison_path": comparison_path, "aggregate_comparison_path": aggregate_comparison_path, "summary_path": summary_path, "comparison": comparison, "summary": summary}


def train_model_matrix(
    dataset_paths_by_strategy: Mapping[str, Mapping[str, str | Path]],
    feature_list_path: str | Path,
    models_directory: str | Path,
    reports_directory: str | Path,
    class_weight_path: str | Path,
    strategies: Iterable[str] = TRAINING_STRATEGIES,
    model_names: Iterable[str] = MODEL_NAMES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the requested sampling-strategy × model matrix with one entry point.

    The default matrix contains four training strategies and four fixed traditional
    models, producing sixteen fitted models. Every strategy is evaluated against
    the same untouched validation and test datasets; no result selects a winner.
    """

    chosen = tuple(strategies)
    chosen_models = tuple(model_names)
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(TRAINING_STRATEGIES):
        raise ValueError(f"strategies must be a unique non-empty subset of {TRAINING_STRATEGIES}.")
    missing = sorted(set(chosen) - set(dataset_paths_by_strategy))
    if missing:
        raise ValueError(f"dataset_paths_by_strategy is missing strategies: {missing}")
    results = {}
    for strategy in chosen:
        results[strategy] = _train_one_strategy(
            dataset_paths_by_strategy[strategy],
            feature_list_path,
            models_directory,
            reports_directory,
            class_weight_path=class_weight_path if strategy == "class_weight" else None,
            model_names=chosen_models,
            training_strategy=strategy,
            random_state=random_state,
        )
    report_root = Path(reports_directory).expanduser().resolve()
    return {
        "strategy_results": results,
        "aggregate_comparison_path": _write_aggregate_comparison(report_root),
        "model_count": len(chosen) * len(chosen_models),
    }
