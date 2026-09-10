"""Optuna tuning for all Stage4 traditional-model and sampling combinations.

This module intentionally has no test-set input.  It searches parameters and
compares models on the validation split only; final test evaluation belongs to
Stage4 Issue3.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .training_matrix import (
    MODEL_NAMES,
    TRAINING_STRATEGIES,
    _class_weights,
    _json_safe,
    _load_features,
    _load_split,
    _metrics,
    _write_predictions,
)


SELECTION_METRIC = "auc"
# A uniform five-trial budget keeps the full 16-combination Issue2 run practical
# on the complete training data while preserving an equal search budget.
DEFAULT_N_TRIALS = 5
VALIDATION_FILENAME = "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet"
TRAIN_FILENAMES = {
    "baseline": "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet",
    "smote": "user_item_feature_wide_labeled_train_preprocessed_selected_smote.parquet",
    "undersampled": "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet",
    "class_weight": "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet",
}
TUNING_CANDIDATES = tuple(f"{strategy}__{model}" for strategy in TRAINING_STRATEGIES for model in MODEL_NAMES)


def _candidate_parts(candidate: str) -> tuple[str, str]:
    """Return the training strategy and estimator name encoded in a candidate."""

    try:
        strategy, model_name = candidate.split("__", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Invalid tuning candidate: {candidate}") from error
    if strategy not in TRAINING_STRATEGIES or model_name not in MODEL_NAMES:
        raise ValueError(f"Invalid tuning candidate: {candidate}")
    return strategy, model_name


def _suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Define the Issue2 Optuna search space for one traditional estimator."""

    if model_name == "logistic_regression":
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        penalty = "l2" if solver == "lbfgs" else trial.suggest_categorical("penalty", ["l1", "l2"])
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "penalty": penalty,
            "solver": solver,
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=200),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_int("max_depth", -1, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    raise ValueError(f"Unsupported model name: {model_name}")


def _build_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    class_weights: dict[int, float] | None,
    random_state: int,
    device: str = "cpu",
):
    """Build one candidate estimator with CPU or GPU tree training when supported."""

    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu.")

    if model_name == "logistic_regression":
        return LogisticRegression(**parameters, class_weight=class_weights, random_state=random_state, n_jobs=None)
    if model_name == "random_forest":
        return RandomForestClassifier(**parameters, class_weight=class_weights, n_jobs=-1, random_state=random_state)
    scale_pos_weight = None if class_weights is None else class_weights[1] / class_weights[0]
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        kwargs = {**parameters, "tree_method": "hist", "eval_metric": "logloss", "n_jobs": -1, "random_state": random_state}
        if device == "gpu":
            kwargs["device"] = "cuda"
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**kwargs)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        kwargs = {**parameters, "n_jobs": -1, "random_state": random_state, "verbosity": -1}
        # The project LightGBM wheel is CPU-only in the current Windows setup.
        # Keep LightGBM on CPU when ``device='gpu'`` so that an XGBoost GPU run
        # remains reproducible instead of failing during OpenCL initialization.
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight
        return LGBMClassifier(**kwargs)
    raise ValueError(f"Unsupported model name: {model_name}")


def _baseline_validation_rows(report_root: Path, candidates: Iterable[str]) -> list[dict[str, Any]]:
    """Read only validation baseline rows to preserve the no-test Issue2 boundary."""

    path = report_root.parent / "baseline_model_performance_comparison.csv"
    if not path.is_file():
        return []
    baseline = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        strategy, model_name = _candidate_parts(candidate)
        match = baseline.loc[
            baseline["training_strategy"].eq(strategy)
            & baseline["model_name"].eq(model_name)
            & baseline["dataset_split"].eq("validation")
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append({
            "candidate": candidate,
            "training_strategy": strategy,
            "model_name": model_name,
            "variant": "baseline",
            "dataset_split": "validation",
            **{key: float(row[key]) for key in ("auc", "average_precision", "precision", "recall", "f1", "logloss", "positive_prediction_ratio")},
        })
    return rows


def _trial_row(candidate: str, strategy: str, model_name: str, trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Serialize a completed Optuna trial and its validation metrics."""

    attributes = trial.user_attrs
    return {
        "candidate": candidate,
        "training_strategy": strategy,
        "model_name": model_name,
        "trial_number": trial.number,
        "state": trial.state.name,
        "parameters": json.dumps(_json_safe(trial.params), ensure_ascii=False, sort_keys=True),
        "fit_seconds": attributes.get("fit_seconds"),
        "rss_delta_bytes": attributes.get("rss_delta_bytes"),
        "auc": attributes.get("auc"),
        "average_precision": attributes.get("average_precision"),
        "precision": attributes.get("precision"),
        "recall": attributes.get("recall"),
        "f1": attributes.get("f1"),
        "logloss": attributes.get("logloss"),
        "positive_prediction_ratio": attributes.get("positive_prediction_ratio"),
    }


def tune_all_models(
    split_directory: str | Path,
    feature_list_path: str | Path,
    models_directory: str | Path,
    reports_directory: str | Path,
    class_weight_path: str | Path,
    candidates: Iterable[str] = TUNING_CANDIDATES,
    n_trials: int = DEFAULT_N_TRIALS,
    random_state: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Tune requested Issue2 candidates using training and validation data only.

    No test path is resolved or read.  The function records every Optuna trial,
    retrains the validation-selected configuration, then writes only validation
    predictions, parameter records, comparison tables, and a fusion candidate pool.
    """

    selected = tuple(candidates)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(TUNING_CANDIDATES):
        raise ValueError(f"candidates must be a unique non-empty subset of {TUNING_CANDIDATES}.")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu.")
    split_root = Path(split_directory).expanduser().resolve()
    feature_path = Path(feature_list_path).expanduser().resolve()
    model_root = Path(models_directory).expanduser().resolve()
    report_root = Path(reports_directory).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    features = _load_features(feature_path)
    validation_x, validation_y, validation_tracking = _load_split(split_root / VALIDATION_FILENAME, features)
    weights = _class_weights(Path(class_weight_path).expanduser().resolve())
    process = psutil.Process()
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for candidate_index, candidate in enumerate(selected):
        strategy, model_name = _candidate_parts(candidate)
        train_path = split_root / TRAIN_FILENAMES[strategy]
        train_x, train_y, _ = _load_split(train_path, features)
        fixed_weights = weights if strategy == "class_weight" else None
        sampler = optuna.samplers.TPESampler(seed=random_state + candidate_index)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=candidate)

        def objective(trial: optuna.Trial) -> float:
            parameters = _suggest_parameters(trial, model_name)
            estimator = _build_estimator(model_name, parameters, fixed_weights, random_state, device=device)
            rss_before = process.memory_info().rss
            trial_started = time.perf_counter()
            estimator.fit(train_x, train_y)
            scores = estimator.predict_proba(validation_x)[:, 1]
            metrics = _metrics(validation_y, scores)
            trial.set_user_attr("fit_seconds", round(time.perf_counter() - trial_started, 6))
            trial.set_user_attr("rss_delta_bytes", int(process.memory_info().rss - rss_before))
            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            return metrics[SELECTION_METRIC]

        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        trial_rows.extend(_trial_row(candidate, strategy, model_name, trial) for trial in study.trials)
        best_parameters = study.best_trial.params
        final_estimator = _build_estimator(model_name, best_parameters, fixed_weights, random_state, device=device)
        final_started = time.perf_counter()
        final_estimator.fit(train_x, train_y)
        validation_scores = final_estimator.predict_proba(validation_x)[:, 1]
        validation_metrics = _metrics(validation_y, validation_scores)

        candidate_root = model_root / strategy / model_name
        model_path = candidate_root / "models" / "tuned_model.joblib"
        prediction_path = candidate_root / "validation_predictions" / "tuned_model.parquet"
        log_path = candidate_root / "run_logs" / "tuning_run.json"
        for directory in (model_path.parent, prediction_path.parent, log_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_estimator, model_path)
        _write_predictions(validation_tracking, validation_y, validation_scores, f"{candidate}_tuned", "validation", prediction_path)
        summary = {
            "candidate": candidate,
            "training_strategy": strategy,
            "model_name": model_name,
            "selection_metric": SELECTION_METRIC,
            "selection_split": "validation",
            "n_trials": n_trials,
            "device": device,
            "best_trial_number": study.best_trial.number,
            "best_parameters": _json_safe(best_parameters),
            "feature_count": len(features),
            "feature_list": features,
            "training_data_path": str(train_path),
            "validation_data_path": str(split_root / VALIDATION_FILENAME),
            "class_weights": fixed_weights,
            "validation_metrics": validation_metrics,
            "final_fit_seconds": round(time.perf_counter() - final_started, 6),
            "model_path": str(model_path),
            "validation_prediction_path": str(prediction_path),
            "test_data_read": False,
            "test_used_for_tuning_or_selection": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries[candidate] = summary

    search_path = report_root / "optuna_search_results.csv"
    pd.DataFrame(trial_rows).to_csv(search_path, index=False)
    parameters_path = report_root / "best_tuned_model_parameters.json"
    parameters_path.write_text(json.dumps(_json_safe(summaries), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison_rows = _baseline_validation_rows(report_root, selected)
    for candidate, summary in summaries.items():
        comparison_rows.append({
            "candidate": candidate,
            "training_strategy": summary["training_strategy"],
            "model_name": summary["model_name"],
            "variant": "optuna_tuned",
            "dataset_split": "validation",
            **summary["validation_metrics"],
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["variant", "auc", "average_precision", "model_name"], ascending=[True, False, False, True]
    )
    comparison_path = report_root / "tuned_validation_performance_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    candidate_pool = comparison.loc[comparison["variant"].eq("optuna_tuned")].copy()
    candidate_pool = candidate_pool.sort_values(["auc", "average_precision", "logloss"], ascending=[False, False, True]).reset_index(drop=True)
    candidate_pool.insert(0, "validation_rank", np.arange(1, len(candidate_pool) + 1, dtype="int64"))
    candidate_pool["issue2_5_eligible"] = True
    candidate_pool_path = report_root / "model_fusion_candidate_pool.csv"
    candidate_pool.to_csv(candidate_pool_path, index=False)

    run_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "candidate_count": len(selected),
        "trial_budget_per_candidate": n_trials,
        "device": device,
        "trial_count": len(trial_rows),
        "selection_metric": SELECTION_METRIC,
        "selection_split": "validation",
        "test_data_read": False,
        "test_used_for_tuning_or_selection": False,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_process_rss_bytes": process.memory_info().rss,
        "search_path": str(search_path),
        "parameters_path": str(parameters_path),
        "comparison_path": str(comparison_path),
        "candidate_pool_path": str(candidate_pool_path),
    }
    summary_path = report_root / "model_tuning_summary.json"
    summary_path.write_text(json.dumps(_json_safe(run_summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "search_path": search_path,
        "parameters_path": parameters_path,
        "comparison_path": comparison_path,
        "candidate_pool_path": candidate_pool_path,
        "summary_path": summary_path,
        "summaries": summaries,
    }


# Compatibility alias for callers that used the initial Issue2 prototype.
tune_primary_models = tune_all_models
