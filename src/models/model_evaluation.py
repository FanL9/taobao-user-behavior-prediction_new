"""Final Stage4 Issue3 evaluation for validation-selected model candidates."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .training_matrix import PREDICTION_THRESHOLD, _load_features, _load_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TEST_FILENAME = "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet"
VALIDATION_FILENAME = "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet"
STACKING_CANDIDATE = "stacking_logistic_regression"


def _read_manifest(path: Path) -> dict[str, Any]:
    """Load the validation-only Issue2.5 candidate manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"Issue2.5 candidate manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("base_candidates"), list):
        raise ValueError("Issue2.5 manifest must contain a base_candidates list.")
    if payload.get("test_data_read") is not False:
        raise ValueError("Issue2.5 manifest does not prove that test data was isolated.")
    return payload


def _metric_row(labels: np.ndarray, scores: np.ndarray, threshold: float = PREDICTION_THRESHOLD) -> dict[str, float]:
    """Calculate Issue3 ranking, probability, and fixed-threshold metrics."""

    clipped = np.clip(np.asarray(scores, dtype="float64"), 1e-15, 1 - 1e-15)
    predicted = (clipped >= threshold).astype("int8")
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "average_precision": float(average_precision_score(labels, clipped)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "logloss": float(log_loss(labels, clipped, labels=[0, 1])),
        "positive_prediction_ratio": float(predicted.mean()),
    }


def _plot_roc(labels: np.ndarray, scores: dict[str, np.ndarray], target: Path) -> None:
    """Write one ROC plot covering all selected single and stacking models."""

    figure, axis = plt.subplots(figsize=(8, 6))
    for candidate, values in scores.items():
        fpr, tpr, _ = roc_curve(labels, values)
        axis.plot(fpr, tpr, label=f"{candidate} (AUC={roc_auc_score(labels, values):.4f})")
    axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def _plot_pr(labels: np.ndarray, scores: dict[str, np.ndarray], target: Path) -> None:
    """Write one precision-recall plot covering all evaluated candidates."""

    figure, axis = plt.subplots(figsize=(8, 6))
    for candidate, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        axis.plot(recall, precision, label=f"{candidate} (AP={average_precision_score(labels, values):.4f})")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def _plot_confusion_matrices(labels: np.ndarray, scores: dict[str, np.ndarray], target: Path) -> list[dict[str, int | str]]:
    """Write threshold-0.5 confusion matrices and return their numeric contents."""

    candidates = list(scores)
    figure, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 3.6), squeeze=False)
    rows: list[dict[str, int | str]] = []
    for axis, candidate in zip(axes[0], candidates):
        matrix = confusion_matrix(labels, (scores[candidate] >= PREDICTION_THRESHOLD).astype("int8"), labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(candidate, fontsize=8)
        axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, str(int(matrix[row_index, column_index])), ha="center", va="center")
        rows.append({"candidate": candidate, "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])})
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return rows


def _threshold_rows(labels: np.ndarray, scores: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    """Calculate fixed-grid threshold sensitivity for every Issue3 candidate."""

    rows: list[dict[str, float | str]] = []
    for candidate, values in scores.items():
        for threshold in np.arange(0.05, 1.0, 0.05):
            rows.append({"candidate": candidate, "threshold": round(float(threshold), 2), **_metric_row(labels, values, float(threshold))})
    return rows


def _write_report(path: Path, metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Write a compact technical evaluation report without business interpretation."""

    display_columns = ["candidate", "dataset_split", "auc", "accuracy", "average_precision", "precision", "recall", "f1", "logloss"]
    table = metrics.loc[:, display_columns].copy()
    for column in display_columns[2:]:
        table[column] = table[column].map(lambda value: f"{value:.6f}")
    leader = manifest.get("validation_leader", "not_recorded")
    headers = list(table.columns)
    markdown_table = "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)],
    ])
    report = "\n".join([
        "# 阶段四 Issue3：模型效果综合评估",
        "",
        "## 评估口径",
        "",
        "- 候选方案由 Issue2.5 的验证集结果确定。",
        "- 本步骤首次读取测试集，仅用于最终指标、曲线、混淆矩阵和阈值分析。",
        "- 未使用测试集调参、选择模型或改变融合结构。",
        f"- Issue2.5 验证集领先方案记录为：`{leader}`。",
        "",
        "## 验证集与测试集指标",
        "",
        markdown_table,
        "",
        "详细曲线、混淆矩阵和阈值敏感性结果见同目录输出文件。",
        "",
    ])
    path.write_text(report, encoding="utf-8")


def evaluate_final_candidates(
    split_directory: str | Path,
    feature_list_path: str | Path,
    issue2_5_manifest_path: str | Path,
    models_directory: str | Path,
    reports_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate Issue2.5 candidates once on the untouched test set.

    Models are loaded from Issue2.5 artifacts; this function never tunes, fits,
    or changes any model parameters.
    """

    split_root = Path(split_directory).expanduser().resolve()
    manifest = _read_manifest(Path(issue2_5_manifest_path).expanduser().resolve())
    model_root = Path(models_directory).expanduser().resolve() / "stacking"
    report_root = Path(reports_directory).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    features = _load_features(Path(feature_list_path).expanduser().resolve())
    base_models_path = model_root / "models" / "base_models.joblib"
    meta_model_path = model_root / "models" / "stacking_meta_model.joblib"
    validation_path = model_root / "validation_predictions" / "stacking_validation_predictions.parquet"
    for path in (base_models_path, meta_model_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Issue2.5 artifact does not exist: {path}")
    base_models: dict[str, Any] = joblib.load(base_models_path)
    meta_model = joblib.load(meta_model_path)
    selected = [str(value) for value in manifest["base_candidates"]]
    if set(selected) - set(base_models):
        raise ValueError("Issue2.5 base-model artifact does not match its candidate manifest.")
    started = time.perf_counter()
    process = psutil.Process()

    validation_frame = pd.read_parquet(validation_path)
    validation_labels = validation_frame["label"].to_numpy(dtype="int8")
    validation_scores: dict[str, np.ndarray] = {}
    for candidate in selected:
        column = f"score_{candidate}"
        if column not in validation_frame:
            raise ValueError(f"Issue2.5 validation prediction is missing {column}.")
        validation_scores[candidate] = validation_frame[column].to_numpy(dtype="float64")
    validation_scores[STACKING_CANDIDATE] = validation_frame["stacking_prediction_score"].to_numpy(dtype="float64")

    test_x, test_y, test_tracking = _load_split(split_root / TEST_FILENAME, features)
    test_scores: dict[str, np.ndarray] = {}
    for candidate in selected:
        test_scores[candidate] = base_models[candidate].predict_proba(test_x)[:, 1]
    test_matrix = np.column_stack([test_scores[candidate] for candidate in selected])
    test_scores[STACKING_CANDIDATE] = meta_model.predict_proba(test_matrix)[:, 1]

    prediction_root = model_root / "test_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    test_prediction_path = prediction_root / "issue3_final_candidate_predictions.parquet"
    prediction_frame = test_tracking.copy()
    prediction_frame["label"] = test_y
    for candidate, values in test_scores.items():
        prediction_frame[f"score_{candidate}"] = values
        prediction_frame[f"label_{candidate}"] = (values >= PREDICTION_THRESHOLD).astype("int8")
    prediction_frame.to_parquet(test_prediction_path, index=False)

    metric_rows = []
    for split, labels, scores in (("validation", validation_labels, validation_scores), ("test", test_y, test_scores)):
        for candidate, values in scores.items():
            metric_rows.append({"candidate": candidate, "dataset_split": split, "threshold": PREDICTION_THRESHOLD, **_metric_row(labels, values)})
    metrics = pd.DataFrame(metric_rows).sort_values(["dataset_split", "auc", "average_precision"], ascending=[True, False, False])
    metrics_path = report_root / "final_model_performance_comparison.csv"
    metrics.to_csv(metrics_path, index=False)

    roc_path = report_root / "test_roc_curves.png"
    pr_path = report_root / "test_pr_curves.png"
    confusion_plot_path = report_root / "test_confusion_matrices.png"
    _plot_roc(test_y, test_scores, roc_path)
    _plot_pr(test_y, test_scores, pr_path)
    confusion = pd.DataFrame(_plot_confusion_matrices(test_y, test_scores, confusion_plot_path))
    confusion_path = report_root / "test_confusion_matrices.csv"
    confusion.to_csv(confusion_path, index=False)
    threshold = pd.DataFrame(_threshold_rows(test_y, test_scores))
    threshold_path = report_root / "test_threshold_sensitivity.csv"
    threshold.to_csv(threshold_path, index=False)
    report_path = report_root / "model_evaluation_report.md"
    _write_report(report_path, metrics, manifest)

    summary = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": [*selected, STACKING_CANDIDATE],
        "selection_split": "validation",
        "test_evaluation_count": 1,
        "test_used_for_tuning_or_selection": False,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "test_prediction_path": str(test_prediction_path),
        "metrics_path": str(metrics_path),
        "roc_path": str(roc_path),
        "pr_path": str(pr_path),
        "confusion_path": str(confusion_path),
        "threshold_path": str(threshold_path),
        "report_path": str(report_path),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_process_rss_bytes": process.memory_info().rss,
    }
    summary_path = report_root / "model_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metrics_path": metrics_path, "summary_path": summary_path, "report_path": report_path, "test_prediction_path": test_prediction_path}
