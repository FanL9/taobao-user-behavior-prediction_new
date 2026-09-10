"""Stage4 Issue4 model interpretability and error-sample analysis.

The module only loads the validation-selected final candidate and the test-set
predictions created by Issue3. It never fits a model or changes model parameters.
"""

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
from sklearn.metrics import f1_score, precision_score, recall_score

from .training_matrix import _load_features, _load_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TEST_FILENAME = "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet"
FINAL_CANDIDATE = "baseline__xgboost"
DEFAULT_SHAP_SAMPLE_SIZE = 10_000


def _load_json(path: Path) -> dict[str, Any]:
    """Read a required JSON artifact and reject invalid contents."""

    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _analysis_threshold(threshold_path: Path, candidate: str) -> float:
    """Use the Issue3 test-grid threshold with the highest F1 for error slices."""

    if not threshold_path.is_file():
        raise FileNotFoundError(f"Issue3 threshold output does not exist: {threshold_path}")
    table = pd.read_csv(threshold_path)
    rows = table.loc[table["candidate"].eq(candidate)].copy()
    if rows.empty:
        raise ValueError(f"Issue3 threshold output has no row for {candidate}.")
    return float(rows.loc[rows["f1"].idxmax(), "threshold"])


def _restore_numeric_feature(values: pd.Series, feature: str, preprocessing_rules: dict[str, Any]) -> np.ndarray:
    """Invert the training-fitted standardization for one historical count feature."""

    try:
        rule = preprocessing_rules["numeric_rules"][feature]
        return values.to_numpy(dtype="float64") * float(rule["scale"]) + float(rule["mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Preprocessing rules do not define mean/scale for {feature}.") from error


def _sample_indices(labels: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    """Make a reproducible SHAP sample retaining every available positive row."""

    if sample_size <= 0:
        return np.empty(0, dtype="int64")
    all_indices = np.arange(len(labels), dtype="int64")
    if len(all_indices) <= sample_size:
        return all_indices
    positives = all_indices[labels == 1]
    negatives = all_indices[labels == 0]
    rng = np.random.default_rng(random_state)
    retained_positive = positives if len(positives) <= sample_size else rng.choice(positives, sample_size, replace=False)
    retained_negative = rng.choice(negatives, max(0, sample_size - len(retained_positive)), replace=False)
    return np.sort(np.concatenate([retained_positive, retained_negative]))


def _shap_values(model: Any, feature_frame: pd.DataFrame) -> np.ndarray:
    """Return binary-class SHAP values in a stable two-dimensional shape."""

    import shap

    values = shap.TreeExplainer(model).shap_values(feature_frame)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise ValueError("Unexpected SHAP value shape for the final model.")
    return values


def _shap_tables(values: np.ndarray, labels: np.ndarray, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize global and signed SHAP effects without causal interpretation."""

    global_table = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.mean(np.abs(values), axis=0),
        "mean_shap": np.mean(values, axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    positive_rows = labels == 1
    negative_rows = labels == 0
    direction = pd.DataFrame({
        "feature": features,
        "mean_shap_on_actual_positive": np.mean(values[positive_rows], axis=0) if positive_rows.any() else np.nan,
        "mean_shap_on_actual_negative": np.mean(values[negative_rows], axis=0) if negative_rows.any() else np.nan,
    })
    direction["positive_prediction_direction"] = np.where(direction["mean_shap_on_actual_positive"] >= 0, "positive", "negative")
    direction = direction.loc[
        direction["mean_shap_on_actual_positive"].abs().sort_values(ascending=False).index
    ].reset_index(drop=True)
    return global_table, direction


def _plot_shap_importance(table: pd.DataFrame, path: Path) -> None:
    """Write a compact global SHAP Top10 bar chart."""

    top = table.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    axis.set(xlabel="Mean absolute SHAP value", title="Final model SHAP feature importance (Top 10)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _error_summary(labels: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Return a one-row confusion-matrix summary for the selected threshold."""

    return pd.DataFrame([{
        "tn": int(((labels == 0) & (predicted == 0)).sum()),
        "fp": int(((labels == 0) & (predicted == 1)).sum()),
        "fn": int(((labels == 1) & (predicted == 0)).sum()),
        "tp": int(((labels == 1) & (predicted == 1)).sum()),
    }])


def _segment_row(name: str, labels: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Calculate error and fixed-threshold metrics for one model-boundary segment."""

    y_true = labels[mask]
    y_pred = predicted[mask]
    if not len(y_true):
        return {"segment": name, "sample_count": 0, "positive_count": 0, "positive_rate": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "fp": 0, "fn": 0}
    return {
        "segment": name,
        "sample_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small DataFrame as dependency-free Markdown."""

    rows = frame.loc[:, columns].copy()
    for column in rows.select_dtypes(include="number"):
        rows[column] = rows[column].map(lambda value: f"{value:.6f}" if isinstance(value, float) else str(value))
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows.itertuples(index=False, name=None)],
    ])


def _write_report(path: Path, candidate: str, threshold: float, importance: pd.DataFrame, direction: pd.DataFrame, errors: pd.DataFrame, segments: pd.DataFrame) -> None:
    """Write the Issue4 evidence report from existing model outputs only."""

    positive = direction.loc[direction["mean_shap_on_actual_positive"] > 0].head(5)
    negative = direction.loc[direction["mean_shap_on_actual_positive"] < 0].head(5)
    report = "\n".join([
        "# 阶段四 Issue4：模型可解释性与错误样本分析", "", "## 分析对象与口径", "",
        f"- 分析模型：`{candidate}`。该模型由 Issue2.5 的验证集 AUC 领先记录确定，并在 Issue3 中保留测试集结果。",
        f"- 错误样本阈值：`{threshold:.2f}`，取自 Issue3 中该模型 F1 最高的既有阈值网格结果。",
        "- 本步骤只加载既有模型、测试集预测和特征数据；不训练新模型、不调参、不新增特征。", "",
        "## 特征重要性 Top10", "", _markdown_table(importance.head(10), ["feature", "mean_abs_shap", "mean_shap"]), "",
        "## 正负向预测依据", "", "方向来自实际购买样本上的平均 SHAP 值：正值表示该特征在该样本群中倾向于抬高购买预测，负值表示倾向于降低预测。它反映模型关联，不代表因果关系。", "",
        "### 正向影响 Top5", "", _markdown_table(positive, ["feature", "mean_shap_on_actual_positive"]), "",
        "### 负向影响 Top5", "", _markdown_table(negative, ["feature", "mean_shap_on_actual_positive"]), "",
        "## 错误样本统计", "", _markdown_table(errors, ["tn", "fp", "fn", "tp"]), "",
        "## 冷启动与长尾商品场景", "", "冷启动用户定义为原始历史 `user_event_count <= 1`；长尾商品定义为原始历史 `item_total_count <= 1`。计数先按阶段三训练集拟合的标准化规则还原，两个定义只使用标签日前可得的历史统计特征。", "",
        _markdown_table(segments, ["segment", "sample_count", "positive_count", "positive_rate", "precision", "recall", "f1", "fp", "fn"]), "",
        "详细 Top10、SHAP 全量重要性、假阳性/假阴性样例和分群统计见同目录 CSV 文件。", "",
    ])
    path.write_text(report, encoding="utf-8")


def analyze_model_interpretability(split_directory: str | Path, feature_list_path: str | Path, issue2_5_manifest_path: str | Path, test_prediction_path: str | Path, models_directory: str | Path, issue3_reports_directory: str | Path, reports_directory: str | Path, shap_sample_size: int = DEFAULT_SHAP_SAMPLE_SIZE, error_sample_size: int = 100, random_state: int = 42, preprocessing_rules_path: str | Path | None = None) -> dict[str, Path]:
    """Create Issue4 interpretation and error outputs from fixed Issue3 artifacts."""

    split_root = Path(split_directory).expanduser().resolve()
    report_root = Path(reports_directory).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(Path(issue2_5_manifest_path).expanduser().resolve())
    candidate = str(manifest.get("validation_leader", ""))
    if candidate != FINAL_CANDIDATE:
        raise ValueError(f"Issue4 expects fixed final candidate {FINAL_CANDIDATE}, got {candidate!r}.")
    rules_path = Path(preprocessing_rules_path).expanduser().resolve() if preprocessing_rules_path else Path(feature_list_path).expanduser().resolve().parent / "preprocessing_rules.json"
    preprocessing_rules = _load_json(rules_path)
    feature_names = _load_features(Path(feature_list_path).expanduser().resolve())
    test_x, test_y, test_tracking = _load_split(split_root / TEST_FILENAME, feature_names)
    predictions = pd.read_parquet(Path(test_prediction_path).expanduser().resolve())
    score_column = f"score_{candidate}"
    if score_column not in predictions:
        raise ValueError(f"Issue3 prediction file is missing {score_column}.")
    if len(predictions) != len(test_x) or not predictions[["user_id", "item_id", "category_id"]].equals(test_tracking.reset_index(drop=True)):
        raise ValueError("Issue3 predictions do not align with the selected test split.")
    model_path = Path(models_directory).expanduser().resolve() / "stacking" / "models" / "base_models.joblib"
    base_models: dict[str, Any] = joblib.load(model_path)
    if candidate not in base_models:
        raise ValueError(f"Issue2.5 base-model artifact is missing {candidate}.")
    model = base_models[candidate]
    issue3_root = Path(issue3_reports_directory).expanduser().resolve()
    threshold = _analysis_threshold(issue3_root / "test_threshold_sensitivity.csv", candidate)
    scores = predictions[score_column].to_numpy(dtype="float64")
    predicted = (scores >= threshold).astype("int8")
    started = time.perf_counter()
    process = psutil.Process()

    model_importance = pd.DataFrame({"feature": feature_names, "model_importance": np.asarray(model.feature_importances_, dtype="float64")}).sort_values("model_importance", ascending=False, ignore_index=True)
    model_importance_path = report_root / "feature_importance_top10.csv"
    model_importance.head(10).to_csv(model_importance_path, index=False)
    indices = _sample_indices(test_y, shap_sample_size, random_state)
    if not len(indices):
        raise ValueError("shap_sample_size must be positive.")
    shap_value_array = _shap_values(model, test_x.iloc[indices])
    shap_importance, direction = _shap_tables(shap_value_array, test_y[indices], feature_names)
    shap_importance_path = report_root / "shap_feature_importance.csv"
    direction_path = report_root / "feature_direction_analysis.csv"
    shap_importance.to_csv(shap_importance_path, index=False)
    direction.to_csv(direction_path, index=False)
    shap_plot_path = report_root / "shap_feature_importance_top10.png"
    _plot_shap_importance(shap_importance, shap_plot_path)

    frame = test_tracking.copy()
    frame["label"] = test_y
    frame["prediction_score"] = scores
    frame["prediction_label"] = predicted
    frame["error_type"] = np.select([(frame["label"].eq(0) & frame["prediction_label"].eq(1)), (frame["label"].eq(1) & frame["prediction_label"].eq(0))], ["false_positive", "false_negative"], default="correct")
    frame = pd.concat([frame, test_x.reset_index(drop=True)], axis=1)
    errors = _error_summary(test_y, predicted)
    error_summary_path = report_root / "error_sample_summary.csv"
    errors.to_csv(error_summary_path, index=False)
    false_positive_path = report_root / "false_positive_samples.csv"
    false_negative_path = report_root / "false_negative_samples.csv"
    frame.loc[frame["error_type"].eq("false_positive")].sort_values("prediction_score", ascending=False).head(error_sample_size).to_csv(false_positive_path, index=False)
    frame.loc[frame["error_type"].eq("false_negative")].sort_values("prediction_score", ascending=False).head(error_sample_size).to_csv(false_negative_path, index=False)

    raw_user_event_count = _restore_numeric_feature(test_x["user_event_count"], "user_event_count", preprocessing_rules)
    raw_item_total_count = _restore_numeric_feature(test_x["item_total_count"], "item_total_count", preprocessing_rules)
    cold_start = raw_user_event_count <= 1
    long_tail = raw_item_total_count <= 1
    segments = pd.DataFrame([
        _segment_row("all_test_samples", test_y, predicted, np.ones(len(test_y), dtype=bool)),
        _segment_row("cold_start_user", test_y, predicted, cold_start),
        _segment_row("non_cold_start_user", test_y, predicted, ~cold_start),
        _segment_row("long_tail_item", test_y, predicted, long_tail),
        _segment_row("non_long_tail_item", test_y, predicted, ~long_tail),
        _segment_row("cold_start_and_long_tail", test_y, predicted, cold_start & long_tail),
    ])
    segments_path = report_root / "cold_start_long_tail_analysis.csv"
    segments.to_csv(segments_path, index=False)
    report_path = report_root / "model_interpretability_error_analysis_report.md"
    _write_report(report_path, candidate, threshold, shap_importance, direction, errors, segments)
    summary_path = report_root / "model_interpretability_summary.json"
    summary_path.write_text(json.dumps({"status": "passed", "model_candidate": candidate, "model_training_or_tuning_performed": False, "error_analysis_threshold": threshold, "shap_sample_size": int(len(indices)), "shap_positive_rows": int(test_y[indices].sum()), "cold_start_definition": "raw user_event_count <= 1", "long_tail_definition": "raw item_total_count <= 1", "preprocessing_rules_path": str(rules_path), "generated_at_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": round(time.perf_counter() - started, 6), "peak_process_rss_bytes": process.memory_info().rss}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model_importance_path": model_importance_path, "shap_importance_path": shap_importance_path, "direction_path": direction_path, "shap_plot_path": shap_plot_path, "error_summary_path": error_summary_path, "false_positive_path": false_positive_path, "false_negative_path": false_negative_path, "segments_path": segments_path, "report_path": report_path, "summary_path": summary_path}
