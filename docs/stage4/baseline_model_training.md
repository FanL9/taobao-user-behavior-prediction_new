# 阶段四：传统机器学习基线训练

## 范围与数据版本

本步骤只训练并记录固定参数的传统机器学习基线，不做深度调优、最终模型选择或业务解释。默认使用 Issue5 的原始未采样训练集，并应用 `reports/stage3/class_weight_config.json` 中仅由训练集拟合出的类别权重；验证集和测试集保持 Issue5 的原始分布。

输入数据包括：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet`
- `reports/stage3/final_model_features.json`

模型输入严格使用最终 37 个特征；`user_id`、`item_id`、`category_id`、`label` 与 `is_synthetic` 均不进入模型。

## 执行接口

```powershell
python scripts/train_baseline_models.py
```

默认依次训练 Logistic Regression、Random Forest、XGBoost 与 LightGBM。参数为预先固定的基线参数，不执行搜索或测试集调参。验证集用于记录横向结果，测试集仅在训练后输出泛化预测与指标，不参与模型选择。

可选参数：`--train-input`、`--validation-input`、`--test-input`、`--feature-list`、`--class-weights`、`--models-dir`、`--reports-dir`、`--models`、`--training-strategy` 与 `--random-state`。`--training-strategy` 支持 `baseline`、`class_weight`、`smote`、`undersampled`；默认 `class_weight`，其样本行仍是原始未采样训练集。

## 输出

本地且被 Git 忽略的模型目录重整为：

- `models/baselines/artifacts/`：四个序列化模型；
- `models/baselines/predictions/`：每个模型的验证集和测试集预测；
- `models/baselines/logs/`：每个模型的固定参数、运行时间、资源增量及指标。

可追踪报告写入 `reports/stage4/`：

- `baseline_model_comparison.csv`：模型、数据集、AUC、Average Precision、Precision、Recall、F1、LogLoss 和正例预测比例；
- `baseline_training_summary.json`：输入版本、训练策略、预测阈值、各模型参数与输出索引。

预测文件保留三个追踪主键、真实 `label`、`prediction_score`、阈值为 0.5 的 `prediction_label`、模型名和数据集名。

## Python 接口

```python
train_baseline_models(
    dataset_paths,
    feature_list_path,
    models_directory,
    reports_directory,
    class_weight_path=None,
    model_names=("logistic_regression", "random_forest", "xgboost", "lightgbm"),
    training_strategy="class_weight",
    random_state=42,
) -> dict
```
