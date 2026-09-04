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
python scripts/train_model_matrix.py
```

默认统一运行 4 个训练方案（`baseline`、`smote`、`undersampled`、`class_weight`）与 4 个模型，共 16 个训练结果。参数为预先固定的基线参数，不执行搜索或测试集调参。验证集用于记录横向结果，测试集仅在训练后输出泛化预测与指标，不参与模型选择。

可选参数：`--validation-input`、`--test-input`、`--feature-list`、`--class-weights`、`--models-dir`、`--reports-dir`、`--strategies`、`--models` 与 `--random-state`。默认运行全部四种方案；可以用 `--strategies` 或 `--models` 执行局部验证。

## 输出

本地且被 Git 忽略的模型目录重整为：

- `models/traditional_ml/<training_strategy>/models/`：四个序列化模型；
- `models/traditional_ml/<training_strategy>/validation_predictions/`：四个验证集预测；
- `models/traditional_ml/<training_strategy>/test_predictions/`：四个测试集预测；
- `models/traditional_ml/<training_strategy>/run_logs/`：四个模型的固定参数、运行时间、资源增量及指标。

可追踪报告写入 `reports/stage4/`：

- `baseline_model_comparison_<training_strategy>.csv`：单一训练方案的指标；
- `baseline_model_performance_comparison.csv`：已完成方案的汇总对比；
- `baseline_training_summary_<training_strategy>.json`：输入版本、训练策略、预测阈值、各模型参数与输出索引。

预测文件保留三个追踪主键、真实 `label`、`prediction_score`、阈值为 0.5 的 `prediction_label`、模型名和数据集名。

## Python 接口

```python
train_model_matrix(
    dataset_paths_by_strategy,
    feature_list_path,
    models_directory,
    reports_directory,
    class_weight_path,
    model_names=("logistic_regression", "random_forest", "xgboost", "lightgbm"),
    strategies=("baseline", "smote", "undersampled", "class_weight"),
    random_state=42,
) -> dict
```
