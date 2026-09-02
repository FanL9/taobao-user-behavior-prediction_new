# 阶段三：类别不平衡处理

## 输入与边界

输入为 Issue4 的筛选后训练、验证和测试集。类别不平衡规则仅使用训练集标签分布；验证集和测试集不进行 SMOTE、欠采样或任何重采样。

本步骤不训练模型、不做模型评估、不决定最终采用哪一种训练方案。

## 输出方案

- 原始未采样训练集：用于模型基线对比。
- SMOTE 训练集：将训练集少数类过采样至与多数类 1:1。
- 欠采样训练集：以固定随机种子将训练集多数类减少至与少数类 1:1。
- 类别权重配置：使用 `n_samples / (n_classes * class_count)`，不改变训练样本数。
- 原始验证集和测试集：仅增加 `is_synthetic=False` 元数据列，类别分布和样本值不变。

SMOTE 对连续特征进行线性插值；名称以 `_code` 结尾的类别编码特征从两个少数类父样本之一继承。合成行的 `user_id`、`item_id`、`category_id` 均设为 `-1`，并以 `is_synthetic=True` 标识；这些字段和 `label` 均不作为模型输入。

## 命令行接口

```powershell
python scripts/prepare_class_imbalance.py
```

默认生成：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet`
- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected_smote.parquet`
- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet`
- `reports/stage3/class_weight_config.json`
- `reports/stage3/training_dataset_versions.json`
- `reports/stage3/class_imbalance_report.json`

可选参数为 `--train-input`、`--validation-input`、`--test-input`、`--output-dir`、`--class-weights`、`--versions`、`--report`、`--batch-size`、`--random-state` 与 `--smote-k-neighbors`。

## Python 接口

```python
prepare_class_imbalance_strategies(
    dataset_paths,
    output_directory,
    class_weight_path,
    versions_path,
    report_path,
    batch_size=50_000,
    random_state=42,
    smote_k_neighbors=5,
) -> dict
```

返回值包含各训练方案和原始验证/测试集路径、类别权重配置、数据集版本清单及处理统计。
