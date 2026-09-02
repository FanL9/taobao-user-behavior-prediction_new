# 阶段三：特征预处理

## 输入与输出

输入为按时间固定划分的带标签宽表：

- `data/splits/user_item_feature_wide_labeled_train.parquet`
- `data/splits/user_item_feature_wide_labeled_validation.parquet`
- `data/splits/user_item_feature_wide_labeled_test.parquet`

默认输出为：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed.parquet`
- `reports/stage3/preprocessing_rules.json`
- `reports/stage3/preprocessing_statistics.json`

## 处理规则

- `user_id`、`item_id`、`category_id` 仅保留用于追踪，不进入模型特征。
- `label` 保留为监督目标，不进入模型特征。
- 删除 `dataset_split`、`history_start`、`history_end`、`label_date` 与直接时间戳字段，避免将分割和窗口信息作为模型输入。
- 数值特征的缺失值使用训练集均值填充；均值和标准差均只在训练集拟合。随后按 `(value - 训练集均值) / 训练集总体标准差` 标准化；常量字段使用标准差 `1`。
- 字符串类别特征使用训练集拟合的整数编码；验证集和测试集中训练集未出现的类别编码为 `-1`。
- 验证集和测试集不重新拟合填充、标准化或编码规则。

本步骤不做特征筛选、采样、模型训练或模型评估。

## 命令行接口

```powershell
python scripts/preprocess_features.py
```

可选参数为 `--train-input`、`--validation-input`、`--test-input`、`--output-dir`、`--rules`、`--report` 和 `--batch-size`。

## Python 接口

```python
preprocess_feature_datasets(
    dataset_paths,
    output_directory,
    rules_path,
    report_path,
    batch_size=50_000,
) -> dict
```

`dataset_paths` 必须包含 `train`、`validation` 和 `test` 三个路径。返回值包含三个预处理文件路径、训练集拟合规则和处理统计。
