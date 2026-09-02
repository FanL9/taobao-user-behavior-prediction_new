# 阶段三：特征筛选

## 输入与输出

输入为预处理后的训练、验证和测试集：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed.parquet`

默认输出：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed_selected.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed_selected.parquet`
- `reports/stage3/final_model_features.json`
- `reports/stage3/feature_selection_report.json`

## 筛选规则

筛选规则仅在训练集上拟合，验证集和测试集只应用最终特征清单：

- `user_id`、`item_id`、`category_id` 为追踪字段，`label` 为监督目标，均不作为候选入模特征。
- 训练集缺失率大于 95% 的候选特征删除。
- 训练集方差小于或等于 `1e-12` 的候选特征删除。
- 按输入字段顺序保留第一个特征；与已保留特征的训练集绝对相关系数大于或等于 `0.98` 时，删除后续冗余特征。
- 训练集存在 NaN、正无穷或负无穷的候选特征删除。
- 字段名含 `label`、`target`、`future`、`history_start`、`history_end`、`label_date` 或 `timestamp` 的候选特征视为疑似未来信息泄露并删除、单独记录。
- 绝对值大于 5 的异常值比例会记录在报告中，但不会单独删除数值有效的特征。

本步骤不训练模型、不做模型评估、不做采样处理。

## 命令行接口

```powershell
python scripts/select_features.py
```

可选参数为 `--train-input`、`--validation-input`、`--test-input`、`--output-dir`、`--feature-list`、`--report` 和 `--batch-size`。

## Python 接口

```python
select_model_features(
    dataset_paths,
    output_directory,
    feature_list_path,
    report_path,
    batch_size=50_000,
) -> dict
```

`dataset_paths` 必须包含 `train`、`validation` 和 `test`。返回值包含三份筛选后数据集、最终入模特征清单、删除原因和训练集筛选统计。
