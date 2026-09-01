# 阶段三：训练、验证、测试集生成

## 口径与边界

- 输入：`data/splits/user_item_feature_wide_labeled.parquet`
- 样本粒度：`dataset_split + user_id + item_id`
- 输出：三个保留全部特征和 `label` 的用户—商品特征宽表
- 划分方式：仅按现有 `dataset_split` 分区，不采用随机划分

固定时间窗口如下：

| 数据集 | history_start | history_end | label_date |
| --- | --- | --- | --- |
| train | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| validation | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| test | 2025-12-16 | 2025-12-17 | 2025-12-18 |

生成时逐批验证窗口元数据与上述口径一致，并要求 `history_end < label_date`。本步骤不做特征预处理、特征筛选、采样、模型训练或模型评估。

## 命令行接口

```powershell
python scripts/split_labeled_datasets.py
```

默认输出：

- `data/splits/user_item_feature_wide_labeled_train.parquet`
- `data/splits/user_item_feature_wide_labeled_validation.parquet`
- `data/splits/user_item_feature_wide_labeled_test.parquet`
- `reports/stage3/dataset_split_statistics.json`

可选参数为 `--input`、`--output-dir`、`--report` 和 `--batch-size`。

## Python 接口

```python
generate_time_ordered_datasets(
    labeled_table,
    output_directory,
    report_path,
    batch_size=50_000,
) -> dict
```

函数按 `dataset_split` 过滤原始 Arrow 批次后直接写入对应数据集，因此不会修改特征字段、标签或样本顺序。返回值包含三份输出路径和样本量、正负样本量、正样本比例、窗口检查及性能统计。
