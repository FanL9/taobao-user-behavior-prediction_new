# 阶段三：未来 1 天购买标签生成

## 口径与边界

- 输入样本：`data/features/user_item_feature_wide.parquet`
- 标签事件来源：`data/processed/user_behavior_clean.parquet`
- 样本粒度：`dataset_split + user_id + item_id`
- 标签字段：`label`（`int8`）
- 用户—商品对在对应 `label_date` 发生 `behavior_type=4` 的购买行为时，`label=1`；否则 `label=0`
- 固定标签日：训练集 `2025-12-08`、验证集 `2025-12-15`、测试集 `2025-12-18`

本步骤仅生成标签，不做特征预处理、特征筛选、采样、模型训练或模型评估。

## 命令行接口

```powershell
python scripts/generate_purchase_labels.py
```

默认输出：

- `data/splits/user_item_feature_wide_labeled.parquet`
- `reports/stage3/label_statistics.json`

可选参数为 `--wide-table`、`--clean-data`、`--output`、`--report` 和 `--batch-size`。

## Python 接口

```python
generate_purchase_labels(
    wide_table,
    clean_data,
    output_parquet,
    report_path,
    batch_size=50_000,
) -> dict
```

函数分批读取阶段二宽表，保留其全部字段和值，仅在末尾追加 `label`，并原子写入样本文件和 JSON 报告。

## 数据泄露检查

标签事件只读取 `user_id`、`item_id`、`behavior_type`、`behavior_date`，仅用于目标匹配，不计算或修改任何特征。生成时逐批验证每个数据集的 `history_start`、`history_end`、`label_date` 与固定窗口一致，并要求 `history_end < label_date`。报告中的 `feature_columns_modified` 必须为空，`label_window_used_for_feature_calculation` 必须为 `false`。
