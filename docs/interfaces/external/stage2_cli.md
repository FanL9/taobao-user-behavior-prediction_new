# 阶段二：命令行接口

## 四张中间表

```bash
python scripts/build_stage2_intermediate_tables.py
```

默认读取 `data/processed/user_behavior_clean.parquet`，输出到 `data/interim/`：

```text
user_intermediate.parquet
item_intermediate.parquet
category_intermediate.parquet
time_intermediate.parquet
```

## 八张特征表

```bash
python scripts/build_stage2_feature_tables.py
```

默认读取 `data/processed/user_behavior_clean.parquet`，输出到 `data/features/`：

```text
user_features.parquet
user_activity_features.parquet
user_sequence_features.parquet
item_behavior_features.parquet
item_popularity_features.parquet
category_behavior_features.parquet
time_behavior_features.parquet
conversion_chain_features.parquet
```

上述两个独立表生成命令均支持 `--input <clean-parquet>` 和 `--output-dir <output-directory>`，且不生成标签、模型或采样结果。

## 用户—商品特征宽表

```bash
python scripts/build_user_item_feature_wide.py
```

默认读取 `data/features/` 下的八张特征表，输出：

```text
data/features/user_item_feature_wide.parquet
reports/stage2/user_item_feature_wide_quality_report.json
```

可使用 `--features-dir`、`--output`、`--quality-report` 和 `--batch-size` 覆盖默认参数。该命令只合并初版宽表，不生成标签，不训练或评估模型，不采样，也不筛选最终入模字段。
