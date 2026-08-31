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

两个命令均支持 `--input <clean-parquet>` 和 `--output-dir <output-directory>`。阶段二命令不生成标签、模型、采样结果或最终特征宽表。
