# 阶段二：命令行接口

从仓库根目录运行：

```bash
python scripts/build_stage2_intermediate_tables.py
```

默认输入：

```text
data/processed/user_behavior_clean.parquet
```

默认输出：

```text
data/interim/user_intermediate.parquet
data/interim/item_intermediate.parquet
data/interim/category_intermediate.parquet
data/interim/time_intermediate.parquet
```

可用参数：

```bash
python scripts/build_stage2_intermediate_tables.py \
  --input <clean-parquet> \
  --output-dir <output-directory>
```

该命令只生成四张中间表，不生成标签、样本、用户—商品表、最终特征宽表或模型产物。

## 前四张特征表

```bash
python scripts/build_stage2_feature_tables.py
```

默认读取 `data/processed/user_behavior_clean.parquet`，输出：

```text
data/features/user_features.parquet
data/features/user_activity_features.parquet
data/features/user_sequence_features.parquet
data/features/item_behavior_features.parquet
```

可使用 `--input` 和 `--output-dir` 覆盖路径。命令不生成商品热度、类目行为、时间行为、转化链路、标签或最终宽表。
