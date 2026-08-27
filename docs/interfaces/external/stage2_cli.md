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
