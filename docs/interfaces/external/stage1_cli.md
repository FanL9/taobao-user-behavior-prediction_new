# 阶段一：命令行外部接口

所有命令都从仓库根目录执行；成功退出码为 `0`，失败退出码为 `1`，错误写入标准错误。

## 初始化 SQLite

```bash
python scripts/setup_local_database.py \
  --database database/taobao_user_behavior.db \
  --schema sql/ddl/001_create_schema.sql
```

参数均可省略。输出数据库路径及创建后的空表列表。该命令可重复执行，不提供删除或覆盖选项。

## CSV 转 Parquet

```bash
python scripts/convert_csv_to_parquet.py \
  --input data/raw/user_behavior_processed.csv \
  --output data/raw/user_behavior_processed.parquet \
  --chunksize 250000 \
  --compression snappy \
  --encoding utf-8-sig
```

可选 `--overwrite` 允许替换已有输出。成功输出输入/输出绝对路径、行数、文件字节数和耗时。默认拒绝覆盖，避免误删本地大文件。

## 数据质量检查

```bash
python scripts/check_data_quality.py \
  --input data/raw/user_behavior_processed.csv \
  --output reports/stage1/data_quality_report.json \
  --chunksize 100000 \
  --duplicate-partitions 32
```

该命令只读取原始 CSV 并输出 JSON 检查报告，不修改输入，不删除、标记或去重，也不生成清洗数据。
