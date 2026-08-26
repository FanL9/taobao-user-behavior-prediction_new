# 阶段一 Member1：Python 内部接口

## `src.data.convert_csv_to_parquet`

```python
convert_csv_to_parquet(
    input_path,
    output_path,
    *,
    chunksize=250_000,
    compression="snappy",
    encoding="utf-8-sig",
    overwrite=False,
) -> ConversionResult
```

输入是符合统一五列口径的 CSV 路径和输出 Parquet 路径。成功时返回 `ConversionResult`：`input_path`、`output_path`、`row_count`、`file_size_bytes`、`elapsed_seconds`。

异常约定：输入不存在抛出 `FileNotFoundError`；目标存在且未允许覆盖抛出 `FileExistsError`；参数、表头或字段值不合法抛出 `ValueError`（底层解析异常可能是其子类）。写入失败时不会留下不完整的正式 Parquet。

## `src.database.initialize_database`

```python
initialize_database(database_path, schema_path=DEFAULT_SCHEMA_PATH) -> DatabaseSummary
```

输入是 SQLite 文件路径和可选 DDL 路径。成功时返回 `DatabaseSummary`：`database_path`、`tables`。DDL 必须使用幂等语句；函数不会删除数据库，也不会导入 CSV。

异常约定：DDL 不存在抛出 `FileNotFoundError`；数据库路径是目录抛出 `ValueError`；SQLite 执行错误原样抛出，便于调用方处理。

## 兼容性规则

上述公开导入路径、参数名、返回字段和异常语义均视为内部稳定接口。变更前需要同步更新测试、外部 CLI 文档和 `docs/project_definition.md`。
