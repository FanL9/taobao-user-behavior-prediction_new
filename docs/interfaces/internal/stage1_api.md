# 阶段一：Python 内部接口

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

## `src.data.check_csv_quality`

```python
check_csv_quality(
    csv_path,
    *,
    chunksize=100_000,
    encoding="utf-8-sig",
    duplicate_partitions=32,
) -> dict
```

只读检查原始 CSV，返回数据规模、完整性、合法性、完全重复和疑似重复统计。函数使用临时 Parquet 分区完成跨分块重复检查；临时文件自动删除，不修改输入，也不生成清洗数据。

## `src.data.write_quality_report`

```python
write_quality_report(report, output_path) -> Path
```

将 `check_csv_quality` 的结果写为 UTF-8 JSON，返回输出文件的绝对路径。

## 标准数据清洗 API

核心实现位于：

- `src/data/cleaning.py`
- `src/data/cleaning_pipeline.py`

### clean_chunk

`clean_chunk(chunk: pandas.DataFrame) -> ChunkCleaningResult`

职责：

- 校验阶段一原始五字段结构
- 处理缺失值、非法 ID、非法行为类型和非法时间
- 将 `item_category` 标准化为 `category_id`
- 生成 `behavior_name`、`behavior_date`、`behavior_hour`、`weekday`
- 不在单个 Chunk 内直接执行全局重复删除

### clean_user_behavior_file

`clean_user_behavior_file(input_csv, output_csv, output_parquet, report_json, chunksize=250000, partitions=64)`

职责：

- 分块读取全量原始 CSV
- 调用 `clean_chunk` 完成逐块标准化
- 使用 Hash 分区处理跨 Chunk 重复记录
- 重复判断键统一为 `user_id + item_id + behavior_type + time`
- 同一四元组出现 2～59 次时全部保留
- 同一四元组出现 60 次及以上时仅保留原始输入顺序中的首次记录，其余删除，阈值 60 为包含边界
- 当前 `time` 仅精确到小时，因此该规则属于同小时代理规则，不能解释为分钟级或秒级重复识别
- 同时输出 CSV、Parquet 与 JSON 清洗报告

标准输出字段顺序：

`time,user_id,item_id,category_id,behavior_type,behavior_name,behavior_date,behavior_hour,weekday`

其中 `weekday` 定义为 Monday=0 至 Sunday=6。



---

# EDA Analysis API

## `src/data/EDA_analysis.py::main`

```python
main() -> None
```

运行完整 EDA pipeline。

该函数作为 EDA 分析阶段的正式运行入口，负责加载清洗后的用户行为数据，并调用各统计模块生成分析结果。

输入：
- `data/processed/user_behavior_clean.parquet`

数据规模：
- 12,256,906 条行为记录
- 时间范围：2025-11-18 00:00:00 至 2025-12-18 23:00:00

输出目录：
- data/EDA

生成文件：
- `behavior_distribution.csv`
- `user_purchase_summary.csv`
- `item_statistics.csv`
- `top_10_item.csv`
- `category_statistics.csv`
- `top_10_category.csv`
- `daily_behavior.csv`
- `hourly_behavior.csv`
- `descriptive_funnel.csv`

异常约定：
- 输入文件不存在时抛出 `FileNotFoundError`
- 输入数据缺少必要字段时抛出 `ValueError`
- 输出路径不可写时抛出 `OSError`


## `src/data/EDA_analysis.py::load_data`


```python
load_data(
    file_path
) -> pandas.DataFrame
```

读取清洗后的用户行为 Parquet 数据。

输入：
- `file_path`：Parquet 文件路径

返回：
- `pandas.DataFrame`
该函数仅负责数据加载，不执行额外的数据清洗或转换。

## `src/data/EDA_analysis.py::calculate_behavior_distribution`
```python
calculate_behavior_distribution(
    df
) -> pandas.DataFrame
```

统计整体用户行为分布。

返回字段：
- `behavior_type`
- `behavior_name`
- `behavior_count`
- `percentage`

用于生成：

`behavior_distribution.csv`



## `src/data/EDA_analysis.py::calculate_user_purchase_summary`
```python
calculate_user_purchase_summary(
    df
) -> pandas.DataFrame
```
统计用户购买行为相关指标。

返回字段：
- `total_behavior_count`
- `purchase_count`
- `purchase_users`
- `non_purchase_users`
- `repeat_purchase_users`

其中：
- `repeat_purchase_users` 定义为购买次数大于等于 2 次的用户。


## `src/data/EDA_analysis.py::calculate_item_statistics`
```python
calculate_item_statistics(
    df
) -> pandas.DataFrame
```
统计商品维度的用户行为情况。

返回字段：
- `item_id`
- `pv_count`
- `fav_count`
- `cart_count`
- `buy_count`

用于生成：
- `item_statistics.csv`
- `top_10_item.csv`


## `src/data/EDA_analysis.py::get_top_10_items`
```python
get_top_10_items(
    item_statistics
) -> pandas.DataFrame
```
根据商品购买次数筛选热门商品。

输入：
- `item_statistics`：商品行为统计结果

返回：
-按 `buy_count` 降序排列的 Top 10 商品

用于生成：
- `top_10_item.csv`



## `src/data/EDA_analysis.py::calculate_category_statistics`
```python
calculate_category_statistics(
    df
) -> pandas.DataFrame
```
统计商品类目维度的用户行为情况。

返回字段：
- `category_id`
- `behavior_count`
- `buy_count`
- `buy_percentage`

其中：
- `buy_percentage` 表示该类目购买次数占全部购买次数的比例。

用于生成：
- `category_statistics.csv`
- `top_10_category.csv`



## `src/data/EDA_analysis.py::get_top_10_categories`
```python
get_top_10_categories(
    category_statistics
) -> pandas.DataFrame
```
根据类目购买次数筛选热门类目。

输入：
- `category_statistics`：类目行为统计结果

返回：

- 按 `buy_count` 降序排列的 Top 10 类目

用于生成：

- `top_10_category.csv`


## `src/data/EDA_analysis.py::calculate_daily_behavior`
```python
calculate_daily_behavior(
    df
) -> pandas.DataFrame
```
统计日期维度的用户行为分布。

返回字段：
- `behavior_date`
- `pv_count`
- `fav_count`
- `cart_count`
- `buy_count`

用于生成：
- `daily_behavior.csv`



## `src/data/EDA_analysis.py::calculate_hourly_behavior`
```python
calculate_hourly_behavior(
    df
) -> pandas.DataFrame
```
统计小时维度的用户行为分布。

返回字段：
- `behavior_hour`
- `pv_count`
- `fav_count`
- `cart_count`
- `buy_count`

用于生成：
- `hourly_behavior.csv`


## `src/data/EDA_analysis.py::calculate_descriptive_funnel`
```python
calculate_descriptive_funnel(
    df
) -> pandas.DataFrame
```
构建描述性用户行为转化漏斗。

统计阶段：
- `PV`
- `Favorite`
- `Cart`
- `Purchase`

返回字段：
- `stage`
- `behavior_count`
- `relative_to_pv_percentage`

用于生成：
- `descriptive_funnel.csv`


# EDA Testing
## Functional Test

测试文件：
- `tests/functional/test_EDA.py`

功能测试用于验证 EDA pipeline 是否正常运行，并检查：
- 所有输出 CSV 文件是否成功生成
- 输出字段是否符合预期
- 行为类型统计是否完整
- Top 10 商品和类目是否按照购买次数排序
- 日期和小时维度统计结构是否正确
- 转化漏斗阶段是否完整


## Performance Test

测试文件：
- `tests/performance/test_EDA_analysis_performance.py`

性能测试用于记录 EDA pipeline 的运行效率。

测试指标包括：
- `runtime_seconds`
- `CPU usage`
- `GPU usage`

测试结果保存至：
- `data/EDA/performance_test_result.csv`

## Compatibility Rules

上述 EDA 接口的：
- 函数名称
- 输入参数
- 返回格式
- 输出文件结构

均视为稳定接口。

若修改 EDA 计算逻辑或接口定义，需要同步更新：

- `tests/functional/test_EDA.py`
- `tests/performance/test_EDA_analysis_performance.py`
- EDA 输出说明文档













