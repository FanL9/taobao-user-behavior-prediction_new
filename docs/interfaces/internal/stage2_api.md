# 阶段二：Python 内部接口

## 固定时间窗口

`HISTORY_WINDOWS` 包含训练、验证和测试三个历史窗口，以及各自排除的 `label_date`。所有中间表和特征表接口默认使用该配置。

## 中间表接口

```python
build_intermediate_tables(clean_data, windows=HISTORY_WINDOWS) -> dict[str, DataFrame]
generate_intermediate_tables(input_parquet, output_directory, windows=HISTORY_WINDOWS) -> dict[str, Path]
```

返回或写出用户、商品、类目和时间四张中间表，不生成标签、特征宽表或模型产物。

## 八张特征表统一接口

```python
build_all_feature_tables(clean_data, windows=HISTORY_WINDOWS) -> dict[str, DataFrame]
generate_all_feature_tables(input_parquet, output_directory, windows=HISTORY_WINDOWS) -> dict[str, Path]
```

返回或原子写出 `user_basic`、`user_activity`、`user_sequence`、`item_behavior`、`item_popularity`、`category_behavior`、`time_behavior`、`conversion_chain` 八张表。完整阶段二特征生成应使用上述统一接口。

## 单表构建接口

```python
build_user_basic_features(history, window) -> DataFrame
build_user_activity_features(history, window) -> DataFrame
build_user_sequence_features(history, window) -> DataFrame
build_item_behavior_features(history, window) -> DataFrame
build_item_popularity_features(item_behavior) -> DataFrame
build_category_behavior_features(clean_data, item_behavior) -> DataFrame
build_time_behavior_features(clean_data, item_behavior) -> DataFrame
build_conversion_chain_features(item_behavior) -> DataFrame
```

- 前四个函数输入已经限定到单个历史窗口的事件。
- 商品热度和转化链路复用商品行为特征表。
- 类目和时间接口使用商品行为表中的窗口元数据，确保八张表窗口一致。
- `build_feature_tables` 和 `generate_feature_tables` 作为只生成前四张的兼容接口保留。

所有接口均不生成 `label`、不采样、不训练模型，也不合并最终宽表。
