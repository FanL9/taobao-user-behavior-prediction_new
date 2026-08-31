# 阶段二：Python 内部接口

## 固定窗口

`HISTORY_WINDOWS` 包含训练、验证、测试三个历史窗口及其排除的标签日。`HistoryWindow` 字段为 `dataset_split`、`history_start`、`history_end`、`label_date`。

## `build_intermediate_tables`

```python
build_intermediate_tables(clean_data, windows=HISTORY_WINDOWS) -> dict[str, DataFrame]
```

输入阶段一标准 clean DataFrame，返回且只返回 `user`、`item`、`category`、`time` 四张中间表。函数校验 clean 字段、时间派生字段、商品—类目映射和窗口定义；不会构造标签、用户—商品表或最终特征表。

## 四个单表函数

```python
build_user_intermediate(history, window) -> DataFrame
build_item_intermediate(history, window) -> DataFrame
build_category_intermediate(history, window) -> DataFrame
build_time_intermediate(history, window) -> DataFrame
```

输入必须已经由 `select_history` 限定到一个历史窗口；输出分别为用户、商品、类目和日期—小时粒度的中间统计。

## `generate_intermediate_tables`

```python
generate_intermediate_tables(input_parquet, output_directory, windows=HISTORY_WINDOWS) -> dict[str, Path]
```

读取 clean Parquet，生成四个 Parquet 文件，并返回表名到绝对路径的映射。正式默认路径由命令行脚本指定。

## 前四张特征表接口

```python
build_feature_tables(clean_data, windows=HISTORY_WINDOWS) -> dict[str, DataFrame]
```

输入阶段一 clean DataFrame，返回且只返回 `user_basic`、`user_activity`、`user_sequence`、`item_behavior` 四张特征表。用户序列表为 `user_id + item_id` 粒度；活跃等级阈值只由训练窗口计算。

```python
generate_feature_tables(input_parquet, output_directory, windows=HISTORY_WINDOWS) -> dict[str, Path]
```

读取 clean Parquet 并原子写入四张特征表。该接口不生成其余四张特征表、标签或最终宽表。

## 后四张特征表接口

### uild_item_popularity_features

uild_item_popularity_features(item_behavior) 基于商品行为特征表生成商品热度特征，粒度为 dataset_split + item_id，排名只在各自 dataset_split 内计算。

### uild_category_behavior_features

uild_category_behavior_features(clean_data, item_behavior) 生成类目行为特征，粒度为 dataset_split + category_id。

### uild_time_behavior_features

uild_time_behavior_features(clean_data, item_behavior) 生成时间行为特征，粒度为 dataset_split + behavior_date + behavior_hour，只输出历史窗口内实际出现的日期—小时组合。

### uild_conversion_chain_features

uild_conversion_chain_features(item_behavior) 生成商品粒度转化链路特征：

- uy_per_pv = item_buy_count / item_pv_count
- uy_per_fav = item_buy_count / item_fav_count
- uy_per_cart = item_buy_count / item_cart_count

分母为 0 时结果为缺失值，不进行平滑。

### uild_assigned_feature_tables

uild_assigned_feature_tables(clean_data, item_behavior) 一次返回后四张特征表：item_popularity、category_behavior、	ime_behavior、conversion_chain。

## 八张特征表统一接口

### uild_all_feature_tables

uild_all_feature_tables(clean_data, *args, **kwargs) 在现有前四张特征表基础上追加后四张，统一返回八张阶段二特征表。

### generate_all_feature_tables

generate_all_feature_tables(input_parquet, output_directory, *args, **kwargs) 读取 clean Parquet，并原子写出完整八张阶段二特征表。

前四张兼容接口 uild_feature_tables 和 generate_feature_tables 继续保留；完整阶段二八张特征表应使用 uild_all_feature_tables 或 generate_all_feature_tables。
