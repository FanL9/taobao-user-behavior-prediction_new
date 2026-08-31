# 阶段二用户—商品特征宽表

## 范围

本任务只将已有八张特征表合成为用户—商品粒度的初版特征宽表。宽表不包含 `label`，不训练或评估模型，不采样，也不执行最终入模特征筛选。

## 输入与输出

输入目录：`data/features/`

输入文件：

- `user_features.parquet`
- `user_activity_features.parquet`
- `user_sequence_features.parquet`
- `item_behavior_features.parquet`
- `item_popularity_features.parquet`
- `category_behavior_features.parquet`
- `time_behavior_features.parquet`
- `conversion_chain_features.parquet`

本地输出：`data/features/user_item_feature_wide.parquet`

质量报告：`reports/stage2/user_item_feature_wide_quality_report.json`

大型宽表不上传 GitHub，由本地脚本生成并同步至 Google Drive。

## 主键与合并口径

宽表唯一主键为 `dataset_split + user_id + item_id`，以 `user_sequence_features.parquet` 作为用户—商品候选行集合。

| 来源 | 合并键 | 使用方式 |
| --- | --- | --- |
| 用户基础行为、用户活跃度 | `dataset_split + user_id` | 为候选行补充用户特征 |
| 商品行为、商品热度、转化链路 | `dataset_split + item_id` | 为候选行补充商品特征和 `category_id` |
| 类目行为 | `dataset_split + category_id` | 经商品所属类目补充类目特征 |
| 时间行为 | `dataset_split + last_behavior_date + last_behavior_hour` | 补充该用户—商品最后一次历史行为所在日期—小时的时间特征 |

`last_behavior_date` 由 `label_date` 和 `last_behavior_days_ago` 还原。每次合并必须保持行数不变，并按原表粒度执行多对一或一对一校验。

八张表必须使用相同的三个时间窗口：

| dataset_split | history_start | history_end | label_date（排除） |
| --- | --- | --- | --- |
| `train` | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| `validation` | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| `test` | 2025-12-16 | 2025-12-17 | 2025-12-18 |

## 字段角色

- 主键字段：`dataset_split`、`user_id`、`item_id`。
- 追踪字段：`category_id`、`history_start`、`history_end`、`label_date`、`last_behavior_date`。
- 候选特征字段：除主键和追踪字段外的其余字段；阶段三可再进行编码、预处理和筛选。
- 禁止直接入模字段：全部主键字段和追踪字段。宽表中不存在 `label`。

## 字段说明

### 主键与追踪字段

| 字段 | 含义 | 角色 |
| --- | --- | --- |
| `dataset_split` | `train`、`validation` 或 `test` 历史窗口 | 主键；禁止直接入模 |
| `user_id` | 用户标识 | 主键；禁止直接入模 |
| `item_id` | 商品标识 | 主键；禁止直接入模 |
| `category_id` | 商品所属类目标识 | 追踪；禁止直接入模 |
| `history_start` | 当前历史窗口开始日 | 追踪；禁止直接入模 |
| `history_end` | 当前历史窗口结束日 | 追踪；禁止直接入模 |
| `label_date` | 后续标签日，本表不生成标签 | 追踪；禁止直接入模 |
| `last_behavior_date` | 用户对商品最后一次历史行为日期 | 追踪；禁止直接入模 |

### 用户基础行为字段

| 字段 | 含义 |
| --- | --- |
| `user_event_count` | 用户历史行为总数 |
| `user_pv_count` | 用户历史浏览数 |
| `user_fav_count` | 用户历史收藏数 |
| `user_cart_count` | 用户历史加购数 |
| `user_buy_count` | 用户历史购买数 |

### 用户活跃度字段

| 字段 | 含义 |
| --- | --- |
| `user_window_days` | 当前历史窗口天数 |
| `user_active_day_count` | 用户历史活跃天数 |
| `user_active_day_ratio` | 活跃天数占窗口天数的比例 |
| `user_avg_daily_event_count` | 窗口日均行为数 |
| `user_avg_active_day_event_count` | 活跃日内日均行为数 |
| `user_days_since_last_event` | 标签日前距用户最后一次行为的天数 |
| `user_unique_item_count` | 用户历史交互商品数 |
| `user_unique_category_count` | 用户历史交互类目数 |
| `user_pv_count_per_day` | 用户日均浏览数 |
| `user_fav_count_per_day` | 用户日均收藏数 |
| `user_cart_count_per_day` | 用户日均加购数 |
| `user_buy_count_per_day` | 用户日均购买数 |
| `user_activity_level` | 使用训练窗口固定 P25/P75 阈值得到的活跃等级 |

### 用户行为序列字段

| 字段 | 含义 |
| --- | --- |
| `last_behavior_type` | 用户对商品最后一次历史行为类型 |
| `last_behavior_hour` | 最后一次历史行为小时 |
| `last_behavior_days_ago` | 标签日前距该用户—商品最后一次行为的天数 |
| `last_10_behavior_sequence` | 用户对商品最近最多 10 次历史行为序列 |

### 商品行为字段

| 字段 | 含义 |
| --- | --- |
| `item_total_count` | 商品历史行为总数 |
| `item_pv_count` | 商品历史浏览数 |
| `item_fav_count` | 商品历史收藏数 |
| `item_cart_count` | 商品历史加购数 |
| `item_buy_count` | 商品历史购买数 |
| `item_unique_user_count` | 商品历史交互用户数 |
| `item_unique_buyer_count` | 商品历史购买用户数 |
| `item_active_day_count` | 商品历史活跃天数 |

### 商品热度字段

| 字段 | 含义 |
| --- | --- |
| `item_total_count_rank` | 商品行为总数在当前数据集内的降序 dense rank |
| `item_unique_user_count_rank` | 商品交互用户数在当前数据集内的降序 dense rank |
| `item_active_day_count_rank` | 商品活跃天数在当前数据集内的降序 dense rank |
| `item_buy_count_rank` | 商品购买数在当前数据集内的降序 dense rank |

### 转化链路字段

| 字段 | 含义 |
| --- | --- |
| `buy_per_pv` | 商品历史购买数除以浏览数；分母为 0 时缺失 |
| `buy_per_fav` | 商品历史购买数除以收藏数；分母为 0 时缺失 |
| `buy_per_cart` | 商品历史购买数除以加购数；分母为 0 时缺失 |

### 类目行为字段

| 字段 | 含义 |
| --- | --- |
| `category_total_count` | 类目历史行为总数 |
| `category_pv_count` | 类目历史浏览数 |
| `category_fav_count` | 类目历史收藏数 |
| `category_cart_count` | 类目历史加购数 |
| `category_buy_count` | 类目历史购买数 |
| `category_unique_user_count` | 类目历史交互用户数 |
| `category_unique_item_count` | 类目历史商品数 |
| `category_active_day_count` | 类目历史活跃天数 |
| `category_first_event_time` | 类目在窗口内的最早行为时间 |
| `category_last_event_time` | 类目在窗口内的最晚行为时间 |

### 时间行为字段

| 字段 | 含义 |
| --- | --- |
| `time_weekday` | 最后行为日期的星期序号，Monday=0 |
| `time_total_count` | 最后行为日期—小时的历史行为总数 |
| `time_pv_count` | 最后行为日期—小时的浏览数 |
| `time_fav_count` | 最后行为日期—小时的收藏数 |
| `time_cart_count` | 最后行为日期—小时的加购数 |
| `time_buy_count` | 最后行为日期—小时的购买数 |
| `time_unique_user_count` | 最后行为日期—小时的用户数 |
| `time_unique_item_count` | 最后行为日期—小时的商品数 |
| `time_unique_category_count` | 最后行为日期—小时的类目数 |

## 接口说明

命令行：

```powershell
python scripts/build_user_item_feature_wide.py
```

可选参数：`--features-dir`、`--output`、`--quality-report`、`--batch-size`。

Python：

```python
generate_user_item_feature_wide(
    feature_directory,
    output_parquet,
    quality_report_path,
    batch_size=200_000,
) -> dict
```

函数读取八张 Parquet，分批生成宽表并写出质量报告，返回输出路径和报告内容。

## 质量检查与交付

质量报告记录样本量、字段数、各数据集行数、逐字段缺失值、重复主键、异常取值和时间窗口违规数。除转化率分母为 0 产生的缺失值外，其余字段不允许缺失。

该宽表作为阶段三标签生成、特征预处理和特征筛选的输入；阶段三负责生成标签并决定候选字段是否入模。

