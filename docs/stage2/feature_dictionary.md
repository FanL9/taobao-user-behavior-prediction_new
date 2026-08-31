# 阶段二特征字典

本字典描述当前四张中间表字段。它们是后续特征构造的基础输入，本阶段均不直接用于建模。

## 公共字段

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 粒度 | 用于建模 |
| --- | --- | --- | --- | --- | --- |
| `dataset_split` | 历史窗口名称 | 按固定训练/验证/测试历史区间赋值 | 阶段二窗口配置 | 所有中间表主键组成部分 | 否 |
| `history_start` | 历史起点 | 取窗口固定起始日 | 阶段二窗口配置 | 每个中间表行 | 否 |
| `history_end` | 历史终点 | 取窗口固定结束日 | 阶段二窗口配置 | 每个中间表行 | 否 |
| `label_date` | 后续标签日 | 取窗口固定标签日，仅用于审计排除 | 阶段二窗口配置 | 每个中间表行 | 否 |
| `event_count` | 历史行为总数 | 窗口内事件行数 | clean Parquet | 对应表粒度 | 否，中间统计字段 |
| `pv_count` | 历史浏览数 | 窗口内 `behavior_type=1` 行数 | clean Parquet | 对应表粒度 | 否，中间统计字段 |
| `fav_count` | 历史收藏数 | 窗口内 `behavior_type=2` 行数 | clean Parquet | 对应表粒度 | 否，中间统计字段 |
| `cart_count` | 历史加购数 | 窗口内 `behavior_type=3` 行数 | clean Parquet | 对应表粒度 | 否，中间统计字段 |
| `buy_count` | 历史购买数 | 窗口内 `behavior_type=4` 行数 | clean Parquet | 对应表粒度 | 否，中间统计字段 |

## 用户中间表字段

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 粒度 | 用于建模 |
| --- | --- | --- | --- | --- | --- |
| `user_id` | 用户标识 | 原字段 | clean Parquet | `dataset_split + user_id` | 否，主键 |
| `unique_item_count` | 历史交互商品数 | 窗口内 `item_id` 去重计数 | clean Parquet | 用户 | 否，中间统计字段 |
| `unique_category_count` | 历史交互类目数 | 窗口内 `category_id` 去重计数 | clean Parquet | 用户 | 否，中间统计字段 |
| `active_day_count` | 历史活跃天数 | 窗口内 `behavior_date` 去重计数 | clean Parquet | 用户 | 否，中间统计字段 |
| `first_event_time` | 窗口内首次行为时间 | `time` 最小值 | clean Parquet | 用户 | 否，中间审计字段 |
| `last_event_time` | 窗口内末次行为时间 | `time` 最大值 | clean Parquet | 用户 | 否，中间审计字段 |

## 商品中间表字段

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 粒度 | 用于建模 |
| --- | --- | --- | --- | --- | --- |
| `item_id` | 商品标识 | 原字段 | clean Parquet | `dataset_split + item_id` | 否，主键 |
| `category_id` | 商品所属类目 | 窗口内唯一类目映射 | clean Parquet | 商品 | 否，中间关联字段 |
| `unique_user_count` | 历史交互用户数 | 窗口内 `user_id` 去重计数 | clean Parquet | 商品 | 否，中间统计字段 |
| `active_day_count` | 历史活跃天数 | 窗口内 `behavior_date` 去重计数 | clean Parquet | 商品 | 否，中间统计字段 |
| `first_event_time` | 窗口内首次行为时间 | `time` 最小值 | clean Parquet | 商品 | 否，中间审计字段 |
| `last_event_time` | 窗口内末次行为时间 | `time` 最大值 | clean Parquet | 商品 | 否，中间审计字段 |

## 类目中间表字段

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 粒度 | 用于建模 |
| --- | --- | --- | --- | --- | --- |
| `category_id` | 类目标识 | 原字段 | clean Parquet | `dataset_split + category_id` | 否，主键 |
| `unique_user_count` | 历史交互用户数 | 窗口内 `user_id` 去重计数 | clean Parquet | 类目 | 否，中间统计字段 |
| `unique_item_count` | 历史商品数 | 窗口内 `item_id` 去重计数 | clean Parquet | 类目 | 否，中间统计字段 |
| `active_day_count` | 历史活跃天数 | 窗口内 `behavior_date` 去重计数 | clean Parquet | 类目 | 否，中间统计字段 |
| `first_event_time` | 窗口内首次行为时间 | `time` 最小值 | clean Parquet | 类目 | 否，中间审计字段 |
| `last_event_time` | 窗口内末次行为时间 | `time` 最大值 | clean Parquet | 类目 | 否，中间审计字段 |

## 时间中间表字段

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 粒度 | 用于建模 |
| --- | --- | --- | --- | --- | --- |
| `behavior_date` | 行为日期 | 由 `time` 解析并按日截断 | clean Parquet | `dataset_split + behavior_date + behavior_hour` | 否，主键 |
| `behavior_hour` | 行为小时 | 由 `time` 解析，0～23 | clean Parquet | 日期—小时 | 否，主键 |
| `weekday` | 星期序号 | 由 `time` 解析，Monday=0 | clean Parquet | 日期—小时 | 否，中间字段 |
| `unique_user_count` | 日期—小时用户数 | `user_id` 去重计数 | clean Parquet | 日期—小时 | 否，中间统计字段 |
| `unique_item_count` | 日期—小时商品数 | `item_id` 去重计数 | clean Parquet | 日期—小时 | 否，中间统计字段 |
| `unique_category_count` | 日期—小时类目数 | `category_id` 去重计数 | clean Parquet | 日期—小时 | 否，中间统计字段 |

`event_count` 和四类行为计数字段适用于上述四张表，其具体粒度由所在表决定。转化率字段本阶段不落表，公式见 `feature_engineering_spec.md`。

## 已完成的前四张特征表

四张表共同包含 `dataset_split`、`history_start`、`history_end` 和 `label_date`。这些字段用于窗口关联和防泄露审计，不作为模型输入。

### `user_features.parquet`

粒度：`dataset_split + user_id`。

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 用于建模 |
| --- | --- | --- | --- | --- |
| `user_id` | 用户标识 | 原字段 | clean Parquet | 关联键，不直接使用 |
| `event_count` | 用户行为总数 | 历史窗口事件行数 | clean Parquet | 可选，需注意窗口长度 |
| `pv_count` | 浏览数 | `behavior_name=pv` 行数 | clean Parquet | 可选 |
| `fav_count` | 收藏数 | `behavior_name=fav` 行数 | clean Parquet | 可选 |
| `cart_count` | 加购数 | `behavior_name=cart` 行数 | clean Parquet | 可选 |
| `buy_count` | 购买数 | `behavior_name=buy` 行数 | clean Parquet | 可选 |

### `user_activity_features.parquet`

粒度：`dataset_split + user_id`。`activity_level` 的 P25/P75 只由训练窗口的 `avg_daily_event_count` 计算，并固定应用到验证和测试窗口。

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 用于建模 |
| --- | --- | --- | --- | --- |
| `user_id` | 用户标识 | 原字段 | clean Parquet | 关联键，不直接使用 |
| `window_days` | 窗口天数 | `history_end-history_start+1` | 窗口配置 | 否，审计字段 |
| `event_count` | 用户行为总数 | 历史窗口事件行数 | clean Parquet | 建议使用日均字段替代 |
| `active_day_count` | 活跃天数 | 行为日期去重数 | clean Parquet | 是 |
| `active_day_ratio` | 活跃日比例 | `active_day_count/window_days` | 派生 | 是 |
| `avg_daily_event_count` | 日均行为数 | `event_count/window_days` | 派生 | 是 |
| `avg_active_day_event_count` | 活跃日日均行为数 | `event_count/active_day_count` | 派生 | 是 |
| `days_since_last_event` | 距末次行为天数 | `label_date-last_event_time` | 派生 | 是 |
| `unique_item_count` | 交互商品数 | `item_id` 去重数 | clean Parquet | 是 |
| `unique_category_count` | 交互类目数 | `category_id` 去重数 | clean Parquet | 是 |
| `pv_count_per_day` | 日均浏览数 | `pv_count/window_days` | 派生 | 是 |
| `fav_count_per_day` | 日均收藏数 | `fav_count/window_days` | 派生 | 是 |
| `cart_count_per_day` | 日均加购数 | `cart_count/window_days` | 派生 | 是 |
| `buy_count_per_day` | 日均购买数 | `buy_count/window_days` | 派生 | 是 |
| `activity_level` | 活跃等级 | 训练集 P25/P75 分为 low/medium/high | 派生 | 编码后可用 |

### `user_sequence_features.parquet`

粒度：`dataset_split + user_id + item_id`。

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 用于建模 |
| --- | --- | --- | --- | --- |
| `user_id` | 用户标识 | 原字段 | clean Parquet | 关联键，不直接使用 |
| `item_id` | 商品标识 | 原字段 | clean Parquet | 关联键，不直接使用 |
| `last_behavior_type` | 最后行为类型 | 历史窗口末次行为 | clean Parquet | 编码后可用 |
| `last_behavior_hour` | 最后行为小时 | 末次行为时间的小时 | clean Parquet | 是 |
| `last_behavior_days_ago` | 距最后行为天数 | `label_date-last_behavior_time` | 派生 | 是 |
| `last_10_behavior_sequence` | 最近行为序列 | 每个用户—商品最近最多10次行为按时间连接 | clean Parquet | 序列编码后可用 |

同一小时内无法从原数据获得更细时间顺序，使用 clean Parquet 中的稳定原始行序作为并列顺序。转移次数不在本表生成，留给后续转化链路特征表。

### `item_behavior_features.parquet`

粒度：`dataset_split + item_id`。

| 字段名 | 字段含义 | 计算逻辑 | 数据来源 | 用于建模 |
| --- | --- | --- | --- | --- |
| `item_id` | 商品标识 | 原字段 | clean Parquet | 关联键，不直接使用 |
| `category_id` | 所属类目 | 窗口内唯一类目映射 | clean Parquet | 关联或编码后使用 |
| `item_total_count` | 商品行为总数 | 历史窗口事件行数 | clean Parquet | 可选，需注意窗口长度 |
| `item_pv_count` | 商品浏览数 | `behavior_name=pv` 行数 | clean Parquet | 可选 |
| `item_fav_count` | 商品收藏数 | `behavior_name=fav` 行数 | clean Parquet | 可选 |
| `item_cart_count` | 商品加购数 | `behavior_name=cart` 行数 | clean Parquet | 可选 |
| `item_buy_count` | 商品购买数 | `behavior_name=buy` 行数 | clean Parquet | 可选 |
| `item_unique_user_count` | 交互用户数 | `user_id` 去重数 | clean Parquet | 是 |
| `item_unique_buyer_count` | 购买用户数 | 购买行为中的 `user_id` 去重数 | clean Parquet | 是 |
| `item_active_day_count` | 商品活跃天数 | 行为日期去重数 | clean Parquet | 是 |

商品热度等级不在本表生成，转化率也不在本表生成，分别留给后续商品热度特征表和转化链路特征表。

