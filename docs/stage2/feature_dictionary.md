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
