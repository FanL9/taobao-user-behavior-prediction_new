# 阶段二中间表结构设计

## 输出位置

```text
data/interim/
├── user_intermediate.parquet
├── item_intermediate.parquet
├── category_intermediate.parquet
└── time_intermediate.parquet
```

四个文件属于中间数据，统一放 Google Drive，不上传 GitHub。每个文件同时保存 `train`、`validation`、`test` 三个窗口，通过 `dataset_split` 区分。

## 公共窗口字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dataset_split` | string | `train`、`validation` 或 `test` |
| `history_start` | date | 该行全部统计的最早允许日期 |
| `history_end` | date | 该行全部统计的最晚允许日期 |
| `label_date` | date | 后续标签日；当前统计明确排除该日 |

## 用户维度中间表

主键：`dataset_split + user_id`。

| 字段 | 类型 | 统计窗口 |
| --- | --- | --- |
| `user_id` | int64 | 粒度字段 |
| `event_count` | int64 | `history_start` 至 `history_end` 全部行为数 |
| `pv_count` / `fav_count` / `cart_count` / `buy_count` | int64 | 同一历史窗口内对应行为数 |
| `unique_item_count` | int64 | 同一历史窗口内交互商品去重数 |
| `unique_category_count` | int64 | 同一历史窗口内交互类目去重数 |
| `active_day_count` | int64 | 同一历史窗口内有行为的自然日数 |
| `first_event_time` / `last_event_time` | timestamp | 同一历史窗口内首次/末次行为时间 |

## 商品维度中间表

主键：`dataset_split + item_id`。

| 字段 | 类型 | 统计窗口 |
| --- | --- | --- |
| `item_id` | int64 | 粒度字段 |
| `category_id` | int64 | 该商品对应类目；窗口内必须唯一 |
| `event_count` | int64 | 对应历史窗口内全部行为数 |
| `pv_count` / `fav_count` / `cart_count` / `buy_count` | int64 | 对应历史窗口内各行为数 |
| `unique_user_count` | int64 | 对应历史窗口内交互用户去重数 |
| `active_day_count` | int64 | 对应历史窗口内有行为的自然日数 |
| `first_event_time` / `last_event_time` | timestamp | 对应历史窗口内首次/末次行为时间 |

## 类目维度中间表

主键：`dataset_split + category_id`。

| 字段 | 类型 | 统计窗口 |
| --- | --- | --- |
| `category_id` | int64 | 粒度字段 |
| `event_count` | int64 | 对应历史窗口内全部行为数 |
| `pv_count` / `fav_count` / `cart_count` / `buy_count` | int64 | 对应历史窗口内各行为数 |
| `unique_user_count` | int64 | 对应历史窗口内交互用户去重数 |
| `unique_item_count` | int64 | 对应历史窗口内商品去重数 |
| `active_day_count` | int64 | 对应历史窗口内有行为的自然日数 |
| `first_event_time` / `last_event_time` | timestamp | 对应历史窗口内首次/末次行为时间 |

## 时间维度中间表

主键：`dataset_split + behavior_date + behavior_hour`。

| 字段 | 类型 | 统计窗口 |
| --- | --- | --- |
| `behavior_date` | date | 历史窗口内的行为日期 |
| `behavior_hour` | int8 | 小时，0～23 |
| `weekday` | int8 | Monday=0，Sunday=6 |
| `event_count` | int64 | 该日期—小时全部行为数 |
| `pv_count` / `fav_count` / `cart_count` / `buy_count` | int64 | 该日期—小时各行为数 |
| `unique_user_count` | int64 | 该日期—小时用户去重数 |
| `unique_item_count` | int64 | 该日期—小时商品去重数 |
| `unique_category_count` | int64 | 该日期—小时类目去重数 |

## 一致性要求

- 每张表在每个 `dataset_split` 内的 `event_count` 合计必须等于该历史窗口输入行数。
- 每行 `pv_count + fav_count + cart_count + buy_count = event_count`。
- 主键必须唯一；窗口字段必须与 `dataset_split` 的固定定义一致。
- 表中不得出现标签、模型预测、采样结果或用户—商品组合字段。
