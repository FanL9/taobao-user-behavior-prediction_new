# 阶段二特征工程口径

## 范围

阶段二特征工程分为三层：

1. 基础统计中间表层：基于 `data/processed/user_behavior_clean.parquet` 生成用户、商品、类目和时间四张中间表。
2. 特征表层：基于阶段二既定时间窗口和清洗后的行为数据，由 `src/features/feature.py` 统一生成八类特征表，输出至 `data/features/`。
3. 初版宽表层：以用户行为序列表为用户—商品主键集合，合并八张特征表，输出 `user_item_feature_wide.parquet`。

本阶段只负责特征构造和初版宽表整合，不生成监督学习标签，不执行采样、模型训练、模型评估或最终入模特征筛选。

## 时间窗口

| `dataset_split` | 历史统计起点 | 历史统计终点 | 标签日（特征统计排除） |
| --- | --- | --- | --- |
| `train` | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| `validation` | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| `test` | 2025-12-16 | 2025-12-17 | 2025-12-18 |

起止日期均为闭区间。所有中间表和特征表均写入或保留 `history_start`、`history_end` 和 `label_date`，统计字段只能使用对应历史闭区间内的事件。

## 防止未来信息泄露

- 标签日事件不得进入对应历史特征统计。
- 标签日之后的事件不得进入对应历史特征统计。
- `train`、`validation`、`test` 三个历史窗口分别计算，不跨窗口累计。
- 所有特征均满足 `history_end < label_date`。
- 时间行为特征中的 `behavior_date` 必须位于对应 `history_start` 与 `history_end` 之间。
- 商品热度排名只在当前 `dataset_split` 内计算，不允许不同窗口之间互相参与排名。
- 本阶段不生成监督学习 `label` 字段。

## 中间表字段与粒度

- 用户中间表：每个 `dataset_split + user_id` 一行。
- 商品中间表：每个 `dataset_split + item_id` 一行；同一窗口内 `item_id` 必须只对应一个 `category_id`。
- 类目中间表：每个 `dataset_split + category_id` 一行。
- 时间中间表：每个 `dataset_split + behavior_date + behavior_hour` 一行，只输出实际出现的日期—小时组合。
- 行为编码：`1=pv`、`2=fav`、`3=cart`、`4=buy`。
- 所有计数均为对应历史窗口内的原始行为事件数。

## 八类特征表

统一实现入口：`src/features/feature.py`

统一输出目录：`data/features/`

八张特征表为：

| 序号 | 输出文件 | 特征类别 |
| --- | --- | --- |
| 1 | `user_features.parquet` | 用户基础行为特征 |
| 2 | `user_activity_features.parquet` | 用户活跃度特征 |
| 3 | `user_sequence_features.parquet` | 用户行为序列特征 |
| 4 | `item_behavior_features.parquet` | 商品行为特征 |
| 5 | `item_popularity_features.parquet` | 商品热度特征 |
| 6 | `category_behavior_features.parquet` | 类目行为特征 |
| 7 | `time_behavior_features.parquet` | 时间行为特征 |
| 8 | `conversion_chain_features.parquet` | 转化链路特征 |

八张表的完整字段、计算逻辑和粒度以 `feature_dictionary.md` 为准。

### 商品热度特征表

粒度：`dataset_split + item_id`

该表直接复用商品行为特征中的历史统计，并在每个 `dataset_split` 内独立计算以下降序 dense rank：

- `item_total_count_rank`
- `item_unique_user_count_rank`
- `item_active_day_count_rank`
- `item_buy_count_rank`

排名值越小表示对应历史统计指标越高。不同数据窗口之间不会互相参与排名。

### 类目行为特征表

粒度：`dataset_split + category_id`

统计内容包括：

- 历史行为总量；
- PV / FAV / CART / BUY 行为次数；
- 去重用户数；
- 去重商品数；
- 活跃天数；
- 历史窗口内首次和末次行为时间。

所有统计均只使用当前历史窗口内数据。

### 时间行为特征表

粒度：`dataset_split + behavior_date + behavior_hour`

只输出真实出现的日期—小时组合，不人为补齐不存在的小时记录。

统计内容包括：

- `weekday`
- 行为总量
- 去重用户数
- 去重商品数
- 去重类目数
- PV / FAV / CART / BUY 行为次数

### 转化链路特征表

粒度：`dataset_split + item_id`

转化链路使用同一商品、同一 `dataset_split`、同一历史窗口内的行为次数计算：

| 字段 | 公式 |
| --- | --- |
| `buy_per_pv` | `item_buy_count / item_pv_count` |
| `buy_per_fav` | `item_buy_count / item_fav_count` |
| `buy_per_cart` | `item_buy_count / item_cart_count` |

分母为 0 时结果定义为缺失值，不填 0，也不进行平滑。

这些值属于历史行为次数比，不表示严格按时间顺序发生的用户转化概率，因此结果不要求限制在 `[0, 1]`。

## 输出与存储约定

- 八张特征表统一输出至 `data/features/`。
- 大型真实 Parquet 数据文件不提交 GitHub。
- GitHub 只提交源码、测试、接口说明和 Markdown 报告。
- `data/` 和 `*.parquet` 已由 `.gitignore` 排除。
