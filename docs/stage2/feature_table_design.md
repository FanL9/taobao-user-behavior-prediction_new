# 阶段二八张特征表说明

## 范围

阶段二基于 `data/processed/user_behavior_clean.parquet` 统一生成八张独立特征表。本文只定义八张独立表；后续用户—商品初版宽表合并见 `user_item_feature_wide.md`。两部分均不生成 `label`，不训练或评估模型，也不采样。

## 时间窗口

| dataset_split | history_start | history_end | label_date（排除） |
| --- | --- | --- | --- |
| train | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| validation | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| test | 2025-12-16 | 2025-12-17 | 2025-12-18 |

所有统计只读取闭区间 `history_start` 至 `history_end`。`label_date` 及之后的数据不得参与特征计算。

## 八张输出

| 文件 | 粒度 | 内容 |
| --- | --- | --- |
| `user_features.parquet` | `dataset_split + user_id` | 用户四类基础行为计数 |
| `user_activity_features.parquet` | `dataset_split + user_id` | 活跃天数、日均行为、最近活跃及活跃等级 |
| `user_sequence_features.parquet` | `dataset_split + user_id + item_id` | 用户—商品最近行为及最近10次序列 |
| `item_behavior_features.parquet` | `dataset_split + item_id` | 商品行为、用户、购买用户及活跃天数统计 |
| `item_popularity_features.parquet` | `dataset_split + item_id` | 商品行为量、用户量、活跃天数和购买量排名 |
| `category_behavior_features.parquet` | `dataset_split + category_id` | 类目行为及覆盖用户、商品、活跃天数统计 |
| `time_behavior_features.parquet` | `dataset_split + behavior_date + behavior_hour` | 实际日期—小时行为及实体数量统计 |
| `conversion_chain_features.parquet` | `dataset_split + item_id` | 商品购买数相对浏览、收藏和加购次数的比率 |

八张 Parquet 统一输出到 `data/features/`。详细字段见 `feature_dictionary.md`。

## 统一约束

- 每张表的主键在其粒度内唯一。
- 商品热度 dense rank 只在当前 `dataset_split` 内计算，排名越小代表历史指标越高。
- 转化链路分母为 0 时结果为缺失值，不填 0、不平滑，也不限制结果必须小于等于 1。
- 时间行为表只输出历史窗口内真实存在的日期—小时组合。
- 大型 Parquet 只在本地生成并同步至 Google Drive，不上传 GitHub。
