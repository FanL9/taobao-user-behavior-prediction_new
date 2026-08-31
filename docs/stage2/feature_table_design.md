# 阶段二前四张特征表说明

## 范围

当前只生成用户基础行为、用户活跃度、用户—商品行为序列和商品行为四张特征表。不生成商品热度、类目行为、时间行为、转化链路、标签或最终特征宽表。

## 时间窗口

| dataset_split | history_start | history_end | label_date（排除） |
| --- | --- | --- | --- |
| train | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| validation | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| test | 2025-12-16 | 2025-12-17 | 2025-12-18 |

所有特征只读取闭区间 `history_start` 至 `history_end`；`label_date` 及之后的数据不参与计算。

## 输出

| 文件 | 粒度 | 内容 |
| --- | --- | --- |
| `data/features/user_features.parquet` | `dataset_split + user_id` | 用户四类行为计数 |
| `data/features/user_activity_features.parquet` | `dataset_split + user_id` | 活跃天数、日均行为、最近活跃和活跃等级 |
| `data/features/user_sequence_features.parquet` | `dataset_split + user_id + item_id` | 用户—商品最近行为与最近10次序列 |
| `data/features/item_behavior_features.parquet` | `dataset_split + item_id` | 商品行为、交互用户、购买用户和活跃天数统计 |

## 职责边界

- 商品热度及等级由后续商品热度特征表负责。
- 行为转化率和相邻转移统计由后续转化链路特征表负责。
- 本阶段不构造 `label`，不合并最终宽表。
- 大型 Parquet 只在本地生成并同步至 Google Drive，不上传 GitHub。
