# 阶段一数据清洗结果报告

## 1. 清洗任务概览

本次清洗输入文件：

data/raw/user_behavior_processed.csv

正式输出文件：

- data/processed/user_behavior_clean.csv
- data/processed/user_behavior_clean.parquet

清洗后的数据文件同步存放于团队 Google Drive 的 taobao/data/processed/，GitHub 不保存数据文件本体。

## 2. 清洗前后数据规模

| 指标 | 数值 |
| --- | ---: |
| 原始记录数 | 12,256,906 |
| 清洗后记录数 | 6,213,379 |
| 删除记录数 | 6,043,527 |
| 数据保留率 | 50.69% |
| 数据去重率 | 49.31% |

## 3. 数据质量处理结果

| 清洗项目 | 删除记录数 |
| --- | ---: |
| 缺失关键字段 | 0 |
| 非法 ID | 0 |
| 非法行为类型 | 0 |
| 非法时间 | 0 |
| 完全重复记录 | 6,043,527 |

本批数据未发现关键字段缺失、非法 ID、非法行为类型或非法时间记录。

主要数据质量问题为完全重复记录，共删除 6,043,527 条，占原始数据的 49.31%。

## 4. 重复记录处理策略

完全重复记录由以下五个字段共同确定：

- time
- user_id
- item_id
- category_id
- behavior_type

完全重复记录执行全局去重，每组仅保留一条。

疑似重复规则：

user_id + item_id + behavior_type + time

原始数据检测结果：

| 指标 | 数值 |
| --- | ---: |
| 疑似重复组数 | 3,843,197 |
| 涉及记录数 | 9,886,724 |
| 超额记录数 | 6,043,527 |

完成精确去重后，剩余疑似重复组数为 0。

因此，本数据集中小时级疑似重复均可由完全重复记录解释，没有额外删除可能属于真实用户行为的小时级记录。

## 5. 字段标准化与新增字段

清洗后标准字段：

- time
- user_id
- item_id
- category_id
- behavior_type
- behavior_name
- behavior_date
- behavior_hour
- weekday

字段处理规则：

- item_category 标准化为 category_id
- behavior_type 映射生成 behavior_name
- 从 time 派生 behavior_date
- 从 time 派生 behavior_hour
- 从 time 派生 weekday
- weekday 定义为 Monday=0 至 Sunday=6

行为类型映射：

| behavior_type | behavior_name |
| ---: | --- |
| 1 | pv |
| 2 | fav |
| 3 | cart |
| 4 | buy |

## 6. 工程实现与性能

清洗流程采用分块读取与 Hash 分区方式，避免一次性将全量数据载入内存。

| 参数 | 数值 |
| --- | ---: |
| chunksize | 250000 |
| Hash 分区数 | 64 |
| 全量处理耗时 | 97.07 秒 |

跨 Chunk 的记录通过 Hash 分区汇聚后执行全局精确去重，避免仅在单个 Chunk 内去重产生遗漏。

## 7. 输出一致性验证

最终验证结果：

- CSV 行数：6,213,379
- Parquet 行数：6,213,379
- CSV 与 Parquet 行数一致
- 清洗结果完全重复记录数为 0
- 全项目自动化测试：19 passed

## 8. 结论

阶段一标准数据清洗流程已完成。

原始数据共 12,256,906 条，清洗后保留 6,213,379 条，数据保留率为 50.69%。

共删除 6,043,527 条完全重复记录，占原始数据的 49.31%。

完成精确去重后，没有剩余需要额外删除的小时级疑似重复行为。清洗结果可作为后续 EDA、转化漏斗分析、特征工程和建模阶段的标准输入数据。
