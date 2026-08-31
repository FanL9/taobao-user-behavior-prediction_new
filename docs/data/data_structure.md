# Data 目录说明

`data/` 下的所有数据文件统一存放在团队 Google Drive 的 `taobao/data/`，GitHub 只保留目录骨架中的 `.gitkeep`。下载或回传数据时保持相同的相对路径。

## 目录结构

| 目录 | 简短说明 |
| --- | --- |
| `data/raw/` | 原始行为数据及其格式转换结果 |
| `data/processed/` | 完成清洗和标准化的数据 |
| `data/EDA/` | 阶段一 EDA 统计输出 |
| `data/interim/` | 阶段二四张基础中间表 |
| `data/features/` | 后续特征工程输出 |
| `data/sample/` | 调试和快速验证用样例数据 |
| `data/splits/` | 后续训练、验证和测试数据 |
| `data/predictions/` | 后续模型预测结果 |

## Raw 与 processed 数据

| 文件 | 简短说明 |
| --- | --- |
| `data/raw/user_behavior_processed.csv` | 阶段一使用的原始用户行为数据 |
| `data/raw/user_behavior_processed.parquet` | 原始 CSV 的 Parquet 转换结果 |
| `data/processed/user_behavior_clean.csv` | 云盘保留的标准清洗 CSV 副本，不作为后续阶段输入 |
| `data/processed/user_behavior_clean.parquet` | 清洗和标准化后的正式数据，后续阶段统一读取此文件 |

## EDA 数据

| 文件 | 简短说明 |
| --- | --- |
| `data/EDA/behavior_distribution.csv` | 四类行为数量分布 |
| `data/EDA/category_statistics.csv` | 类目维度行为统计 |
| `data/EDA/daily_behavior.csv` | 日期维度行为统计 |
| `data/EDA/descriptive_funnel.csv` | 基础行为漏斗统计 |
| `data/EDA/hourly_behavior.csv` | 小时维度行为统计 |
| `data/EDA/item_statistics.csv` | 商品维度行为统计 |
| `data/EDA/performance_test_result.csv` | EDA 性能测试结果 |
| `data/EDA/top_10_category.csv` | 行为量最高的 10 个类目 |
| `data/EDA/top_10_item.csv` | 行为量最高的 10 个商品 |
| `data/EDA/user_purchase_summary.csv` | 用户购买情况汇总 |

## 阶段二中间表

| 文件 | 简短说明 |
| --- | --- |
| `data/interim/user_intermediate.parquet` | 用户粒度历史行为中间统计 |
| `data/interim/item_intermediate.parquet` | 商品粒度历史行为中间统计 |
| `data/interim/category_intermediate.parquet` | 类目粒度历史行为中间统计 |
| `data/interim/time_intermediate.parquet` | 日期和小时粒度历史行为中间统计 |

## 已完成的前四张特征表

| 文件 | 简短说明 |
| --- | --- |
| `data/features/user_features.parquet` | 用户基础行为计数特征 |
| `data/features/user_activity_features.parquet` | 用户活跃度与日均行为特征 |
| `data/features/user_sequence_features.parquet` | 用户—商品最近行为序列特征 |
| `data/features/item_behavior_features.parquet` | 商品基础行为统计特征 |

禁止使用 `git add -f`、Git LFS 或修改忽略规则将任何 `data/` 数据文件上传到 GitHub。
