# Data 目录说明

`data/` 下的所有数据文件统一存放在团队 Google Drive 的 `taobao/data/`，GitHub 只保留目录骨架中的 `.gitkeep`。下载或回传数据时保持相同的相对路径。

## 目录结构

| 目录 | 简短说明 |
| --- | --- |
| `data/raw/` | 原始行为数据及其格式转换结果 |
| `data/processed/` | 完成清洗和标准化的数据 |
| `data/EDA/` | 阶段一 EDA 统计输出 |
| `data/interim/` | 阶段二四张基础中间表 |
| `data/features/` | 阶段二特征表和用户—商品初版特征宽表 |
| `data/sample/` | 调试和快速验证用样例数据 |
| `data/splits/` | 阶段三带标签用户—商品样本及后续数据划分结果 |
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

## 阶段二八张特征表

| 文件 | 简短说明 |
| --- | --- |
| `data/features/user_features.parquet` | 用户基础行为计数特征 |
| `data/features/user_activity_features.parquet` | 用户活跃度与日均行为特征 |
| `data/features/user_sequence_features.parquet` | 用户—商品最近行为序列特征 |
| `data/features/item_behavior_features.parquet` | 商品基础行为统计特征 |
| `data/features/item_popularity_features.parquet` | 商品热度排名特征 |
| `data/features/category_behavior_features.parquet` | 类目行为统计特征 |
| `data/features/time_behavior_features.parquet` | 日期—小时行为统计特征 |
| `data/features/conversion_chain_features.parquet` | 商品历史转化链路特征 |

## 阶段二初版宽表

| 文件 | 简短说明 |
| --- | --- |
| `data/features/user_item_feature_wide.parquet` | 八张特征表合成的用户—商品粒度初版特征宽表 |

## 阶段三带标签样本

| 文件 | 简短说明 |
| --- | --- |
| `data/splits/user_item_feature_wide_labeled.parquet` | 保留阶段二初版特征宽表全部字段，并追加未来 1 天购买 `label` 的用户—商品样本 |
| `data/splits/user_item_feature_wide_labeled_train.parquet` | 训练集：特征窗口 2025-11-18 至 2025-12-07，标签日 2025-12-08 |
| `data/splits/user_item_feature_wide_labeled_validation.parquet` | 验证集：特征窗口 2025-12-09 至 2025-12-14，标签日 2025-12-15 |
| `data/splits/user_item_feature_wide_labeled_test.parquet` | 测试集：特征窗口 2025-12-16 至 2025-12-17，标签日 2025-12-18 |

禁止使用 `git add -f`、Git LFS 或修改忽略规则将任何 `data/` 数据文件上传到 GitHub。
