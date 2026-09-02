# 项目统一口径

本文档是项目字段、文件和处理边界的权威定义。实现与本文档冲突时，以本文档为准并先通过评审修改口径。

## 1. 项目目标

使用历史浏览、收藏、加购和购买事件，构建可复现的数据处理、特征工程与购买预测流程。阶段一完成数据接入、质量检查和标准清洗；阶段二生成基础中间表、八张独立特征表和用户—商品初版特征宽表；阶段三完成标签生成、数据集划分、特征预处理、特征筛选和类别不平衡处理；模型训练、评估及预测由后续任务负责。

## 2. 阶段一输入

- 文件名：`user_behavior_processed.csv`
- 本地路径：`data/raw/user_behavior_processed.csv`
- 文本编码：默认 `UTF-8 with BOM`（`utf-8-sig` 同时兼容普通 UTF-8）
- 表头和顺序：`time,user_id,item_id,item_category,behavior_type`
- 一行代表一次用户行为事件；阶段一不假设组合键唯一。

| 字段 | Parquet / SQLite 类型 | 允许值与说明 |
| --- | --- | --- |
| `time` | string / TEXT | `YYYY-MM-DD HH`，精确到小时 |
| `user_id` | int64 / INTEGER | 正整数用户标识 |
| `item_id` | int64 / INTEGER | 正整数商品标识 |
| `item_category` | int64 / INTEGER | 正整数类目标识；全项目统一使用该名称 |
| `behavior_type` | int8 / INTEGER | 行为类型编码：`1`、`2`、`3`、`4` |

## 2.1 阶段一清洗后标准数据

阶段一标准清洗输出：

- `data/processed/user_behavior_clean.csv`
- `data/processed/user_behavior_clean.parquet`

清洗后的标准字段顺序为：

`time,user_id,item_id,category_id,behavior_type,behavior_name,behavior_date,behavior_hour,weekday`

字段约定：

- 原始输入继续使用 `item_category`
- 阶段一标准清洗输出统一使用 `category_id`
- `behavior_name`：由 `behavior_type` 映射生成
- `behavior_date`：行为日期
- `behavior_hour`：行为小时，取值 0~23
- `weekday`：Monday=0，Sunday=6

原始数据接口保持不变；`item_category` 到 `category_id` 的标准化仅发生在清洗输出层。

## 2.2 阶段二输入与时间窗口

阶段二正式输入统一为 `data/processed/user_behavior_clean.parquet`，不读取 clean CSV 参与特征计算。

| dataset_split | history_start | history_end | label_date（不参与统计） |
| --- | --- | --- | --- |
| `train` | 2025-11-18 | 2025-12-07 | 2025-12-08 |
| `validation` | 2025-12-09 | 2025-12-14 | 2025-12-15 |
| `test` | 2025-12-16 | 2025-12-17 | 2025-12-18 |

阶段二所有统计只能使用 `history_start` 至 `history_end` 闭区间内的数据，必须满足 `history_end < label_date`。阶段二不生成 `label`，不训练模型，不采样，也不执行最终入模特征筛选。

## 2.3 阶段三未来 1 天购买标签

阶段三标签生成读取阶段二用户—商品初版特征宽表和阶段一标准清洗 Parquet。样本粒度为 `dataset_split + user_id + item_id`，仅在用户—商品对于对应 `label_date` 发生 `behavior_type=4` 的购买行为时生成 `label=1`，否则生成 `label=0`。

标签日固定为训练集 `2025-12-08`、验证集 `2025-12-15`、测试集 `2025-12-18`。标签日数据只能用于目标匹配，不能参与特征计算。

带标签宽表必须按既有 `dataset_split` 固定划分为训练、验证和测试集，分别对应 `2025-11-18` 至 `2025-12-07`、`2025-12-09` 至 `2025-12-14`、`2025-12-16` 至 `2025-12-17` 的特征窗口；禁止随机划分。

特征预处理仅使用训练集拟合规则：数值缺失值用训练集均值填充并以训练集统计量标准化，类别字段按训练集编码，验证集和测试集只能复用这些规则。`user_id`、`item_id`、`category_id` 只用于追踪，`label` 只作为监督目标；二者均不得作为模型输入。`dataset_split`、窗口字段和直接时间戳字段不得作为模型输入。

特征筛选仅基于预处理后的训练集确定：检查缺失率、低方差、高相关、非有限值、异常值比例和疑似未来信息泄露字段，输出最终入模特征清单及删除原因。验证集和测试集只应用训练集确定的清单。

类别不平衡处理仅作用于训练集：保留原始未采样训练集，构建 SMOTE 过采样和随机欠采样训练集，并输出不改变样本数的类别权重配置。验证集和测试集必须保持原始分布。是否采用某一方案由后续模型在验证集上的结果决定；阶段三不训练模型或做模型评估。

## 3. 产物与职责

| 产物 | 固定位置 | 说明 |
| --- | --- | --- |
| 原始 CSV | `data/raw/user_behavior_processed.csv` | Google Drive 下载，只读源文件 |
| 原始 Parquet | `data/raw/user_behavior_processed.parquet` | CSV 的类型化副本，行数和列顺序保持一致 |
| SQLite | `database/taobao_user_behavior.db` | 任务要求的空表结构，不参与后续 Parquet 数据流 |
| DDL | `sql/ddl/001_create_schema.sql` | 仅定义空表结构 |
| 标准清洗数据 | `data/processed/user_behavior_clean.parquet` | 阶段二唯一正式输入 |
| 四张中间表 | `data/interim/` | 用户、商品、类目和时间维度的历史统计基础输入 |
| 八张特征表 | `data/features/` | 八类独立特征输出，不包含标签 |
| 用户—商品初版特征宽表 | `data/features/user_item_feature_wide.parquet` | 合并八张特征表，不包含标签和最终入模筛选 |
| 带标签用户—商品特征宽表 | `data/splits/user_item_feature_wide_labeled.parquet` | 保留初版特征宽表全部字段并新增未来 1 天购买标签 `label` |
| 训练、验证、测试集 | `data/splits/user_item_feature_wide_labeled_{train,validation,test}.parquet` | 按固定时间窗口从带标签宽表确定性划分的三个数据集 |
| 预处理训练、验证、测试集 | `data/splits/user_item_feature_wide_labeled_{train,validation,test}_preprocessed.parquet` | 使用训练集拟合规则预处理后的三个数据集 |
| 筛选后训练、验证、测试集 | `data/splits/user_item_feature_wide_labeled_{train,validation,test}_preprocessed_selected.parquet` | 应用训练集最终特征清单后的三个数据集 |
| 不平衡处理训练方案 | `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected_{baseline,smote,undersampled}.parquet` | 原始基线、SMOTE 和欠采样三种训练数据版本 |
| 阶段一测试结果 | `reports/stage1/stage1_test_results.md` | 阶段一功能与性能测试结果 |
| 阶段二测试结果 | `reports/stage2/` | 中间表、八张特征表和初版宽表的功能、性能及质量记录 |
| 阶段三标签报告 | `reports/stage3/` | 标签统计、数据泄露检查和功能、性能测试记录 |

CSV→Parquet 的职责仅为：校验表头、按声明类型解析、分块写入、返回行数/文件大小/耗时。它不负责修正缺失值、过滤异常、删除重复、排序、导入 SQLite 或生成分析指标。

阶段二通过 `scripts/build_stage2_intermediate_tables.py` 生成四张中间表，通过 `scripts/build_stage2_feature_tables.py` 统一生成八张特征表，再通过 `scripts/build_user_item_feature_wide.py` 合成用户—商品初版特征宽表。宽表口径见 `docs/stage2/user_item_feature_wide.md`。

阶段三通过 `scripts/generate_purchase_labels.py` 为初版宽表追加未来 1 天购买标签，再通过 `scripts/split_labeled_datasets.py` 按固定时间窗口生成训练、验证和测试集，通过 `scripts/preprocess_features.py` 拟合训练集预处理规则并复用到验证和测试集，通过 `scripts/select_features.py` 确定最终入模特征清单，最后通过 `scripts/prepare_class_imbalance.py` 准备不同的训练不平衡处理方案。标签口径见 `docs/stage3/purchase_label_generation.md`，数据集划分口径见 `docs/stage3/dataset_split_generation.md`，预处理口径见 `docs/stage3/feature_preprocessing.md`，筛选口径见 `docs/stage3/feature_selection.md`，类别不平衡口径见 `docs/stage3/class_imbalance.md`。

## 4. SQLite 对象

- 仅创建表 `user_behavior`，包含五个核心字段；新建数据库时为 0 行。
- 后续数据处理统一使用 Parquet。

初始化脚本幂等：重复执行不会删除或覆盖已有行。阶段一初始化不导入真实数据。

## 5. 命名与版本规则

- Python、SQL、文档文件使用小写 `snake_case`。
- 可复用逻辑放在 `src/`，可执行入口放在 `scripts/`，测试放在 `tests/`。
- 后续阶段使用阶段一标准清洗数据时统一采用 `category_id`；若直接读取原始数据，则仍使用 `item_category`。
- 代码、DDL 或接口发生不兼容变更时，应在 Pull Request 中说明迁移方式。

## 6. 存储与可复现性

GitHub 不保存 `data/` 下的任何数据文件。数据统一放团队 Google Drive，并按 `docs/data/data_structure.md` 的目录说明放回本地路径。正式处理只通过 `scripts/` 中的必要入口执行。
