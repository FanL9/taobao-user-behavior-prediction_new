# 项目统一口径

本文档是项目字段、文件和处理边界的权威定义。实现与本文档冲突时，以本文档为准并先通过评审修改口径。

## 1. 项目目标

使用历史浏览、收藏、加购和购买事件，构建可复现的数据处理与购买预测流程。阶段一只负责数据接入基础设施，不在格式转换时进行清洗、EDA、特征计算或建模。

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

## 3. 产物与职责

| 产物 | 固定位置 | 说明 |
| --- | --- | --- |
| 原始 CSV | `data/raw/user_behavior_processed.csv` | Google Drive 下载，只读源文件 |
| 原始 Parquet | `data/raw/user_behavior_processed.parquet` | CSV 的类型化副本，行数和列顺序保持一致 |
| SQLite | `database/taobao_user_behavior.db` | 任务要求的空表结构，不参与后续 Parquet 数据流 |
| DDL | `sql/ddl/001_create_schema.sql` | 仅定义空表结构 |
| 测试结果 | `reports/stage1/stage1_test_results.md` | 阶段一功能与性能测试结果 |

CSV→Parquet 的职责仅为：校验表头、按声明类型解析、分块写入、返回行数/文件大小/耗时。它不负责修正缺失值、过滤异常、删除重复、排序、导入 SQLite 或生成分析指标。

## 4. SQLite 对象

- 仅创建表 `user_behavior`，包含五个核心字段；新建数据库时为 0 行。
- 后续数据处理统一使用 Parquet。

初始化脚本幂等：重复执行不会删除或覆盖已有行。阶段一初始化不导入真实数据。

## 5. 命名与版本规则

- Python、SQL、文档文件使用小写 `snake_case`。
- 可复用逻辑放在 `src/`，可执行入口放在 `scripts/`，测试放在 `tests/`。
- 后续阶段不得把 `item_category` 私自改为 `category_id`；如需更名，必须先更新本文件和所有接口。
- 代码、DDL 或接口发生不兼容变更时，应在 Pull Request 中说明迁移方式。

## 6. 存储与可复现性

GitHub 不保存 `data/` 下的任何数据文件。数据统一放团队 Google Drive，并按 `docs/data/data_structure.md` 的目录说明放回本地路径。正式处理只通过 `scripts/` 中的必要入口执行。
