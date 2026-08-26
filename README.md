# 淘宝用户行为预测

本项目基于淘宝用户行为记录，逐阶段完成数据检查、特征工程和购买预测。当前仓库包含阶段一的数据接入、只读质量检查和标准数据清洗实现。

## 数据口径

阶段一输入文件固定为 `data/raw/user_behavior_processed.csv`，包含且仅包含以下列（顺序固定）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `time` | 字符串 | 行为发生时间，格式 `YYYY-MM-DD HH` |
| `user_id` | 整数 | 用户标识 |
| `item_id` | 整数 | 商品标识 |
| `item_category` | 整数 | 商品类目标识 |
| `behavior_type` | 整数 | 行为类型编码，允许值为 `1`、`2`、`3`、`4` |

完整约定见 [统一口径文档](docs/project_definition.md)。CSV 转 Parquet 只做格式转换和类型校验，不清洗、不去重、不修改业务值。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. 从团队 Google Drive 下载 `user_behavior_processed.csv`，放入 `data/raw/`。
2. 初始化仅含空表结构的本地数据库：

```bash
python scripts/setup_local_database.py
```

3. 分块转换 CSV，避免一次将全量数据载入内存：

```bash
python scripts/convert_csv_to_parquet.py
```

默认输出为 `data/raw/user_behavior_processed.parquet`。已有输出不会被覆盖；确认需要重建时添加 `--overwrite`。查看所有参数可运行 `python scripts/convert_csv_to_parquet.py --help`。

4. 对原始 CSV 做只读质量检查：

```bash
python scripts/check_data_quality.py
```

默认输出 `reports/stage1/data_quality_report.json`，不会修改原始 CSV，也不会生成清洗数据。


### 阶段一标准数据清洗

运行：

```bash
python scripts/clean_user_behavior.py
```

默认输入：

`data/raw/user_behavior_processed.csv`

默认输出：

- `data/processed/user_behavior_clean.csv`
- `data/processed/user_behavior_clean.parquet`
- `reports/stage1/cleaning_report.json`

清洗流程采用分块读取与 Hash 分区完成跨 Chunk 全局精确去重，不修改原始数据。

正式清洗数据需同步至团队 Google Drive 的 `taobao/data/processed/`，GitHub 不保存数据文件本体。

## 验证

```bash
python -m pytest
```

## 数据文件规则

`data/` 目录下的所有数据文件均存放在团队 Google Drive，禁止上传 GitHub；仓库只保留 `.gitkeep`。各目录用途见 [Data 目录说明](docs/data/data_structure.md)。

## 文档入口

- [统一口径](docs/project_definition.md)
- [项目工作流程](docs/project_workflow.md)
- [阶段一 Python 内部接口](docs/interfaces/internal/stage1_api.md)
- [阶段一命令行接口](docs/interfaces/external/stage1_cli.md)
- [Data 目录说明](docs/data/data_structure.md)
- [数据质量检查与后续处理规则](docs/data/data_quality_rules.md)
- [阶段一功能与性能测试结果](reports/stage1/stage1_test_results.md)
