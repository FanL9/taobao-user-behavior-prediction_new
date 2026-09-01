# 项目工作流程

| 做了什么 | 输出什么 |
| --- | --- |
| 安装 `requirements.txt` 中的依赖 | 可运行的 Python 环境 |
| 从 Google Drive 下载阶段一源数据 | `data/raw/user_behavior_processed.csv` |
| 运行 `python scripts/setup_local_database.py` | `database/taobao_user_behavior.db` 空表结构 |
| 运行 `python scripts/convert_csv_to_parquet.py` | `data/raw/user_behavior_processed.parquet` |
| 运行 `python scripts/check_data_quality.py` | `reports/stage1/data_quality_report.json`，不生成清洗数据 |
| 运行 `python scripts/clean_user_behavior.py` | `data/processed/user_behavior_clean.parquet`，供后续阶段统一读取 |
| 运行 `python scripts/build_stage2_intermediate_tables.py` | `data/interim/` 下的用户、商品、类目、时间四张中间表 |
| 运行 `python scripts/build_stage2_feature_tables.py` | `data/features/` 下的八张独立特征表 |
| 运行 `python scripts/build_user_item_feature_wide.py` | 用户—商品初版特征宽表及其质量检查报告 |
| 运行 `python scripts/generate_purchase_labels.py` | `data/splits/user_item_feature_wide_labeled.parquet` 和 `reports/stage3/label_statistics.json` |
| 运行 `python scripts/split_labeled_datasets.py` | 固定时间顺序的训练、验证、测试集和 `reports/stage3/dataset_split_statistics.json` |
| 运行阶段一、阶段二和阶段三相关功能、性能测试并记录结果 | `reports/stage1/`、`reports/stage2/` 和 `reports/stage3/` 下的测试记录 |

所有 `data/` 文件从 Google Drive 获取或回传，不上传 GitHub；目录用途见 `docs/data/data_structure.md`。
