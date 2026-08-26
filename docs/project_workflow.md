# 项目工作流程

| 做了什么 | 输出什么 |
| --- | --- |
| 安装 `requirements.txt` 中的依赖 | 可运行的 Python 环境 |
| 从 Google Drive 下载阶段一源数据 | `data/raw/user_behavior_processed.csv` |
| 运行 `python scripts/setup_local_database.py` | `database/taobao_user_behavior.db` 空表结构 |
| 运行 `python scripts/convert_csv_to_parquet.py` | `data/raw/user_behavior_processed.parquet` |
| 运行 `python -m pytest` | 功能和性能测试结果 |

所有 `data/` 文件从 Google Drive 获取或回传，不上传 GitHub；目录用途见 `docs/data/data_structure.md`。
