# Data 目录说明

`data/` 下的所有数据文件统一存放在团队 Google Drive，GitHub 只保留目录中的 `.gitkeep`。

| 目录 | 简短说明 |
| --- | --- |
| `data/raw/` | 原始 CSV 和由其直接转换的原始 Parquet |
| `data/interim/` | 数据处理过程中的临时中间结果 |
| `data/processed/` | 完成清洗和标准化后的数据 |
| `data/features/` | 特征工程生成的特征数据 |
| `data/samples/` | 调试、抽样和快速验证使用的数据 |
| `data/splits/` | 训练集、验证集和测试集 |
| `data/predictions/` | 模型预测结果 |

阶段一使用 `data/raw/user_behavior_processed.csv`，转换后输出 `data/raw/user_behavior_processed.parquet`。其他目录由后续阶段按实际任务使用。

从 Google Drive 下载文件后放回相同的相对路径；新增数据产物也上传到 Google Drive 的对应目录。禁止使用 `git add -f`、Git LFS 或修改忽略规则把 `data/` 文件上传到 GitHub。
