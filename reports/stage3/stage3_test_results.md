# 阶段三未来 1 天购买标签与数据集划分测试记录

测试日期：2026-09-01

## 全量生成结果

执行命令：

```powershell
python scripts/generate_purchase_labels.py
```

输出样本：`data/splits/user_item_feature_wide_labeled.parquet`（本地生成，不上传 GitHub）

机器可读统计与泄露检查：`reports/stage3/label_statistics.json`

| 数据集 | 样本数 | 正样本数 | 负样本数 | 正样本比例 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,944,576 | 895 | 2,943,681 | 0.030395% |
| validation | 1,110,131 | 886 | 1,109,245 | 0.079810% |
| test | 329,938 | 616 | 329,322 | 0.186702% |
| 总计 | 4,384,645 | 2,397 | 4,382,248 | 0.054668% |

输出共 65 个字段，完整保留阶段二宽表的 64 个字段，并仅在末尾新增 `int8` 类型的 `label`。

## 数据泄露检查

| 检查内容 | 结果 |
| --- | --- |
| 三个数据集的历史窗口元数据与固定口径一致 | 通过 |
| 所有样本满足 `history_end < label_date` | 通过 |
| 特征窗口与标签窗口重叠行数 | 0 |
| 标签窗口参与特征计算 | 否 |
| 被修改的特征字段 | 无 |
| 标签事件读取字段 | `user_id`、`item_id`、`behavior_type`、`behavior_date` |

## 功能测试记录

```powershell
python -m pytest tests/functional/test_purchase_labels.py -q
# 3 passed
```

功能测试覆盖精确标签日购买匹配、非标签日行为排除、正负样本统计、原特征字段和值保持不变、窗口元数据校验和命令行输出。

## 性能测试记录

50,000 行临时样本测试：

```powershell
python -m pytest tests/performance/test_purchase_labels_performance.py -q -s
# 1 passed
```

全量 4,384,645 行本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 42.705222 秒 |
| 进程 CPU 时间 | 43.421875 秒 |
| 进程 RSS 峰值 | 1,450,725,376 bytes |
| 输出文件大小 | 172,684,985 bytes |
| GPU | 未使用 |

## 类别不平衡处理

执行命令：

```powershell
python scripts/prepare_class_imbalance.py
```

输出文件（本地生成，不上传 GitHub）：

- 原始训练基线、SMOTE 训练集、欠采样训练集。
- 原始分布验证集、原始分布测试集。
- `reports/stage3/class_weight_config.json`
- `reports/stage3/training_dataset_versions.json`
- `reports/stage3/class_imbalance_report.json`

| 训练方案 | 样本数 | 负样本数 | 正样本数 | 正样本比例 |
| --- | ---: | ---: | ---: | ---: |
| 原始基线 | 2,944,576 | 2,943,681 | 895 | 0.030395% |
| SMOTE | 5,887,362 | 2,943,681 | 2,943,681 | 50.000000% |
| 欠采样 | 1,790 | 895 | 895 | 50.000000% |
| 类别权重 | 2,944,576 | 2,943,681 | 895 | 0.030395% |

验证集和测试集均保留原始样本分布，未进行任何采样。类别权重使用训练集 balanced 公式，类别 `0` 权重为 `0.500152`，类别 `1` 权重为 `1645.014525`。

### 功能测试记录

```powershell
python -m pytest tests/functional/test_class_imbalance.py -q
# 3 passed
```

功能测试覆盖基线保留、SMOTE 平衡、欠采样平衡、类别权重、验证/测试集不采样、合成行标识和命令行输出。

### 性能测试记录

```powershell
python -m pytest tests/performance/test_class_imbalance_performance.py -q -s
# 1 passed
```

全量本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 41.543707 秒 |
| 进程 CPU 时间 | 43.656250 秒 |
| 进程 RSS 峰值 | 1,250,488,320 bytes |
| SMOTE 文件大小 | 510,009,172 bytes |
| GPU | 未使用 |

## 特征筛选

执行命令：

```powershell
python scripts/select_features.py
```

输出文件（本地生成，不上传 GitHub）：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed_selected.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed_selected.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed_selected.parquet`

最终特征清单：`reports/stage3/final_model_features.json`；筛选报告：`reports/stage3/feature_selection_report.json`。

| 检查内容 | 结果 |
| --- | --- |
| 筛选规则拟合数据 | 仅训练集 |
| 初始候选特征数 | 54 |
| 最终入模特征数 | 37 |
| 低方差删除特征数 | 1 |
| 高相关删除特征数 | 16 |
| 缺失/非有限值删除特征数 | 0 |
| 疑似未来信息泄露字段 | 0 |
| 验证集和测试集重新拟合筛选规则 | 否 |
| 随机采样 | 未使用 |

### 功能测试记录

```powershell
python -m pytest tests/functional/test_feature_selection.py -q
# 3 passed
```

功能测试覆盖低方差、高相关、非有限值和疑似未来字段删除，训练集规则复用，追踪/标签字段保留方式，以及命令行输出。

### 性能测试记录

```powershell
python -m pytest tests/performance/test_feature_selection_performance.py -q -s
# 1 passed
```

全量 4,384,645 行本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 12.128226 秒 |
| 进程 CPU 时间 | 15.625000 秒 |
| 进程 RSS 峰值 | 1,112,248,320 bytes |
| train 文件大小 | 90,426,311 bytes |
| validation 文件大小 | 31,214,391 bytes |
| test 文件大小 | 8,066,147 bytes |
| GPU | 未使用 |

## 特征预处理

执行命令：

```powershell
python scripts/preprocess_features.py
```

输出文件（本地生成，不上传 GitHub）：

- `data/splits/user_item_feature_wide_labeled_train_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_validation_preprocessed.parquet`
- `data/splits/user_item_feature_wide_labeled_test_preprocessed.parquet`

规则文件：`reports/stage3/preprocessing_rules.json`；机器可读统计和检查：`reports/stage3/preprocessing_statistics.json`。

训练集拟合 51 个数值特征的均值填充和标准化规则，以及 3 个字符串类别特征的整数编码规则；最终每份数据集保留 3 个追踪字段、`label` 和 54 个模型候选特征，共 58 个字段。

| 检查内容 | 结果 |
| --- | --- |
| `user_id`、`item_id`、`category_id` 不进入模型特征 | 通过 |
| `label` 不进入模型特征 | 通过 |
| 分割、窗口和直接时间戳字段被移除 | 通过 |
| 缺失填充、标准化和编码仅在训练集拟合 | 通过 |
| 验证集和测试集重新拟合规则 | 否 |
| 验证集/测试集未知类别编码 | `-1` |
| 随机采样 | 未使用 |

### 功能测试记录

```powershell
python -m pytest tests/functional/test_feature_preprocessing.py -q
# 3 passed
```

功能测试覆盖训练集拟合、验证/测试集规则复用、数值缺失填充、标准化、类别未知值编码、追踪/标签/时间字段处理、输入校验和命令行输出。

### 性能测试记录

```powershell
python -m pytest tests/performance/test_feature_preprocessing_performance.py -q -s
# 1 passed
```

全量 4,384,645 行本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 43.517188 秒 |
| 进程 CPU 时间 | 53.062500 秒 |
| 进程 RSS 峰值 | 1,968,148,480 bytes |
| train 文件大小 | 116,112,430 bytes |
| validation 文件大小 | 39,933,288 bytes |
| test 文件大小 | 10,376,394 bytes |
| GPU | 未使用 |

## 训练、验证、测试集划分

执行命令：

```powershell
python scripts/split_labeled_datasets.py
```

输出文件（本地生成，不上传 GitHub）：

- `data/splits/user_item_feature_wide_labeled_train.parquet`
- `data/splits/user_item_feature_wide_labeled_validation.parquet`
- `data/splits/user_item_feature_wide_labeled_test.parquet`

机器可读统计和时间窗口检查：`reports/stage3/dataset_split_statistics.json`。

| 数据集 | 样本数 | 正样本数 | 负样本数 | 正样本比例 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,944,576 | 895 | 2,943,681 | 0.030395% |
| validation | 1,110,131 | 886 | 1,109,245 | 0.079810% |
| test | 329,938 | 616 | 329,322 | 0.186702% |
| 总计 | 4,384,645 | 2,397 | 4,382,248 | 0.054668% |

三个输出均为 65 字段，保留带标签宽表的所有字段和值。划分仅按既有 `dataset_split` 进行，未使用随机划分；三个固定窗口的元数据均通过验证，窗口重叠行数为 0。

### 功能测试记录

```powershell
python -m pytest tests/functional/test_dataset_splits.py -q
# 3 passed
```

功能测试覆盖固定训练/验证/测试窗口、原始行与字段保持不变、样本与正负样本统计、窗口错误拒绝和命令行输出。

### 性能测试记录

```powershell
python -m pytest tests/performance/test_dataset_splits_performance.py -q -s
# 1 passed
```

全量 4,384,645 行本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 23.451965 秒 |
| 进程 CPU 时间 | 29.921875 秒 |
| 进程 RSS 峰值 | 2,020,827,136 bytes |
| train 文件大小 | 120,660,794 bytes |
| validation 文件大小 | 41,363,791 bytes |
| test 文件大小 | 10,713,971 bytes |
| GPU | 未使用 |
