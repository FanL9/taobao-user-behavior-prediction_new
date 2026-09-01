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
