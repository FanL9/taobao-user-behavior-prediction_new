# 阶段二中间表与八张特征表测试记录

测试日期：2026-08-31

输入数据：`data/processed/user_behavior_clean.parquet`

统一实现：`src/features/feature.py`

## 测试范围

- 四张中间表：用户、商品、类目、时间维度中间表。
- 八张特征表：用户基础行为、用户活跃度、用户行为序列、商品行为、商品热度、类目行为、时间行为、转化链路。
- 时间窗口、主键、统计平衡、排名、转化率、序列和未来信息泄露检查。
- 不包含标签生成、模型训练、采样、模型评估和最终特征宽表整合。

## 八张特征表生成结果

执行命令：

```powershell
python scripts/build_stage2_feature_tables.py
```

全量生成耗时为 294.175 秒，八张 Parquet 均成功写入 `data/features/`。

| 特征表 | 行数 | 列数 | 文件大小（bytes） |
| --- | ---: | ---: | ---: |
| `user_features.parquet` | 27,235 | 10 | 292,029 |
| `user_activity_features.parquet` | 27,235 | 19 | 587,478 |
| `user_sequence_features.parquet` | 4,384,645 | 10 | 39,503,902 |
| `item_behavior_features.parquet` | 3,050,688 | 14 | 32,041,254 |
| `item_popularity_features.parquet` | 3,050,688 | 17 | 25,410,774 |
| `category_behavior_features.parquet` | 20,925 | 15 | 346,799 |
| `time_behavior_features.parquet` | 672 | 15 | 33,844 |
| `conversion_chain_features.parquet` | 3,050,688 | 13 | 25,008,722 |

以上数据文件受 Git 忽略规则限制，不上传 GitHub。

## 功能检查结果

| 检查内容 | 结果 |
| --- | --- |
| 八张表均只包含 `train`、`validation`、`test` 三个时间窗口 | 通过 |
| 所有表均满足 `history_end < label_date` | 通过 |
| 八张表各自主键无重复 | 通过 |
| 用户、商品、类目、时间表的行为总数与对应历史输入一致 | 通过 |
| `pv + fav + cart + buy = event_count` | 通过 |
| 商品热度四个 dense rank 均在各 `dataset_split` 内独立计算 | 通过 |
| 转化率公式正确，分母为 0 时结果为缺失值 | 通过 |
| 活跃度分档使用训练窗口固定 P25/P75 阈值 | 通过 |
| 时间行为记录全部位于对应历史窗口且早于标签日 | 通过 |
| 行为序列类型合法、回溯天数非负、序列长度不超过 10 | 通过 |

功能测试命令与结果：

```powershell
python -m pytest tests/functional/test_feature_functional.py -q
# 8 passed in 14.30s
```

阶段二中间表和特征表合并回归结果：

```text
12 passed in 3.46s
```

## 性能测试结果

性能测试使用 50,000 行临时 clean 格式 Parquet，通过统一接口生成八张特征表。

```powershell
python -m pytest tests/performance/test_feature_engineering_performance.py -q -s
```

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 0.981779 秒 |
| 进程 CPU 时间 | 0.890625 秒 |
| 进程 RSS 峰值 | 152,555,520 bytes |
| GPU | 未使用 |

测试结果：`1 passed in 2.36s`。

## 全仓库回归说明

执行 `python -m pytest -q` 的结果为 `49 passed, 2 failed in 66.35s`。两个失败均来自阶段一 EDA 看板测试，原因是当前环境缺少 `plotly`：

- `tests/functional/test_eda_dashboard.py::test_dashboard_app_smoke_run`
- `tests/performance/test_eda_dashboard_performance.py::test_eda_dashboard_startup_performance`

这两个失败与本次阶段二特征表代码无关；本次范围内的阶段二测试全部通过。

## 结论

八张特征表已由同一脚本生成，并通过时间窗口、主键、统计口径和未来信息泄露检查。前四张与后四张的接口、字段文档、功能测试和性能测试现已统一，不包含阶段三内容。
