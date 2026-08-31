# 阶段二前四张特征表功能与性能测试结果

测试日期：2026-08-31  
测试环境：Windows 11、Python 3.12.6、pandas 2.3.3、PyArrow 24.0.0、psutil 7.1.3。

## 功能测试

执行命令：

```text
python -m pytest tests/functional/test_feature_functional.py -q
```

结果：`6 passed`。

| 验证内容 | 结果 |
| --- | --- |
| 只生成用户基础、用户活跃度、用户序列和商品行为四张表 | 通过 |
| 三个历史窗口严格排除标签日及之后数据 | 通过 |
| 四张表主键唯一且三种 `dataset_split` 完整 | 通过 |
| 用户和商品行为总量与历史输入一致 | 通过 |
| 活跃等级只使用训练集 P25/P75 阈值 | 通过 |
| `history_end` 与 `label_date` 存在间隔时仍严格执行历史终点 | 通过 |
| 未提前生成商品热度和转化链路字段 | 通过 |
| clean 时间字段不一致时拒绝生成 | 通过 |

## 性能测试

执行命令：

```text
python -m pytest tests/performance/test_feature_engineering_performance.py -q -s
```

测试使用临时生成的 50,000 行 clean 格式 Parquet，通过正式接口落盘四张特征表。

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 0.673090 秒 |
| 进程 CPU 时间 | 0.562500 秒 |
| 进程 RSS 峰值 | 150,519,808 bytes |
| GPU | 未使用 |

## 全量生成验证

执行 `python scripts/build_stage2_feature_tables.py`，正式 clean Parquet 全量生成耗时 375.526 秒。

| 特征表 | 行数 | 文件大小 |
| --- | ---: | ---: |
| 用户基础行为 | 27,235 | 292,029 bytes |
| 用户活跃度 | 27,235 | 587,478 bytes |
| 用户—商品行为序列 | 4,384,645 | 39,503,902 bytes |
| 商品行为 | 3,050,688 | 32,041,254 bytes |

四张全量结果均通过主键唯一、窗口元数据和标签日排除检查。Parquet 位于 `data/features/`，不上传 GitHub。

## 阶段二回归

阶段二中间表与前四张特征表共 `11 passed`。完整仓库测试为 `48 passed, 2 failed`；两项失败均来自阶段一 EDA 看板环境缺少 `plotly`，与本次阶段二特征表改动无关。
