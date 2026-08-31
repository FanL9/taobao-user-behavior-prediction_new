# 阶段二中间表与八类特征表功能及性能测试结果

测试日期：2026-08-31
测试平台：Windows 11
Python：3.11.9
pytest：9.1.1

## 1. 测试范围

本报告覆盖阶段二：

- 四张基础统计中间表；
- 八类最终特征表；
- 时间窗口和未来信息泄露检查；
- 最终特征表真实数据生成；
- 全仓库回归测试；
- 八表统一性能测试。

正式输入数据：

`data/processed/user_behavior_clean.parquet`

数据规模：

12,256,906 条行为记录。

阶段二最终特征统一实现：

`src/features/feature.py`

最终特征统一输出：

`data/features/`

## 2. 四张中间表功能测试

执行命令：

`python -m pytest tests/functional/test_stage2_intermediate_tables.py -q`

历史测试结果：

`3 passed in 1.07s`

| 验证内容 | 结果 |
| --- | --- |
| 只生成用户、商品、类目、时间四张中间表 | 通过 |
| 三个历史窗口与标签日严格分离 | 通过 |
| 各表主键唯一 | 通过 |
| 四类行为数之和等于总行为数 | 通过 |
| 每个窗口各表总量与历史输入一致 | 通过 |
| 不包含标签、用户—商品表或最终宽表 | 通过 |
| clean 时间派生字段不一致时拒绝生成 | 通过 |

## 3. 中间表性能测试

执行命令：

`python -m pytest tests/performance/test_stage2_intermediate_tables_performance.py -q`

测试使用临时生成的 50,000 行 clean 格式 Parquet，通过正式接口生成四张中间表。

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 0.347858 秒 |
| 进程 CPU 时间 | 0.312500 秒 |
| 进程 RSS 峰值 | 148,578,304 bytes |
| GPU | 未使用 |

正式 clean Parquet 的四张中间表全量生成历史记录为 111.804 秒。

## 4. 八类特征表功能测试

执行命令：

`python -m pytest tests/functional/test_feature_functional.py -q`

结果：

`8 passed in 11.03s`

八张最终输出：

| 序号 | 特征表 | 输出文件 |
| --- | --- | --- |
| 1 | 用户基础行为特征表 | `user_features.parquet` |
| 2 | 用户活跃度特征表 | `user_activity_features.parquet` |
| 3 | 用户行为序列特征表 | `user_sequence_features.parquet` |
| 4 | 商品行为特征表 | `item_behavior_features.parquet` |
| 5 | 商品热度特征表 | `item_popularity_features.parquet` |
| 6 | 类目行为特征表 | `category_behavior_features.parquet` |
| 7 | 时间行为特征表 | `time_behavior_features.parquet` |
| 8 | 转化链路特征表 | `conversion_chain_features.parquet` |

后四张特征表重点验证结果：

| 验证内容 | 结果 |
| --- | --- |
| 商品热度表 `dataset_split + item_id` 主键唯一 | 通过 |
| 商品热度排名在各 `dataset_split` 内独立计算 | 通过 |
| 商品热度 dense rank 计算正确 | 通过 |
| 类目行为表 `dataset_split + category_id` 主键唯一 | 通过 |
| 类目行为四类行为数之和等于总行为量 | 通过 |
| 时间行为表只包含真实日期—小时组合 | 通过 |
| 时间行为日期严格位于对应历史窗口内 | 通过 |
| 转化链路 `dataset_split + item_id` 主键唯一 | 通过 |
| `buy_per_pv` 计算正确 | 通过 |
| `buy_per_fav` 计算正确 | 通过 |
| `buy_per_cart` 计算正确 | 通过 |
| 转化链路分母为 0 时结果为缺失值 | 通过 |
| 四张表均满足 `history_end < label_date` | 通过 |

## 5. 八张特征表真实生成验证

统一执行 `src/features/feature.py` 后，八张表均成功生成。

| 特征表 | 行数 | 列数 |
| --- | ---: | ---: |
| 用户基础行为特征表 | 27,235 | 11 |
| 用户活跃度特征表 | 27,235 | 19 |
| 用户行为序列特征表 | 4,384,645 | 14 |
| 商品行为特征表 | 3,050,688 | 18 |
| 商品热度特征表 | 3,050,688 | 18 |
| 类目行为特征表 | 20,925 | 15 |
| 时间行为特征表 | 672 | 15 |
| 转化链路特征表 | 3,050,688 | 13 |

真实输出均位于：

`data/features/`

大型 Parquet 输出不提交 GitHub。

## 6. 八表统一性能测试

性能脚本：

`tests/performance/test_feature_engineering_performance.py`

测试方式：

- 子进程完整执行 `src/features/feature.py`；
- 监控任务进程及其全部子进程；
- 统计进程树 CPU 使用率；
- 统计进程树 RSS 峰值；
- 验证八张最终 Parquet 均成功输出；
- GPU 不参与特征工程计算，`nvidia-smi` 结果仅作为系统环境参考。

最终结果：

| 指标 | 结果 |
| --- | ---: |
| 八表总运行时间 | 136.66 秒 |
| 外层实际耗时 | 136.791 秒 |
| 平均进程树 CPU 使用率 | 97.79% |
| 进程树峰值内存 | 6,627.76 MB |
| 系统 GPU 利用率参考值 | 34.33% |
| 任务 GPU 加速 | 未使用 |
| 子进程退出码 | 0 |
| 最终状态 | SUCCESS |

其中 GPU 利用率来自系统级 `nvidia-smi` 查询，仅用于说明测试机器当时的 GPU 环境状态，不表示阶段二特征工程任务使用了 GPU 计算。

## 7. 全仓库回归测试

执行命令：

`python -m pytest`

最终结果：

`51 passed in 39.73s`

结果说明：

- 无失败测试；
- 无测试错误；
- 阶段一已有功能未发生回归；
- 阶段二中间表测试全部通过；
- 八类最终特征表功能测试全部通过；
- 现有性能测试全部通过。

## 8. 结论

阶段二八类特征表已经完成统一生成和验证。

所有特征均遵守既定 train / validation / test 历史窗口，标签日及未来行为不会进入对应历史特征统计。

后四张特征表已经与前四张统一进入 `src/features/feature.py`、统一功能测试和统一性能测试流程。

大型真实特征 Parquet 保存在本地数据目录，不提交 GitHub。
