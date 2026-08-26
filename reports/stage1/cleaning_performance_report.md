# 阶段一数据清洗性能测试报告

## 1. 测试范围

本报告记录阶段一标准数据清洗流程的运行时间及 CPU/GPU 使用情况。

清洗实现：

- `src/data/cleaning.py`
- `src/data/cleaning_pipeline.py`
- `scripts/clean_user_behavior.py`

## 2. 自动化性能测试

测试数据规模：50,000 行。

| 指标 | 结果 |
| --- | ---: |
| 测试行数 | 50,000 |
| Wall time | 0.7192 秒 |
| Process CPU time | 0.6406 秒 |
| 逻辑 CPU 数 | 32 |
| 整机 CPU 容量平均占用 | 2.78% |
| 单核等效 CPU 利用率 | 约 89.07% |
| GPU 使用 | 未使用 |

CPU 指标基于 Python `time.process_time()` 与实际 Wall time 计算。

整机 CPU 容量平均占用按 32 个逻辑 CPU 的总计算容量折算，因此不能直接等同于任务管理器中的瞬时 CPU 百分比。

当前清洗 Pipeline 基于 pandas 与 pyarrow，为 CPU-only 流程，不调用 GPU。

## 3. 全量数据运行结果

正式清洗输入：

`data/raw/user_behavior_processed.csv`

| 指标 | 结果 |
| --- | ---: |
| 原始记录数 | 12,256,906 |
| 清洗后记录数 | 12,256,906 |
| 高频重复删除记录 | 0 |
| chunksize | 250,000 |
| Hash 分区数 | 64 |
| 全量处理耗时 | 109.04 秒 |

全量处理过程中采用分块读取与 Hash 分区方式处理跨 Chunk 重复记录，避免一次性加载全部数据。

## 4. 性能测试结论

性能测试和全量运行均成功完成。

- 50,000 行自动化性能测试通过
- 全量 12,256,906 行数据约 109.04 秒完成
- 清洗过程不依赖 GPU
- 当前性能满足阶段一批处理和后续 EDA 数据准备需求

完整项目自动化测试结果：19 passed。
