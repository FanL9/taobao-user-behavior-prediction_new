# 数据质量检查性能记录

| 项目 | 记录 |
| --- | --- |
| 测试入口 | `tests/performance/test_data_quality_performance.py` |
| 测试数据 | 临时生成 10,000 行，不读取或修改原始数据 |
| 运行时间 | 0.323637 秒 |
| 进程 CPU 时间 | 0.312500 秒 |
| 进程内存峰值 | 123,305,984 bytes |
| GPU | 未使用 |

运行 `python -m pytest tests/performance/test_data_quality_performance.py -q` 可在 pytest 临时目录生成本机性能 JSON。该测试只执行质量检查，不执行清洗。
