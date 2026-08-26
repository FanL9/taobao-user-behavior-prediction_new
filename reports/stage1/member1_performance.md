# 阶段一 Member1 性能记录

## 2026-08-26 烟雾基准

对应测试：

```bash
python -m pytest tests/performance/test_csv_to_parquet_performance.py -q
```

环境：Windows 11、Python 3.12.6、pandas 2.3.3、PyArrow 24.0.0、psutil 7.1.3，逻辑 CPU 数 22。

| 指标 | 结果 |
| --- | ---: |
| 合成输入行数 | 10,000 |
| 运行时间 | 0.163472 秒 |
| 吞吐量 | 61,172.41 行/秒 |
| 进程 CPU 时间 | 0.093750 秒 |
| 单核口径进程 CPU 利用率 | 57.35% |
| 进程 RSS 峰值 | 91,930,624 bytes |
| Parquet 大小 | 140,605 bytes |
| GPU | 未使用 |

该结果用于验证性能采集链路能够工作，不是全量数据 SLA。CSV→Parquet 实现只调用 CPU、内存和磁盘接口，因此 GPU 状态固定记录为未使用。不同机器、磁盘和缓存状态下的数值不可直接横向比较；正式全量测试应保留相同参数和环境信息。

测试运行时的 JSON 记录写入 pytest 临时目录，不提交 GitHub。
