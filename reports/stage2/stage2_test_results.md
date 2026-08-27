# 阶段二中间表功能与性能测试结果

测试日期：2026-08-27  
测试环境：Windows 11、Python 3.12.6、pandas 2.3.3、PyArrow 24.0.0、psutil 7.1.3。

## 功能测试

执行命令：`python -m pytest tests/functional/test_stage2_intermediate_tables.py -q`

结果：`3 passed in 1.07s`

| 验证内容 | 结果 |
| --- | --- |
| 只生成用户、商品、类目、时间四张中间表 | 通过 |
| 三个历史窗口与标签日严格分离 | 通过 |
| 各表主键唯一 | 通过 |
| 四类行为数之和等于总行为数 | 通过 |
| 每个窗口各表总量与历史输入一致 | 通过 |
| 不包含标签、用户—商品表或最终特征表 | 通过 |
| clean 时间派生字段不一致时拒绝生成 | 通过 |

## 性能测试

执行命令：`python -m pytest tests/performance/test_stage2_intermediate_tables_performance.py -q`

测试使用临时生成的 50,000 行 clean 格式 Parquet，通过正式生成接口落盘四张中间表。

| 指标 | 结果 |
| --- | ---: |
| 运行时间 | 0.347858 秒 |
| 进程 CPU 时间 | 0.312500 秒 |
| 进程 RSS 峰值 | 148,578,304 bytes |
| GPU | 未使用 |

## 全量生成验证

正式 clean Parquet 生成耗时 111.804 秒，四张表直接输出至 `data/interim/`：

| 中间表 | 行数 | 文件大小 |
| --- | ---: | ---: |
| 用户 | 27,235 | 462,197 bytes |
| 商品 | 3,050,688 | 38,536,230 bytes |
| 类目 | 20,925 | 346,075 bytes |
| 时间 | 672 | 33,530 bytes |

三个窗口的事件总数分别为：训练 7,506,554、验证 2,809,856、测试 779,876。四张表均通过标签日排除检查。正式 Parquet 属于 `data/` 数据文件，不上传 GitHub。
