# 阶段四 Issue2 测试记录

## 功能测试

执行：

```powershell
python -m pytest tests/functional/test_model_tuning.py -q
```

验证内容：

- 16 个“训练方案 × 模型”组合均可执行统一预算的 Optuna 搜索。
- 每个组合均生成最优参数、验证集预测、运行日志和融合候选池记录。
- 运行摘要和单模型日志均标记 `test_data_read=false`。
- 命令行支持选择单个模型组合和指定试验次数。

## 性能测试

执行：

```powershell
python -m pytest tests/performance/test_model_tuning_performance.py -q
```

测试使用 3,000 行训练样本、1,200 行验证样本，对一个 LightGBM 组合运行 2 次 Optuna 试验；记录总耗时、试验次数和进程 RSS 信息。

## 最近验证结果

执行：

```powershell
python -m pytest tests/functional/test_model_tuning.py tests/performance/test_model_tuning_performance.py -q
```

结果：`3 passed`。测试集未被读取。
