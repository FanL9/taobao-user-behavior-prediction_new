# 阶段四 Issue2.5 测试记录

## 功能测试

执行：

```powershell
python -m pytest tests/functional/test_model_stacking.py -q
```

验证 OOF 预测可生成、元模型可训练、验证集比较表和 Issue3 候选清单可输出，并确认运行记录标记 `test_data_read=false`。

## 性能测试

执行：

```powershell
python -m pytest tests/performance/test_model_stacking_performance.py -q
```

使用 2,000 行训练样本、600 行验证样本，对两个基模型执行三折 OOF Stacking，记录总耗时和进程 RSS。

## 最近验证结果

执行：

```powershell
python -m pytest tests/functional/test_model_stacking.py tests/performance/test_model_stacking_performance.py -q
```

结果：`3 passed`。测试集未被读取。
