# 阶段四 Issue3 测试记录

## 功能测试

```powershell
python -m pytest tests/functional/test_model_evaluation.py -q
```

验证固定 Issue2.5 模型可生成测试集预测、验证/测试指标、ROC 曲线、PR 曲线、混淆矩阵、阈值敏感性结果和技术报告。

## 性能测试

```powershell
python -m pytest tests/performance/test_model_evaluation_performance.py -q
```

记录已确定模型在测试集上的评估耗时和进程资源信息。

## 最近验证结果

执行：

```powershell
python -m pytest tests/functional/test_model_evaluation.py tests/performance/test_model_evaluation_performance.py -q
```

结果：`3 passed`。测试集仅在评估函数内读取一次，未参与调参或候选方案选择。
