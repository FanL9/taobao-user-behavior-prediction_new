# 阶段四：传统机器学习基线训练记录

## 本次运行

执行命令：

```powershell
python scripts/train_model_matrix.py
```

训练策略为 `class_weight`：训练数据使用原始未采样的筛选后训练集，并使用仅由训练集生成的类别权重；验证集和测试集均保持原始分布。模型输入为 Issue4 确定的 37 个特征，追踪字段、标签和 `is_synthetic` 均已排除。测试集仅在模型拟合完成后记录预测和指标，没有参与调参或模型选择。

| 模型 | 拟合时间（秒） | 验证集 AUC | 验证集 Precision | 验证集 Recall | 验证集 F1 | 验证集 LogLoss | 测试集 AUC | 测试集 Precision | 测试集 Recall | 测试集 F1 | 测试集 LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 56.921975 | 0.856414 | 0.004470 | 0.671558 | 0.008881 | 0.419360 | 0.831024 | 0.012190 | 0.100649 | 0.021747 | 0.133343 |
| Random Forest | 119.454990 | 0.848801 | 0.011257 | 0.048533 | 0.018275 | 0.135893 | 0.714929 | 0.000000 | 0.000000 | 0.000000 | 0.136395 |
| XGBoost | 18.815613 | 0.849602 | 0.009297 | 0.428894 | 0.018199 | 0.130125 | 0.750830 | 0.011074 | 0.139610 | 0.020520 | 0.130630 |
| LightGBM | 12.541014 | 0.714533 | 0.001804 | 0.772009 | 0.003599 | 11.783813 | 0.556911 | 0.002133 | 0.910714 | 0.004256 | 27.478674 |

每个方案的四个模型均生成验证集与测试集预测，分别包含 1,110,131 行和 329,938 行。完整指标见 `baseline_model_performance_comparison.csv`；各模型的固定参数、输出路径、运行时间及进程 RSS 增量见 `models/traditional_ml/<方案>/run_logs/*_run.json` 和对应的训练汇总 JSON。

Logistic Regression 在固定的 `max_iter=200` 上达到迭代上限，收敛警告已保留在各方案的 `models/traditional_ml/<方案>/run_logs/training_stderr.log`。本步骤不改变参数进行深度调优。

## 四种类别不平衡方案补充运行

在既有 `class_weight` 结果基础上，已补跑 `baseline`（原始未采样）、`smote` 和 `undersampled`。四种方案均训练 Logistic Regression、Random Forest、XGBoost 和 LightGBM，使用相同的最终特征、固定参数、原始验证集和原始测试集。

- 汇总表：`reports/stage4/baseline_model_performance_comparison.csv`
- 单方案表：`baseline_model_comparison_baseline.csv`、`baseline_model_comparison_smote.csv`、`baseline_model_comparison_undersampled.csv`，以及已有的 `baseline_model_comparison.csv`（类别权重）。
- 汇总表包含 32 行：4 个方案 × 4 个模型 × 验证/测试 2 个数据集。
- 四种方案的模型、预测和日志均位于 `models/traditional_ml/<方案>/`；每种方案包含 4 个模型、4 个验证预测、4 个测试预测和 4 份模型日志。

该汇总仅记录可比较的基线结果，不据此作最终模型或最终采样方案选择。

## 功能测试记录

```powershell
python -m pytest tests/functional/test_training_matrix.py -q
# 2 passed
```

覆盖内容：四种模型训练、模型序列化、验证集/测试集预测字段与行数、指标比较表、参数/日志写入、测试集未参与选择，以及命令行单模型调用。

## 性能测试记录

```powershell
python -m pytest tests/performance/test_training_matrix_performance.py -q -s
# 1 passed
```

性能用例使用 12,000 行训练集、5,000 行验证集和 3,000 行测试集运行 Logistic Regression 基线；本地实测总耗时为 0.175805 秒，模型拟合耗时为 0.078355 秒。完整数据上的四模型训练总耗时为 224.229912 秒；运行日志同时保留各模型的进程 RSS 增量，未使用 GPU。
