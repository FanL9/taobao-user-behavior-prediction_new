# 传统机器学习训练结果

本目录按“训练方案 × 模型”保存本地结果，共 16 个模型：

```text
traditional_ml/
├── baseline/       # 原始未采样训练集
├── smote/          # SMOTE 过采样训练集
├── undersampled/   # 欠采样训练集
└── class_weight/   # 原始训练集 + 类别权重
```

每个方案目录结构完全相同：

- `models/`：Logistic Regression、Random Forest、XGBoost、LightGBM 四个模型文件；
- `validation_predictions/`：四个模型的验证集预测；
- `test_predictions/`：四个模型的测试集预测；
- `run_logs/`：四个模型的参数、指标和运行日志。

模型、预测和日志是本地生成文件，不提交到 Git。四方案的统一指标见 `reports/stage4/baseline_model_performance_comparison.csv`。
