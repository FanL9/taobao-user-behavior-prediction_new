# 阶段四 Issue4：模型可解释性与错误样本分析

本步骤加载 Issue2.5 确定的验证集领先单模型及 Issue3 已生成的测试集预测结果，输出特征重要性、SHAP 解释、错误样本和模型边界统计。不会重新训练模型、调参或创建特征。

## 运行

```powershell
python scripts/analyze_model_interpretability.py
```

默认分析 `baseline__xgboost`。错误样本阈值取自 Issue3 已输出的该模型最高 F1 阈值；SHAP 使用包含全部测试集正样本的最多 10,000 行固定随机样本。冷启动和长尾判断会先用阶段三的预处理规则将计数特征还原为原始历史计数。

## 输出

- `feature_importance_top10.csv`：XGBoost 内置特征重要性 Top10。
- `shap_feature_importance.csv`、`shap_feature_importance_top10.png`：SHAP 全局重要性及图表。
- `feature_direction_analysis.csv`：实际购买样本与未购买样本上的 SHAP 方向统计。
- `error_sample_summary.csv`、`false_positive_samples.csv`、`false_negative_samples.csv`：错误数量和可追溯样例。
- `cold_start_long_tail_analysis.csv`：冷启动用户与长尾商品的固定阈值表现。
- `model_interpretability_error_analysis_report.md`：解释与错误分析报告。
