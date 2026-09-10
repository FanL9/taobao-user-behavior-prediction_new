# 阶段四 Issue3：模型效果综合评估

该步骤读取 Issue2.5 已确定的基模型、Stacking 元模型和候选方案清单，首次在测试集上生成最终预测和综合评估结果。它不重新训练、不调参，也不改变候选方案。

## 运行

```powershell
python scripts/evaluate_final_models.py
```

## 输出

- `final_model_performance_comparison.csv`：验证集和测试集的 AUC、Accuracy、Average Precision、Precision、Recall、F1、LogLoss。
- `test_roc_curves.png`、`test_pr_curves.png`：测试集曲线图。
- `test_confusion_matrices.png` 与 `test_confusion_matrices.csv`：阈值 0.5 的混淆矩阵。
- `test_threshold_sensitivity.csv`：0.05 至 0.95 阈值下的指标变化。
- `model_evaluation_report.md`：技术评估报告。
- `model_evaluation_summary.json`：测试集只评估一次的运行记录。

测试集预测文件只在 `models/traditional_ml/stacking/test_predictions/` 本地生成，不提交 GitHub。
