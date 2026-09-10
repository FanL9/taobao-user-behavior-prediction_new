# 阶段四 Issue2：模型调优与结果对比

脚本使用 Optuna 对四种训练方案（baseline、SMOTE、undersampled、class_weight）下的 Logistic Regression、Random Forest、XGBoost、LightGBM 进行统一试验次数预算的参数搜索，共 16 个组合。

参数选择和模型比较只使用训练集与验证集。选择指标为验证集 AUC，并同时记录 Average Precision、Precision、Recall、F1、LogLoss。脚本不会读取测试集，也不会执行模型融合。

## 运行

```powershell
python scripts/tune_primary_models.py --n-trials 5 --device gpu
```

默认每个组合执行 5 次 Optuna 试验；`--n-trials` 对每个被选组合使用相同预算。树模型搜索上限为 200 棵树，并收窄最大深度范围以控制全量运行时间。`--device gpu` 会让 XGBoost 使用 GPU；当前环境的 LightGBM 保持 CPU。

## 输出

- `optuna_search_results.csv`：全部 Optuna 试验的参数、验证集指标、耗时和内存变化。
- `best_tuned_model_parameters.json`：16 个组合的最优参数和验证集结果。
- `tuned_validation_performance_comparison.csv`：基线与调优后的验证集指标对比。
- `model_fusion_candidate_pool.csv`：供 Issue2.5 使用的单模型候选池。
- `model_tuning_summary.json`：运行范围、试验预算、总耗时和测试集隔离记录。

模型、验证集预测和单次运行日志在 `models/traditional_ml/tuned/<训练方案>/<模型>/` 本地生成，不提交 GitHub。
