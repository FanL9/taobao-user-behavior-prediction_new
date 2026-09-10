# 阶段四 Issue2.5：模型融合与候选方案确定

该步骤读取 Issue2 的最优参数和候选池，在训练集内部生成 OOF 概率，再以这些概率作为一级特征训练 Logistic Regression 元模型。

每个 OOF 折只使用该折训练部分进行模型训练；欠采样、SMOTE 和类别权重也只在该折训练部分处理。验证集只用于比较单模型和 Stacking，测试集不会被读取。

## 运行

```powershell
python scripts/run_stacking_fusion.py --folds 5 --base-model-count 3
```

默认从 Issue2 候选池中按验证集指标选择三个不同模型类型；至少需要两个不同模型类型。

## 输出

- `stacking_validation_performance_comparison.csv`：基模型和 Stacking 的验证集指标。
- `issue3_candidate_manifest.json`：Issue3 可评估方案、基模型与元模型参数引用。
- `stacking_oof_fold_records.csv`：各基模型 OOF 折的样本量、耗时和内存记录。
- `stacking_fusion_summary.json`：融合过程、候选方案和测试集隔离记录。

OOF、验证集预测、元模型和基模型文件在 `models/traditional_ml/stacking/` 本地生成，不提交 GitHub。
