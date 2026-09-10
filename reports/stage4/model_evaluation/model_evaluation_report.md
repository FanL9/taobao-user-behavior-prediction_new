# 阶段四 Issue3：模型效果综合评估

## 评估口径

- 候选方案由 Issue2.5 的验证集结果确定。
- 本步骤首次读取测试集，仅用于最终指标、曲线、混淆矩阵和阈值分析。
- 未使用测试集调参、选择模型或改变融合结构。
- Issue2.5 验证集领先方案记录为：`baseline__xgboost`。

## 验证集与测试集指标

| candidate | dataset_split | auc | accuracy | average_precision | precision | recall | f1 | logloss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline__xgboost | test | 0.874953 | 0.998133 | 0.029451 | 0.000000 | 0.000000 | 0.000000 | 0.011206 |
| undersampled__lightgbm | test | 0.864176 | 0.447466 | 0.018270 | 0.003161 | 0.938312 | 0.006301 | 1.005017 |
| stacking_logistic_regression | test | 0.861004 | 0.998133 | 0.012590 | 0.000000 | 0.000000 | 0.000000 | 0.012200 |
| undersampled__random_forest | test | 0.846724 | 0.512039 | 0.009718 | 0.003474 | 0.910714 | 0.006921 | 0.751291 |
| baseline__xgboost | validation | 0.899130 | 0.999202 | 0.024073 | 0.000000 | 0.000000 | 0.000000 | 0.005194 |
| undersampled__lightgbm | validation | 0.896693 | 0.672573 | 0.014365 | 0.002213 | 0.909707 | 0.004415 | 0.707454 |
| stacking_logistic_regression | validation | 0.896450 | 0.999202 | 0.012429 | 0.000000 | 0.000000 | 0.000000 | 0.005466 |
| undersampled__random_forest | validation | 0.885850 | 0.655444 | 0.009929 | 0.002090 | 0.904063 | 0.004171 | 0.651848 |

详细曲线、混淆矩阵和阈值敏感性结果见同目录输出文件。
