# 阶段四 Issue5：模型评估与解释看板

模型看板已加入现有 Streamlit 应用 `dashboards/eda/app.py`，作为 `Model Evaluation` 区域。它只展示阶段四已经生成的结果文件，不训练模型、不调参，也不重新计算完整特征工程。

## 运行

先完成阶段四 Issue2、Issue2.5、Issue3 和 Issue4 的离线脚本，再运行：

```powershell
streamlit run dashboards/eda/app.py
```

## 展示内容

- 测试集核心指标和多模型性能对比。
- ROC 曲线、PR 曲线、混淆矩阵和阈值敏感性。
- 冷启动用户、长尾商品等分群效果。
- SHAP 特征重要性、方向统计、假阳性和假阴性样例。

## 数据来源

看板读取 `reports/stage4/model_evaluation/` 和 `reports/stage4/model_interpretability/` 的预计算 CSV、PNG 和 JSON 文件。
