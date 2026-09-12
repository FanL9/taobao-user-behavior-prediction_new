# 项目完整运行流程

本文档给出 Stage 1–5 的推荐执行顺序、输入输出和关键复现约束。默认从仓库根目录执行所有命令。

## 0. 环境准备

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

数据和模型二进制不从 GitHub 获取。数据通过团队共享存储同步，模型由 Stage 4 本地产出或由团队模型产物提供。

## 1. Stage 1：数据接入、质量检查、清洗与 EDA

### 1.1 原始输入

```text
data/raw/user_behavior_processed.csv
```

### 1.2 执行

```bash
python scripts/setup_local_database.py
python scripts/convert_csv_to_parquet.py
python scripts/check_data_quality.py
python scripts/clean_user_behavior.py
```

### 1.3 主要输出

```text
data/raw/user_behavior_processed.parquet
data/processed/user_behavior_clean.csv
data/processed/user_behavior_clean.parquet
reports/stage1/data_quality_report.json
reports/stage1/cleaning_report.json
```

清洗后的正式行为数据为后续阶段唯一主输入。

### 1.4 EDA 与看板

Stage 1 EDA 统计由 `src/data/EDA_analysis.py` 生成，轻量结果由 EDA 看板读取。

```bash
streamlit run dashboards/eda/app.py
```

关键历史规模：

- 12,256,906 条行为记录
- 10,000 名用户
- 2,876,947 个商品
- 8,916 个类目

## 2. Stage 2：中间表、八张特征表与用户—商品宽表

### 2.1 四张中间表

```bash
python scripts/build_stage2_intermediate_tables.py
```

输出位于：

```text
data/interim/
```

### 2.2 八张 Stage 2 特征表

```bash
python scripts/build_stage2_feature_tables.py
```

生成：

```text
user_features.parquet
user_activity_features.parquet
user_sequence_features.parquet
item_behavior_features.parquet
item_popularity_features.parquet
category_behavior_features.parquet
time_behavior_features.parquet
conversion_chain_features.parquet
```

全部位于：

```text
data/features/
```

### 2.3 用户—商品特征宽表

```bash
python scripts/build_user_item_feature_wide.py
```

输出：

```text
data/features/user_item_feature_wide.parquet
```

正式结果为：

```text
4,384,645 rows × 64 columns
```

### 2.4 Stage 2 看板统计

```bash
python scripts/build_stage2_dashboard_stats.py
streamlit run dashboards/eda/app.py
```

看板包含行为漏斗、转移矩阵、行为深度、用户活跃度、商品热度和类目流量与购买转化的描述性关系。

## 3. Stage 3：标签、时间划分、预处理、筛选与类别不平衡

Stage 3 的关键原则是：**所有拟合规则只使用训练集；验证集和测试集只应用既定规则。**

### 3.1 未来 1 天购买标签

```bash
python scripts/generate_purchase_labels.py
```

输出：

```text
data/splits/user_item_feature_wide_labeled.parquet
reports/stage3/label_statistics.json
```

标签总规模：

```text
4,384,645 samples
2,397 positives
0.054668% positive rate
```

### 3.2 固定时间顺序划分

```bash
python scripts/split_labeled_datasets.py
```

| split | history | label date | rows | positives |
| --- | --- | --- | ---: | ---: |
| train | 2025-11-18 ~ 2025-12-07 | 2025-12-08 | 2,944,576 | 895 |
| validation | 2025-12-09 ~ 2025-12-14 | 2025-12-15 | 1,110,131 | 886 |
| test | 2025-12-16 ~ 2025-12-17 | 2025-12-18 | 329,938 | 616 |

不使用随机划分。

### 3.3 预处理

```bash
python scripts/preprocess_features.py
```

规则：

```text
reports/stage3/preprocessing_rules.json
```

训练集拟合数值缺失填充、标准化和类别编码；validation/test 不重新拟合。

### 3.4 特征筛选

```bash
python scripts/select_features.py
```

最终：

```text
54 candidate features
→ 37 final model features
```

最终清单：

```text
reports/stage3/final_model_features.json
```

### 3.5 类别不平衡版本

```bash
python scripts/prepare_class_imbalance.py
```

生成 baseline、SMOTE、undersampled 和 class-weight 四种训练方案；validation/test 保留原始分布。

## 4. Stage 4：模型训练、调参、融合、最终评估与解释

### 4.1 基线训练矩阵

```bash
python scripts/train_model_matrix.py
```

比较：

```text
4 training strategies × 4 model families
```

模型和预测产物保存在本地：

```text
models/traditional_ml/
```

轻量比较结果：

```text
reports/stage4/
```

### 4.2 Optuna 调参

```bash
python scripts/tune_primary_models.py --n-trials 5 --device gpu
```

只使用 train/validation；测试集不参与调参或选择。

### 4.3 Stacking 候选比较

```bash
python scripts/run_stacking_fusion.py --folds 5 --base-model-count 3
```

验证集领先候选：

```text
baseline__xgboost
validation AUC = 0.899130
```

### 4.4 最终测试集评估

```bash
python scripts/evaluate_final_models.py
```

最终 XGBoost：

```text
test AUC = 0.874953
test Average Precision = 0.029451
```

详细输出：

```text
reports/stage4/model_evaluation/
```

### 4.5 模型解释与错误样本

```bash
python scripts/analyze_model_interpretability.py
```

输出：

```text
reports/stage4/model_interpretability/
```

主要分析包括 SHAP、特征方向、FP/FN、冷启动和长尾商品。

### 4.6 模型看板

```bash
streamlit run dashboards/eda/app.py
```

`Model Evaluation` 区域读取 Stage 4 预计算结果，不重新训练模型。

## 5. Stage 5：本地批量购买概率推理

Stage 5 不重新训练模型、不重新拟合预处理规则，也不重新筛选特征。

### 5.1 必需文件

```text
models/traditional_ml/stacking/models/base_models.joblib
reports/stage3/preprocessing_rules.json
reports/stage3/final_model_features.json
data/features/user_item_feature_wide.parquet
```

模型 bundle key：

```text
baseline__xgboost
```

先检查：

```bash
python scripts/run_batch_inference.py --check-only
```

### 5.2 按配置运行

```bash
python scripts/run_batch_inference.py --config configs/stage5_inference.json
```

或者显式指定：

```bash
python scripts/run_batch_inference.py \
  --input data/interim/stage5_inference_test.parquet \
  --output outputs/predictions/stage5_test_predictions.parquet \
  --summary-output reports/stage5/stage5_test_inference_summary.json \
  --batch-size 50000
```

输出字段：

```text
user_id
item_id
category_id
purchase_probability
predicted_purchase
```

完整 test split 最近验证：

```text
329,938 rows
runtime = 0.609862 s
throughput = 541,004.54 rows/s
status = passed
```

其中 `purchase_probability` 是主要业务输出；默认 `0.5` 阈值仅作为固定分类字段，不应直接视为最终运营阈值。

## 6. 测试

全仓库：

```bash
python -m pytest -q
```

单阶段测试位于：

```text
tests/functional/
tests/performance/
```

各阶段最近测试证据保存在：

```text
reports/stage1/
reports/stage2/
reports/stage3/
reports/stage4/
reports/stage5/
```

## 7. 数据、模型和 GitHub 边界

### GitHub 保留

- `src/`
- `scripts/`
- `configs/`
- `dashboards/`
- `tests/`
- `docs/`
- `reports/` 中的轻量报告、JSON、CSV 和图表

### 不上传 GitHub

- `data/` 数据文件
- `models/` 模型二进制
- `outputs/` 预测结果
- 大型实验运行产物

`.gitignore` 提供保护；最终检查见：

```text
reports/stage6/repository_hygiene_report.md
```

## 8. 最终复现最小路径

如果 Stage 1–4 的数据和模型产物已经由团队共享，只需要验证 Stage 5：

```bash
python -m pip install -r requirements.txt
python scripts/run_batch_inference.py --check-only
python -m pytest tests/functional/test_batch_inference.py -q
python scripts/run_batch_inference.py --config configs/stage5_inference.json
```

如果需要从原始数据完整重建，则按本文档 Stage 1 → Stage 5 顺序执行。
