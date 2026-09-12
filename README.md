# 淘宝用户行为分析与购买概率预测

本项目基于淘宝用户行为记录，完成从数据质量检查、清洗、EDA、特征工程、标签与数据集构建、传统机器学习建模与调优、模型解释，到本地批量购买概率推理的完整链路。

项目当前已完成 **Stage 1–5**，最终模型为 `baseline__xgboost`。GitHub 仅保存代码、配置、文档和轻量结果；大数据、模型二进制和预测文件均保留在本地/团队共享存储中。

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| 原始行为记录 | 12,256,906 |
| 用户数 | 10,000 |
| 商品数 | 2,876,947 |
| 用户—商品特征宽表 | 4,384,645 行 × 64 列 |
| 总正样本 | 2,397 |
| 总正样本率 | 0.054668% |
| 最终入模特征 | 37 |
| 最终模型 | `baseline__xgboost` |
| 验证集 AUC | 0.899130 |
| 测试集 AUC | 0.874953 |
| 测试集 Average Precision | 0.029451 |
| 全量 test 批量推理 | 329,938 行 |
| 推理吞吐 | 541,004.54 rows/s |

> 数据高度类别不平衡，因此 AUC、Average Precision、概率排序和业务阈值比默认 0.5 分类准确率更有解释价值。最终推理的核心输出是 `purchase_probability`。

## 项目流程

```text
Stage 1
原始行为数据
  → 数据质量检查
  → 标准清洗
  → EDA / 看板

Stage 2
清洗数据
  → 四张中间表
  → 八张特征表
  → 用户-商品 64 列初版特征宽表
  → 特征工程关系看板

Stage 3
特征宽表
  → 未来 1 天购买标签
  → 时间顺序 train / validation / test
  → 训练集拟合预处理
  → 54 → 37 特征筛选
  → 类别不平衡数据版本

Stage 4
37 个最终特征
  → 4 种不平衡策略 × 4 类模型
  → Optuna 调参
  → Stacking 候选比较
  → 测试集最终评估
  → SHAP / 错误样本 / 冷启动与长尾分析
  → 模型评估看板

Stage 5
既定模型 + Stage 3 规则
  → 本地批量推理
  → 用户-商品购买概率
```

完整运行顺序见 [`docs/project_workflow.md`](docs/project_workflow.md)。

## 环境

建议使用 Python 3.10+。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

`scikit-learn` 在 `requirements.txt` 中固定为与当前序列化模型一致的 `1.8.0`，避免跨版本反序列化警告。

## 数据与模型文件

### 数据

原始和派生数据均位于 `data/`，通过团队共享存储传递，不提交 GitHub。

阶段一原始输入：

```text
data/raw/user_behavior_processed.csv
```

Stage 5 默认推理输入：

```text
data/features/user_item_feature_wide.parquet
```

### 模型

最终推理使用：

```text
models/traditional_ml/stacking/models/base_models.joblib
```

模型 bundle key：

```text
baseline__xgboost
```

模型二进制受 `.gitignore` 保护，不提交 GitHub。进行 Stage 5 推理前，需要从团队模型产物或完成 Stage 4 的本地环境取得该文件。

## 快速运行

### Stage 1：数据接入、检查与清洗

```bash
python scripts/setup_local_database.py
python scripts/convert_csv_to_parquet.py
python scripts/check_data_quality.py
python scripts/clean_user_behavior.py
```

### Stage 2：中间表、特征表与宽表

```bash
python scripts/build_stage2_intermediate_tables.py
python scripts/build_stage2_feature_tables.py
python scripts/build_user_item_feature_wide.py
python scripts/build_stage2_dashboard_stats.py
```

EDA/特征看板：

```bash
streamlit run dashboards/eda/app.py
```

### Stage 3：标签、划分、预处理与特征筛选

```bash
python scripts/generate_purchase_labels.py
python scripts/split_labeled_datasets.py
python scripts/preprocess_features.py
python scripts/select_features.py
python scripts/prepare_class_imbalance.py
```

### Stage 4：模型训练、调参、融合与评估

```bash
python scripts/train_model_matrix.py
python scripts/tune_primary_models.py --n-trials 5 --device gpu
python scripts/run_stacking_fusion.py --folds 5 --base-model-count 3
python scripts/evaluate_final_models.py
python scripts/analyze_model_interpretability.py
```

模型评估看板与 Stage 2 EDA 共用 Streamlit 入口：

```bash
streamlit run dashboards/eda/app.py
```

### Stage 5：本地批量推理

先检查模型、预处理规则和 37 个最终特征是否齐全：

```bash
python scripts/run_batch_inference.py --check-only
```

按配置执行：

```bash
python scripts/run_batch_inference.py --config configs/stage5_inference.json
```

也可以显式指定输入输出：

```bash
python scripts/run_batch_inference.py \
  --input data/interim/stage5_inference_test.parquet \
  --output outputs/predictions/stage5_test_predictions.parquet \
  --summary-output reports/stage5/stage5_test_inference_summary.json \
  --batch-size 50000
```

预测字段说明见 [`docs/stage5/prediction_output_schema.md`](docs/stage5/prediction_output_schema.md)。

## 测试

运行全仓库测试：

```bash
python -m pytest -q
```

也可按阶段运行 `tests/functional/` 与 `tests/performance/` 中的测试。

Stage 5 最近验证：

```text
Functional: 5 passed
Performance: 1 passed
Full test inference: 329,938 rows, status=passed
```

## 关键业务发现

- 浏览行为占全部行为的 **94.24%**，购买行为仅占 **0.98%**，流量与购买之间存在显著数量落差。
- 晚间 **20–22 点**是行为最集中的时段；浏览/收藏在 21 点达到峰值，加购/购买在 22 点达到峰值。
- 最终模型的重要信号集中在**行为新鲜度、最近行为序列、商品加购、用户历史购买与用户/商品活跃度**等特征。
- 测试集正样本率仅 **0.1867%**。默认阈值 0.5 对最终 XGBoost 过于保守，实际运营更适合使用购买概率排序、Top-K 人群或基于触达成本/收益确定阈值。
- 长尾商品样本上的识别能力明显弱于非长尾商品，适合增加单独的长尾召回/探索策略，而不是仅依赖当前分类阈值。

完整数据、模型结论和限制见 [`reports/stage6/final_project_report.md`](reports/stage6/final_project_report.md)。

## 仓库结构

```text
configs/       运行配置
dashboards/    Streamlit EDA / 模型看板
data/          本地大数据目录，仅保留目录骨架
docs/          口径、接口与运行文档
models/        本地模型产物，仅保留说明/目录骨架
outputs/       本地预测与运行产物
reports/       各阶段轻量统计、测试记录与最终报告
scripts/       CLI / pipeline 入口
src/           核心实现
tests/         功能与性能测试
```

## 最终文档入口

- [项目定义与统一口径](docs/project_definition.md)
- [完整项目运行流程](docs/project_workflow.md)
- [Stage 2 特征看板说明](docs/stage2/feature_dashboard_guide.md)
- [Stage 4 模型评估与解释看板](docs/stage4/model_dashboard.md)
- [Stage 5 本地批量推理](docs/stage5/batch_inference.md)
- [Stage 5 预测字段说明](docs/stage5/prediction_output_schema.md)
- [最终项目报告](reports/stage6/final_project_report.md)
- [项目复盘](reports/stage6/project_retrospective.md)
- [GitHub 仓库与大文件检查](reports/stage6/repository_hygiene_report.md)

## GitHub 文件规则

以下内容不上传 GitHub：

- `data/` 下的 CSV / Parquet 等数据文件
- `models/` 下的模型二进制
- `outputs/` 下的预测结果
- 本地运行缓存、日志和临时压缩包

仓库只保留能够复现流程的代码、配置、接口文档、测试与轻量报告。
