# 阶段五：本地批量推理

## 目标

基于阶段三已经冻结的预处理规则和最终 37 个入模特征，以及阶段四已经完成验证/测试评估的最终候选模型，提供本地批量购买概率推理能力。

本阶段严格禁止：

- 重新训练模型；
- 重新调参；
- 重新拟合缺失值、标准化或类别编码规则；
- 重新执行特征筛选；
- 使用测试集重新选择模型。

## 最终模型

阶段四 `reports/stage4/stacking_fusion/issue3_candidate_manifest.json` 记录的验证集领先方案为：

`baseline__xgboost`

阶段四最终测试评估实际从以下模型 bundle 读取该候选：

`models/traditional_ml/stacking/models/base_models.joblib`

bundle key：

`baseline__xgboost`

阶段五配置固定在：

`configs/stage5_inference.json`

推理代码会再次检查配置中的候选是否仍与阶段四验证集 leader 一致，并确认该候选包含在阶段四最终评估候选列表中。

> 模型二进制文件受 `.gitignore` 管理，不上传 GitHub。运行机器必须从阶段四产物保存位置获取**同一个已评估模型文件**，不能通过重新训练替代。

## 阶段三规则复用

推理只读取：

- `reports/stage3/preprocessing_rules.json`
- `reports/stage3/final_model_features.json`

当前最终入模特征数：37。

原始输入中的数值特征严格使用训练集已经确定的 `fill_value`、`mean` 和 `scale`；类别特征严格使用训练集已经确定的 ordinal mapping，未知类别编码为 `-1`。

## 输入

推荐输入为阶段二用户—商品特征宽表或同 schema 的待预测文件：

`data/features/user_item_feature_wide.parquet`

推理不需要 `label`。输入可以包含额外字段，但必须包含：

- `user_id`
- `item_id`
- `category_id`
- 最终 37 个入模特征对应的原始字段

支持 `.parquet` 和 `.csv`。

## 输出字段

| 字段 | 含义 |
| --- | --- |
| `user_id` | 用户标识，仅用于结果追踪，不进入模型 |
| `item_id` | 商品标识，仅用于结果追踪，不进入模型 |
| `category_id` | 商品类目标识，仅用于结果追踪，不进入模型 |
| `purchase_probability` | 最终模型输出的正类（购买）概率，范围 `[0, 1]` |
| `predicted_purchase` | 使用阶段四固定阈值 `0.5` 得到的二值预测；主要交付仍是概率 |

默认输出：

`outputs/predictions/purchase_probability_predictions.parquet`

运行摘要默认写入：

`outputs/performance/runtime/stage5_batch_inference_summary.json`

## 运行

先检查模型和规则是否齐全：

```powershell
python scripts/run_batch_inference.py --check-only
```

批量推理：

```powershell
python scripts/run_batch_inference.py `
  --input data/features/user_item_feature_wide.parquet `
  --output outputs/predictions/purchase_probability_predictions.parquet
```

可通过 `--batch-size` 控制每批行数，默认 `100000`。

## 验收

功能测试：

```powershell
pytest tests/functional/test_batch_inference.py -q
```

性能测试：

```powershell
pytest tests/performance/test_batch_inference_performance.py -q -s
```
