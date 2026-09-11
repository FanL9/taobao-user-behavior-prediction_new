# Stage 5 Batch Inference Test Results

Date: 2026-09-11

## Scope

This record validates the Stage 5 local batch inference workflow using the final Stage 4 candidate model and the Stage 3 preprocessing/feature contracts.

The inference workflow does **not** retrain the model, refit preprocessing rules, or rerun feature selection.

## Final inference contract

- Final candidate: `baseline__xgboost`
- Model artifact: `models/traditional_ml/stacking/models/base_models.joblib`
- Model bundle key: `baseline__xgboost`
- Preprocessing rules: `reports/stage3/preprocessing_rules.json`
- Feature list: `reports/stage3/final_model_features.json`
- Feature count: 37
- Prediction threshold: 0.5

## Functional test

Command:

```bash
python -m pytest tests/functional/test_batch_inference.py -q
```

Observed result:

```text
5 passed in 3.45s
```

Result: **PASS**

## Performance test

Command:

```bash
python -m pytest tests/performance/test_batch_inference_performance.py -q -s
```

Observed result:

```text
1 passed in 1.83s
rows: 50000
runtime_seconds: 0.122782
throughput_rows_per_second: 407224.49
process_cpu_seconds: 0.125
peak_process_rss_bytes: 181665792
```

Result: **PASS**

## Full test-split batch inference

The Stage 5 test inference input was produced from:

`data/features/user_item_feature_wide.parquet`

with `dataset_split == "test"`.

Input/output:

- Rows: 329,938
- Batch size: 50,000
- Output: `outputs/predictions/stage5_test_predictions.parquet`
- Runtime summary: `reports/stage5/stage5_test_inference_summary.json`

Observed runtime:

| Metric | Result |
| --- | ---: |
| Rows | 329,938 |
| Runtime | 0.609862 s |
| Throughput | 541,004.54 rows/s |
| Process CPU time | 15.125 s |
| RSS at start | 298,053,632 bytes |
| RSS at end | 533,438,464 bytes |
| Peak process RSS | 537,501,696 bytes |
| GPU used by inference wrapper | No |

The runtime emitted an XGBoost device-mismatch warning because the serialized booster referenced CUDA while the inference input used CPU memory. XGBoost fell back to DMatrix prediction and the inference completed successfully.

Result: **PASS**

## Prediction output

The full prediction file contains 329,938 rows and these columns:

- `user_id`
- `item_id`
- `category_id`
- `purchase_probability`
- `predicted_purchase`

The inference summary reports:

- preprocessing refit performed: `false`
- feature selection performed: `false`
- model training performed: `false`
- status: `passed`

At the configured threshold of 0.5, the full test inference produced zero positive `predicted_purchase` values. This does not invalidate the required probability output; `purchase_probability` remains the primary Stage 5 prediction field.

## Overall result

**PASS**

The Stage 5 local batch inference script, prediction output, functional validation, and performance validation completed successfully.
