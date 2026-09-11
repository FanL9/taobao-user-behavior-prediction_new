# Stage 5 Local Batch Inference CLI

Entry point:

```powershell
python scripts/run_batch_inference.py
```

## Artifact validation

```powershell
python scripts/run_batch_inference.py --check-only
```

This verifies that the locally available model file, Stage 3 rules/features, and Stage 4 model-selection records agree.

## Prediction

```powershell
python scripts/run_batch_inference.py `
  --input data/features/user_item_feature_wide.parquet `
  --output outputs/predictions/purchase_probability_predictions.parquet `
  --batch-size 100000
```

Arguments:

- `--input`: required raw user-item feature `.parquet` or `.csv` file unless `--check-only` is used;
- `--output`: prediction `.parquet` or `.csv` path;
- `--config`: frozen Stage 5 inference configuration;
- `--summary-output`: JSON runtime/resource record;
- `--batch-size`: rows processed per batch;
- `--check-only`: validate artifacts without scoring.

Exit code `0` means success. Failures print `Batch inference failed: ...` to stderr and return code `1`.
