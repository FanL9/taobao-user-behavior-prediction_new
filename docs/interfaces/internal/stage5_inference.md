# Stage 5 Batch Inference Internal Interface

## Contract

Implementation: `src/inference/batch_inference.py`

Configuration: `configs/stage5_inference.json`

The module is inference-only. It never calls `.fit()`, never runs feature-selection logic, and never recomputes preprocessing statistics.

## `load_inference_contract(...)`

Loads and cross-validates:

- Stage 5 fixed candidate and model artifact path;
- Stage 4 validation-only selection manifest;
- Stage 4 final evaluation summary;
- Stage 3 preprocessing rules;
- Stage 3 final model feature list.

The configured candidate must equal the Stage 4 `validation_leader`. The model binary must exist when `require_model=True`.

## `transform_inference_features(frame, contract)`

Input: raw user-item feature rows.

Output: numeric matrix with exactly the Stage 3 final model columns, in the frozen order.

Numeric rule: `(value - mean) / scale`, with the train-fitted `fill_value` for invalid/missing values.

Categorical rule: train-fitted ordinal map; unseen values use the frozen unknown-category code.

## `predict_purchase_probabilities(frame, model, contract)`

Returns tracking IDs plus:

- `purchase_probability`
- `predicted_purchase`

The probability is `predict_proba(...)[:, 1]`.

## `run_batch_inference(...)`

Streams CSV or Parquet in bounded batches, writes CSV or Parquet atomically, and records runtime/CPU/RSS/throughput metadata.

It records the following audit flags as `false`:

- `preprocessing_refit_performed`
- `feature_selection_performed`
- `model_training_performed`
