# Stage 2 EDA Dashboard Internal Interface

## Purpose

The Stage 2 EDA dashboard consumes already-built feature engineering outputs and exposes descriptive feature-to-conversion relationships. It must not create labels, train models, or evaluate models.

## Input contract

Feature root: `data/features/`.

The canonical eight-table mapping is `STAGE2_FEATURE_FILES` in `dashboards/eda/data_loader.py`.

Routine startup should consume pre-aggregated CSV statistics from `data/interim/stage2_dashboard/`. If those files are absent, the loader may build a projected fallback for the selected split.

## Public loader functions

`get_stage2_dataset_splits(output_dir)` returns available split names.

`load_stage2_feature_inventory(output_dir)` reads lightweight Parquet metadata for all eight Stage 2 feature files.

`load_stage2_feature_table(name, output_dir, columns, dataset_split)` loads one Stage 2 table with projected columns and optional Parquet split filtering.

`build_stage2_dashboard_statistics(output_dir, dataset_split, transition_sequence_limit)` returns:

- `conversion_funnel`
- `transition_matrix`
- `behavior_depth`
- `user_activity`
- `item_popularity`
- `top_items`
- `category_traffic`

`write_stage2_dashboard_statistics(statistics_by_split, output_dir)` persists the summaries.

`load_stage2_dashboard_outputs(output_dir, dataset_split)` loads the precomputed summaries for one split.

## Failure behavior

Missing Stage 2 files raise `FileNotFoundError`; missing required fields raise `ValueError`. The Streamlit app catches these failures so the existing Stage 1 dashboard remains usable.
