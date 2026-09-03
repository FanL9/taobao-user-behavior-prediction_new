# Stage 2 Feature Engineering Dashboard Guide

## Scope

The Stage 2 section extends the existing EDA dashboard with feature-engineering relationship analysis only. It does **not** train models, evaluate models, generate labels, or clean raw data.

Dashboard entry point: `dashboards/eda/app.py`

Stage 2 data interface: `dashboards/eda/data_loader.py`

## Eight Stage 2 data sources

| Logical table | File | Dashboard use |
| --- | --- | --- |
| user_basic | `user_features.parquet` | Contract/inventory |
| user_activity | `user_activity_features.parquet` | Behavior depth and user activity conversion |
| user_sequence | `user_sequence_features.parquet` | Behavior transition matrix |
| item_behavior | `item_behavior_features.parquet` | Contract/inventory |
| item_popularity | `item_popularity_features.parquet` | Item popularity conversion |
| category_behavior | `category_behavior_features.parquet` | Category traffic conversion |
| time_behavior | `time_behavior_features.parquet` | Contract/inventory |
| conversion_chain | `conversion_chain_features.parquet` | Full-chain behavior funnel |

The dashboard prefers lightweight statistics under `data/interim/stage2_dashboard/` so it does not load every multi-million-row feature table at Streamlit startup.

## Build lightweight dashboard statistics

From the repository root:

```bash
python scripts/build_stage2_dashboard_stats.py
```

The default scans every sequence when building the transition matrix. For development only:

```bash
python scripts/build_stage2_dashboard_stats.py --transition-sequence-limit 250000
```

If the statistics files do not exist, the dashboard uses a projected fallback and caps the interactive transition scan at 250,000 sequences.

## Chart definitions

- **Stage 2 full-chain funnel**: sums `item_pv_count`, `item_fav_count`, `item_cart_count`, and `item_buy_count` from `conversion_chain_features.parquet`.
- **Transition matrix**: counts adjacent pairs from `last_10_behavior_sequence` and row-normalizes by source behavior.
- **Behavior depth vs. purchase conversion**: defines depth as user `event_count`, creates up to five quantile bands, and calculates aggregated BUY/PV.
- **User activity vs. purchase conversion**: groups by existing `activity_level` and calculates aggregated BUY/PV.
- **Item popularity vs. purchase conversion**: groups by `item_total_count_rank` quantile bands and calculates aggregated BUY/PV.
- **Category traffic vs. purchase conversion**: groups by `category_total_count` quantile bands and calculates aggregated BUY/PV.

The funnel ratios are descriptive historical behavior-count ratios, not strict user-level sequential conversion probabilities.

## Loader interfaces

- `get_stage2_dataset_splits(output_dir)`
- `load_stage2_feature_inventory(output_dir)`
- `load_stage2_feature_table(name, output_dir, columns, dataset_split)`
- `build_stage2_dashboard_statistics(output_dir, dataset_split, transition_sequence_limit)`
- `write_stage2_dashboard_statistics(statistics_by_split, output_dir)`
- `load_stage2_dashboard_outputs(output_dir, dataset_split)`

## Run

```bash
streamlit run dashboards/eda/app.py
```

## Functional test

```bash
pytest tests/functional/test_eda_dashboard.py -q
```

## Performance test

```bash
pytest tests/performance/test_eda_dashboard_performance.py -q -s
```

Performance results are written to:

`outputs/performance/runtime/eda_dashboard_stage2_performance.json`
