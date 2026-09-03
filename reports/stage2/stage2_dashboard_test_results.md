# Stage 2 EDA / Feature Engineering Dashboard Test Results

Date: 2026-09-03

## Scope

This record covers the Stage 2 feature-engineering additions to the existing EDA dashboard.

Implemented dashboard views:

- Stage 2 full-chain behavior conversion funnel
- Behavior transition probability matrix
- Behavior depth vs. purchase conversion
- User activity vs. purchase conversion
- Item popularity vs. purchase conversion
- Category traffic vs. purchase conversion
- Stage 2 data-source and interface panel

The dashboard consumes existing Stage 2 feature outputs/statistics only. It does not train models, evaluate models, generate labels, or clean raw data.

## Stage 2 statistics build

Command:

```bash
python scripts/build_stage2_dashboard_stats.py
```

Observed result:

- Verified all 8 Stage 2 feature tables.
- Built dashboard statistics for `test`, `train`, and `validation`.
- Generated:
  - `data/interim/stage2_dashboard/conversion_funnel.csv`
  - `data/interim/stage2_dashboard/behavior_transition_matrix.csv`
  - `data/interim/stage2_dashboard/behavior_depth_conversion.csv`
  - `data/interim/stage2_dashboard/user_activity_conversion.csv`
  - `data/interim/stage2_dashboard/item_popularity_conversion.csv`
  - `data/interim/stage2_dashboard/top_items.csv`
  - `data/interim/stage2_dashboard/category_traffic_conversion.csv`

Result: PASS

## Functional test

Command:

```bash
pytest tests/functional/test_eda_dashboard.py -q
```

Observed result:

```text
11 passed in 2.61s
```

Result: PASS

Coverage includes the existing EDA dashboard smoke path plus Stage 2 feature-table contract, projected split/column loading, aggregate dashboard statistics, transition matrix generation, and precomputed-statistics round trip.

## Performance test

Command:

```bash
pytest tests/performance/test_eda_dashboard_performance.py -q -s
```

Observed metrics:

| Metric | Result |
| --- | ---: |
| Startup time | 1.5212 s |
| Process CPU time | 1.5000 s |
| Average CPU percent | 98.60% |
| Memory before | 75.83 MB |
| Memory after | 162.45 MB |
| Memory delta | 86.62 MB |
| GPU used | No |

Observed pytest result:

```text
1 passed in 2.20s
```

Machine-readable performance output:

`outputs/performance/runtime/eda_dashboard_stage2_performance.json`

Result: PASS

## Manual dashboard validation

The Streamlit application started successfully at the local development server and rendered the Stage 2 sections without visible runtime exceptions.

Validated visible sections:

- Stage 2 Full-Chain Behavior Conversion Funnel
- Behavior Transition Probability Matrix
- Behavior Depth vs. Purchase Conversion
- User Activity vs. Purchase Conversion
- Item Popularity vs. Purchase Conversion
- Category Traffic vs. Purchase Conversion

Result: PASS

## Interpretation note

The Stage 2 funnel is based on aggregated historical behavior counts. `Favorite` and `Cart` are not guaranteed to form strict nested sequential populations, so a later listed behavior can have a larger count than an earlier listed behavior. The PV-relative percentages are the primary descriptive ratios; the chart must not be interpreted as a strict user-level sequential funnel.

## Overall result

PASS

The Stage 2 EDA / feature-engineering dashboard implementation, functional tests, statistics generation, and startup performance test completed successfully.
