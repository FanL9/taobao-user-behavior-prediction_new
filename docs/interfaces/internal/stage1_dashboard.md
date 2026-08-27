# Stage 1 EDA Dashboard Internal Interface

## 1. Purpose

This document describes the internal interface between the Stage 1 EDA outputs and the Issue #5 visualization dashboard.

The dashboard is a visualization consumer of Issue #4 outputs.

It must not reimplement the complete EDA pipeline or repeatedly aggregate the full clean dataset.

---

## 2. Dashboard Entry Point

Dashboard application:

`dashboards/eda/app.py`

Start command:

`streamlit run .\dashboards\eda\app.py`

The application renders the Stage 1 exploratory analysis workspace and consumes data returned by the dashboard data loader.

---

## 3. Data Loader Interface

Data loader module:

`dashboards/eda/data_loader.py`

Primary function:

`load_eda_outputs(output_dir=EDA_OUTPUT_DIR) -> dict[str, pandas.DataFrame]`

Responsibilities:

- Validate that required Stage 1 EDA output files exist.
- Load lightweight aggregate CSV files used by the dashboard.
- Cache loaded data with Streamlit.
- Avoid loading large detailed outputs during normal dashboard startup.

Default EDA output directory:

`data/interim/`

---

## 4. Lightweight Dashboard Inputs

The returned dashboard dataset dictionary contains:

- `behavior_distribution`
- `user_purchase_summary`
- `top_10_item`
- `category_statistics`
- `top_10_category`
- `daily_behavior`
- `hourly_behavior`
- `descriptive_funnel`

Each key maps to a pandas DataFrame loaded from the corresponding Stage 1 EDA CSV output.

---

## 5. Large EDA Output Boundary

`item_statistics.csv` is a valid Stage 1 EDA output and its existence is checked by the loader.

However, it is intentionally excluded from the normal returned dashboard dataset because it contains millions of rows.

The dashboard uses `top_10_item.csv` for Top Items visualization instead.

This boundary prevents unnecessary startup memory and latency overhead.

---

## 6. Dashboard Data Contracts

### Behavior Distribution

Expected fields include:

- `behavior_name`
- `behavior_count`
- `percentage`

### User Purchase Summary

Expected fields include:

- `purchase_count`
- `purchase_users`
- `non_purchase_users`
- `repeat_purchase_users`

### Top Items

Expected fields include:

- `item_id`
- `buy_count`

### Top Categories

Expected fields include:

- `category_id`
- `buy_count`
- `buy_percentage`

### Daily Behavior

Expected fields include:

- `behavior_date`
- `pv_count`
- `fav_count`
- `cart_count`
- `buy_count`

### Hourly Behavior

Expected fields include:

- `behavior_hour`
- `pv_count`
- `fav_count`
- `cart_count`
- `buy_count`

The expected hour domain is 0 through 23.

### Descriptive Funnel

Expected fields include:

- `stage`
- `behavior_count`
- `relative_to_pv_percentage`

Funnel percentages are descriptive ratios relative to PV and are not strict sequential conversion probabilities.

---

## 7. Trend Interaction Contract

Daily and Hourly Trend charts expose the following behavior focus options:

- All Behaviors
- PV
- FAV
- CART
- BUY

When a single behavior is selected, only that behavior is plotted and the Y axis is allowed to rescale automatically.

The same behavior type uses a consistent semantic color across Daily and Hourly Trend charts.

---

## 8. Error Contract

If one or more required EDA files are missing, `load_eda_outputs` raises `FileNotFoundError`.

The dashboard catches this condition, displays the error to the user, and stops rendering dependent sections.

Missing Stage 1 outputs should be regenerated with:

`python .\src\data\EDA_analysis.py`

---

## 9. Performance Contract

Normal dashboard startup must use Stage 1 aggregate outputs rather than the full 12,256,906-row clean dataset.

Dashboard data loading is cached with Streamlit.

Large detailed outputs must not be loaded unless a future feature explicitly requires them.

---

## 10. Dependency Boundary

Issue #3 provides the formal cleaned dataset.

Issue #4 generates the Stage 1 EDA aggregate outputs.

Issue #5 consumes those outputs for visualization.

The dashboard must preserve this dependency direction.
