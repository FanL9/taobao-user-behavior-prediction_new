from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_OUTPUT_DIR = REPO_ROOT / "data" / "interim"
FEATURE_OUTPUT_DIR = REPO_ROOT / "data" / "features"
STAGE2_DASHBOARD_STATS_DIR = EDA_OUTPUT_DIR / "stage2_dashboard"

EDA_FILES = {
    "behavior_distribution": "behavior_distribution.csv",
    "user_purchase_summary": "user_purchase_summary.csv",
    "top_10_item": "top_10_item.csv",
    "category_statistics": "category_statistics.csv",
    "top_10_category": "top_10_category.csv",
    "daily_behavior": "daily_behavior.csv",
    "hourly_behavior": "hourly_behavior.csv",
    "descriptive_funnel": "descriptive_funnel.csv",
}

LARGE_EDA_FILES = {"item_statistics": "item_statistics.csv"}

# Complete Stage 2 feature-table contract.
STAGE2_FEATURE_FILES = {
    "user_basic": "user_features.parquet",
    "user_activity": "user_activity_features.parquet",
    "user_sequence": "user_sequence_features.parquet",
    "item_behavior": "item_behavior_features.parquet",
    "item_popularity": "item_popularity_features.parquet",
    "category_behavior": "category_behavior_features.parquet",
    "time_behavior": "time_behavior_features.parquet",
    "conversion_chain": "conversion_chain_features.parquet",
}

STAGE2_DASHBOARD_STAT_FILES = {
    "conversion_funnel": "conversion_funnel.csv",
    "transition_matrix": "behavior_transition_matrix.csv",
    "behavior_depth": "behavior_depth_conversion.csv",
    "user_activity": "user_activity_conversion.csv",
    "item_popularity": "item_popularity_conversion.csv",
    "top_items": "top_items.csv",
    "category_traffic": "category_traffic_conversion.csv",
}

BEHAVIOR_ORDER = ["pv", "fav", "cart", "buy"]


def _rate_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return float percentage ratios with NaN for zero denominators."""
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    den = den.mask(den.eq(0))
    return num.div(den).mul(100)


def _quantile_labels(bucket_count: int, *, high_first: bool = False) -> list[str]:
    """Return readable labels for one to five ordered quantile buckets."""
    labels = {
        1: ["All"],
        2: ["Low", "High"],
        3: ["Low", "Medium", "High"],
        4: ["Very Low", "Low", "High", "Very High"],
        5: ["Very Low", "Low", "Medium", "High", "Very High"],
    }[bucket_count]
    return list(reversed(labels)) if high_first else labels


@st.cache_data(show_spinner=False)
def load_eda_outputs(output_dir: str | Path = EDA_OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    """Load lightweight Stage 1 EDA aggregate outputs."""
    output_path = Path(output_dir)
    required_files = {**EDA_FILES, **LARGE_EDA_FILES}
    missing = [
        filename for filename in required_files.values()
        if not (output_path / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing EDA output files: " + ", ".join(missing))
    return {
        name: pd.read_csv(output_path / filename)
        for name, filename in EDA_FILES.items()
    }


@st.cache_data(show_spinner=False)
def get_stage2_dataset_splits(output_dir: str | Path = FEATURE_OUTPUT_DIR) -> list[str]:
    """Return available Stage 2 train/validation/test split names."""
    path = Path(output_dir) / STAGE2_FEATURE_FILES["user_activity"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 feature file: {path.name}")
    frame = pd.read_parquet(path, columns=["dataset_split"])
    return sorted(frame["dataset_split"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def load_stage2_feature_table(
    name: str,
    output_dir: str | Path = FEATURE_OUTPUT_DIR,
    columns: tuple[str, ...] | None = None,
    dataset_split: str | None = None,
) -> pd.DataFrame:
    """Load one Stage 2 table with projection and optional split pushdown filter."""
    if name not in STAGE2_FEATURE_FILES:
        raise KeyError(
            f"Unknown Stage 2 feature table {name!r}. "
            f"Expected one of: {', '.join(STAGE2_FEATURE_FILES)}"
        )

    path = Path(output_dir) / STAGE2_FEATURE_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 feature file: {path.name}")

    requested = list(columns) if columns is not None else None
    read_columns = requested
    if (
        dataset_split is not None
        and requested is not None
        and "dataset_split" not in requested
    ):
        read_columns = ["dataset_split", *requested]

    filters = [("dataset_split", "==", dataset_split)] if dataset_split else None
    frame = pd.read_parquet(path, columns=read_columns, filters=filters)
    return frame.loc[:, requested] if requested is not None else frame


@st.cache_data(show_spinner=False)
def load_stage2_feature_inventory(
    output_dir: str | Path = FEATURE_OUTPUT_DIR,
) -> pd.DataFrame:
    """Read Parquet metadata for all eight Stage 2 feature tables."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to inspect Stage 2 Parquet metadata.") from exc

    output_path = Path(output_dir)
    rows, missing = [], []
    for name, filename in STAGE2_FEATURE_FILES.items():
        path = output_path / filename
        if not path.is_file():
            missing.append(filename)
            continue
        parquet_file = pq.ParquetFile(path)
        rows.append({
            "logical_table": name,
            "feature_file": filename,
            "row_count": int(parquet_file.metadata.num_rows),
            "column_count": len(parquet_file.schema.names),
        })
    if missing:
        raise FileNotFoundError("Missing Stage 2 feature files: " + ", ".join(missing))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_stage2_feature_outputs(
    output_dir: str | Path = FEATURE_OUTPUT_DIR,
    names: tuple[str, ...] = ("user_activity", "item_popularity"),
    dataset_split: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible loader for selected raw Stage 2 tables."""
    return {
        name: load_stage2_feature_table(
            name, output_dir=output_dir, dataset_split=dataset_split
        )
        for name in names
    }


def summarize_user_activity(
    user_activity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build activity-level and behavior-depth BUY/PV summaries."""
    required = {
        "user_id", "activity_level", "event_count", "avg_daily_event_count",
        "pv_count_per_day", "buy_count_per_day",
    }
    missing = sorted(required - set(user_activity.columns))
    if missing:
        raise ValueError("Missing user activity columns: " + ", ".join(missing))

    activity = (
        user_activity.groupby("activity_level", observed=True)
        .agg(
            user_count=("user_id", "nunique"),
            avg_daily_events=("avg_daily_event_count", "mean"),
            pv_per_day=("pv_count_per_day", "sum"),
            buy_per_day=("buy_count_per_day", "sum"),
        )
        .reset_index()
    )
    activity["activity_level"] = activity["activity_level"].astype(str)
    activity["buy_to_pv_rate"] = _rate_percent(
        activity["buy_per_day"], activity["pv_per_day"]
    )

    depth = user_activity[
        ["user_id", "event_count", "pv_count_per_day", "buy_count_per_day"]
    ].copy()
    bucket_count = min(5, int(depth["event_count"].nunique()), len(depth))
    if bucket_count <= 1:
        depth["depth_band"] = "All"
    else:
        depth["depth_band"] = pd.qcut(
            depth["event_count"].rank(method="first"),
            q=bucket_count,
            labels=_quantile_labels(bucket_count),
        )

    depth_summary = (
        depth.groupby("depth_band", observed=True)
        .agg(
            user_count=("user_id", "nunique"),
            avg_event_count=("event_count", "mean"),
            pv_per_day=("pv_count_per_day", "sum"),
            buy_per_day=("buy_count_per_day", "sum"),
        )
        .reset_index()
    )
    depth_summary["depth_band"] = depth_summary["depth_band"].astype(str)
    depth_summary["buy_to_pv_rate"] = _rate_percent(
        depth_summary["buy_per_day"], depth_summary["pv_per_day"]
    )
    return activity, depth_summary


def summarize_item_popularity(
    item_popularity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build popularity-band BUY/PV summary and top-ten audit table."""
    required = {
        "item_id", "item_total_count_rank", "item_pv_count",
        "item_buy_count", "item_total_count",
    }
    missing = sorted(required - set(item_popularity.columns))
    if missing:
        raise ValueError("Missing item popularity columns: " + ", ".join(missing))

    popularity = item_popularity.copy()
    bucket_count = min(
        5, int(popularity["item_total_count_rank"].nunique()), len(popularity)
    )
    if bucket_count <= 1:
        popularity["popularity_band"] = "All"
    else:
        popularity["popularity_band"] = pd.qcut(
            popularity["item_total_count_rank"].rank(method="first"),
            q=bucket_count,
            labels=_quantile_labels(bucket_count, high_first=True),
        )

    summary = (
        popularity.groupby("popularity_band", observed=True)
        .agg(
            item_count=("item_id", "nunique"),
            total_events=("item_total_count", "sum"),
            pv_count=("item_pv_count", "sum"),
            buy_count=("item_buy_count", "sum"),
        )
        .reset_index()
    )
    summary["popularity_band"] = summary["popularity_band"].astype(str)
    summary["buy_to_pv_rate"] = _rate_percent(summary["buy_count"], summary["pv_count"])

    top_items = popularity.nsmallest(10, "item_total_count_rank")[
        [
            "item_id", "item_total_count_rank", "item_total_count",
            "item_pv_count", "item_buy_count",
        ]
    ].copy()
    top_items["buy_to_pv_rate"] = _rate_percent(
        top_items["item_buy_count"], top_items["item_pv_count"]
    )
    return summary, top_items


def summarize_category_traffic(category_behavior: pd.DataFrame) -> pd.DataFrame:
    """Bucket categories by traffic and summarize BUY/PV conversion."""
    required = {
        "category_id", "category_total_count", "category_pv_count",
        "category_buy_count",
    }
    missing = sorted(required - set(category_behavior.columns))
    if missing:
        raise ValueError("Missing category behavior columns: " + ", ".join(missing))

    category = category_behavior.copy()
    bucket_count = min(
        5, int(category["category_total_count"].nunique()), len(category)
    )
    if bucket_count <= 1:
        category["traffic_band"] = "All"
    else:
        category["traffic_band"] = pd.qcut(
            category["category_total_count"].rank(method="first"),
            q=bucket_count,
            labels=_quantile_labels(bucket_count),
        )

    summary = (
        category.groupby("traffic_band", observed=True)
        .agg(
            category_count=("category_id", "nunique"),
            total_events=("category_total_count", "sum"),
            pv_count=("category_pv_count", "sum"),
            buy_count=("category_buy_count", "sum"),
        )
        .reset_index()
    )
    summary["traffic_band"] = summary["traffic_band"].astype(str)
    summary["buy_to_pv_rate"] = _rate_percent(summary["buy_count"], summary["pv_count"])
    return summary


def summarize_conversion_funnel(conversion_chain: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stage 2 item behavior counts into a four-stage funnel."""
    required = {
        "item_pv_count", "item_fav_count", "item_cart_count", "item_buy_count"
    }
    missing = sorted(required - set(conversion_chain.columns))
    if missing:
        raise ValueError("Missing conversion-chain columns: " + ", ".join(missing))

    funnel = pd.DataFrame({
        "stage": ["PV", "Favorite", "Cart", "Purchase"],
        "behavior_count": [
            float(conversion_chain["item_pv_count"].sum()),
            float(conversion_chain["item_fav_count"].sum()),
            float(conversion_chain["item_cart_count"].sum()),
            float(conversion_chain["item_buy_count"].sum()),
        ],
    })
    pv = funnel.loc[0, "behavior_count"]
    funnel["relative_to_pv_percentage"] = (
        funnel["behavior_count"] / pv * 100 if pv else float("nan")
    )
    funnel["from_previous_percentage"] = _rate_percent(
        funnel["behavior_count"], funnel["behavior_count"].shift(1)
    )
    funnel.loc[0, "from_previous_percentage"] = 100.0
    return funnel


def summarize_transition_sequences(sequences: pd.Series) -> pd.DataFrame:
    """Calculate row-normalized adjacent behavior transition probabilities."""
    counts: Counter[tuple[str, str]] = Counter()
    valid = set(BEHAVIOR_ORDER)

    for value in sequences.dropna().astype(str):
        behaviors = [p.strip().lower() for p in value.split("→") if p.strip()]
        for source, target in zip(behaviors, behaviors[1:]):
            if source in valid and target in valid:
                counts[(source, target)] += 1

    rows = []
    for source in BEHAVIOR_ORDER:
        total = sum(counts[(source, target)] for target in BEHAVIOR_ORDER)
        for target in BEHAVIOR_ORDER:
            count = counts[(source, target)]
            rows.append({
                "from_behavior": source,
                "to_behavior": target,
                "transition_count": count,
                "transition_probability": count / total * 100 if total else 0.0,
            })
    return pd.DataFrame(rows)


def summarize_transition_parquet(
    path: str | Path,
    dataset_split: str,
    sequence_limit: int | None = None,
) -> pd.DataFrame:
    """Stream user sequence Parquet and build an adjacent transition matrix."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 feature file: {parquet_path.name}")

    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for streamed Stage 2 transition statistics."
        ) from exc

    dataset = pa_dataset.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(
        columns=["last_10_behavior_sequence"],
        filter=pa_dataset.field("dataset_split") == dataset_split,
        batch_size=65_536,
    )

    counts: Counter[tuple[str, str]] = Counter()
    valid = set(BEHAVIOR_ORDER)
    processed = 0

    for batch in scanner.to_batches():
        idx = batch.schema.get_field_index("last_10_behavior_sequence")
        values = batch.column(idx).to_pylist()
        if sequence_limit is not None:
            remaining = sequence_limit - processed
            if remaining <= 0:
                break
            values = values[:remaining]

        for value in values:
            if value is None:
                continue
            behaviors = [p.strip().lower() for p in str(value).split("→") if p.strip()]
            for source, target in zip(behaviors, behaviors[1:]):
                if source in valid and target in valid:
                    counts[(source, target)] += 1

        processed += len(values)
        if sequence_limit is not None and processed >= sequence_limit:
            break

    rows = []
    for source in BEHAVIOR_ORDER:
        total = sum(counts[(source, target)] for target in BEHAVIOR_ORDER)
        for target in BEHAVIOR_ORDER:
            count = counts[(source, target)]
            rows.append({
                "from_behavior": source,
                "to_behavior": target,
                "transition_count": count,
                "transition_probability": count / total * 100 if total else 0.0,
                "sequences_scanned": processed,
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_stage2_dashboard_statistics(
    output_dir: str | Path = FEATURE_OUTPUT_DIR,
    dataset_split: str = "train",
    transition_sequence_limit: int | None = 250_000,
) -> dict[str, pd.DataFrame]:
    """Build only the aggregate Stage 2 datasets required by the EDA dashboard."""
    output_path = Path(output_dir)

    user_activity = load_stage2_feature_table(
        "user_activity",
        output_path,
        columns=(
            "user_id", "activity_level", "event_count", "avg_daily_event_count",
            "pv_count_per_day", "buy_count_per_day",
        ),
        dataset_split=dataset_split,
    )
    item_popularity = load_stage2_feature_table(
        "item_popularity",
        output_path,
        columns=(
            "item_id", "item_total_count_rank", "item_pv_count",
            "item_buy_count", "item_total_count",
        ),
        dataset_split=dataset_split,
    )
    category_behavior = load_stage2_feature_table(
        "category_behavior",
        output_path,
        columns=(
            "category_id", "category_total_count",
            "category_pv_count", "category_buy_count",
        ),
        dataset_split=dataset_split,
    )
    conversion_chain = load_stage2_feature_table(
        "conversion_chain",
        output_path,
        columns=(
            "item_pv_count", "item_fav_count",
            "item_cart_count", "item_buy_count",
        ),
        dataset_split=dataset_split,
    )

    activity, depth = summarize_user_activity(user_activity)
    popularity, top_items = summarize_item_popularity(item_popularity)
    transition = summarize_transition_parquet(
        output_path / STAGE2_FEATURE_FILES["user_sequence"],
        dataset_split=dataset_split,
        sequence_limit=transition_sequence_limit,
    )

    return {
        "conversion_funnel": summarize_conversion_funnel(conversion_chain),
        "transition_matrix": transition,
        "behavior_depth": depth,
        "user_activity": activity,
        "item_popularity": popularity,
        "top_items": top_items,
        "category_traffic": summarize_category_traffic(category_behavior),
    }


def write_stage2_dashboard_statistics(
    statistics_by_split: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path = STAGE2_DASHBOARD_STATS_DIR,
) -> dict[str, Path]:
    """Persist lightweight per-split dashboard statistics as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for stat_name, filename in STAGE2_DASHBOARD_STAT_FILES.items():
        parts = []
        for split, statistics in statistics_by_split.items():
            frame = statistics[stat_name].copy()
            frame.insert(0, "dataset_split", split)
            parts.append(frame)
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        path = output_path / filename
        combined.to_csv(path, index=False)
        written[stat_name] = path

    return written


@st.cache_data(show_spinner=False)
def load_stage2_dashboard_outputs(
    output_dir: str | Path = STAGE2_DASHBOARD_STATS_DIR,
    dataset_split: str = "train",
) -> dict[str, pd.DataFrame]:
    """Load precomputed Stage 2 dashboard statistics for one split."""
    output_path = Path(output_dir)
    missing = [
        filename for filename in STAGE2_DASHBOARD_STAT_FILES.values()
        if not (output_path / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 2 dashboard statistics: " + ", ".join(missing)
        )

    outputs = {}
    for name, filename in STAGE2_DASHBOARD_STAT_FILES.items():
        frame = pd.read_csv(output_path / filename)
        if "dataset_split" in frame.columns:
            frame = frame[
                frame["dataset_split"].astype(str).eq(dataset_split)
            ].copy()
            frame = frame.drop(columns="dataset_split")
        outputs[name] = frame.reset_index(drop=True)
    return outputs
