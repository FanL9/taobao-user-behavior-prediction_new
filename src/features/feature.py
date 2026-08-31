"""Build the first four stage-two feature tables.

This module creates only user basic behavior, user activity, user-item behavior
sequence, and item behavior feature tables. Item popularity, category behavior,
time behavior, conversion-chain features, labels, and a final wide table are
outside this module's scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .stage2_intermediate_tables import HISTORY_WINDOWS, HistoryWindow


REQUIRED_COLUMNS = (
    "time",
    "user_id",
    "item_id",
    "category_id",
    "behavior_name",
    "behavior_date",
)

VALID_BEHAVIORS = ("pv", "fav", "cart", "buy")

OUTPUT_FILENAMES = {
    "user_basic": "user_features.parquet",
    "user_activity": "user_activity_features.parquet",
    "user_sequence": "user_sequence_features.parquet",
    "item_behavior": "item_behavior_features.parquet",
}

USER_SEQUENCE_COLUMNS = [
    "dataset_split",
    "user_id",
    "item_id",
    "history_start",
    "history_end",
    "label_date",
    "last_behavior_type",
    "last_behavior_hour",
    "last_behavior_days_ago",
    "last_10_behavior_sequence",
]


def _validate_windows(
    windows: Iterable[HistoryWindow],
) -> tuple[HistoryWindow, ...]:
    """Validate feature history windows.

    Args:
        windows: Candidate history windows and excluded label dates.

    Returns:
        Validated windows as an immutable tuple.
    """

    normalized = tuple(windows)
    if not normalized:
        raise ValueError("At least one history window is required.")
    names = [window.dataset_split for window in normalized]
    if len(names) != len(set(names)):
        raise ValueError("dataset_split values must be unique.")

    for window in normalized:
        start = pd.Timestamp(window.history_start)
        end = pd.Timestamp(window.history_end)
        label = pd.Timestamp(window.label_date)
        if not start <= end < label:
            raise ValueError(
                f"Invalid window {window.dataset_split!r}: expected "
                "history_start <= history_end < label_date."
            )
    return normalized


def _prepare_clean_data(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Validate clean input and create deterministic time fields.

    Args:
        clean_data: Stage-one clean user behavior rows.

    Returns:
        Required input columns plus parsed event time, normalized event date,
        and stable source-row order.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(clean_data.columns))
    if missing:
        raise ValueError(f"Clean data is missing columns: {missing}")

    frame = clean_data.loc[:, REQUIRED_COLUMNS].copy()
    event_time = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H",
        errors="coerce",
        exact=True,
    )
    behavior_date = pd.to_datetime(
        frame["behavior_date"],
        errors="coerce",
    ).dt.normalize()
    if event_time.isna().any() or behavior_date.isna().any():
        raise ValueError("Clean data contains invalid time fields.")
    if not behavior_date.eq(event_time.dt.normalize()).all():
        raise ValueError("behavior_date is inconsistent with time.")
    if not frame["behavior_name"].isin(VALID_BEHAVIORS).all():
        raise ValueError("Clean data contains an unsupported behavior_name.")
    if frame[["user_id", "item_id", "category_id"]].isna().any().any():
        raise ValueError("Clean data contains missing dimension identifiers.")

    frame["event_time"] = event_time
    frame["event_date"] = event_time.dt.normalize()
    frame["source_order"] = np.arange(len(frame), dtype="int64")
    return frame


def select_feature_history(
    clean_data: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Select one inclusive history window and exclude its label date.

    Args:
        clean_data: Prepared clean rows containing ``event_time``.
        window: History boundaries and excluded label date.

    Returns:
        A copy containing only events from ``history_start`` through
        ``history_end``. Label-date and later rows are excluded.
    """

    start = pd.Timestamp(window.history_start)
    end_exclusive = pd.Timestamp(window.history_end) + pd.Timedelta(days=1)
    label = pd.Timestamp(window.label_date)
    mask = (
        clean_data["event_time"].ge(start)
        & clean_data["event_time"].lt(end_exclusive)
        & clean_data["event_time"].lt(label)
    )
    return clean_data.loc[mask].copy()


def _behavior_counts(history: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Count the four behavior types for a requested grain.

    Args:
        history: Rows from one validated history window.
        keys: Columns defining the output grain.

    Returns:
        One row per key with integer ``pv``, ``fav``, ``cart``, and ``buy``
        count columns.
    """

    counts = (
        history.groupby(keys + ["behavior_name"], observed=True, sort=False)
        .size()
        .unstack("behavior_name", fill_value=0)
        .reindex(columns=VALID_BEHAVIORS, fill_value=0)
        .reset_index()
    )
    for behavior in VALID_BEHAVIORS:
        counts[behavior] = counts[behavior].astype("int64")
    return counts


def _add_window_metadata(
    table: pd.DataFrame,
    window: HistoryWindow,
    key_count: int,
) -> pd.DataFrame:
    """Attach fixed split metadata immediately after grain keys.

    Args:
        table: Feature rows for one history window.
        window: Window metadata to attach.
        key_count: Number of grain-key columns following ``dataset_split``.

    Returns:
        Feature rows with split, history boundaries, and label-date metadata.
    """

    table.insert(0, "dataset_split", window.dataset_split)
    insert_at = key_count + 1
    table.insert(insert_at, "history_start", pd.Timestamp(window.history_start))
    table.insert(insert_at + 1, "history_end", pd.Timestamp(window.history_end))
    table.insert(insert_at + 2, "label_date", pd.Timestamp(window.label_date))
    return table


def build_user_basic_features(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build user basic behavior counts for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Metadata written to every output row.

    Returns:
        One row per user containing total and four behavior counts. Conversion
        fields are deliberately excluded for the later conversion-chain table.
    """

    total = (
        history.groupby("user_id", observed=True, sort=False)
        .size()
        .rename("event_count")
        .reset_index()
    )
    counts = _behavior_counts(history, ["user_id"]).rename(
        columns={behavior: f"{behavior}_count" for behavior in VALID_BEHAVIORS}
    )
    table = total.merge(counts, on="user_id", validate="one_to_one")
    return _add_window_metadata(table, window, key_count=1)


def build_user_activity_features(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build continuous user activity features for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Metadata written to every output row.

    Returns:
        One row per user with length-normalized activity, recency, interaction
        breadth, and per-day behavior counts. Activity level is added after all
        windows are combined so train-only thresholds can be used.
    """

    window_days = (
        pd.Timestamp(window.history_end) - pd.Timestamp(window.history_start)
    ).days + 1
    base = (
        history.groupby("user_id", observed=True, sort=False)
        .agg(
            event_count=("user_id", "size"),
            active_day_count=("event_date", "nunique"),
            last_event_time=("event_time", "max"),
            unique_item_count=("item_id", "nunique"),
            unique_category_count=("category_id", "nunique"),
        )
        .reset_index()
    )
    counts = _behavior_counts(history, ["user_id"])
    table = base.merge(counts, on="user_id", validate="one_to_one")
    table["window_days"] = window_days
    table["active_day_ratio"] = table["active_day_count"] / window_days
    table["avg_daily_event_count"] = table["event_count"] / window_days
    table["avg_active_day_event_count"] = (
        table["event_count"] / table["active_day_count"]
    )
    table["days_since_last_event"] = (
        pd.Timestamp(window.label_date) - table["last_event_time"]
    ).dt.total_seconds() / 86_400
    for behavior in VALID_BEHAVIORS:
        table[f"{behavior}_count_per_day"] = table[behavior] / window_days

    table = table.drop(columns=["last_event_time", *VALID_BEHAVIORS])
    numeric_columns = [
        "active_day_ratio",
        "avg_daily_event_count",
        "avg_active_day_event_count",
        "days_since_last_event",
        *[f"{behavior}_count_per_day" for behavior in VALID_BEHAVIORS],
    ]
    table[numeric_columns] = table[numeric_columns].round(4)
    table = _add_window_metadata(table, window, key_count=1)
    ordered_columns = [
        "dataset_split",
        "user_id",
        "history_start",
        "history_end",
        "label_date",
        "window_days",
        "event_count",
        "active_day_count",
        "active_day_ratio",
        "avg_daily_event_count",
        "avg_active_day_event_count",
        "days_since_last_event",
        "unique_item_count",
        "unique_category_count",
        *[f"{behavior}_count_per_day" for behavior in VALID_BEHAVIORS],
    ]
    return table.loc[:, ordered_columns]


def build_user_sequence_features(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build deterministic user-item behavior sequence features.

    Args:
        history: Events already restricted to ``window``.
        window: Metadata written to every output row.

    Returns:
        One row per user-item pair with its latest behavior and last ten
        behavior names. Input row order resolves events tied within one hour.
        Transition and conversion fields are excluded for the later dedicated
        conversion-chain table.
    """

    if history.empty:
        return pd.DataFrame(columns=USER_SEQUENCE_COLUMNS)

    keys = ["user_id", "item_id"]
    ordered = history.sort_values(
        keys + ["event_time", "source_order"],
        kind="mergesort",
    )
    latest = ordered.groupby(keys, observed=True, sort=False).tail(1).copy()
    latest = latest.loc[:, keys + ["behavior_name", "event_time"]].rename(
        columns={
            "behavior_name": "last_behavior_type",
            "event_time": "last_behavior_time",
        }
    )
    latest["last_behavior_hour"] = latest["last_behavior_time"].dt.hour.astype(
        "int8"
    )
    latest["last_behavior_days_ago"] = (
        pd.Timestamp(window.label_date) - latest["last_behavior_time"]
    ).dt.total_seconds().div(86_400).round(4)

    recent = ordered.groupby(keys, observed=True, sort=False).tail(10)
    sequences = (
        recent.groupby(keys, observed=True, sort=False)["behavior_name"]
        .agg("→".join)
        .rename("last_10_behavior_sequence")
        .reset_index()
    )
    table = latest.drop(columns="last_behavior_time").merge(
        sequences,
        on=keys,
        validate="one_to_one",
    )
    table = _add_window_metadata(table, window, key_count=2)
    return table.loc[:, USER_SEQUENCE_COLUMNS]


def build_item_behavior_features(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build item behavior counts for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Metadata written to every output row.

    Returns:
        One row per item with category, behavior counts, user breadth, buyer
        count, and active days. Heat and conversion fields are deliberately
        excluded for their later dedicated feature tables.
    """

    category_counts = history.groupby("item_id", observed=True)[
        "category_id"
    ].nunique()
    if category_counts.gt(1).any():
        raise ValueError("An item_id maps to multiple category_id values.")

    base = (
        history.groupby("item_id", observed=True, sort=False)
        .agg(
            category_id=("category_id", "first"),
            item_total_count=("item_id", "size"),
            item_unique_user_count=("user_id", "nunique"),
            item_active_day_count=("event_date", "nunique"),
        )
        .reset_index()
    )
    counts = _behavior_counts(history, ["item_id"]).rename(
        columns={behavior: f"item_{behavior}_count" for behavior in VALID_BEHAVIORS}
    )
    buyers = (
        history.loc[history["behavior_name"].eq("buy")]
        .groupby("item_id", observed=True)["user_id"]
        .nunique()
        .rename("item_unique_buyer_count")
        .reset_index()
    )
    table = base.merge(counts, on="item_id", validate="one_to_one").merge(
        buyers,
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    table["item_unique_buyer_count"] = (
        table["item_unique_buyer_count"].fillna(0).astype("int64")
    )
    table = _add_window_metadata(table, window, key_count=2)
    ordered_columns = [
        "dataset_split",
        "item_id",
        "category_id",
        "history_start",
        "history_end",
        "label_date",
        "item_total_count",
        *[f"item_{behavior}_count" for behavior in VALID_BEHAVIORS],
        "item_unique_user_count",
        "item_unique_buyer_count",
        "item_active_day_count",
    ]
    return table.loc[:, ordered_columns]


def build_feature_tables(
    clean_data: pd.DataFrame,
    windows: Iterable[HistoryWindow] = HISTORY_WINDOWS,
) -> dict[str, pd.DataFrame]:
    """Build exactly the first four stage-two feature tables.

    Args:
        clean_data: Stage-one clean behavior data.
        windows: Leakage-safe history windows. Defaults to the three fixed
            stage-two windows.

    Returns:
        Mapping containing exactly ``user_basic``, ``user_activity``,
        ``user_sequence``, and ``item_behavior`` tables. No labels, remaining
        feature tables, or final wide table are created.
    """

    prepared = _prepare_clean_data(clean_data)
    validated_windows = _validate_windows(windows)
    builders = {
        "user_basic": build_user_basic_features,
        "user_activity": build_user_activity_features,
        "user_sequence": build_user_sequence_features,
        "item_behavior": build_item_behavior_features,
    }
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in builders}

    for window in validated_windows:
        history = select_feature_history(prepared, window)
        for name, builder in builders.items():
            parts[name].append(builder(history, window))

    tables = {
        name: pd.concat(window_parts, ignore_index=True)
        for name, window_parts in parts.items()
    }
    activity = tables["user_activity"]
    train_values = activity.loc[
        activity["dataset_split"].eq("train"),
        "avg_daily_event_count",
    ]
    if train_values.empty:
        raise ValueError("Train rows are required to define activity thresholds.")
    p25 = train_values.quantile(0.25)
    p75 = train_values.quantile(0.75)
    activity["activity_level"] = np.select(
        [
            activity["avg_daily_event_count"].le(p25),
            activity["avg_daily_event_count"].le(p75),
        ],
        ["low", "medium"],
        default="high",
    )
    return tables


def generate_feature_tables(
    input_parquet: str | Path,
    output_directory: str | Path,
    windows: Iterable[HistoryWindow] = HISTORY_WINDOWS,
) -> dict[str, Path]:
    """Read clean Parquet and atomically write exactly four feature tables.

    Args:
        input_parquet: Stage-one clean Parquet path.
        output_directory: Directory receiving four feature Parquet files.
        windows: History windows passed to :func:`build_feature_tables`.

    Returns:
        Mapping from logical table name to its resolved output path.
    """

    source = Path(input_parquet).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Clean Parquet does not exist: {source}")

    clean_data = pd.read_parquet(source, columns=list(REQUIRED_COLUMNS))
    tables = build_feature_tables(clean_data, windows)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    temporary_paths: list[Path] = []

    try:
        for name, table in tables.items():
            target = destination / OUTPUT_FILENAMES[name]
            temporary = target.with_suffix(".parquet.tmp")
            temporary_paths.append(temporary)
            table.to_parquet(temporary, index=False)
            temporary.replace(target)
            output_paths[name] = target
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    return output_paths
# ============================================================
# ????????
# ============================================================

from pathlib import Path as _FeaturePath


ASSIGNED_OUTPUT_FILENAMES = {
    "item_popularity": "item_popularity_features.parquet",
    "category_behavior": "category_behavior_features.parquet",
    "time_behavior": "time_behavior_features.parquet",
    "conversion_chain": "conversion_chain_features.parquet",
}

ALL_OUTPUT_FILENAMES = {
    **OUTPUT_FILENAMES,
    **ASSIGNED_OUTPUT_FILENAMES,
}


def _item_behavior_table_key() -> str:
    """?????????????????"""

    for name, filename in OUTPUT_FILENAMES.items():
        if filename == "item_behavior_features.parquet":
            return name

    raise KeyError(
        "OUTPUT_FILENAMES ???? item_behavior_features.parquet"
    )


def _assigned_behavior_counts(
    history: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate PV/FAV/CART/BUY counts for one feature grain."""

    if "behavior_name" in history.columns:
        behavior_column = "behavior_name"
        expected_values = ["pv", "fav", "cart", "buy"]
        rename_map = {
            "pv": f"{prefix}_pv_count",
            "fav": f"{prefix}_fav_count",
            "cart": f"{prefix}_cart_count",
            "buy": f"{prefix}_buy_count",
        }
    elif "behavior_type" in history.columns:
        behavior_column = "behavior_type"
        expected_values = [1, 2, 3, 4]
        rename_map = {
            1: f"{prefix}_pv_count",
            2: f"{prefix}_fav_count",
            3: f"{prefix}_cart_count",
            4: f"{prefix}_buy_count",
        }
    else:
        raise ValueError(
            "Clean data must contain behavior_name or behavior_type."
        )

    counts = (
        history.groupby(
            keys + [behavior_column],
            observed=True,
            sort=False,
        )
        .size()
        .unstack(
            behavior_column,
            fill_value=0,
        )
        .reindex(
            columns=expected_values,
            fill_value=0,
        )
        .rename(columns=rename_map)
        .reset_index()
    )

    count_columns = [
        f"{prefix}_pv_count",
        f"{prefix}_fav_count",
        f"{prefix}_cart_count",
        f"{prefix}_buy_count",
    ]

    counts[count_columns] = counts[count_columns].astype("int64")

    return counts

def build_item_popularity_features(
    item_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """????????????????"""

    columns = [
        "dataset_split",
        "item_id",
        "category_id",
        "history_start",
        "history_end",
        "label_date",
        "item_total_count",
        "item_pv_count",
        "item_fav_count",
        "item_cart_count",
        "item_buy_count",
        "item_unique_user_count",
        "item_active_day_count",
    ]

    missing = [
        column
        for column in columns
        if column not in item_behavior.columns
    ]
    if missing:
        raise ValueError(
            "???????????????: "
            + ", ".join(missing)
        )

    result = item_behavior.loc[:, columns].copy()

    rank_sources = {
        "item_total_count": "item_total_count_rank",
        "item_unique_user_count": "item_unique_user_count_rank",
        "item_active_day_count": "item_active_day_count_rank",
        "item_buy_count": "item_buy_count_rank",
    }

    for source, target in rank_sources.items():
        result[target] = (
            result.groupby(
                "dataset_split",
                observed=True,
                sort=False,
            )[source]
            .rank(
                method="dense",
                ascending=False,
            )
            .astype("int64")
        )

    return (
        result.sort_values(
            [
                "dataset_split",
                "item_total_count_rank",
                "item_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _assigned_window_metadata(
    item_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """??????????????????"""

    metadata = (
        item_behavior[
            [
                "dataset_split",
                "history_start",
                "history_end",
                "label_date",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if metadata["dataset_split"].duplicated().any():
        raise ValueError(
            "?? dataset_split ????????????"
        )

    return metadata


def build_category_behavior_features(
    clean_data: pd.DataFrame,
    item_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """?????????????"""

    prepared = _prepare_clean_data(clean_data)
    metadata = _assigned_window_metadata(item_behavior)

    if "event_time" not in prepared.columns:
        prepared = prepared.copy()

        if "time" in prepared.columns:
            prepared["event_time"] = pd.to_datetime(
                prepared["time"],
                errors="coerce",
            )
        else:
            prepared["event_time"] = (
                pd.to_datetime(prepared["behavior_date"])
                + pd.to_timedelta(
                    prepared["behavior_hour"],
                    unit="h",
                )
            )

    results = []

    for window in metadata.itertuples(index=False):
        history_start = pd.Timestamp(window.history_start)
        history_end = pd.Timestamp(window.history_end)
        label_date = pd.Timestamp(window.label_date)

        history = prepared[
            pd.to_datetime(prepared["behavior_date"]).between(
                history_start,
                history_end,
                inclusive="both",
            )
        ].copy()

        base = (
            history.groupby(
                "category_id",
                observed=True,
                sort=False,
            )
            .agg(
                category_total_count=("category_id", "size"),
                category_unique_user_count=("user_id", "nunique"),
                category_unique_item_count=("item_id", "nunique"),
                category_active_day_count=("behavior_date", "nunique"),
                category_first_event_time=("event_time", "min"),
                category_last_event_time=("event_time", "max"),
            )
            .reset_index()
        )

        counts = _assigned_behavior_counts(
            history,
            ["category_id"],
            "category",
        )

        table = base.merge(
            counts,
            on="category_id",
            how="left",
            validate="one_to_one",
        )

        table.insert(0, "dataset_split", window.dataset_split)
        table.insert(2, "history_start", history_start)
        table.insert(3, "history_end", history_end)
        table.insert(4, "label_date", label_date)

        results.append(table)

    return pd.concat(results, ignore_index=True)


def build_time_behavior_features(
    clean_data: pd.DataFrame,
    item_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """Build observed date-hour historical behavior features."""

    prepared = _prepare_clean_data(clean_data).copy()

    if "event_time" in prepared.columns:
        event_time = pd.to_datetime(
            prepared["event_time"],
            errors="coerce",
        )
    else:
        event_time = pd.to_datetime(
            prepared["time"],
            errors="coerce",
        )

    if "behavior_hour" not in prepared.columns:
        prepared["behavior_hour"] = event_time.dt.hour

    if "weekday" not in prepared.columns:
        prepared["weekday"] = event_time.dt.weekday

    metadata = _assigned_window_metadata(item_behavior)
    results = []

    for window in metadata.itertuples(index=False):
        history_start = pd.Timestamp(window.history_start)
        history_end = pd.Timestamp(window.history_end)
        label_date = pd.Timestamp(window.label_date)

        history = prepared[
            pd.to_datetime(prepared["behavior_date"]).between(
                history_start,
                history_end,
                inclusive="both",
            )
        ].copy()

        keys = [
            "behavior_date",
            "behavior_hour",
        ]

        base = (
            history.groupby(
                keys,
                observed=True,
                sort=False,
            )
            .agg(
                weekday=("weekday", "first"),
                time_total_count=("behavior_hour", "size"),
                time_unique_user_count=("user_id", "nunique"),
                time_unique_item_count=("item_id", "nunique"),
                time_unique_category_count=("category_id", "nunique"),
            )
            .reset_index()
        )

        counts = _assigned_behavior_counts(
            history,
            keys,
            "time",
        )

        table = base.merge(
            counts,
            on=keys,
            how="left",
            validate="one_to_one",
        )

        table.insert(0, "dataset_split", window.dataset_split)
        table.insert(3, "history_start", history_start)
        table.insert(4, "history_end", history_end)
        table.insert(5, "label_date", label_date)

        table = table.sort_values(
            ["behavior_date", "behavior_hour"],
            kind="stable",
        ).reset_index(drop=True)

        results.append(table)

    return pd.concat(results, ignore_index=True)


def build_conversion_chain_features(
    item_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """??????????????"""

    columns = [
        "dataset_split",
        "item_id",
        "category_id",
        "history_start",
        "history_end",
        "label_date",
        "item_pv_count",
        "item_fav_count",
        "item_cart_count",
        "item_buy_count",
    ]

    missing = [
        column
        for column in columns
        if column not in item_behavior.columns
    ]
    if missing:
        raise ValueError(
            "???????????????: "
            + ", ".join(missing)
        )

    result = item_behavior.loc[:, columns].copy()
    numerator = result["item_buy_count"].astype("Float64")

    denominators = {
        "buy_per_pv": "item_pv_count",
        "buy_per_fav": "item_fav_count",
        "buy_per_cart": "item_cart_count",
    }

    for feature, denominator_column in denominators.items():
        denominator = (
            result[denominator_column]
            .astype("Float64")
            .mask(
                result[denominator_column].eq(0)
            )
        )

        result[feature] = numerator / denominator

    return (
        result.sort_values(
            ["dataset_split", "item_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_assigned_feature_tables(
    clean_data: pd.DataFrame,
    item_behavior: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """?? Issue #8 ???????"""

    return {
        "item_popularity": build_item_popularity_features(
            item_behavior
        ),
        "category_behavior": build_category_behavior_features(
            clean_data,
            item_behavior,
        ),
        "time_behavior": build_time_behavior_features(
            clean_data,
            item_behavior,
        ),
        "conversion_chain": build_conversion_chain_features(
            item_behavior
        ),
    }


def build_all_feature_tables(
    clean_data: pd.DataFrame,
    *args,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """?????????????????????????"""

    tables = build_feature_tables(
        clean_data,
        *args,
        **kwargs,
    )

    item_key = _item_behavior_table_key()
    item_behavior = tables[item_key]

    tables.update(
        build_assigned_feature_tables(
            clean_data,
            item_behavior,
        )
    )

    return tables


def generate_all_feature_tables(
    input_parquet,
    output_directory,
    *args,
    **kwargs,
):
    """???????????????"""

    source = _FeaturePath(input_parquet).expanduser().resolve()
    destination = (
        _FeaturePath(output_directory)
        .expanduser()
        .resolve()
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Clean Parquet does not exist: {source}"
        )

    clean_data = pd.read_parquet(source)

    tables = build_all_feature_tables(
        clean_data,
        *args,
        **kwargs,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {}
    temporary_paths = []

    try:
        for name, table in tables.items():
            target = destination / ALL_OUTPUT_FILENAMES[name]
            temporary = target.with_suffix(".parquet.tmp")
            temporary_paths.append(temporary)

            table.to_parquet(
                temporary,
                index=False,
            )
            temporary.replace(target)

            output_paths[name] = target

    finally:
        for temporary in temporary_paths:
            temporary.unlink(
                missing_ok=True
            )

    return output_paths
