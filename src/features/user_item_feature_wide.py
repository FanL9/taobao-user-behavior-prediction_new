"""Merge the eight stage-two feature tables into one user-item wide table."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .stage2_intermediate_tables import HISTORY_WINDOWS


FEATURE_TABLE_FILES = {
    "user_basic": "user_features.parquet",
    "user_activity": "user_activity_features.parquet",
    "user_sequence": "user_sequence_features.parquet",
    "item_behavior": "item_behavior_features.parquet",
    "item_popularity": "item_popularity_features.parquet",
    "category_behavior": "category_behavior_features.parquet",
    "time_behavior": "time_behavior_features.parquet",
    "conversion_chain": "conversion_chain_features.parquet",
}

WIDE_TABLE_FILENAME = "user_item_feature_wide.parquet"
QUALITY_REPORT_FILENAME = "user_item_feature_wide_quality_report.json"
PRIMARY_KEY = ("dataset_split", "user_id", "item_id")
TRACKING_COLUMNS = (
    "category_id",
    "history_start",
    "history_end",
    "label_date",
    "last_behavior_date",
)
ALLOWED_MISSING_COLUMNS = ("buy_per_pv", "buy_per_fav", "buy_per_cart")

REQUIRED_COLUMNS = {
    "user_basic": {
        "dataset_split", "user_id", "history_start", "history_end",
        "label_date", "event_count", "pv_count", "fav_count",
        "cart_count", "buy_count",
    },
    "user_activity": {
        "dataset_split", "user_id", "history_start", "history_end",
        "label_date", "window_days", "event_count", "active_day_count",
        "active_day_ratio", "avg_daily_event_count",
        "avg_active_day_event_count", "days_since_last_event",
        "unique_item_count", "unique_category_count", "pv_count_per_day",
        "fav_count_per_day", "cart_count_per_day", "buy_count_per_day",
        "activity_level",
    },
    "user_sequence": {
        "dataset_split", "user_id", "item_id", "history_start",
        "history_end", "label_date", "last_behavior_type",
        "last_behavior_hour", "last_behavior_days_ago",
        "last_10_behavior_sequence",
    },
    "item_behavior": {
        "dataset_split", "item_id", "category_id", "history_start",
        "history_end", "label_date", "item_total_count", "item_pv_count",
        "item_fav_count", "item_cart_count", "item_buy_count",
        "item_unique_user_count", "item_unique_buyer_count",
        "item_active_day_count",
    },
    "item_popularity": {
        "dataset_split", "item_id", "category_id", "history_start",
        "history_end", "label_date", "item_total_count", "item_pv_count",
        "item_fav_count", "item_cart_count", "item_buy_count",
        "item_unique_user_count", "item_active_day_count",
        "item_total_count_rank", "item_unique_user_count_rank",
        "item_active_day_count_rank", "item_buy_count_rank",
    },
    "category_behavior": {
        "dataset_split", "category_id", "history_start", "history_end",
        "label_date", "category_total_count", "category_unique_user_count",
        "category_unique_item_count", "category_active_day_count",
        "category_first_event_time", "category_last_event_time",
        "category_pv_count", "category_fav_count", "category_cart_count",
        "category_buy_count",
    },
    "time_behavior": {
        "dataset_split", "behavior_date", "behavior_hour", "history_start",
        "history_end", "label_date", "weekday", "time_total_count",
        "time_unique_user_count", "time_unique_item_count",
        "time_unique_category_count", "time_pv_count", "time_fav_count",
        "time_cart_count", "time_buy_count",
    },
    "conversion_chain": {
        "dataset_split", "item_id", "category_id", "history_start",
        "history_end", "label_date", "item_pv_count", "item_fav_count",
        "item_cart_count", "item_buy_count", "buy_per_pv",
        "buy_per_fav", "buy_per_cart",
    },
}

WINDOWS_BY_SPLIT = {
    window.dataset_split: {
        "history_start": pd.Timestamp(window.history_start),
        "history_end": pd.Timestamp(window.history_end),
        "label_date": pd.Timestamp(window.label_date),
    }
    for window in HISTORY_WINDOWS
}


def _feature_paths(feature_directory: str | Path) -> dict[str, Path]:
    """Resolve and validate the eight input feature-table paths.

    Args:
        feature_directory: Directory containing the eight feature Parquets.

    Returns:
        Mapping from logical table name to resolved Parquet path.
    """

    directory = Path(feature_directory).expanduser().resolve()
    paths = {
        name: directory / filename
        for name, filename in FEATURE_TABLE_FILES.items()
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing feature tables: " + ", ".join(missing_files)
        )

    for name, path in paths.items():
        actual = set(pq.ParquetFile(path).schema_arrow.names)
        missing_columns = sorted(REQUIRED_COLUMNS[name] - actual)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(missing_columns)}"
            )
    return paths


def _validate_window_metadata(
    path: Path,
    table_name: str,
    dataset_split: str,
) -> None:
    """Validate one table's window columns without loading the full table.

    Args:
        path: Feature-table Parquet path.
        table_name: Logical name used in validation errors.
        dataset_split: Split whose rows are checked.

    Returns:
        None. A ``ValueError`` is raised for missing or inconsistent metadata.
    """

    expected = WINDOWS_BY_SPLIT[dataset_split]
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=["history_start", "history_end", "label_date"],
        filter=ds.field("dataset_split") == dataset_split,
        batch_size=131_072,
    )
    found_rows = False
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        found_rows = True
        frame = batch.to_pandas()
        for column, expected_value in expected.items():
            actual = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
            if actual.isna().any() or not actual.eq(expected_value).all():
                raise ValueError(
                    f"{table_name} has inconsistent {column} in "
                    f"dataset_split={dataset_split}."
                )
    if not found_rows:
        raise ValueError(
            f"{table_name} has no rows for dataset_split={dataset_split}."
        )


def _read_split(
    path: Path,
    dataset_split: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Read selected columns for one dataset split.

    Args:
        path: Feature-table Parquet path.
        dataset_split: Split used as a Parquet filter.
        columns: Columns to load.

    Returns:
        A pandas DataFrame containing only the requested split and columns.
    """

    return pd.read_parquet(
        path,
        columns=list(columns),
        filters=[("dataset_split", "==", dataset_split)],
    )


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    """Require a non-null, unique key for one feature lookup table.

    Args:
        frame: Table being checked.
        keys: Columns forming its expected key.
        name: Table name used in errors.

    Returns:
        None. Invalid keys raise ``ValueError``.
    """

    if frame[keys].isna().any(axis=None):
        raise ValueError(f"{name} contains a missing primary-key value.")
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate keys.")


def _merge_exact_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    right_name: str,
) -> pd.DataFrame:
    """Merge two lookup tables and require identical key coverage.

    Args:
        left: Base lookup table.
        right: Lookup table being attached.
        key: Shared unique key.
        right_name: Source name used in validation errors.

    Returns:
        One-to-one merged lookup table.
    """

    merged = left.merge(
        right,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator="_source_match",
        sort=False,
    )
    if not merged["_source_match"].eq("both").all():
        raise ValueError(f"{right_name} key coverage does not match its base table.")
    return merged.drop(columns="_source_match")


def _require_equal_columns(
    frame: pd.DataFrame,
    pairs: Mapping[str, str],
    source_name: str,
) -> None:
    """Require duplicated source statistics to agree before columns are dropped.

    Args:
        frame: Merged lookup containing both copies.
        pairs: Mapping from canonical column to comparison column.
        source_name: Source name used in validation errors.

    Returns:
        None. Any inconsistent value raises ``ValueError``.
    """

    for canonical, comparison in pairs.items():
        if not frame[canonical].eq(frame[comparison]).all():
            raise ValueError(
                f"{source_name} is inconsistent with {canonical}."
            )


def _build_user_lookup(paths: Mapping[str, Path], split: str) -> pd.DataFrame:
    """Build the user-keyed lookup used by wide-table batches.

    Args:
        paths: Eight logical feature-table paths.
        split: Dataset split to read.

    Returns:
        One row per user with basic and activity fields.
    """

    basic_columns = [
        "user_id", "event_count", "pv_count", "fav_count", "cart_count",
        "buy_count",
    ]
    activity_columns = [
        "user_id", "window_days", "event_count", "active_day_count",
        "active_day_ratio", "avg_daily_event_count",
        "avg_active_day_event_count", "days_since_last_event",
        "unique_item_count", "unique_category_count", "pv_count_per_day",
        "fav_count_per_day", "cart_count_per_day", "buy_count_per_day",
        "activity_level",
    ]
    basic = _read_split(paths["user_basic"], split, basic_columns)
    activity = _read_split(paths["user_activity"], split, activity_columns)
    _require_unique(basic, ["user_id"], "user_basic")
    _require_unique(activity, ["user_id"], "user_activity")
    activity = activity.rename(columns={"event_count": "activity_event_count"})
    lookup = _merge_exact_one_to_one(
        basic, activity, "user_id", "user_activity"
    )
    _require_equal_columns(
        lookup,
        {"event_count": "activity_event_count"},
        "user_activity",
    )
    lookup = lookup.drop(columns="activity_event_count")
    return lookup.rename(
        columns={
            column: f"user_{column}"
            for column in lookup.columns
            if column != "user_id"
        }
    )


def _build_item_lookup(paths: Mapping[str, Path], split: str) -> pd.DataFrame:
    """Build and cross-check the item behavior, heat, and ratio lookup.

    Args:
        paths: Eight logical feature-table paths.
        split: Dataset split to read.

    Returns:
        One row per item with non-redundant fields from three item tables.
    """

    behavior_columns = [
        "item_id", "category_id", "item_total_count", "item_pv_count",
        "item_fav_count", "item_cart_count", "item_buy_count",
        "item_unique_user_count", "item_unique_buyer_count",
        "item_active_day_count",
    ]
    popularity_columns = [
        "item_id", "category_id", "item_total_count", "item_pv_count",
        "item_fav_count", "item_cart_count", "item_buy_count",
        "item_unique_user_count", "item_active_day_count",
        "item_total_count_rank", "item_unique_user_count_rank",
        "item_active_day_count_rank", "item_buy_count_rank",
    ]
    conversion_columns = [
        "item_id", "category_id", "item_pv_count", "item_fav_count",
        "item_cart_count", "item_buy_count", "buy_per_pv", "buy_per_fav",
        "buy_per_cart",
    ]

    behavior = _read_split(paths["item_behavior"], split, behavior_columns)
    popularity = _read_split(
        paths["item_popularity"], split, popularity_columns
    )
    conversion = _read_split(
        paths["conversion_chain"], split, conversion_columns
    )
    for name, frame in {
        "item_behavior": behavior,
        "item_popularity": popularity,
        "conversion_chain": conversion,
    }.items():
        _require_unique(frame, ["item_id"], name)

    popularity = popularity.rename(
        columns={
            column: f"{column}_popularity_source"
            for column in popularity.columns
            if column not in {
                "item_id", "item_total_count_rank",
                "item_unique_user_count_rank", "item_active_day_count_rank",
                "item_buy_count_rank",
            }
        }
    )
    lookup = _merge_exact_one_to_one(
        behavior, popularity, "item_id", "item_popularity"
    )
    popularity_pairs = {
        column: f"{column}_popularity_source"
        for column in [
            "category_id", "item_total_count", "item_pv_count",
            "item_fav_count", "item_cart_count", "item_buy_count",
            "item_unique_user_count", "item_active_day_count",
        ]
    }
    _require_equal_columns(lookup, popularity_pairs, "item_popularity")
    lookup = lookup.drop(columns=list(popularity_pairs.values()))

    conversion = conversion.rename(
        columns={
            column: f"{column}_conversion_source"
            for column in conversion.columns
            if column not in {"item_id", *ALLOWED_MISSING_COLUMNS}
        }
    )
    lookup = _merge_exact_one_to_one(
        lookup, conversion, "item_id", "conversion_chain"
    )
    conversion_pairs = {
        column: f"{column}_conversion_source"
        for column in [
            "category_id", "item_pv_count", "item_fav_count",
            "item_cart_count", "item_buy_count",
        ]
    }
    _require_equal_columns(lookup, conversion_pairs, "conversion_chain")
    return lookup.drop(columns=list(conversion_pairs.values()))


def _build_category_lookup(paths: Mapping[str, Path], split: str) -> pd.DataFrame:
    """Build the category-keyed lookup used by wide-table batches.

    Args:
        paths: Eight logical feature-table paths.
        split: Dataset split to read.

    Returns:
        One row per category containing category behavior fields.
    """

    columns = sorted(
        REQUIRED_COLUMNS["category_behavior"]
        - {"dataset_split", "history_start", "history_end", "label_date"}
    )
    lookup = _read_split(paths["category_behavior"], split, columns)
    _require_unique(lookup, ["category_id"], "category_behavior")
    return lookup


def _build_time_lookup(paths: Mapping[str, Path], split: str) -> pd.DataFrame:
    """Build the last-behavior date-hour lookup used by wide-table batches.

    Args:
        paths: Eight logical feature-table paths.
        split: Dataset split to read.

    Returns:
        One row per observed date-hour with renamed wide-table join keys.
    """

    columns = sorted(
        REQUIRED_COLUMNS["time_behavior"]
        - {"dataset_split", "history_start", "history_end", "label_date"}
    )
    lookup = _read_split(paths["time_behavior"], split, columns)
    lookup["behavior_date"] = pd.to_datetime(
        lookup["behavior_date"], errors="raise"
    ).dt.normalize()
    _require_unique(
        lookup,
        ["behavior_date", "behavior_hour"],
        "time_behavior",
    )
    return lookup.rename(
        columns={
            "behavior_date": "last_behavior_date",
            "behavior_hour": "last_behavior_hour",
            "weekday": "time_weekday",
        }
    )


def build_split_lookups(
    paths: Mapping[str, Path],
    dataset_split: str,
) -> dict[str, pd.DataFrame]:
    """Load and validate all dimension lookups for one history split.

    Args:
        paths: Mapping returned by ``_feature_paths``.
        dataset_split: One of ``train``, ``validation``, or ``test``.

    Returns:
        User-, item-, category-, and time-keyed lookup DataFrames.
    """

    if dataset_split not in WINDOWS_BY_SPLIT:
        raise ValueError(f"Unsupported dataset_split: {dataset_split}")
    for table_name, path in paths.items():
        _validate_window_metadata(path, table_name, dataset_split)
    return {
        "user": _build_user_lookup(paths, dataset_split),
        "item": _build_item_lookup(paths, dataset_split),
        "category": _build_category_lookup(paths, dataset_split),
        "time": _build_time_lookup(paths, dataset_split),
    }


def _merge_required_lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    source_name: str,
) -> pd.DataFrame:
    """Attach a many-to-one lookup and reject missing source matches.

    Args:
        left: Current wide-table batch.
        right: Unique lookup table being attached.
        keys: Shared merge columns.
        source_name: Source name used in validation errors.

    Returns:
        Merged batch with the same row count as ``left``.
    """

    marker = f"_{source_name}_match"
    merged = left.merge(
        right,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=marker,
        sort=False,
    )
    missing = int(merged[marker].ne("both").sum())
    if missing:
        raise ValueError(f"{source_name} is missing for {missing} wide-table rows.")
    return merged.drop(columns=marker)


def merge_user_item_feature_batch(
    sequence_batch: pd.DataFrame,
    lookups: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge one user-item sequence batch with all seven other feature tables.

    Args:
        sequence_batch: Rows from ``user_sequence_features.parquet`` for one
            dataset split.
        lookups: Validated user, item, category, and time lookup tables for the
            same split.

    Returns:
        A user-item feature batch with one canonical set of window metadata.
    """

    missing = sorted(REQUIRED_COLUMNS["user_sequence"] - set(sequence_batch))
    if missing:
        raise ValueError(
            "user_sequence batch is missing columns: " + ", ".join(missing)
        )
    if sequence_batch.empty:
        return sequence_batch.copy()
    if sequence_batch[list(PRIMARY_KEY)].isna().any(axis=None):
        raise ValueError("user_sequence contains a missing primary-key value.")

    wide = sequence_batch.copy()
    for column in ["history_start", "history_end", "label_date"]:
        wide[column] = pd.to_datetime(wide[column], errors="raise").dt.normalize()

    days_ago = pd.to_numeric(
        wide["last_behavior_days_ago"], errors="raise"
    )
    if days_ago.lt(0).any():
        raise ValueError("last_behavior_days_ago contains a negative value.")
    wide["last_behavior_date"] = wide["label_date"] - pd.to_timedelta(
        np.ceil(days_ago).astype("int64"), unit="D"
    )

    wide = _merge_required_lookup(
        wide, lookups["user"], ["user_id"], "user_features"
    )
    wide = _merge_required_lookup(
        wide, lookups["item"], ["item_id"], "item_features"
    )
    wide = _merge_required_lookup(
        wide,
        lookups["category"],
        ["category_id"],
        "category_behavior",
    )
    wide = _merge_required_lookup(
        wide,
        lookups["time"],
        ["last_behavior_date", "last_behavior_hour"],
        "time_behavior",
    )

    ordered = [
        *PRIMARY_KEY,
        "category_id",
        "history_start",
        "history_end",
        "label_date",
        "last_behavior_date",
    ]
    ordered.extend(column for column in wide.columns if column not in ordered)
    return wide.loc[:, ordered]


def feature_role_mapping(columns: Iterable[str]) -> dict[str, list[str]]:
    """Classify wide-table fields by key, trace, feature, and model prohibition.

    Args:
        columns: Final wide-table columns.

    Returns:
        Four field-role lists used by documentation and the quality report.
    """

    available = list(columns)
    primary = [column for column in PRIMARY_KEY if column in available]
    tracking = [column for column in TRACKING_COLUMNS if column in available]
    prohibited = primary + tracking
    candidates = [column for column in available if column not in prohibited]
    return {
        "primary_key_fields": primary,
        "tracking_fields": tracking,
        "candidate_feature_fields": candidates,
        "prohibited_model_input_fields": prohibited,
    }


def _update_quality_counts(
    wide: pd.DataFrame,
    missing_counts: dict[str, int],
    abnormal_counts: dict[str, int],
) -> int:
    """Accumulate missing, abnormal, and time-window counts for one batch.

    Args:
        wide: One completed wide-table batch.
        missing_counts: Mutable per-field missing-value totals.
        abnormal_counts: Mutable per-field abnormal-value totals.

    Returns:
        Number of time-window violations found in this batch.
    """

    for column, count in wide.isna().sum().items():
        missing_counts[column] = missing_counts.get(column, 0) + int(count)

    numeric = wide.select_dtypes(include=["number"])
    for column in numeric.columns:
        infinity_count = int(np.isinf(numeric[column].to_numpy()).sum())
        if infinity_count:
            abnormal_counts[column] = (
                abnormal_counts.get(column, 0) + infinity_count
            )
        if (
            column.endswith(("_count", "_rank", "_ratio", "_days", "_days_ago"))
            or column.startswith("days_since_")
            or column in ALLOWED_MISSING_COLUMNS
        ):
            negative_count = int(numeric[column].lt(0).sum())
            if negative_count:
                abnormal_counts[column] = (
                    abnormal_counts.get(column, 0) + negative_count
                )

    for column in ["user_id", "item_id", "category_id"]:
        invalid = int(pd.to_numeric(wide[column], errors="coerce").le(0).sum())
        if invalid:
            abnormal_counts[column] = abnormal_counts.get(column, 0) + invalid
    for column in [
        "item_total_count_rank",
        "item_unique_user_count_rank",
        "item_active_day_count_rank",
        "item_buy_count_rank",
    ]:
        invalid = int(pd.to_numeric(wide[column], errors="coerce").le(0).sum())
        if invalid:
            abnormal_counts[column] = abnormal_counts.get(column, 0) + invalid

    invalid_behavior = int(
        (~wide["last_behavior_type"].isin(["pv", "fav", "cart", "buy"])).sum()
    )
    sequence_pattern = r"(?:pv|fav|cart|buy)(?:→(?:pv|fav|cart|buy)){0,9}"
    invalid_sequence = int(
        (~wide["last_10_behavior_sequence"].str.fullmatch(sequence_pattern)).sum()
    )
    inconsistent_last_behavior = int(
        wide["last_10_behavior_sequence"]
        .str.rsplit("→", n=1)
        .str[-1]
        .ne(wide["last_behavior_type"])
        .sum()
    )
    invalid_activity_level = int(
        (~wide["user_activity_level"].isin(["low", "medium", "high"])).sum()
    )
    for column, count in {
        "last_behavior_type": invalid_behavior,
        "last_10_behavior_sequence": invalid_sequence,
        "last_behavior_sequence_tail": inconsistent_last_behavior,
        "user_activity_level": invalid_activity_level,
    }.items():
        if count:
            abnormal_counts[column] = abnormal_counts.get(column, 0) + count

    invalid_active_ratio = int(
        (~wide["user_active_day_ratio"].between(0, 1)).sum()
    )
    if invalid_active_ratio:
        abnormal_counts["user_active_day_ratio"] = (
            abnormal_counts.get("user_active_day_ratio", 0)
            + invalid_active_ratio
        )

    hour_invalid = int((~wide["last_behavior_hour"].between(0, 23)).sum())
    if hour_invalid:
        abnormal_counts["last_behavior_hour"] = (
            abnormal_counts.get("last_behavior_hour", 0) + hour_invalid
        )
    weekday_invalid = int((~wide["time_weekday"].between(0, 6)).sum())
    if weekday_invalid:
        abnormal_counts["time_weekday"] = (
            abnormal_counts.get("time_weekday", 0) + weekday_invalid
        )
    weekday_mismatch = int(
        wide["time_weekday"].ne(wide["last_behavior_date"].dt.weekday).sum()
    )
    if weekday_mismatch:
        abnormal_counts["time_weekday_mismatch"] = (
            abnormal_counts.get("time_weekday_mismatch", 0)
            + weekday_mismatch
        )

    category_first = pd.to_datetime(
        wide["category_first_event_time"], errors="coerce"
    )
    category_last = pd.to_datetime(
        wide["category_last_event_time"], errors="coerce"
    )
    valid_category_time = (
        category_first.notna()
        & category_last.notna()
        & category_first.le(category_last)
        & category_first.ge(wide["history_start"])
        & category_last.lt(wide["history_end"] + pd.Timedelta(days=1))
        & category_last.lt(wide["label_date"])
    )
    invalid_category_time = int((~valid_category_time).sum())
    if invalid_category_time:
        abnormal_counts["category_event_time"] = (
            abnormal_counts.get("category_event_time", 0)
            + invalid_category_time
        )

    valid_time = (
        wide["history_start"].le(wide["last_behavior_date"])
        & wide["last_behavior_date"].le(wide["history_end"])
        & wide["history_end"].lt(wide["label_date"])
    )
    return int((~valid_time).sum())


def _validate_quality_report(report: Mapping[str, Any]) -> None:
    """Reject a generated wide table that violates the Issue #9 checks.

    Args:
        report: Completed quality metrics for the generated wide table.

    Returns:
        None. Failed required checks raise ``ValueError``.
    """

    unexpected_missing = {
        column: count
        for column, count in report["missing_values"].items()
        if count and column not in ALLOWED_MISSING_COLUMNS
    }
    failures = {
        "row_count_mismatch": report["row_count"] != report["anchor_row_count"],
        "duplicate_primary_keys": report["duplicate_primary_key_count"] != 0,
        "missing_primary_keys": report["missing_primary_key_count"] != 0,
        "unexpected_missing_values": bool(unexpected_missing),
        "abnormal_values": bool(report["abnormal_values"]),
        "time_window_violations": report["time_window_violation_count"] != 0,
    }
    failed = [name for name, value in failures.items() if value]
    if failed:
        raise ValueError("Wide-table quality checks failed: " + ", ".join(failed))


def generate_user_item_feature_wide(
    feature_directory: str | Path,
    output_parquet: str | Path,
    quality_report_path: str | Path,
    batch_size: int = 200_000,
) -> dict[str, Any]:
    """Generate and quality-check the user-item feature wide table.

    Args:
        feature_directory: Directory containing exactly the eight required
            stage-two feature-table inputs.
        output_parquet: Destination for the local wide-table Parquet.
        quality_report_path: Destination for the JSON quality report.
        batch_size: Maximum user-item sequence rows merged per batch.

    Returns:
        A result containing output paths and the complete quality report.
        No label, model, sampling, evaluation, or feature selection is created.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paths = _feature_paths(feature_directory)
    output = Path(output_parquet).expanduser().resolve()
    report_path = Path(quality_report_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    temporary_report.unlink(missing_ok=True)

    anchor_keys = pd.read_parquet(
        paths["user_sequence"], columns=list(PRIMARY_KEY)
    )
    anchor_row_count = len(anchor_keys)
    duplicate_primary_keys = int(anchor_keys.duplicated(list(PRIMARY_KEY)).sum())
    missing_primary_keys = int(anchor_keys[list(PRIMARY_KEY)].isna().any(axis=1).sum())
    split_row_counts = {
        str(name): int(count)
        for name, count in anchor_keys["dataset_split"].value_counts().items()
    }
    actual_splits = set(split_row_counts)
    expected_splits = set(WINDOWS_BY_SPLIT)
    if actual_splits != expected_splits:
        raise ValueError(
            "user_sequence dataset splits do not match the fixed stage-two "
            f"windows: {sorted(actual_splits)}"
        )
    del anchor_keys

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    row_count = 0
    missing_counts: dict[str, int] = {}
    abnormal_counts: dict[str, int] = {}
    time_window_violations = 0
    output_columns: list[str] = []

    try:
        sequence_dataset = ds.dataset(paths["user_sequence"], format="parquet")
        for split in WINDOWS_BY_SPLIT:
            lookups = build_split_lookups(paths, split)
            scanner = sequence_dataset.scanner(
                filter=ds.field("dataset_split") == split,
                batch_size=batch_size,
            )
            for record_batch in scanner.to_batches():
                if record_batch.num_rows == 0:
                    continue
                sequence_batch = record_batch.to_pandas()
                wide = merge_user_item_feature_batch(sequence_batch, lookups)
                if "label" in wide.columns:
                    raise ValueError("The wide table must not contain label.")
                row_count += len(wide)
                time_window_violations += _update_quality_counts(
                    wide, missing_counts, abnormal_counts
                )
                arrow_table = pa.Table.from_pandas(wide, preserve_index=False)
                if writer is None:
                    output_schema = arrow_table.schema
                    output_columns = list(wide.columns)
                    writer = pq.ParquetWriter(
                        temporary_output,
                        output_schema,
                        compression="snappy",
                    )
                elif arrow_table.schema != output_schema:
                    arrow_table = arrow_table.cast(output_schema)
                writer.write_table(arrow_table)
                del sequence_batch, wide, arrow_table
            del lookups
        if writer is None:
            raise ValueError("No user-item rows were generated.")
        writer.close()
        writer = None

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_feature_tables": {
                name: str(path) for name, path in paths.items()
            },
            "output_path": str(output),
            "primary_key": list(PRIMARY_KEY),
            "anchor_row_count": anchor_row_count,
            "row_count": row_count,
            "column_count": len(output_columns),
            "dataset_split_row_counts": split_row_counts,
            "duplicate_primary_key_count": duplicate_primary_keys,
            "missing_primary_key_count": missing_primary_keys,
            "missing_values": missing_counts,
            "abnormal_values": abnormal_counts,
            "time_window_violation_count": time_window_violations,
            "field_roles": feature_role_mapping(output_columns),
        }
        _validate_quality_report(report)
        report["status"] = "passed"
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(output)
        report["output_file_size_bytes"] = output.stat().st_size
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_path)
        return {
            "output_path": output,
            "quality_report_path": report_path,
            "quality_report": report,
        }
    finally:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
