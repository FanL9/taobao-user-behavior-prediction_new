"""Build the four stage-two intermediate tables from cleaned behavior data.

The module only creates user, item, category, and time aggregates. It does not
create modeling samples, labels, user-item rows, or a final feature table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "time",
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "behavior_date",
    "behavior_hour",
    "weekday",
)

BEHAVIOR_COUNT_COLUMNS = {
    1: "pv_count",
    2: "fav_count",
    3: "cart_count",
    4: "buy_count",
}


@dataclass(frozen=True)
class HistoryWindow:
    """One leakage-safe history window and its following label date.

    Attributes:
        dataset_split: Window name used in every intermediate-table key.
        history_start: First included history date.
        history_end: Last included history date.
        label_date: Excluded date reserved for later label construction.
    """

    dataset_split: str
    history_start: str
    history_end: str
    label_date: str


HISTORY_WINDOWS = (
    HistoryWindow("train", "2025-11-18", "2025-12-07", "2025-12-08"),
    HistoryWindow("validation", "2025-12-09", "2025-12-14", "2025-12-15"),
    HistoryWindow("test", "2025-12-16", "2025-12-17", "2025-12-18"),
)

OUTPUT_FILENAMES = {
    "user": "user_intermediate.parquet",
    "item": "item_intermediate.parquet",
    "category": "category_intermediate.parquet",
    "time": "time_intermediate.parquet",
}


def _validate_windows(windows: Iterable[HistoryWindow]) -> tuple[HistoryWindow, ...]:
    """Validate names, ordering, and label-date exclusion for all windows.

    Args:
        windows: Candidate history-window definitions.

    Returns:
        Validated windows as an immutable tuple.
    """

    normalized = tuple(windows)
    if not normalized:
        raise ValueError("At least one history window is required.")
    names = [window.dataset_split for window in normalized]
    if len(names) != len(set(names)):
        raise ValueError("dataset_split values must be unique.")

    previous_label: pd.Timestamp | None = None
    for window in normalized:
        start = pd.Timestamp(window.history_start)
        end = pd.Timestamp(window.history_end)
        label = pd.Timestamp(window.label_date)
        if not start <= end < label:
            raise ValueError(
                f"Invalid window {window.dataset_split!r}: expected "
                "history_start <= history_end < label_date."
            )
        if previous_label is not None and start <= previous_label:
            raise ValueError("History windows and label dates must not overlap.")
        previous_label = label
    return normalized


def _prepare_clean_data(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the clean-data contract and attach a parsed event timestamp.

    Args:
        clean_data: Stage-one clean behavior rows.

    Returns:
        Required clean columns plus parsed ``event_time`` and normalized time
        fields.
    """

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(clean_data.columns))
    if missing_columns:
        raise ValueError(f"Clean data is missing columns: {missing_columns}")

    frame = clean_data.loc[:, REQUIRED_COLUMNS].copy()
    event_time = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H",
        errors="coerce",
        exact=True,
    )
    if event_time.isna().any():
        raise ValueError("Clean data contains invalid time values.")
    if not frame["behavior_type"].isin(BEHAVIOR_COUNT_COLUMNS).all():
        raise ValueError("Clean data contains behavior_type values outside 1-4.")

    expected_date = event_time.dt.strftime("%Y-%m-%d")
    expected_hour = event_time.dt.hour
    expected_weekday = event_time.dt.weekday
    if not frame["behavior_date"].astype("string").eq(expected_date).all():
        raise ValueError("behavior_date is inconsistent with time.")
    if not pd.to_numeric(frame["behavior_hour"], errors="coerce").eq(expected_hour).all():
        raise ValueError("behavior_hour is inconsistent with time.")
    if not pd.to_numeric(frame["weekday"], errors="coerce").eq(expected_weekday).all():
        raise ValueError("weekday is inconsistent with time.")

    frame["event_time"] = event_time
    frame["behavior_date"] = event_time.dt.normalize()
    frame["behavior_hour"] = event_time.dt.hour.astype("int8")
    frame["weekday"] = event_time.dt.weekday.astype("int8")
    return frame


def select_history(clean_data: pd.DataFrame, window: HistoryWindow) -> pd.DataFrame:
    """Select only events in one history window, excluding its label date.

    Args:
        clean_data: Data prepared by :func:`_prepare_clean_data` or a compatible
            frame containing ``event_time``.
        window: Inclusive history dates and the excluded later label date.

    Returns:
        A copy containing events from history_start 00:00 through history_end
        23:59. No label-date or later event is returned.
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
    """Count the four behavior codes for one intermediate-table grain.

    Args:
        history: Events from one validated history window.
        keys: Columns defining the target table grain.

    Returns:
        One row per key with ``pv``, ``fav``, ``cart``, and ``buy`` counts.
    """

    counts = (
        history.groupby(keys + ["behavior_type"], observed=True, sort=False)
        .size()
        .unstack("behavior_type", fill_value=0)
        .reindex(columns=list(BEHAVIOR_COUNT_COLUMNS), fill_value=0)
        .rename(columns=BEHAVIOR_COUNT_COLUMNS)
        .reset_index()
    )
    for column in BEHAVIOR_COUNT_COLUMNS.values():
        counts[column] = counts[column].astype("int64")
    return counts


def _add_window_columns(table: pd.DataFrame, window: HistoryWindow) -> pd.DataFrame:
    """Attach the exact history and label dates used for every aggregate row.

    Args:
        table: Aggregated rows for one history window.
        window: Window metadata to attach.

    Returns:
        The table with split, history boundary, and excluded label-date fields.
    """

    table.insert(0, "dataset_split", window.dataset_split)
    table["history_start"] = pd.Timestamp(window.history_start)
    table["history_end"] = pd.Timestamp(window.history_end)
    table["label_date"] = pd.Timestamp(window.label_date)
    return table


def build_user_intermediate(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build one row per user for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Window metadata written to every output row.

    Returns:
        User-grain counts, distinct-object counts, active days, and observed
        first/last event timestamps. All statistics use only ``history``.
    """

    keys = ["user_id"]
    base = (
        history.groupby(keys, observed=True, sort=False)
        .agg(
            event_count=("user_id", "size"),
            unique_item_count=("item_id", "nunique"),
            unique_category_count=("category_id", "nunique"),
            active_day_count=("behavior_date", "nunique"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    table = base.merge(_behavior_counts(history, keys), on=keys, validate="one_to_one")
    return _add_window_columns(table, window)


def build_item_intermediate(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build one row per item for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Window metadata written to every output row.

    Returns:
        Item-grain category mapping, behavior counts, user/day counts, and
        observed first/last event timestamps. Multi-category items are rejected.
    """

    category_counts = history.groupby("item_id", observed=True)["category_id"].nunique()
    if category_counts.gt(1).any():
        raise ValueError("An item_id maps to more than one category_id in a window.")

    keys = ["item_id"]
    base = (
        history.groupby(keys, observed=True, sort=False)
        .agg(
            category_id=("category_id", "first"),
            event_count=("item_id", "size"),
            unique_user_count=("user_id", "nunique"),
            active_day_count=("behavior_date", "nunique"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    table = base.merge(_behavior_counts(history, keys), on=keys, validate="one_to_one")
    return _add_window_columns(table, window)


def build_category_intermediate(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build one row per category for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Window metadata written to every output row.

    Returns:
        Category-grain behavior, user, item, and day counts plus observed
        first/last event timestamps, all calculated from ``history``.
    """

    keys = ["category_id"]
    base = (
        history.groupby(keys, observed=True, sort=False)
        .agg(
            event_count=("category_id", "size"),
            unique_user_count=("user_id", "nunique"),
            unique_item_count=("item_id", "nunique"),
            active_day_count=("behavior_date", "nunique"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    table = base.merge(_behavior_counts(history, keys), on=keys, validate="one_to_one")
    return _add_window_columns(table, window)


def build_time_intermediate(
    history: pd.DataFrame,
    window: HistoryWindow,
) -> pd.DataFrame:
    """Build one row per observed date and hour for one history window.

    Args:
        history: Events already restricted to ``window``.
        window: Window metadata written to every output row.

    Returns:
        Date-hour-grain behavior and distinct-entity counts. Only observed
        date-hour combinations are emitted; no future or synthetic rows appear.
    """

    keys = ["behavior_date", "behavior_hour"]
    base = (
        history.groupby(keys, observed=True, sort=False)
        .agg(
            weekday=("weekday", "first"),
            event_count=("behavior_hour", "size"),
            unique_user_count=("user_id", "nunique"),
            unique_item_count=("item_id", "nunique"),
            unique_category_count=("category_id", "nunique"),
        )
        .reset_index()
    )
    table = base.merge(_behavior_counts(history, keys), on=keys, validate="one_to_one")
    table = table.sort_values(keys).reset_index(drop=True)
    return _add_window_columns(table, window)


def build_intermediate_tables(
    clean_data: pd.DataFrame,
    windows: Iterable[HistoryWindow] = HISTORY_WINDOWS,
) -> dict[str, pd.DataFrame]:
    """Build the four stage-two intermediate tables for all history windows.

    Args:
        clean_data: Stage-one clean data using the documented standard schema.
        windows: Leakage-safe history windows. Defaults to train, validation,
            and test windows fixed by the stage-two contract.

    Returns:
        Mapping with exactly ``user``, ``item``, ``category``, and ``time``
        tables. No modeling label, user-item grain, or final feature table is
        created.
    """

    prepared = _prepare_clean_data(clean_data)
    validated_windows = _validate_windows(windows)
    builders = {
        "user": build_user_intermediate,
        "item": build_item_intermediate,
        "category": build_category_intermediate,
        "time": build_time_intermediate,
    }
    results: dict[str, list[pd.DataFrame]] = {name: [] for name in builders}

    for window in validated_windows:
        history = select_history(prepared, window)
        for name, builder in builders.items():
            results[name].append(builder(history, window))

    return {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in results.items()
    }


def generate_intermediate_tables(
    input_parquet: str | Path,
    output_directory: str | Path,
    windows: Iterable[HistoryWindow] = HISTORY_WINDOWS,
) -> dict[str, Path]:
    """Read clean Parquet and atomically write exactly four intermediate tables.

    Args:
        input_parquet: Stage-one ``user_behavior_clean.parquet`` path.
        output_directory: Directory receiving the four Parquet tables.
        windows: History windows passed to :func:`build_intermediate_tables`.

    Returns:
        Mapping from table name to its resolved Parquet path.

    Raises:
        FileNotFoundError: If ``input_parquet`` does not exist.
        ValueError: If data or window definitions violate the contract.
    """

    source = Path(input_parquet).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Clean Parquet does not exist: {source}")

    clean_data = pd.read_parquet(source, columns=list(REQUIRED_COLUMNS))
    tables = build_intermediate_tables(clean_data, windows)
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
