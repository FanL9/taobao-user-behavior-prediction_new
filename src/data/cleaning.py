"""Core cleaning utilities for stage-one user-behavior data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


RAW_COLUMNS = (
    "time",
    "user_id",
    "item_id",
    "item_category",
    "behavior_type",
)

CLEAN_COLUMNS = (
    "time",
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "behavior_name",
    "behavior_date",
    "behavior_hour",
    "weekday",
)

BEHAVIOR_NAMES = {
    1: "pv",
    2: "fav",
    3: "cart",
    4: "buy",
}


@dataclass(frozen=True)
class ChunkCleaningStats:
    """Statistics collected while cleaning one input chunk."""

    input_rows: int
    output_rows: int
    removed_missing_rows: int
    removed_invalid_id_rows: int
    removed_invalid_behavior_rows: int
    removed_invalid_time_rows: int


@dataclass(frozen=True)
class ChunkCleaningResult:
    """Cleaned chunk plus its cleaning statistics."""

    frame: pd.DataFrame
    stats: ChunkCleaningStats


def clean_chunk(chunk: pd.DataFrame) -> ChunkCleaningResult:
    """Validate and normalize one chunk of raw user-behavior data.

    This function handles field validation, invalid-row removal, type
    normalization, field renaming, and derived time/behavior fields.

    Global exact-duplicate removal is intentionally handled by the full
    cleaning pipeline rather than inside this function because duplicate
    rows may occur in different CSV chunks.
    """
    actual_columns = tuple(chunk.columns)
    if actual_columns != RAW_COLUMNS:
        raise ValueError(
            "Input columns do not match the raw-data contract. "
            f"Expected {list(RAW_COLUMNS)}, got {list(actual_columns)}."
        )

    frame = chunk.copy()

    # Normalize all raw values to stripped strings first.
    for column in RAW_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()

    input_rows = len(frame)

    missing_mask = (
        frame[list(RAW_COLUMNS)].isna().any(axis=1)
        | frame[list(RAW_COLUMNS)].eq("").any(axis=1)
    )
    removed_missing_rows = int(missing_mask.sum())
    frame = frame.loc[~missing_mask].copy()

    numeric_ids = {}
    invalid_id_mask = pd.Series(False, index=frame.index)

    for column in ("user_id", "item_id", "item_category"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.notna() & numeric.gt(0) & numeric.le(2**63 - 1)
        invalid_id_mask |= ~valid
        numeric_ids[column] = numeric

    removed_invalid_id_rows = int(invalid_id_mask.sum())
    frame = frame.loc[~invalid_id_mask].copy()

    for column in ("user_id", "item_id", "item_category"):
        frame[column] = numeric_ids[column].loc[frame.index].astype("int64")

    behavior_numeric = pd.to_numeric(
        frame["behavior_type"],
        errors="coerce",
    )
    valid_behavior = behavior_numeric.isin((1, 2, 3, 4))
    removed_invalid_behavior_rows = int((~valid_behavior).sum())
    frame = frame.loc[valid_behavior].copy()
    frame["behavior_type"] = (
        behavior_numeric.loc[frame.index].astype("int8")
    )

    parsed_time = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H",
        errors="coerce",
        exact=True,
    )
    valid_time = parsed_time.notna()
    removed_invalid_time_rows = int((~valid_time).sum())

    frame = frame.loc[valid_time].copy()
    parsed_time = parsed_time.loc[frame.index]

    # Issue #3 standardizes item_category to category_id.
    frame = frame.rename(columns={"item_category": "category_id"})

    frame["behavior_name"] = (
        frame["behavior_type"].map(BEHAVIOR_NAMES).astype("string")
    )

    # Preserve the documented raw time representation.
    frame["time"] = parsed_time.dt.strftime("%Y-%m-%d %H").astype("string")
    frame["behavior_date"] = parsed_time.dt.strftime("%Y-%m-%d").astype(
        "string"
    )
    frame["behavior_hour"] = parsed_time.dt.hour.astype("int8")

    # Monday=0 ... Sunday=6.
    frame["weekday"] = parsed_time.dt.weekday.astype("int8")

    frame = frame.loc[:, CLEAN_COLUMNS].reset_index(drop=True)

    stats = ChunkCleaningStats(
        input_rows=input_rows,
        output_rows=len(frame),
        removed_missing_rows=removed_missing_rows,
        removed_invalid_id_rows=removed_invalid_id_rows,
        removed_invalid_behavior_rows=removed_invalid_behavior_rows,
        removed_invalid_time_rows=removed_invalid_time_rows,
    )

    return ChunkCleaningResult(frame=frame, stats=stats)
