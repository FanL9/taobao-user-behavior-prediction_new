"""Read-only quality checks for the stage-one user-behavior CSV."""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_COLUMNS = (
    "time",
    "user_id",
    "item_id",
    "item_category",
    "behavior_type",
)
SUSPECTED_DUPLICATE_KEY = (
    "user_id",
    "item_id",
    "behavior_type",
    "time",
)
ID_COLUMNS = ("user_id", "item_id", "item_category")
STRING_SCHEMA = pa.schema(
    [pa.field(column, pa.string()) for column in EXPECTED_COLUMNS]
)
SUSPECTED_SCHEMA = pa.schema(
    [pa.field(column, pa.string()) for column in SUSPECTED_DUPLICATE_KEY]
)


def _validate_header(csv_path: Path, encoding: str) -> None:
    """Require the five fields and their documented order."""

    actual = tuple(pd.read_csv(csv_path, encoding=encoding, nrows=0).columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV columns do not match the project contract. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {list(actual)}."
        )


def _valid_positive_integer(values: pd.Series) -> pd.Series:
    """Return whether each string is a positive signed-64-bit integer."""

    stripped = values.str.strip()
    syntax_valid = stripped.str.fullmatch(r"[1-9][0-9]*", na=False)
    numeric = pd.to_numeric(stripped.where(syntax_valid), errors="coerce")
    return syntax_valid & numeric.le(2**63 - 1)


def _write_partitions(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    prefix: str,
    partition_count: int,
    directory: Path,
    writers: dict[tuple[str, int], pq.ParquetWriter],
) -> None:
    """Hash-partition rows so global duplicate counts remain memory bounded."""

    selected = frame.loc[:, columns]
    buckets = (
        pd.util.hash_pandas_object(selected, index=False) % partition_count
    ).astype("uint16")
    schema = STRING_SCHEMA if columns == EXPECTED_COLUMNS else SUSPECTED_SCHEMA

    for bucket in buckets.unique():
        bucket_number = int(bucket)
        part = selected.loc[buckets.eq(bucket)].reset_index(drop=True)
        table = pa.Table.from_pandas(
            part,
            schema=schema,
            preserve_index=False,
        )
        writer_key = (prefix, bucket_number)
        if writer_key not in writers:
            path = directory / f"{prefix}_{bucket_number:03d}.parquet"
            writers[writer_key] = pq.ParquetWriter(path, schema, compression="snappy")
        writers[writer_key].write_table(table)


def _duplicate_summary(
    directory: Path,
    prefix: str,
    columns: tuple[str, ...],
) -> dict[str, int]:
    """Count repeated groups and rows from all hash partitions."""

    group_count = 0
    record_count = 0
    excess_count = 0
    for path in sorted(directory.glob(f"{prefix}_*.parquet")):
        frame = pd.read_parquet(path, columns=list(columns))
        counts = frame.value_counts(subset=list(columns), dropna=False)
        repeated = counts[counts > 1]
        group_count += len(repeated)
        record_count += int(repeated.sum())
        excess_count += int((repeated - 1).sum())
    return {
        "group_count": group_count,
        "record_count": record_count,
        "excess_count": excess_count,
    }


def check_csv_quality(
    csv_path: str | Path,
    *,
    chunksize: int = 100_000,
    encoding: str = "utf-8-sig",
    duplicate_partitions: int = 32,
) -> dict[str, Any]:
    """Inspect the source CSV without modifying or cleaning it.

    The function checks data scale, missing fields, illegal identifiers,
    behavior codes, timestamp format, exact duplicate rows, and repeated
    ``user_id + item_id + behavior_type + time`` keys. Duplicate calculations
    use temporary hash-partitioned Parquet files so records in different CSV
    chunks are still compared correctly.

    Args:
        csv_path: Source ``user_behavior_processed.csv`` path.
        chunksize: Positive number of rows read per chunk.
        encoding: Source CSV text encoding.
        duplicate_partitions: Positive number of temporary hash partitions.

    Returns:
        A JSON-serializable dictionary containing quality metrics and the
        documented read-only decisions. No source or cleaned data is written.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If arguments or the CSV header are invalid.
    """

    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CSV input does not exist: {source}")
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    if duplicate_partitions <= 0:
        raise ValueError("duplicate_partitions must be a positive integer.")
    _validate_header(source, encoding)

    started_at = time.perf_counter()
    total_rows = 0
    rows_with_any_missing = 0
    missing_by_column = Counter({column: 0 for column in EXPECTED_COLUMNS})
    invalid_by_column = Counter({column: 0 for column in EXPECTED_COLUMNS})
    unique_values = {column: set() for column in ID_COLUMNS}
    behavior_counts: Counter[str] = Counter()
    earliest_time: pd.Timestamp | None = None
    latest_time: pd.Timestamp | None = None

    with tempfile.TemporaryDirectory(prefix="taobao-quality-check-") as temp_dir:
        partition_directory = Path(temp_dir)
        writers: dict[tuple[str, int], pq.ParquetWriter] = {}
        try:
            reader = pd.read_csv(
                source,
                encoding=encoding,
                dtype="string",
                chunksize=chunksize,
            )
            for chunk in reader:
                total_rows += len(chunk)
                stripped = chunk.apply(lambda column: column.str.strip())
                missing = stripped.isna() | stripped.eq("")
                rows_with_any_missing += int(missing.any(axis=1).sum())
                for column in EXPECTED_COLUMNS:
                    missing_by_column[column] += int(missing[column].sum())

                for column in ID_COLUMNS:
                    valid = _valid_positive_integer(stripped[column])
                    invalid = ~missing[column] & ~valid
                    invalid_by_column[column] += int(invalid.sum())
                    unique_values[column].update(stripped.loc[valid, column].tolist())

                valid_behavior = stripped["behavior_type"].isin(("1", "2", "3", "4"))
                invalid_by_column["behavior_type"] += int(
                    (~missing["behavior_type"] & ~valid_behavior).sum()
                )
                behavior_counts.update(
                    stripped.loc[valid_behavior, "behavior_type"].tolist()
                )

                parsed_time = pd.to_datetime(
                    stripped["time"],
                    format="%Y-%m-%d %H",
                    errors="coerce",
                    exact=True,
                )
                valid_time = parsed_time.notna()
                invalid_by_column["time"] += int(
                    (~missing["time"] & ~valid_time).sum()
                )
                if valid_time.any():
                    chunk_min = parsed_time.loc[valid_time].min()
                    chunk_max = parsed_time.loc[valid_time].max()
                    earliest_time = (
                        chunk_min
                        if earliest_time is None
                        else min(earliest_time, chunk_min)
                    )
                    latest_time = (
                        chunk_max if latest_time is None else max(latest_time, chunk_max)
                    )

                _write_partitions(
                    chunk,
                    EXPECTED_COLUMNS,
                    "exact",
                    duplicate_partitions,
                    partition_directory,
                    writers,
                )
                _write_partitions(
                    chunk,
                    SUSPECTED_DUPLICATE_KEY,
                    "suspected",
                    duplicate_partitions,
                    partition_directory,
                    writers,
                )
        finally:
            for writer in writers.values():
                writer.close()

        exact_duplicates = _duplicate_summary(
            partition_directory,
            "exact",
            EXPECTED_COLUMNS,
        )
        suspected_duplicates = _duplicate_summary(
            partition_directory,
            "suspected",
            SUSPECTED_DUPLICATE_KEY,
        )

    issue_count = (
        sum(missing_by_column.values())
        + sum(invalid_by_column.values())
        + exact_duplicates["record_count"]
    )
    return {
        "input": {
            "file_name": source.name,
            "size_bytes": source.stat().st_size,
            "columns": list(EXPECTED_COLUMNS),
        },
        "scale": {
            "row_count": total_rows,
            "column_count": len(EXPECTED_COLUMNS),
            "unique_user_count": len(unique_values["user_id"]),
            "unique_item_count": len(unique_values["item_id"]),
            "unique_category_count": len(unique_values["item_category"]),
        },
        "completeness": {
            "rows_with_any_missing": rows_with_any_missing,
            "missing_by_column": dict(missing_by_column),
        },
        "validity": {
            "invalid_by_column": dict(invalid_by_column),
            "valid_behavior_counts": dict(sorted(behavior_counts.items())),
            "valid_time_min": (
                earliest_time.strftime("%Y-%m-%d %H") if earliest_time is not None else None
            ),
            "valid_time_max": (
                latest_time.strftime("%Y-%m-%d %H") if latest_time is not None else None
            ),
        },
        "duplicates": {
            "exact_rows": exact_duplicates,
            "suspected_key": {
                "columns": list(SUSPECTED_DUPLICATE_KEY),
                **suspected_duplicates,
            },
        },
        "decisions": {
            "source_modified": False,
            "cleaning_performed": False,
            "exact_duplicates_removed": False,
            "suspected_duplicates_removed": False,
            "id_outlier_method": "none",
        },
        "status": "PASS" if issue_count == 0 else "REVIEW",
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
    }


def write_quality_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a quality-check result as UTF-8 JSON and return its path.

    Args:
        report: Result returned by :func:`check_csv_quality`.
        output_path: Destination JSON path.

    Returns:
        The resolved path of the written report.
    """

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
