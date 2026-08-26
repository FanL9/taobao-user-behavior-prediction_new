"""Validated, chunked conversion from the stage-one CSV to Parquet."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

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

ARROW_SCHEMA = pa.schema(
    [
        pa.field("time", pa.string()),
        pa.field("user_id", pa.int64()),
        pa.field("item_id", pa.int64()),
        pa.field("item_category", pa.int64()),
        pa.field("behavior_type", pa.int8()),
    ]
)

PANDAS_DTYPES = {
    "time": "string",
    "user_id": "Int64",
    "item_id": "Int64",
    "item_category": "Int64",
    "behavior_type": "Int8",
}


@dataclass(frozen=True)
class ConversionResult:
    """Summary returned after a successful conversion.

    Attributes:
        input_path: CSV file that was read.
        output_path: Parquet file written atomically.
        row_count: Number of converted data rows.
        file_size_bytes: Final Parquet file size.
        elapsed_seconds: Wall-clock conversion time.
    """

    input_path: Path
    output_path: Path
    row_count: int
    file_size_bytes: int
    elapsed_seconds: float


def _validate_header(input_path: Path, encoding: str) -> None:
    """Validate that the CSV header exactly matches the project contract."""

    actual_columns = tuple(
        pd.read_csv(input_path, encoding=encoding, nrows=0).columns
    )
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV columns do not match the stage-one contract. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {list(actual_columns)}."
        )


def _temporary_output(output_path: Path) -> Path:
    """Create a temporary path beside the target for an atomic replacement."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".parquet.tmp",
        dir=output_path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _validate_chunk(chunk: pd.DataFrame, row_offset: int) -> None:
    """Reject nulls and values outside the documented stage-one contract."""

    null_columns = chunk.columns[chunk.isna().any()].tolist()
    if null_columns:
        raise ValueError(
            f"Rows after offset {row_offset} contain null or unparseable values "
            f"in columns: {null_columns}."
        )

    invalid_time = pd.to_datetime(
        chunk["time"],
        format="%Y-%m-%d %H",
        errors="coerce",
        exact=True,
    ).isna()
    if invalid_time.any():
        raise ValueError(
            f"Rows after offset {row_offset} contain time values that do not "
            "match YYYY-MM-DD HH."
        )

    id_columns = ("user_id", "item_id", "item_category")
    invalid_ids = [column for column in id_columns if chunk[column].le(0).any()]
    if invalid_ids:
        raise ValueError(
            f"Rows after offset {row_offset} contain non-positive identifiers "
            f"in columns: {invalid_ids}."
        )

    if not chunk["behavior_type"].isin((1, 2, 3, 4)).all():
        raise ValueError(
            f"Rows after offset {row_offset} contain behavior_type values "
            "outside 1, 2, 3, 4."
        )


def convert_csv_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    chunksize: int = 250_000,
    compression: str = "snappy",
    encoding: str = "utf-8-sig",
    overwrite: bool = False,
) -> ConversionResult:
    """Convert the contracted stage-one CSV to a typed Parquet file.

    The source is read in chunks, so peak memory is controlled by ``chunksize``.
    Output is first written to a sibling temporary file and only moved into place
    after every chunk succeeds. The function performs schema conversion only; it
    does not clean, deduplicate, sort, or otherwise change business values.

    Args:
        input_path: Source CSV path.
        output_path: Destination Parquet path. It must differ from ``input_path``.
        chunksize: Positive number of CSV rows read per chunk.
        compression: A compression codec supported by PyArrow.
        encoding: Source CSV text encoding.
        overwrite: Replace an existing destination when ``True``.

    Returns:
        A :class:`ConversionResult` with paths, row count, size, and elapsed time.

    Raises:
        FileNotFoundError: If the input does not exist.
        FileExistsError: If the output exists and ``overwrite`` is ``False``.
        ValueError: If arguments, columns, or field values violate the contract.
    """

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"CSV input does not exist: {source}")
    if source == destination:
        raise ValueError("Input and output paths must be different.")
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Parquet output already exists: {destination}. "
            "Pass overwrite=True to replace it."
        )

    _validate_header(source, encoding)
    temporary_path = _temporary_output(destination)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    started_at = time.perf_counter()

    try:
        reader = pd.read_csv(
            source,
            encoding=encoding,
            dtype=PANDAS_DTYPES,
            chunksize=chunksize,
        )
        for chunk in reader:
            _validate_chunk(chunk, row_count)
            table = pa.Table.from_pandas(
                chunk,
                schema=ARROW_SCHEMA,
                preserve_index=False,
                safe=True,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    ARROW_SCHEMA,
                    compression=compression,
                )
            writer.write_table(table)
            row_count += len(chunk)

        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path,
                ARROW_SCHEMA,
                compression=compression,
            )
        writer.close()
        writer = None
        temporary_path.replace(destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    elapsed_seconds = time.perf_counter() - started_at
    return ConversionResult(
        input_path=source,
        output_path=destination,
        row_count=row_count,
        file_size_bytes=destination.stat().st_size,
        elapsed_seconds=elapsed_seconds,
    )
