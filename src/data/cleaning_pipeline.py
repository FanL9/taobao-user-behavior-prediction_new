"""Full stage-one cleaning pipeline.

The pipeline reads the raw CSV in chunks, validates each chunk,
partitions cleaned rows by the suspected-duplicate key, performs
global exact deduplication, and writes cleaned CSV/Parquet outputs.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.cleaning import ChunkCleaningStats, clean_chunk


SUSPECTED_DUPLICATE_KEY = (
    "time",
    "user_id",
    "item_id",
    "behavior_type",
)

EXACT_DUPLICATE_KEY = (
    "time",
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
)


def _merge_chunk_stats(
    total: dict[str, int],
    stats: ChunkCleaningStats,
) -> None:
    """Accumulate statistics from one cleaned chunk."""
    values = asdict(stats)

    for key, value in values.items():
        total[key] = total.get(key, 0) + int(value)


def clean_user_behavior_file(
    input_csv: str | Path,
    output_csv: str | Path,
    output_parquet: str | Path,
    report_json: str | Path,
    *,
    chunksize: int = 250_000,
    partitions: int = 64,
    temp_dir: str | Path | None = None,
) -> dict:
    """Clean the complete raw user-behavior CSV.

    Exact duplicates are removed globally, including duplicates that
    occur in different input chunks.

    Suspected duplicates sharing user/item/behavior/hour are reported
    but are retained.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")

    if partitions <= 0:
        raise ValueError("partitions must be greater than zero.")

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)
    report_json = Path(report_json)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        temp_dir = output_parquet.parent / ".cleaning_tmp"

    temp_dir = Path(temp_dir)

    temp_output_csv = output_csv.with_suffix(
        output_csv.suffix + ".tmp"
    )
    temp_output_parquet = output_parquet.with_suffix(
        output_parquet.suffix + ".tmp"
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_output_csv.unlink(missing_ok=True)
    temp_output_parquet.unlink(missing_ok=True)

    partition_paths = [
        temp_dir / f"partition_{index:03d}.parquet"
        for index in range(partitions)
    ]

    partition_writers: dict[int, pq.ParquetWriter] = {}
    final_writer: pq.ParquetWriter | None = None

    aggregate_stats: dict[str, int] = {}

    started_at = time.perf_counter()

    try:
        # Pass 1:
        # Clean each input chunk and hash-partition rows.
        # Rows sharing the suspected-duplicate key always go to
        # the same partition, including rows from different chunks.
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                input_csv,
                dtype="string",
                chunksize=chunksize,
            ),
            start=1,
        ):
            print(
                f"[pass 1] cleaning chunk {chunk_number} "
                f"({len(chunk):,} rows)",
                flush=True,
            )

            result = clean_chunk(chunk)
            cleaned = result.frame

            _merge_chunk_stats(
                aggregate_stats,
                result.stats,
            )

            if cleaned.empty:
                continue

            hashes = pd.util.hash_pandas_object(
                cleaned[list(SUSPECTED_DUPLICATE_KEY)],
                index=False,
            ).to_numpy()

            partition_ids = hashes % partitions

            for partition_id in pd.unique(partition_ids):
                partition_id = int(partition_id)

                mask = partition_ids == partition_id

                part = cleaned.loc[mask].copy()

                if part.empty:
                    continue

                table = pa.Table.from_pandas(
                    part,
                    preserve_index=False,
                )

                if partition_id not in partition_writers:
                    partition_writers[partition_id] = (
                        pq.ParquetWriter(
                            partition_paths[partition_id],
                            table.schema,
                            compression="snappy",
                        )
                    )

                partition_writers[
                    partition_id
                ].write_table(table)

        for writer in partition_writers.values():
            writer.close()

        partition_writers.clear()

        exact_duplicates_removed = 0

        suspected_raw_groups = 0
        suspected_raw_records = 0
        suspected_raw_excess = 0

        suspected_retained_groups = 0
        suspected_retained_records = 0

        final_rows = 0
        csv_header_written = False

        # Pass 2:
        # Duplicate-key groups are now contained within individual
        # partitions, so exact deduplication is global.
        for partition_number, partition_path in enumerate(
            partition_paths,
            start=1,
        ):
            if not partition_path.exists():
                continue

            print(
                f"[pass 2] processing partition "
                f"{partition_number}/{partitions}",
                flush=True,
            )

            frame = pd.read_parquet(partition_path)

            raw_group_sizes = frame.groupby(
                list(SUSPECTED_DUPLICATE_KEY),
                dropna=False,
                sort=False,
            ).size()

            raw_suspected = raw_group_sizes[
                raw_group_sizes > 1
            ]

            suspected_raw_groups += int(
                len(raw_suspected)
            )
            suspected_raw_records += int(
                raw_suspected.sum()
            )
            suspected_raw_excess += int(
                (raw_suspected - 1).sum()
            )

            before_dedup = len(frame)

            frame = frame.drop_duplicates(
                subset=list(EXACT_DUPLICATE_KEY),
                keep="first",
            )

            exact_duplicates_removed += (
                before_dedup - len(frame)
            )

            retained_group_sizes = frame.groupby(
                list(SUSPECTED_DUPLICATE_KEY),
                dropna=False,
                sort=False,
            ).size()

            retained_suspected = retained_group_sizes[
                retained_group_sizes > 1
            ]

            suspected_retained_groups += int(
                len(retained_suspected)
            )

            suspected_retained_records += int(
                retained_suspected.sum()
            )

            frame = frame.reset_index(drop=True)

            final_rows += len(frame)

            frame.to_csv(
                temp_output_csv,
                mode="a",
                index=False,
                header=not csv_header_written,
                encoding="utf-8",
            )

            csv_header_written = True

            table = pa.Table.from_pandas(
                frame,
                preserve_index=False,
            )

            if final_writer is None:
                final_writer = pq.ParquetWriter(
                    temp_output_parquet,
                    table.schema,
                    compression="snappy",
                )

            final_writer.write_table(table)

        if final_writer is not None:
            final_writer.close()
            final_writer = None

        if not temp_output_csv.exists():
            raise RuntimeError(
                "Cleaning completed without producing CSV output."
            )

        if not temp_output_parquet.exists():
            raise RuntimeError(
                "Cleaning completed without producing Parquet output."
            )

        elapsed_seconds = (
            time.perf_counter() - started_at
        )

        input_rows = aggregate_stats.get(
            "input_rows",
            0,
        )

        report = {
            "input": {
                "path": str(input_csv),
                "rows": input_rows,
            },
            "output": {
                "csv": str(output_csv),
                "parquet": str(output_parquet),
                "rows": final_rows,
            },
            "removed": {
                "missing_rows": aggregate_stats.get(
                    "removed_missing_rows",
                    0,
                ),
                "invalid_id_rows": aggregate_stats.get(
                    "removed_invalid_id_rows",
                    0,
                ),
                "invalid_behavior_rows": aggregate_stats.get(
                    "removed_invalid_behavior_rows",
                    0,
                ),
                "invalid_time_rows": aggregate_stats.get(
                    "removed_invalid_time_rows",
                    0,
                ),
                "exact_duplicate_rows": (
                    exact_duplicates_removed
                ),
            },
            "suspected_duplicates": {
                "rule": (
                    "same user_id + item_id + "
                    "behavior_type + time"
                ),
                "raw_group_count": (
                    suspected_raw_groups
                ),
                "raw_record_count": (
                    suspected_raw_records
                ),
                "raw_excess_count": (
                    suspected_raw_excess
                ),
                "retained_group_count_after_exact_dedup": (
                    suspected_retained_groups
                ),
                "retained_record_count_after_exact_dedup": (
                    suspected_retained_records
                ),
                "action": "reported_and_retained",
            },
            "schema": {
                "item_category_renamed_to": (
                    "category_id"
                ),
                "weekday_definition": (
                    "Monday=0 ... Sunday=6"
                ),
            },
            "processing": {
                "chunksize": chunksize,
                "partition_count": partitions,
                "elapsed_seconds": elapsed_seconds,
            },
        }

        # Only replace formal data outputs after the entire
        # cleaning process has completed successfully.
        temp_output_csv.replace(output_csv)
        temp_output_parquet.replace(output_parquet)

        with report_json.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return report

    finally:
        for writer in partition_writers.values():
            writer.close()

        if final_writer is not None:
            final_writer.close()

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_output_csv.unlink(missing_ok=True)
        temp_output_parquet.unlink(missing_ok=True)
