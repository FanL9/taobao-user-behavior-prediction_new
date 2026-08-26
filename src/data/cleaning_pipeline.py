"""Full stage-one cleaning pipeline.

The pipeline reads the raw CSV in chunks, validates each chunk,
partitions cleaned rows by the duplicate-frequency key, applies the
hour-level frequency rule globally, and writes cleaned CSV/Parquet outputs.
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


DUPLICATE_FREQUENCY_KEY = (
    "time",
    "user_id",
    "item_id",
    "behavior_type",
)

HIGH_FREQUENCY_DUPLICATE_THRESHOLD = 60


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

    Duplicate handling uses the four-field key
    user_id + item_id + behavior_type + time. Groups occurring 2-59
    times are retained in full. Groups occurring 60 times or more retain
    only the first record encountered in the original input stream.

    Because ``time`` is hour-level, this is an hourly proxy rule rather
    than minute- or second-level duplicate detection.
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
        # Rows sharing the duplicate-frequency key always go to
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
                cleaned[list(DUPLICATE_FREQUENCY_KEY)],
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

        normal_frequency_groups = 0
        normal_frequency_records = 0

        high_frequency_groups = 0
        high_frequency_records = 0
        high_frequency_rows_removed = 0

        final_rows = 0
        csv_header_written = False

        # Pass 2:
        # Duplicate-frequency groups are fully contained within individual
        # partitions, so the 60+ threshold is enforced globally.
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

            group_sizes = frame.groupby(
                list(DUPLICATE_FREQUENCY_KEY),
                dropna=False,
                sort=False,
            ).size()

            normal_sizes = group_sizes[
                (group_sizes >= 2)
                & (
                    group_sizes
                    < HIGH_FREQUENCY_DUPLICATE_THRESHOLD
                )
            ]

            high_sizes = group_sizes[
                group_sizes
                >= HIGH_FREQUENCY_DUPLICATE_THRESHOLD
            ]

            normal_frequency_groups += int(
                len(normal_sizes)
            )
            normal_frequency_records += int(
                normal_sizes.sum()
            )

            high_frequency_groups += int(
                len(high_sizes)
            )
            high_frequency_records += int(
                high_sizes.sum()
            )

            row_group_sizes = frame.groupby(
                list(DUPLICATE_FREQUENCY_KEY),
                dropna=False,
                sort=False,
            )["category_id"].transform("size")

            # Partition writers append chunks in source order and each
            # partition preserves row order, so keep="first" preserves
            # the first encountered record from the input stream.
            repeated_after_first = frame.duplicated(
                subset=list(DUPLICATE_FREQUENCY_KEY),
                keep="first",
            )

            remove_mask = (
                row_group_sizes
                >= HIGH_FREQUENCY_DUPLICATE_THRESHOLD
            ) & repeated_after_first

            high_frequency_rows_removed += int(
                remove_mask.sum()
            )

            frame = frame.loc[~remove_mask].copy()
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
                "high_frequency_duplicate_rows": (
                    high_frequency_rows_removed
                ),
            },
            "duplicate_handling": {
                "key": (
                    "user_id + item_id + behavior_type + time"
                ),
                "threshold": HIGH_FREQUENCY_DUPLICATE_THRESHOLD,
                "normal_frequency_rule": (
                    "2-59 occurrences: retain all records"
                ),
                "normal_frequency_group_count": (
                    normal_frequency_groups
                ),
                "normal_frequency_record_count": (
                    normal_frequency_records
                ),
                "high_frequency_rule": (
                    "60 or more occurrences: retain first record only"
                ),
                "high_frequency_group_count": (
                    high_frequency_groups
                ),
                "high_frequency_record_count": (
                    high_frequency_records
                ),
                "removed_high_frequency_rows": (
                    high_frequency_rows_removed
                ),
                "retained_high_frequency_rows": (
                    high_frequency_groups
                ),
                "time_granularity_note": (
                    "time is hourly; this is an hourly proxy rule, "
                    "not minute-level detection"
                ),
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
