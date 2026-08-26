"""Command-line entry point for validated CSV-to-Parquet conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import convert_csv_to_parquet  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse conversion paths and storage options."""

    parser = argparse.ArgumentParser(
        description="Convert the stage-one CSV to typed Parquet in chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "user_behavior_processed.csv",
        help="Source CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "user_behavior_processed.parquet",
        help="Destination Parquet path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per input chunk (default: 250000).",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="PyArrow compression codec (default: snappy).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding (default: utf-8-sig).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    """Run conversion and print its observable result."""

    args = parse_args()
    try:
        result = convert_csv_to_parquet(
            args.input,
            args.output,
            chunksize=args.chunksize,
            compression=args.compression,
            encoding=args.encoding,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1

    print(f"Input: {result.input_path}")
    print(f"Output: {result.output_path}")
    print(f"Rows: {result.row_count:,}")
    print(f"Size: {result.file_size_bytes:,} bytes")
    print(f"Elapsed: {result.elapsed_seconds:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
