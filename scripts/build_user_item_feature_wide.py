"""Command-line entry point for the stage-two user-item feature wide table."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.user_item_feature_wide import (  # noqa: E402
    QUALITY_REPORT_FILENAME,
    WIDE_TABLE_FILENAME,
    generate_user_item_feature_wide,
)


def parse_args() -> argparse.Namespace:
    """Parse the eight-table input, wide output, and quality-report paths.

    Returns:
        Parsed paths and batch-size arguments.
    """

    parser = argparse.ArgumentParser(
        description="Merge eight stage-two tables into one user-item wide table."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features",
        help="Directory containing the eight stage-two feature Parquets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / WIDE_TABLE_FILENAME,
        help="Local user-item feature-wide Parquet path.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage2" / QUALITY_REPORT_FILENAME,
        help="JSON quality-report path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help="Maximum user-item sequence rows merged per batch.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the wide table and print its checked output summary.

    Returns:
        Process exit code: zero on success and one on generation failure.
    """

    args = parse_args()
    started_at = time.perf_counter()
    try:
        result = generate_user_item_feature_wide(
            args.features_dir,
            args.output,
            args.quality_report,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Wide-table generation failed: {error}", file=sys.stderr)
        return 1

    report = result["quality_report"]
    print(f"wide_table: {result['output_path']}")
    print(f"quality_report: {result['quality_report_path']}")
    print(f"rows: {report['row_count']}")
    print(f"columns: {report['column_count']}")
    print(f"status: {report['status']}")
    print(f"Elapsed: {time.perf_counter() - started_at:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
