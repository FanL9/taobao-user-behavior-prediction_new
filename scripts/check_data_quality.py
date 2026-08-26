"""Command-line entry point for the read-only CSV quality check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import check_csv_quality, write_quality_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse input, report, chunk, and duplicate-partition options."""

    parser = argparse.ArgumentParser(
        description="Inspect the source CSV without modifying or cleaning it."
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
        default=PROJECT_ROOT / "reports" / "stage1" / "data_quality_report.json",
        help="Quality report JSON path.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--duplicate-partitions", type=int, default=32)
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def main() -> int:
    """Run the quality check and write only its JSON summary."""

    args = parse_args()
    try:
        report = check_csv_quality(
            args.input,
            chunksize=args.chunksize,
            encoding=args.encoding,
            duplicate_partitions=args.duplicate_partitions,
        )
        output_path = write_quality_report(report, args.output)
    except Exception as error:
        print(f"Quality check failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
