"""Command-line entry point for future-one-day purchase labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.labels import (  # noqa: E402
    LABELED_SAMPLE_FILENAME,
    LABEL_REPORT_FILENAME,
    generate_purchase_labels,
)


def parse_args() -> argparse.Namespace:
    """Parse the wide-table, clean-data, labeled-output, and report paths."""

    parser = argparse.ArgumentParser(
        description="Append future-one-day purchase labels to the stage-two wide table."
    )
    parser.add_argument(
        "--wide-table",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "user_item_feature_wide.parquet",
        help="Stage-two user-item feature-wide Parquet.",
    )
    parser.add_argument(
        "--clean-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet",
        help="Stage-one standard clean behavior Parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / LABELED_SAMPLE_FILENAME,
        help="Labeled user-item sample Parquet.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / LABEL_REPORT_FILENAME,
        help="Label statistics and leakage-check JSON report.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Maximum feature-wide rows processed per batch.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the labeled sample and print its statistics."""

    args = parse_args()
    try:
        result = generate_purchase_labels(
            args.wide_table,
            args.clean_data,
            args.output,
            args.report,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Label generation failed: {error}", file=sys.stderr)
        return 1

    report = result["report"]
    print(f"labeled_sample: {result['output_path']}")
    print(f"report: {result['report_path']}")
    for split, stats in report["statistics"]["by_split"].items():
        print(
            f"{split}: positive={stats['positive_count']}, "
            f"negative={stats['negative_count']}, "
            f"positive_ratio={stats['positive_ratio']:.8f}"
        )
    print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
