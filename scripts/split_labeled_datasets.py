"""Command-line entry point for time-ordered labeled dataset generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_splits import (  # noqa: E402
    SPLIT_REPORT_FILENAME,
    generate_time_ordered_datasets,
)
from src.features.labels import LABELED_SAMPLE_FILENAME  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse labeled-input, output-directory, report, and batch-size options."""

    parser = argparse.ArgumentParser(
        description="Create fixed time-ordered train, validation, and test datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / LABELED_SAMPLE_FILENAME,
        help="Labeled user-item feature-wide Parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory receiving train, validation, and test Parquets.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / SPLIT_REPORT_FILENAME,
        help="JSON path for dataset statistics and time-window checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Maximum labeled rows processed per batch.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the three datasets and print their sample statistics."""

    args = parse_args()
    try:
        result = generate_time_ordered_datasets(
            args.input,
            args.output_dir,
            args.report,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Dataset generation failed: {error}", file=sys.stderr)
        return 1

    report = result["report"]
    for split, path in result["output_paths"].items():
        stats = report["statistics"]["by_split"][split]
        print(f"{split}: {path}")
        print(
            f"  samples={stats['sample_count']}, "
            f"positive={stats['positive_count']}, "
            f"negative={stats['negative_count']}, "
            f"positive_ratio={stats['positive_ratio']:.8f}"
        )
    print(f"report: {result['report_path']}")
    print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
