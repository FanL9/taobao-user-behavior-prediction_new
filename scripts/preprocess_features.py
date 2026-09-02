"""Command-line entry point for train-fitted feature preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessing import (  # noqa: E402
    PREPROCESSING_REPORT_FILENAME,
    PREPROCESSING_RULES_FILENAME,
    preprocess_feature_datasets,
)


def parse_args() -> argparse.Namespace:
    """Parse train/validation/test inputs, outputs, rules, and report paths."""

    parser = argparse.ArgumentParser(
        description="Fit preprocessing rules on train and apply them to all datasets."
    )
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-input",
            type=Path,
            default=(
                PROJECT_ROOT
                / "data"
                / "splits"
                / f"user_item_feature_wide_labeled_{split}.parquet"
            ),
            help=f"Labeled {split} dataset Parquet.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory receiving the three preprocessed Parquets.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / PREPROCESSING_RULES_FILENAME,
        help="JSON path for train-fitted preprocessing rules.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / PREPROCESSING_REPORT_FILENAME,
        help="JSON path for preprocessing statistics and checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Maximum rows processed per batch.",
    )
    return parser.parse_args()


def main() -> int:
    """Run train-fitted preprocessing and print the generated artifacts."""

    args = parse_args()
    try:
        result = preprocess_feature_datasets(
            {
                "train": args.train_input,
                "validation": args.validation_input,
                "test": args.test_input,
            },
            args.output_dir,
            args.rules,
            args.report,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Feature preprocessing failed: {error}", file=sys.stderr)
        return 1

    for split, path in result["output_paths"].items():
        stats = result["report"]["statistics"]["by_split"][split]
        print(f"{split}: {path}")
        print(f"  samples={stats['sample_count']}")
    print(f"rules: {result['rules_path']}")
    print(f"report: {result['report_path']}")
    print(f"status: {result['report']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
