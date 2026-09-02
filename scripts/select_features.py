"""Command-line entry point for training-only feature screening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_selection import (  # noqa: E402
    FEATURE_SELECTION_REPORT_FILENAME,
    FINAL_FEATURE_LIST_FILENAME,
    select_model_features,
)


def parse_args() -> argparse.Namespace:
    """Parse preprocessed input, selected output, and report paths."""

    parser = argparse.ArgumentParser(
        description="Fit train-only feature screening and apply the final list."
    )
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-input",
            type=Path,
            default=(
                PROJECT_ROOT
                / "data"
                / "splits"
                / f"user_item_feature_wide_labeled_{split}_preprocessed.parquet"
            ),
            help=f"Preprocessed {split} dataset Parquet.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory receiving selected-feature Parquets.",
    )
    parser.add_argument(
        "--feature-list",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / FINAL_FEATURE_LIST_FILENAME,
        help="JSON path for the final model-feature list.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3" / FEATURE_SELECTION_REPORT_FILENAME,
        help="JSON path for selection reasons, checks, and statistics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Maximum rows processed per batch.",
    )
    return parser.parse_args()


def main() -> int:
    """Run feature selection and print output locations plus selected count."""

    args = parse_args()
    try:
        result = select_model_features(
            {
                "train": args.train_input,
                "validation": args.validation_input,
                "test": args.test_input,
            },
            args.output_dir,
            args.feature_list,
            args.report,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Feature selection failed: {error}", file=sys.stderr)
        return 1

    for split, path in result["output_paths"].items():
        print(f"{split}: {path}")
    print(f"selected_features: {result['feature_list']['model_feature_count']}")
    print(f"feature_list: {result['feature_list_path']}")
    print(f"report: {result['report_path']}")
    print(f"status: {result['report']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
