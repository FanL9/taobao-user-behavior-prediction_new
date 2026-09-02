"""Command-line entry point for train-only class-imbalance preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling.class_imbalance import (  # noqa: E402
    CLASS_WEIGHT_CONFIG_FILENAME,
    DATASET_VERSIONS_FILENAME,
    IMBALANCE_REPORT_FILENAME,
    prepare_class_imbalance_strategies,
)


def parse_args() -> argparse.Namespace:
    """Parse selected inputs, outputs, reports, and reproducibility settings."""

    parser = argparse.ArgumentParser(
        description="Prepare baseline, SMOTE, undersampling, and class-weight datasets."
    )
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-input",
            type=Path,
            default=(PROJECT_ROOT / "data" / "splits" / f"user_item_feature_wide_labeled_{split}_preprocessed_selected.parquet"),
            help=f"Issue4 selected {split} dataset Parquet.",
        )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "splits", help="Directory receiving class-imbalance datasets.")
    parser.add_argument("--class-weights", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / CLASS_WEIGHT_CONFIG_FILENAME, help="JSON path for balanced class weights.")
    parser.add_argument("--versions", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / DATASET_VERSIONS_FILENAME, help="JSON path for downstream training dataset versions.")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / IMBALANCE_REPORT_FILENAME, help="JSON path for sampling statistics and checks.")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Maximum rows processed per batch.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for SMOTE parent selection and undersampling.")
    parser.add_argument("--smote-k-neighbors", type=int, default=5, help="Maximum minority nearest neighbors used by SMOTE.")
    return parser.parse_args()


def main() -> int:
    """Create all class-imbalance strategies and print their output summaries."""

    args = parse_args()
    try:
        result = prepare_class_imbalance_strategies(
            {"train": args.train_input, "validation": args.validation_input, "test": args.test_input},
            args.output_dir,
            args.class_weights,
            args.versions,
            args.report,
            batch_size=args.batch_size,
            random_state=args.random_state,
            smote_k_neighbors=args.smote_k_neighbors,
        )
    except Exception as error:
        print(f"Class-imbalance preparation failed: {error}", file=sys.stderr)
        return 1
    for name, path in result["output_paths"].items():
        print(f"{name}: {path}")
    print(f"class_weights: {result['class_weight_path']}")
    print(f"versions: {result['versions_path']}")
    print(f"report: {result['report_path']}")
    print(f"status: {result['report']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
