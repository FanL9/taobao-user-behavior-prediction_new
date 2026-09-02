"""Command-line entry point for fixed-parameter baseline model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_training import MODEL_NAMES, train_baseline_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse data-version, output, and fixed baseline-training settings."""

    parser = argparse.ArgumentParser(description="Train Logistic Regression, Random Forest, XGBoost, and LightGBM baselines.")
    split_dir = PROJECT_ROOT / "data" / "splits"
    parser.add_argument("--train-input", type=Path, default=split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet")
    parser.add_argument("--validation-input", type=Path, default=split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet")
    parser.add_argument("--test-input", type=Path, default=split_dir / "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--class-weights", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "class_weight_config.json")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "baselines")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4")
    parser.add_argument("--training-strategy", choices=("baseline", "class_weight", "smote", "undersampled"), default="class_weight")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Train requested baseline models and report tracked output paths."""

    args = parse_args()
    try:
        result = train_baseline_models(
            {"train": args.train_input, "validation": args.validation_input, "test": args.test_input},
            args.feature_list, args.models_dir, args.reports_dir,
            class_weight_path=args.class_weights if args.training_strategy == "class_weight" else None,
            model_names=args.models, training_strategy=args.training_strategy,
            random_state=args.random_state,
        )
    except Exception as error:
        print(f"Baseline model training failed: {error}", file=sys.stderr)
        return 1
    print(f"comparison: {result['comparison_path']}")
    print(f"summary: {result['summary_path']}")
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
