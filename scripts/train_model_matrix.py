"""Run the complete four-strategy by four-model baseline training matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.training_matrix import MODEL_NAMES, TRAINING_STRATEGIES, train_model_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse shared inputs and an optional subset of the 16-model matrix."""

    parser = argparse.ArgumentParser(description="Train the 4 training-strategy × 4 model baseline matrix.")
    split_dir = PROJECT_ROOT / "data" / "splits"
    parser.add_argument("--baseline-train-input", type=Path, default=None)
    parser.add_argument("--smote-train-input", type=Path, default=None)
    parser.add_argument("--undersampled-train-input", type=Path, default=None)
    parser.add_argument("--validation-input", type=Path, default=split_dir / "user_item_feature_wide_labeled_validation_preprocessed_selected_original.parquet")
    parser.add_argument("--test-input", type=Path, default=split_dir / "user_item_feature_wide_labeled_test_preprocessed_selected_original.parquet")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--class-weights", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "class_weight_config.json")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4")
    parser.add_argument("--strategies", nargs="+", choices=TRAINING_STRATEGIES, default=list(TRAINING_STRATEGIES))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Train all requested strategy/model combinations from one CLI command."""

    args = parse_args()
    split_dir = PROJECT_ROOT / "data" / "splits"
    train_paths = {
        "baseline": args.baseline_train_input or (split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet"),
        "smote": args.smote_train_input or (split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_smote.parquet"),
        "undersampled": args.undersampled_train_input or (split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_undersampled.parquet"),
        "class_weight": args.baseline_train_input or (split_dir / "user_item_feature_wide_labeled_train_preprocessed_selected_baseline.parquet"),
    }
    inputs = {
        strategy: {"train": train_paths[strategy], "validation": args.validation_input, "test": args.test_input}
        for strategy in args.strategies
    }
    try:
        result = train_model_matrix(
            inputs, args.feature_list, args.models_dir, args.reports_dir, args.class_weights,
            strategies=args.strategies, model_names=args.models, random_state=args.random_state,
        )
    except Exception as error:
        print(f"Training matrix failed: {error}", file=sys.stderr)
        return 1
    print(f"trained_models: {result['model_count']}")
    print(f"aggregate_comparison: {result['aggregate_comparison_path']}")
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
