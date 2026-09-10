"""Command-line entry point for Stage4 Issue3 final model evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_evaluation import evaluate_final_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse Issue3 artifacts, test data, and report locations."""

    parser = argparse.ArgumentParser(description="Evaluate Issue2.5 candidates once on the test set.")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--issue2-5-manifest", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "stacking_fusion" / "issue3_candidate_manifest.json")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_evaluation")
    return parser.parse_args()


def main() -> int:
    """Run final, non-tuning test evaluation and print result locations."""

    args = parse_args()
    try:
        result = evaluate_final_candidates(args.split_dir, args.feature_list, args.issue2_5_manifest, args.models_dir, args.reports_dir)
    except Exception as error:
        print(f"Model evaluation failed: {error}", file=sys.stderr)
        return 1
    for name, value in result.items():
        print(f"{name}: {value}")
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
