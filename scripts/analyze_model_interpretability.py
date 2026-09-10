"""Run Stage4 Issue4 model interpretation and error-sample analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_interpretability import analyze_model_interpretability  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse paths to fixed Issue2.5/Issue3 artifacts and report controls."""

    parser = argparse.ArgumentParser(description="Create Stage4 Issue4 model interpretation outputs.")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--preprocessing-rules", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "preprocessing_rules.json")
    parser.add_argument("--issue2-5-manifest", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "stacking_fusion" / "issue3_candidate_manifest.json")
    parser.add_argument("--test-predictions", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml" / "stacking" / "test_predictions" / "issue3_final_candidate_predictions.parquet")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml")
    parser.add_argument("--issue3-reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_evaluation")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_interpretability")
    parser.add_argument("--shap-sample-size", type=int, default=10_000)
    parser.add_argument("--error-sample-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run Issue4 without fitting or modifying a model."""

    args = parse_args()
    try:
        result = analyze_model_interpretability(args.split_dir, args.feature_list, args.issue2_5_manifest, args.test_predictions, args.models_dir, args.issue3_reports_dir, args.reports_dir, args.shap_sample_size, args.error_sample_size, args.random_state, args.preprocessing_rules)
    except Exception as error:
        print(f"Model interpretability analysis failed: {error}", file=sys.stderr)
        return 1
    for name, path in result.items():
        print(f"{name}: {path}")
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
