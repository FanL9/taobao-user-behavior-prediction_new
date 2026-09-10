"""Command-line entry point for Stage4 Issue2.5 validation-only stacking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_stacking import run_stacking_fusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse Issue2 artifacts, data paths, and OOF stacking settings."""

    parser = argparse.ArgumentParser(description="Run leakage-safe Stage4 Issue2.5 stacking.")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--class-weights", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "class_weight_config.json")
    parser.add_argument("--issue2-parameters", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_tuning" / "best_tuned_model_parameters.json")
    parser.add_argument("--issue2-candidate-pool", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_tuning" / "model_fusion_candidate_pool.csv")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "stacking_fusion")
    parser.add_argument("--base-model-count", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="Use GPU for XGBoost base models; LightGBM remains CPU in this environment.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run validation-only OOF stacking and print generated artifact locations."""

    args = parse_args()
    try:
        result = run_stacking_fusion(
            args.split_dir, args.feature_list, args.class_weights, args.issue2_parameters,
            args.issue2_candidate_pool, args.models_dir, args.reports_dir,
            base_model_count=args.base_model_count, folds=args.folds, random_state=args.random_state, device=args.device,
        )
    except Exception as error:
        print(f"Stacking fusion failed: {error}", file=sys.stderr)
        return 1
    for name in ("comparison_path", "manifest_path", "summary_path", "oof_path", "validation_path"):
        print(f"{name}: {result[name]}")
    print("selected_candidates: " + ", ".join(result["selected_candidates"]))
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
