"""Run Stage4 Issue2 Optuna tuning for one or all traditional-model combinations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_tuning import DEFAULT_N_TRIALS, TUNING_CANDIDATES, tune_all_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse Issue2 paths, uniformly budgeted Optuna trials, and candidates."""

    parser = argparse.ArgumentParser(description="Tune Stage4 traditional-model combinations with Optuna.")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--feature-list", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "final_model_features.json")
    parser.add_argument("--class-weights", type=Path, default=PROJECT_ROOT / "reports" / "stage3" / "class_weight_config.json")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "traditional_ml" / "tuned")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "stage4" / "model_tuning")
    parser.add_argument("--candidates", nargs="+", choices=TUNING_CANDIDATES, default=list(TUNING_CANDIDATES))
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Identical Optuna trial budget per candidate.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="Use GPU for XGBoost; LightGBM remains CPU in this environment.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run validation-only tuning and print generated report locations."""

    args = parse_args()
    try:
        result = tune_all_models(
            args.split_dir, args.feature_list, args.models_dir, args.reports_dir,
            args.class_weights, candidates=tuple(args.candidates), n_trials=args.n_trials,
            random_state=args.random_state, device=args.device,
        )
    except Exception as error:
        print(f"Model tuning failed: {error}", file=sys.stderr)
        return 1
    for name in ("search_path", "parameters_path", "comparison_path", "candidate_pool_path", "summary_path"):
        print(f"{name}: {result[name]}")
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
