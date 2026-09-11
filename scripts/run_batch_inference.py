"""Command-line entry point for Stage 5 local batch inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.batch_inference import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUMMARY_PATH,
    load_inference_contract,
    run_batch_inference,
)


def parse_args() -> argparse.Namespace:
    """Parse local inference inputs and frozen-artifact configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Score raw user-item feature data with the frozen Stage 3 preprocessing "
            "rules, Stage 3 final feature list, and Stage 4 final model."
        )
    )
    parser.add_argument("--input", type=Path, help="Raw user-item feature .parquet or .csv file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the final model/rules/features contract without scoring data.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate the frozen contract or execute local batch inference."""

    args = parse_args()
    try:
        if args.check_only:
            contract = load_inference_contract(
                args.config,
                project_root=PROJECT_ROOT,
                require_model=True,
            )
            print(f"final_candidate: {contract.final_candidate}")
            print(f"model_artifact: {contract.model_artifact}")
            print(f"model_bundle_key: {contract.model_bundle_key}")
            print(f"preprocessing_rules: {contract.preprocessing_rules_path}")
            print(f"feature_list: {contract.feature_list_path}")
            print(f"model_feature_count: {len(contract.model_features)}")
            print("status: passed")
            return 0

        if args.input is None:
            raise ValueError("--input is required unless --check-only is used.")
        summary = run_batch_inference(
            args.input,
            args.output,
            args.config,
            args.summary_output,
            batch_size=args.batch_size,
            project_root=PROJECT_ROOT,
        )
    except Exception as error:
        print(f"Batch inference failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
