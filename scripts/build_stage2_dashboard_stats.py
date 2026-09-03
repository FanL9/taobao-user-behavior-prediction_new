"""Build lightweight Stage 2 statistics consumed by the EDA dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboards.eda.data_loader import (  # noqa: E402
    FEATURE_OUTPUT_DIR,
    STAGE2_DASHBOARD_STATS_DIR,
    build_stage2_dashboard_statistics,
    get_stage2_dataset_splits,
    load_stage2_feature_inventory,
    write_stage2_dashboard_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lightweight Stage 2 statistics for the Streamlit EDA dashboard."
    )
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STAGE2_DASHBOARD_STATS_DIR)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument(
        "--transition-sequence-limit",
        type=int,
        default=0,
        help="0 scans all sequences; positive values are for development only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = load_stage2_feature_inventory(args.feature_dir)
    print(
        f"[stage2-dashboard] verified {len(inventory)} Stage 2 feature tables "
        f"under {args.feature_dir}"
    )

    splits = args.splits or get_stage2_dataset_splits(args.feature_dir)
    sequence_limit = args.transition_sequence_limit or None

    statistics_by_split = {}
    for split in splits:
        print(f"[stage2-dashboard] building split={split} ...", flush=True)
        statistics_by_split[split] = build_stage2_dashboard_statistics(
            output_dir=args.feature_dir,
            dataset_split=split,
            transition_sequence_limit=sequence_limit,
        )

    written = write_stage2_dashboard_statistics(
        statistics_by_split,
        output_dir=args.output_dir,
    )
    print("[stage2-dashboard] generated:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
