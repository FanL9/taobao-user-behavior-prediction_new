"""Command-line entry point for the first four stage-two feature tables."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature import generate_feature_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse clean input and feature-table output paths.

    Returns:
        Parsed command-line arguments containing input and output paths.
    """

    parser = argparse.ArgumentParser(
        description="Build only the first four stage-two feature tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet",
        help="Stage-one clean Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features",
        help="Directory for the four feature Parquet tables.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate four feature tables and report paths plus runtime.

    Returns:
        Process exit code: zero on success and one on generation failure.
    """

    args = parse_args()
    started_at = time.perf_counter()
    try:
        outputs = generate_feature_tables(args.input, args.output_dir)
    except Exception as error:
        print(f"Feature-table generation failed: {error}", file=sys.stderr)
        return 1

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"Elapsed: {time.perf_counter() - started_at:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
