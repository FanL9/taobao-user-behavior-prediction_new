"""Command-line entry point for stage-one data cleaning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Allow this file to be executed directly with:
# python scripts/clean_user_behavior.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.cleaning_pipeline import clean_user_behavior_file


DEFAULT_INPUT = Path("data/raw/user_behavior_processed.csv")

DEFAULT_OUTPUT_CSV = Path(
    "data/processed/user_behavior_clean.csv"
)

DEFAULT_OUTPUT_PARQUET = Path(
    "data/processed/user_behavior_clean.parquet"
)

DEFAULT_REPORT = Path(
    "reports/stage1/cleaning_report.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Taobao user-behavior data.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the raw user-behavior CSV.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the cleaned CSV.",
    )

    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Path for the cleaned Parquet file.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the cleaning report JSON.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Number of source rows processed per chunk.",
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=64,
        help="Number of temporary hash partitions.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = clean_user_behavior_file(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
        report_json=args.report,
        chunksize=args.chunksize,
        partitions=args.partitions,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
