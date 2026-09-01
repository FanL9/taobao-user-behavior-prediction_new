"""Data ingestion and conversion utilities."""

from .csv_to_parquet import ConversionResult, convert_csv_to_parquet
from .data_quality import check_csv_quality, write_quality_report

__all__ = [
    "ConversionResult",
    "check_csv_quality",
    "convert_csv_to_parquet",
    "write_quality_report",
]
"""Data ingestion, cleaning, and dataset-partition interfaces."""

from .dataset_splits import (
    DATASET_FILENAMES,
    SPLIT_REPORT_FILENAME,
    generate_time_ordered_datasets,
)

__all__ = [
    "DATASET_FILENAMES",
    "SPLIT_REPORT_FILENAME",
    "generate_time_ordered_datasets",
]
