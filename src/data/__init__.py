"""Data ingestion and conversion utilities."""

from .csv_to_parquet import ConversionResult, convert_csv_to_parquet

__all__ = ["ConversionResult", "convert_csv_to_parquet"]
