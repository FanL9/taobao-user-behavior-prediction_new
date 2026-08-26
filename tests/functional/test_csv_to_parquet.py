"""Functional checks for validated CSV-to-Parquet conversion."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data import convert_csv_to_parquet


def _sample_frame() -> pd.DataFrame:
    """Return a small frame that follows the project data contract."""

    return pd.DataFrame(
        {
            "time": [
                "2025-11-18 00",
                "2025-11-18 01",
                "2025-11-18 02",
            ],
            "user_id": [1, 1, 2],
            "item_id": [11, 12, 13],
            "item_category": [101, 102, 101],
            "behavior_type": [1, 3, 4],
        }
    )


def test_convert_csv_to_parquet_preserves_rows_and_schema(tmp_path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.parquet"
    expected = _sample_frame()
    expected.to_csv(source, index=False, encoding="utf-8-sig")

    result = convert_csv_to_parquet(source, destination, chunksize=2)

    assert result.row_count == 3
    assert result.file_size_bytes > 0
    assert result.elapsed_seconds >= 0
    parquet = pq.read_table(destination)
    assert parquet.schema.names == list(expected.columns)
    assert parquet.schema.field("behavior_type").type.bit_width == 8
    actual = parquet.to_pandas()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_convert_supports_header_only_csv(tmp_path) -> None:
    source = tmp_path / "empty.csv"
    destination = tmp_path / "empty.parquet"
    _sample_frame().iloc[:0].to_csv(source, index=False)

    result = convert_csv_to_parquet(source, destination)

    assert result.row_count == 0
    assert pq.read_table(destination).num_rows == 0


def test_convert_rejects_wrong_header_without_partial_output(tmp_path) -> None:
    source = tmp_path / "wrong.csv"
    destination = tmp_path / "output.parquet"
    pd.DataFrame({"user_id": [1]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="columns"):
        convert_csv_to_parquet(source, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.parquet.tmp"))


def test_convert_requires_explicit_overwrite(tmp_path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.parquet"
    _sample_frame().to_csv(source, index=False)
    convert_csv_to_parquet(source, destination)

    with pytest.raises(FileExistsError):
        convert_csv_to_parquet(source, destination)

    result = convert_csv_to_parquet(source, destination, overwrite=True)
    assert result.row_count == 3


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("time", "2025/11/18 00", "YYYY-MM-DD HH"),
        ("user_id", 0, "non-positive"),
        ("behavior_type", 9, "outside 1, 2, 3, 4"),
    ],
)
def test_convert_rejects_values_outside_contract(
    tmp_path, column, value, message
) -> None:
    source = tmp_path / "invalid.csv"
    destination = tmp_path / "invalid.parquet"
    frame = _sample_frame()
    frame.loc[0, column] = value
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match=message):
        convert_csv_to_parquet(source, destination)

    assert not destination.exists()
