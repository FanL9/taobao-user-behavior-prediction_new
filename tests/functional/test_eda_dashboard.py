from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_LOADER_PATH = REPO_ROOT / "dashboards" / "eda" / "data_loader.py"

SPEC = spec_from_file_location(
    "stage1_dashboard_data_loader",
    DATA_LOADER_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

data_loader = module_from_spec(SPEC)
SPEC.loader.exec_module(data_loader)


def _write_sample_outputs(output_dir: Path) -> None:
    """Create small Stage 1 EDA outputs for dashboard functional tests."""
    frames = {
        "behavior_distribution.csv": pd.DataFrame(
            {
                "behavior_type": [1, 2, 3, 4],
                "behavior_name": ["pv", "fav", "cart", "buy"],
                "behavior_count": [1000, 20, 30, 10],
                "percentage": [94.34, 1.89, 2.83, 0.94],
            }
        ),
        "user_purchase_summary.csv": pd.DataFrame(
            {
                "total_behavior_count": [1060],
                "purchase_count": [10],
                "purchase_users": [6],
                "non_purchase_users": [4],
                "repeat_purchase_users": [2],
            }
        ),
        "top_10_item.csv": pd.DataFrame(
            {
                "item_id": [101, 102],
                "pv_count": [50, 40],
                "fav_count": [2, 1],
                "cart_count": [3, 2],
                "buy_count": [5, 4],
            }
        ),
        "item_statistics.csv": pd.DataFrame(
            {
                "item_id": [101, 102, 103],
                "pv_count": [50, 40, 30],
                "fav_count": [2, 1, 1],
                "cart_count": [3, 2, 1],
                "buy_count": [5, 4, 1],
            }
        ),
        "category_statistics.csv": pd.DataFrame(
            {
                "category_id": [201, 202],
                "behavior_count": [100, 80],
                "buy_count": [8, 6],
                "buy_percentage": [8.0, 7.5],
            }
        ),
        "top_10_category.csv": pd.DataFrame(
            {
                "category_id": [201, 202],
                "behavior_count": [100, 80],
                "buy_count": [8, 6],
                "buy_percentage": [8.0, 7.5],
            }
        ),
        "daily_behavior.csv": pd.DataFrame(
            {
                "behavior_date": ["2025-11-18", "2025-11-19"],
                "pv_count": [500, 520],
                "fav_count": [10, 11],
                "cart_count": [15, 16],
                "buy_count": [5, 6],
            }
        ),
        "hourly_behavior.csv": pd.DataFrame(
            {
                "behavior_hour": [0, 23],
                "pv_count": [100, 200],
                "fav_count": [5, 8],
                "cart_count": [6, 9],
                "buy_count": [2, 4],
            }
        ),
        "descriptive_funnel.csv": pd.DataFrame(
            {
                "stage": ["PV", "Favorite", "Cart", "Purchase"],
                "behavior_count": [1000, 20, 30, 10],
                "relative_to_pv_percentage": [100.0, 2.0, 3.0, 1.0],
            }
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, dataframe in frames.items():
        dataframe.to_csv(output_dir / filename, index=False)


def test_loads_only_lightweight_dashboard_outputs(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)

    outputs = data_loader.load_eda_outputs(str(tmp_path))

    assert set(outputs) == set(data_loader.EDA_FILES)
    assert "item_statistics" not in outputs
    assert len(outputs) == 8


def test_missing_required_output_raises_file_not_found(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    (tmp_path / "hourly_behavior.csv").unlink()

    with pytest.raises(FileNotFoundError, match="hourly_behavior.csv"):
        data_loader.load_eda_outputs(str(tmp_path))


def test_behavior_and_user_contracts(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    outputs = data_loader.load_eda_outputs(str(tmp_path))

    behavior = outputs["behavior_distribution"]
    users = outputs["user_purchase_summary"]

    assert list(behavior["behavior_name"]) == ["pv", "fav", "cart", "buy"]
    assert int(behavior["behavior_count"].sum()) == 1060
    assert int(users.iloc[0]["purchase_users"]) == 6
    assert int(users.iloc[0]["repeat_purchase_users"]) == 2


def test_item_and_category_contracts(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    outputs = data_loader.load_eda_outputs(str(tmp_path))

    items = outputs["top_10_item"]
    categories = outputs["top_10_category"]

    assert {"item_id", "buy_count"} <= set(items.columns)
    assert int(items["buy_count"].max()) == 5
    assert {"category_id", "buy_count", "buy_percentage"} <= set(categories.columns)
    assert int(categories["buy_count"].max()) == 8


def test_daily_behavior_contract(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    daily = data_loader.load_eda_outputs(str(tmp_path))["daily_behavior"]

    assert {
        "behavior_date",
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
    } <= set(daily.columns)
    assert len(daily) == 2


def test_hourly_behavior_contract(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    hourly = data_loader.load_eda_outputs(str(tmp_path))["hourly_behavior"]

    assert {
        "behavior_hour",
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
    } <= set(hourly.columns)
    assert list(hourly["behavior_hour"]) == [0, 23]


def test_descriptive_funnel_contract(tmp_path: Path) -> None:
    _write_sample_outputs(tmp_path)
    funnel = data_loader.load_eda_outputs(str(tmp_path))["descriptive_funnel"]

    assert list(funnel["stage"]) == [
        "PV",
        "Favorite",
        "Cart",
        "Purchase",
    ]
    assert float(funnel.iloc[0]["relative_to_pv_percentage"]) == 100.0


def test_dashboard_app_smoke_run() -> None:
    """Verify that the Streamlit dashboard entry point renders without exceptions."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(REPO_ROOT / "dashboards" / "eda" / "app.py")
    )

    app.run(timeout=15)

    assert not app.exception
