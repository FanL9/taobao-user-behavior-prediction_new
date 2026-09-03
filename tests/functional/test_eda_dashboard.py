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


def _write_sample_stage2_features(output_dir: Path) -> None:
    """Create compact Stage 2 Parquet fixtures covering the dashboard contract."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "dataset_split": ["train"] * 6 + ["validation"],
            "user_id": [1, 2, 3, 4, 5, 6, 7],
            "activity_level": ["low", "low", "medium", "medium", "high", "high", "low"],
            "event_count": [4, 8, 12, 20, 40, 80, 5],
            "avg_daily_event_count": [0.4, 0.8, 1.2, 2.0, 4.0, 8.0, 0.5],
            "pv_count_per_day": [0.3, 0.6, 0.9, 1.4, 2.8, 5.0, 0.4],
            "buy_count_per_day": [0.0, 0.1, 0.1, 0.2, 0.5, 1.2, 0.0],
        }
    ).to_parquet(output_dir / "user_activity_features.parquet", index=False)

    pd.DataFrame(
        {
            "dataset_split": ["train"] * 10,
            "item_id": list(range(101, 111)),
            "item_total_count_rank": list(range(1, 11)),
            "item_pv_count": [100, 95, 90, 80, 70, 60, 50, 40, 30, 20],
            "item_buy_count": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "item_total_count": [120, 114, 108, 96, 84, 72, 60, 48, 36, 24],
        }
    ).to_parquet(output_dir / "item_popularity_features.parquet", index=False)

    pd.DataFrame(
        {
            "dataset_split": ["train"] * 5,
            "category_id": [201, 202, 203, 204, 205],
            "category_total_count": [1000, 800, 600, 400, 200],
            "category_pv_count": [800, 650, 500, 330, 170],
            "category_buy_count": [80, 60, 45, 25, 10],
        }
    ).to_parquet(output_dir / "category_behavior_features.parquet", index=False)

    pd.DataFrame(
        {
            "dataset_split": ["train"] * 3,
            "item_pv_count": [100, 200, 300],
            "item_fav_count": [20, 40, 60],
            "item_cart_count": [30, 50, 70],
            "item_buy_count": [10, 20, 30],
        }
    ).to_parquet(output_dir / "conversion_chain_features.parquet", index=False)

    pd.DataFrame(
        {
            "dataset_split": ["train"] * 5,
            "last_10_behavior_sequence": [
                "pv→pv→fav→cart→buy",
                "pv→cart→buy",
                "pv→fav→buy",
                "fav→cart→buy",
                "pv→buy",
            ],
        }
    ).to_parquet(output_dir / "user_sequence_features.parquet", index=False)

    pd.DataFrame({"dataset_split": ["train"], "user_id": [1]}).to_parquet(
        output_dir / "user_features.parquet", index=False
    )
    pd.DataFrame({"dataset_split": ["train"], "item_id": [101]}).to_parquet(
        output_dir / "item_behavior_features.parquet", index=False
    )
    pd.DataFrame({"dataset_split": ["train"], "behavior_hour": [12]}).to_parquet(
        output_dir / "time_behavior_features.parquet", index=False
    )


def test_stage2_feature_contract_lists_all_eight_tables() -> None:
    assert len(data_loader.STAGE2_FEATURE_FILES) == 8
    assert set(data_loader.STAGE2_FEATURE_FILES) == {
        "user_basic",
        "user_activity",
        "user_sequence",
        "item_behavior",
        "item_popularity",
        "category_behavior",
        "time_behavior",
        "conversion_chain",
    }


def test_stage2_dashboard_statistics_and_precomputed_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    feature_dir = tmp_path / "features"
    stats_dir = tmp_path / "stage2_dashboard"
    _write_sample_stage2_features(feature_dir)

    inventory = data_loader.load_stage2_feature_inventory(str(feature_dir))
    assert len(inventory) == 8

    splits = data_loader.get_stage2_dataset_splits(str(feature_dir))
    assert splits == ["train", "validation"]

    statistics = data_loader.build_stage2_dashboard_statistics(
        output_dir=str(feature_dir),
        dataset_split="train",
        transition_sequence_limit=None,
    )

    assert set(statistics) == set(data_loader.STAGE2_DASHBOARD_STAT_FILES)
    assert list(statistics["conversion_funnel"]["stage"]) == [
        "PV", "Favorite", "Cart", "Purchase"
    ]
    assert float(statistics["conversion_funnel"].iloc[0]["behavior_count"]) == 600.0
    assert len(statistics["transition_matrix"]) == 16
    assert statistics["transition_matrix"]["transition_count"].sum() > 0
    assert not statistics["behavior_depth"].empty
    assert not statistics["user_activity"].empty
    assert not statistics["item_popularity"].empty
    assert len(statistics["top_items"]) == 10
    assert not statistics["category_traffic"].empty

    data_loader.write_stage2_dashboard_statistics(
        {"train": statistics},
        output_dir=str(stats_dir),
    )
    loaded = data_loader.load_stage2_dashboard_outputs(
        output_dir=str(stats_dir),
        dataset_split="train",
    )
    assert set(loaded) == set(statistics)
    assert len(loaded["transition_matrix"]) == 16


def test_stage2_projected_loader_filters_split_and_columns(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    feature_dir = tmp_path / "features"
    _write_sample_stage2_features(feature_dir)

    frame = data_loader.load_stage2_feature_table(
        "user_activity",
        output_dir=str(feature_dir),
        columns=("user_id", "event_count"),
        dataset_split="validation",
    )
    assert list(frame.columns) == ["user_id", "event_count"]
    assert frame["user_id"].tolist() == [7]
