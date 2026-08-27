import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "data/EDA"


# ============================================================
# Test 1: All output files exist
# ============================================================

def test_output_files_exist():

    required_files = [

        "behavior_distribution.csv",

        "user_purchase_summary.csv",

        "item_statistics.csv",

        "top_10_item.csv",

        "category_statistics.csv",

        "top_10_category.csv",

        "daily_behavior.csv",

        "hourly_behavior.csv",

        "descriptive_funnel.csv"

    ]


    for file in required_files:

        path = os.path.join(
            OUTPUT_DIR,
            file
        )

        assert os.path.exists(path), (
            f"{file} is missing"
        )


# ============================================================
# Test 2: Behavior distribution
# ============================================================

def test_behavior_distribution():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "behavior_distribution.csv"
        )
    )


    assert len(df) == 4


    assert set(
        df["behavior_name"]
    ) == {
        "pv",
        "fav",
        "cart",
        "buy"
    }


    assert (
        df["behavior_count"].sum()
        == 12256906
    )


# ============================================================
# Test 3: User purchase summary
# ============================================================

def test_user_purchase_summary():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "user_purchase_summary.csv"
        )
    )


    required_columns = [

        "total_behavior_count",

        "purchase_count",

        "purchase_users",

        "non_purchase_users",

        "repeat_purchase_users"

    ]


    for col in required_columns:

        assert col in df.columns


    assert len(df) == 1


# ============================================================
# Test 4: Item statistics
# ============================================================

def test_item_statistics():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "item_statistics.csv"
        )
    )


    required_columns = [

        "item_id",

        "pv_count",

        "fav_count",

        "cart_count",

        "buy_count"

    ]


    for col in required_columns:

        assert col in df.columns


    assert len(df) > 0



# ============================================================
# Test 5: Top 10 items
# ============================================================

def test_top_10_item():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_10_item.csv"
        )
    )


    assert len(df) == 10


    assert (
        df["buy_count"]
        .is_monotonic_decreasing
    )


# ============================================================
# Test 6: Category statistics
# ============================================================

def test_category_statistics():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "category_statistics.csv"
        )
    )


    required_columns = [

        "category_id",

        "behavior_count",

        "buy_count",

        "buy_percentage"

    ]


    for col in required_columns:

        assert col in df.columns


    assert len(df) > 0



# ============================================================
# Test 7: Top 10 categories
# ============================================================

def test_top_10_category():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_10_category.csv"
        )
    )


    assert len(df) == 10


    assert (
        df["buy_count"]
        .is_monotonic_decreasing
    )



# ============================================================
# Test 8: Daily behavior
# ============================================================

def test_daily_behavior():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "daily_behavior.csv"
        )
    )


    required_columns = [

        "behavior_date",

        "pv_count",

        "fav_count",

        "cart_count",

        "buy_count"

    ]


    for col in required_columns:

        assert col in df.columns


    assert len(df) > 0



# ============================================================
# Test 9: Hourly behavior
# ============================================================

def test_hourly_behavior():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "hourly_behavior.csv"
        )
    )


    required_columns = [

        "behavior_hour",

        "pv_count",

        "fav_count",

        "cart_count",

        "buy_count"

    ]


    for col in required_columns:

        assert col in df.columns


    # 24 hours

    assert len(df) == 24



# ============================================================
# Test 10: Funnel
# ============================================================

def test_descriptive_funnel():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "descriptive_funnel.csv"
        )
    )


    assert len(df) == 4


    assert list(
        df["stage"]
    ) == [

        "PV",

        "Favorite",

        "Cart",

        "Purchase"

    ]


    assert (
        df.loc[
            df["stage"] == "PV",
            "relative_to_pv_percentage"
        ].iloc[0]
        == 100
    )
