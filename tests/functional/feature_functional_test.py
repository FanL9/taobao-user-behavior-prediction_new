import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

BASE_PATH = r"C:\Users\Hongshucham\Desktop\827"

RAW_FILE = rf"{BASE_PATH}\user_behavior_clean.parquet"

USER_FEATURE_FILE = rf"{BASE_PATH}\user_features.parquet"
USER_ACTIVITY_FILE = rf"{BASE_PATH}\user_activity_features.parquet"
USER_SEQUENCE_FILE = rf"{BASE_PATH}\user_sequence_features.parquet"
ITEM_FEATURE_FILE = rf"{BASE_PATH}\item_behavior_features.parquet"


# ============================================================
# Temporal Split Definition
# ============================================================

TIME_SPLITS = {
    "train": {
        "history_start": "2025-11-18",
        "history_end": "2025-12-07",
        "label_date": "2025-12-08"
    },
    "validation": {
        "history_start": "2025-12-09",
        "history_end": "2025-12-14",
        "label_date": "2025-12-15"
    },
    "test": {
        "history_start": "2025-12-16",
        "history_end": "2025-12-17",
        "label_date": "2025-12-18"
    }
}


# ============================================================
# Test Result Storage
# ============================================================

total_tests = 0
passed_tests = 0
failed_tests = 0


def test_result(test_name, condition, detail=""):
    """
    Print PASS / FAIL and update counters.
    """

    global total_tests
    global passed_tests
    global failed_tests

    total_tests += 1

    if condition:
        passed_tests += 1
        print(f"PASS  {test_name}")
    else:
        failed_tests += 1
        print(f"FAIL  {test_name}")

        if detail:
            print(f"      {detail}")


# ============================================================
# Common Tests
# ============================================================

def test_common_metadata(df, table_name):

    print("-" * 60)
    print(f"Common metadata tests: {table_name}")
    print("-" * 60)

    required_columns = [
        "dataset_split",
        "history_start",
        "history_end",
        "label_date"
    ]

    # --------------------------------------------------------
    # Required metadata columns
    # --------------------------------------------------------

    test_result(
        "Required metadata columns",
        all(col in df.columns for col in required_columns)
    )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    valid_splits = set(TIME_SPLITS.keys())

    actual_splits = set(
        df["dataset_split"].dropna().unique()
    )

    test_result(
        "Dataset split values",
        actual_splits.issubset(valid_splits),
        f"Unexpected values: {actual_splits - valid_splits}"
    )

    # --------------------------------------------------------
    # Check temporal metadata for each split
    # --------------------------------------------------------

    for split_name, split_info in TIME_SPLITS.items():

        split_df = df[
            df["dataset_split"] == split_name
        ]

        if split_df.empty:
            test_result(
                f"{split_name} rows exist",
                False,
                "No rows found."
            )
            continue

        test_result(
            f"{split_name} history_start",
            split_df["history_start"].astype(str).eq(
                split_info["history_start"]
            ).all()
        )

        test_result(
            f"{split_name} history_end",
            split_df["history_end"].astype(str).eq(
                split_info["history_end"]
            ).all()
        )

        test_result(
            f"{split_name} label_date",
            split_df["label_date"].astype(str).eq(
                split_info["label_date"]
            ).all()
        )


# ============================================================
# 1. User Basic Feature Tests
# ============================================================

def test_user_features():

    print("\n")
    print("=" * 60)
    print("1. USER BASIC FEATURES")
    print("=" * 60)

    df = pd.read_parquet(USER_FEATURE_FILE)

    required_columns = [
        "dataset_split",
        "user_id",
        "history_start",
        "history_end",
        "label_date",
        "event_count",
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
        "buy_conversion_rate"
    ]

    test_result(
        "Required columns",
        all(col in df.columns for col in required_columns)
    )

    # --------------------------------------------------------
    # Common metadata
    # --------------------------------------------------------

    test_common_metadata(
        df,
        "user_features.parquet"
    )

    # --------------------------------------------------------
    # User ID uniqueness
    #
    # Each user should appear once per dataset split.
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["dataset_split", "user_id"]
    ).sum()

    test_result(
        "User-level uniqueness",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}"
    )

    # --------------------------------------------------------
    # Count features should be non-negative
    # --------------------------------------------------------

    count_columns = [
        "event_count",
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count"
    ]

    negative_count = (
        df[count_columns] < 0
    ).sum().sum()

    test_result(
        "Behavior counts non-negative",
        negative_count == 0
    )

    # --------------------------------------------------------
    # Behavior counts should not exceed total event count
    # --------------------------------------------------------

    count_sum = (
        df[
            [
                "pv_count",
                "fav_count",
                "cart_count",
                "buy_count"
            ]
        ].sum(axis=1)
    )

    test_result(
        "Behavior counts <= event_count",
        (count_sum <= df["event_count"]).all()
    )

    # --------------------------------------------------------
    # Purchase conversion rate
    #
    # buy_count / event_count
    # --------------------------------------------------------

    expected_rate = np.where(
        df["event_count"] > 0,
        df["buy_count"] / df["event_count"],
        np.nan
    )

    actual_rate = df["buy_conversion_rate"].values

    rate_match = np.isclose(
        actual_rate,
        expected_rate,
        atol=0.0001,
        equal_nan=True
    )

    test_result(
        "Buy conversion rate",
        rate_match.all()
    )

    # --------------------------------------------------------
    # Conversion rate range
    # --------------------------------------------------------

    test_result(
        "Buy conversion rate >= 0",
        (df["buy_conversion_rate"] >= 0).all()
    )

    test_result(
        "Buy conversion rate <= 1",
        (df["buy_conversion_rate"] <= 1).all()
    )

    return df


# ============================================================
# 2. User Activity Feature Tests
# ============================================================

def test_user_activity_features():

    print("\n")
    print("=" * 60)
    print("2. USER ACTIVITY FEATURES")
    print("=" * 60)

    df = pd.read_parquet(USER_ACTIVITY_FILE)

    required_columns = [
        "dataset_split",
        "user_id",
        "history_start",
        "history_end",
        "label_date",
        "window_days",
        "event_count",
        "active_day_count",
        "active_day_ratio",
        "avg_daily_event_count",
        "avg_active_day_event_count",
        "days_since_last_event",
        "unique_item_count",
        "unique_category_count",
        "pv_count_per_day",
        "fav_count_per_day",
        "cart_count_per_day",
        "buy_count_per_day",
        "activity_level"
    ]

    test_result(
        "Required columns",
        all(col in df.columns for col in required_columns)
    )

    # --------------------------------------------------------
    # Common metadata
    # --------------------------------------------------------

    test_common_metadata(
        df,
        "user_activity_features.parquet"
    )

    # --------------------------------------------------------
    # User-level uniqueness
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["dataset_split", "user_id"]
    ).sum()

    test_result(
        "User-level uniqueness",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}"
    )

    # --------------------------------------------------------
    # Window days
    # --------------------------------------------------------

    expected_window_days = {
        "train": 20,
        "validation": 6,
        "test": 2
    }

    for split, expected_days in expected_window_days.items():

        split_df = df[
            df["dataset_split"] == split
        ]

        if not split_df.empty:

            test_result(
                f"{split} window_days",
                (split_df["window_days"] == expected_days).all()
            )

    # --------------------------------------------------------
    # Count features
    # --------------------------------------------------------

    count_columns = [
        "event_count",
        "active_day_count",
        "unique_item_count",
        "unique_category_count"
    ]

    negative_count = (
        df[count_columns] < 0
    ).sum().sum()

    test_result(
        "Activity counts non-negative",
        negative_count == 0
    )

    # --------------------------------------------------------
    # Active day count cannot exceed window days
    # --------------------------------------------------------

    test_result(
        "Active days <= window days",
        (
            df["active_day_count"]
            <= df["window_days"]
        ).all()
    )

    # --------------------------------------------------------
    # Active day ratio
    # --------------------------------------------------------

    expected_ratio = (
        df["active_day_count"]
        / df["window_days"]
    )

    test_result(
        "Active day ratio",
        np.isclose(
            df["active_day_ratio"],
            expected_ratio,
            atol=0.0001
        ).all()
    )

    # --------------------------------------------------------
    # Average daily event count
    # --------------------------------------------------------

    expected_avg_daily = (
        df["event_count"]
        / df["window_days"]
    )

    test_result(
        "Average daily event count",
        np.isclose(
            df["avg_daily_event_count"],
            expected_avg_daily,
            atol=0.0001
        ).all()
    )

    # --------------------------------------------------------
    # Days since last event
    # --------------------------------------------------------

    test_result(
        "Days since last event non-negative",
        (df["days_since_last_event"] >= 0).all()
    )

    # --------------------------------------------------------
    # Activity level
    # --------------------------------------------------------

    valid_levels = {
        "low",
        "medium",
        "high"
    }

    actual_levels = set(
        df["activity_level"].dropna().unique()
    )

    test_result(
        "Activity level values",
        actual_levels.issubset(valid_levels),
        f"Unexpected values: {actual_levels - valid_levels}"
    )

    return df


# ============================================================
# 3. User-Item Sequence Feature Tests
# ============================================================

def test_user_sequence_features():

    print("\n")
    print("=" * 60)
    print("3. USER-ITEM SEQUENCE FEATURES")
    print("=" * 60)

    df = pd.read_parquet(USER_SEQUENCE_FILE)

    required_columns = [
        "dataset_split",
        "user_id",
        "item_id",
        "history_start",
        "history_end",
        "label_date",
        "last_behavior_type",
        "last_behavior_hour",
        "last_behavior_days_ago",
        "last_10_behavior_sequence",
        "pv_to_cart_count",
        "cart_to_buy_count",
        "pv_to_buy_count",
        "fav_to_buy_count"
    ]

    test_result(
        "Required columns",
        all(col in df.columns for col in required_columns)
    )

    # --------------------------------------------------------
    # Common metadata
    # --------------------------------------------------------

    test_common_metadata(
        df,
        "user_sequence_features.parquet"
    )

    # --------------------------------------------------------
    # User-item uniqueness
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "dataset_split",
            "user_id",
            "item_id"
        ]
    ).sum()

    test_result(
        "User-item uniqueness",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}"
    )

    # --------------------------------------------------------
    # Valid behavior types
    # --------------------------------------------------------

    valid_behaviors = {
        "pv",
        "fav",
        "cart",
        "buy"
    }

    actual_behaviors = set(
        df["last_behavior_type"]
        .dropna()
        .unique()
    )

    test_result(
        "Last behavior type values",
        actual_behaviors.issubset(valid_behaviors),
        f"Unexpected values: {actual_behaviors - valid_behaviors}"
    )

    # --------------------------------------------------------
    # Last behavior hour
    # --------------------------------------------------------

    test_result(
        "Last behavior hour range",
        (
            df["last_behavior_hour"].between(0, 23)
        ).all()
    )

    # --------------------------------------------------------
    # Days since last behavior
    # --------------------------------------------------------

    test_result(
        "Last behavior days ago non-negative",
        (
            df["last_behavior_days_ago"] >= 0
        ).all()
    )

    # --------------------------------------------------------
    # Sequence should not be empty
    # --------------------------------------------------------

    test_result(
        "Last 10 behavior sequence not empty",
        df["last_10_behavior_sequence"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    # --------------------------------------------------------
    # Sequence should contain only valid behaviors
    # --------------------------------------------------------

    invalid_sequence = 0

    for sequence in (
        df["last_10_behavior_sequence"]
        .dropna()
    ):

        behaviors = sequence.split("→")

        if (
            len(behaviors) > 10
            or not all(
                behavior in valid_behaviors
                for behavior in behaviors
            )
        ):
            invalid_sequence += 1

    test_result(
        "Last 10 behavior sequence format",
        invalid_sequence == 0,
        f"Invalid sequences: {invalid_sequence}"
    )

    # --------------------------------------------------------
    # Transition counts
    # --------------------------------------------------------

    transition_columns = [
        "pv_to_cart_count",
        "cart_to_buy_count",
        "pv_to_buy_count",
        "fav_to_buy_count"
    ]

    negative_transition_count = (
        df[transition_columns] < 0
    ).sum().sum()

    test_result(
        "Transition counts non-negative",
        negative_transition_count == 0
    )

    # --------------------------------------------------------
    # Transition counts should be integers
    # --------------------------------------------------------

    integer_check = all(
        pd.api.types.is_integer_dtype(
            df[col]
        )
        for col in transition_columns
    )

    test_result(
        "Transition counts are integers",
        integer_check
    )

    return df


# ============================================================
# 4. Item Behavior Feature Tests
# ============================================================

def test_item_features():

    print("\n")
    print("=" * 60)
    print("4. ITEM BEHAVIOR FEATURES")
    print("=" * 60)

    df = pd.read_parquet(ITEM_FEATURE_FILE)

    required_columns = [
        "dataset_split",
        "item_id",
        "category_id",
        "history_start",
        "history_end",
        "label_date",
        "item_total_count",
        "item_pv_count",
        "item_fav_count",
        "item_cart_count",
        "item_buy_count",
        "item_unique_user_count",
        "item_unique_buyer_count",
        "item_active_day_count",
        "item_fav_to_pv_rate",
        "item_cart_to_pv_rate",
        "item_buy_to_pv_rate",
        "item_heat_level"
    ]

    test_result(
        "Required columns",
        all(col in df.columns for col in required_columns)
    )

    # --------------------------------------------------------
    # Common metadata
    # --------------------------------------------------------

    test_common_metadata(
        df,
        "item_behavior_features.parquet"
    )

    # --------------------------------------------------------
    # Item uniqueness
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "dataset_split",
            "item_id"
        ]
    ).sum()

    test_result(
        "Item-level uniqueness",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}"
    )

    # --------------------------------------------------------
    # Count columns
    # --------------------------------------------------------

    count_columns = [
        "item_total_count",
        "item_pv_count",
        "item_fav_count",
        "item_cart_count",
        "item_buy_count",
        "item_unique_user_count",
        "item_unique_buyer_count",
        "item_active_day_count"
    ]

    negative_count = (
        df[count_columns] < 0
    ).sum().sum()

    test_result(
        "Item counts non-negative",
        negative_count == 0
    )

    # --------------------------------------------------------
    # Behavior counts <= total count
    # --------------------------------------------------------

    behavior_sum = (
        df[
            [
                "item_pv_count",
                "item_fav_count",
                "item_cart_count",
                "item_buy_count"
            ]
        ].sum(axis=1)
    )

    test_result(
        "Behavior counts <= item_total_count",
        (
            behavior_sum
            <= df["item_total_count"]
        ).all()
    )

    # --------------------------------------------------------
    # Unique buyers <= unique users
    # --------------------------------------------------------

    test_result(
        "Unique buyers <= unique users",
        (
            df["item_unique_buyer_count"]
            <= df["item_unique_user_count"]
        ).all()
    )

    # --------------------------------------------------------
    # Active days > 0
    # --------------------------------------------------------

    test_result(
        "Active day count > 0",
        (
            df["item_active_day_count"] > 0
        ).all()
    )

    # --------------------------------------------------------
    # Conversion rates
    # --------------------------------------------------------

    expected_fav_rate = np.where(
        df["item_pv_count"] > 0,
        df["item_fav_count"]
        / df["item_pv_count"],
        0
    )

    expected_cart_rate = np.where(
        df["item_pv_count"] > 0,
        df["item_cart_count"]
        / df["item_pv_count"],
        0
    )

    expected_buy_rate = np.where(
        df["item_pv_count"] > 0,
        df["item_buy_count"]
        / df["item_pv_count"],
        0
    )

    test_result(
        "Favorite-to-PV conversion rate",
        np.isclose(
            df["item_fav_to_pv_rate"],
            expected_fav_rate,
            atol=0.0001
        ).all()
    )

    test_result(
        "Cart-to-PV conversion rate",
        np.isclose(
            df["item_cart_to_pv_rate"],
            expected_cart_rate,
            atol=0.0001
        ).all()
    )

    test_result(
        "Buy-to-PV conversion rate",
        np.isclose(
            df["item_buy_to_pv_rate"],
            expected_buy_rate,
            atol=0.0001
        ).all()
    )

    # --------------------------------------------------------
    # Heat level
    # --------------------------------------------------------

    valid_heat_levels = {
        "low",
        "medium",
        "high"
    }

    actual_heat_levels = set(
        df["item_heat_level"]
        .dropna()
        .unique()
    )

    test_result(
        "Item heat level values",
        actual_heat_levels.issubset(
            valid_heat_levels
        ),
        f"Unexpected values: "
        f"{actual_heat_levels - valid_heat_levels}"
    )

    # --------------------------------------------------------
    # Verify train P25/P75 thresholds
    # --------------------------------------------------------

    train_values = df.loc[
        df["dataset_split"] == "train",
        "item_total_count"
    ]

    p25 = train_values.quantile(0.25)
    p75 = train_values.quantile(0.75)

    expected_heat = np.select(
        [
            df["item_total_count"] <= p25,
            df["item_total_count"] <= p75
        ],
        [
            "low",
            "medium"
        ],
        default="high"
    )

    test_result(
        "Item heat level follows train P25/P75",
        (
            df["item_heat_level"].values
            == expected_heat
        ).all()
    )

    return df


# ============================================================
# Main Test Pipeline
# ============================================================

def main():

    global total_tests
    global passed_tests
    global failed_tests

    print("=" * 60)
    print("FEATURE TABLE FUNCTION TEST")
    print("=" * 60)

    print("\nLoading raw data...")

    raw_df = pd.read_parquet(
        RAW_FILE,
        columns=[
            "time",
            "user_id",
            "item_id",
            "category_id",
            "behavior_name",
            "behavior_date"
        ]
    )

    print(
        "Raw data rows:",
        f"{len(raw_df):,}"
    )

    # --------------------------------------------------------
    # Check raw data exists
    # --------------------------------------------------------

    test_result(
        "Raw data loaded",
        len(raw_df) > 0
    )

    # --------------------------------------------------------
    # Run four feature table tests
    # --------------------------------------------------------

    test_user_features()

    test_user_activity_features()

    test_user_sequence_features()

    test_item_features()

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")
    print("=" * 60)
    print("OVERALL TEST RESULT")
    print("=" * 60)

    print(
        f"TOTAL TESTS:  {total_tests}"
    )

    print(
        f"PASSED:       {passed_tests}"
    )

    print(
        f"FAILED:       {failed_tests}"
    )

    print("-" * 60)

    if failed_tests == 0:

        print(
            "OVERALL RESULT: PASS"
        )

    else:

        print(
            "OVERALL RESULT: FAIL"
        )

    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()