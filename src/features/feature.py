import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

# 以后如果路径变了，只需要修改这里
INPUT_FILE = (
    "data/processed/user_behavior_clean.parquet"
)

OUTPUT_DIR = (
    "data/feature"
)

# Output files
USER_FEATURES_FILE = (
    rf"{OUTPUT_DIR}\user_features.parquet"
)

USER_ACTIVITY_FEATURES_FILE = (
    rf"{OUTPUT_DIR}\user_activity_features.parquet"
)

USER_SEQUENCE_FEATURES_FILE = (
    rf"{OUTPUT_DIR}\user_sequence_features.parquet"
)

ITEM_BEHAVIOR_FEATURES_FILE = (
    rf"{OUTPUT_DIR}\item_behavior_features.parquet"
)


# ============================================================
# Temporal Split Definition
# ============================================================

TIME_SPLITS = [
    {
        "dataset_split": "train",
        "history_start": "2025-11-18",
        "history_end": "2025-12-07",
        "label_date": "2025-12-08"
    },
    {
        "dataset_split": "validation",
        "history_start": "2025-12-09",
        "history_end": "2025-12-14",
        "label_date": "2025-12-15"
    },
    {
        "dataset_split": "test",
        "history_start": "2025-12-16",
        "history_end": "2025-12-17",
        "label_date": "2025-12-18"
    }
]


# ============================================================
# 1. User Basic Behavior Features
# ============================================================

def calculate_user_basic_features(
        df,
        dataset_split,
        history_start,
        history_end,
        label_date
):
    """
    Calculate user-level historical behavior features
    within a specified time window.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean user behavior dataset.

    dataset_split : str
        train / validation / test.

    history_start : str
        Start date of historical behavior window.

    history_end : str
        End date of historical behavior window.

    label_date : str
        Prediction date.

    Returns
    -------
    pandas.DataFrame
        User basic behavior features.
    """

    # Filter historical window only
    mask = (
        (df["behavior_date"] >= history_start)
        &
        (df["behavior_date"] <= history_end)
    )

    history_df = df.loc[mask].copy()

    # Aggregate user behaviors
    result = (
        history_df
        .groupby("user_id")
        .agg(
            event_count=(
                "behavior_name",
                "count"
            ),
            pv_count=(
                "behavior_name",
                lambda x: (x == "pv").sum()
            ),
            fav_count=(
                "behavior_name",
                lambda x: (x == "fav").sum()
            ),
            cart_count=(
                "behavior_name",
                lambda x: (x == "cart").sum()
            ),
            buy_count=(
                "behavior_name",
                lambda x: (x == "buy").sum()
            )
        )
        .reset_index()
    )

    # Purchase conversion rate
    result["buy_conversion_rate"] = (
        result["buy_count"]
        /
        result["event_count"]
    ).round(4)

    # Add temporal metadata
    result.insert(
        0,
        "dataset_split",
        dataset_split
    )

    result.insert(
        2,
        "history_start",
        history_start
    )

    result.insert(
        3,
        "history_end",
        history_end
    )

    result.insert(
        4,
        "label_date",
        label_date
    )

    return result


# ============================================================
# 2. User Activity Features
# ============================================================

def calculate_user_activity_features(
        df,
        dataset_split,
        history_start,
        history_end,
        label_date
):

    print("\n" + "-" * 60)
    print(f"Processing {dataset_split}")
    print("-" * 60)

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    history_start_dt = pd.Timestamp(history_start)
    history_end_dt = pd.Timestamp(history_end)
    label_date_dt = pd.Timestamp(label_date)

    # --------------------------------------------------------
    # Calculate window days
    # --------------------------------------------------------

    window_days = (
        history_end_dt - history_start_dt
    ).days + 1

    print(f"History: {history_start} -> {history_end}")
    print(f"Label date: {label_date}")
    print(f"Window days: {window_days}")

    # --------------------------------------------------------
    # Filter history only
    #
    # IMPORTANT:
    # No data from label_date is used.
    # --------------------------------------------------------

    mask = (
        (df["behavior_date"] >= history_start_dt)
        &
        (df["behavior_date"] < history_end_dt + pd.Timedelta(days=1))
    )

    history_df = df.loc[mask].copy()

    print(f"History events: {len(history_df):,}")

    # --------------------------------------------------------
    # Convert actual behavior timestamp
    # --------------------------------------------------------

    history_df["behavior_time"] = pd.to_datetime(
        history_df["time"],
        errors="coerce"
    )

    # Remove invalid timestamps
    history_df = history_df.dropna(
        subset=["behavior_time"]
    )

    # --------------------------------------------------------
    # Create event date
    # --------------------------------------------------------

    history_df["event_date"] = (
        history_df["behavior_time"]
        .dt.normalize()
    )

    # --------------------------------------------------------
    # Aggregate user-level activity
    # --------------------------------------------------------

    result = (
        history_df
        .groupby("user_id")
        .agg(

            # Total number of behaviors
            event_count=(
                "behavior_name",
                "count"
            ),

            # Number of unique active days
            active_day_count=(
                "event_date",
                "nunique"
            ),

            # Last behavior timestamp
            last_event_time=(
                "behavior_time",
                "max"
            ),

            # Unique items
            unique_item_count=(
                "item_id",
                "nunique"
            ),

            # Unique categories
            unique_category_count=(
                "category_id",
                "nunique"
            ),

            # PV
            pv_count=(
                "behavior_name",
                lambda x: (x == "pv").sum()
            ),

            # Favorite
            fav_count=(
                "behavior_name",
                lambda x: (x == "fav").sum()
            ),

            # Cart
            cart_count=(
                "behavior_name",
                lambda x: (x == "cart").sum()
            ),

            # Purchase
            buy_count=(
                "behavior_name",
                lambda x: (x == "buy").sum()
            )
        )
        .reset_index()
    )

    # ========================================================
    # Continuous Activity Features
    # ========================================================

    # --------------------------------------------------------
    # Active day ratio
    # --------------------------------------------------------

    result["active_day_ratio"] = (
        result["active_day_count"]
        / window_days
    )

    # --------------------------------------------------------
    # Average daily event count
    # --------------------------------------------------------

    result["avg_daily_event_count"] = (
        result["event_count"]
        / window_days
    )

    # --------------------------------------------------------
    # Average events on active days
    # --------------------------------------------------------

    result["avg_active_day_event_count"] = (
        result["event_count"]
        / result["active_day_count"]
    )

    result["avg_active_day_event_count"] = (
        result["avg_active_day_event_count"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Days since last event
    # --------------------------------------------------------

    result["days_since_last_event"] = (
        (
            label_date_dt
            - result["last_event_time"]
        )
        .dt.total_seconds()
        / (24 * 60 * 60)
    )

    # --------------------------------------------------------
    # Behavior counts per day
    # --------------------------------------------------------

    result["pv_count_per_day"] = (
        result["pv_count"]
        / window_days
    )

    result["fav_count_per_day"] = (
        result["fav_count"]
        / window_days
    )

    result["cart_count_per_day"] = (
        result["cart_count"]
        / window_days
    )

    result["buy_count_per_day"] = (
        result["buy_count"]
        / window_days
    )

    # --------------------------------------------------------
    # Round continuous features
    # --------------------------------------------------------

    continuous_columns = [
        "active_day_ratio",
        "avg_daily_event_count",
        "avg_active_day_event_count",
        "days_since_last_event",
        "pv_count_per_day",
        "fav_count_per_day",
        "cart_count_per_day",
        "buy_count_per_day"
    ]

    result[continuous_columns] = (
        result[continuous_columns]
        .round(4)
    )

    # ========================================================
    # Add Temporal Metadata
    # ========================================================

    result.insert(
        0,
        "dataset_split",
        dataset_split
    )

    result.insert(
        2,
        "history_start",
        history_start
    )

    result.insert(
        3,
        "history_end",
        history_end
    )

    result.insert(
        4,
        "label_date",
        label_date
    )

    result.insert(
        5,
        "window_days",
        window_days
    )

    # ========================================================
    # Reorder columns
    # ========================================================

    result = result[
        [
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
            "buy_count_per_day"
        ]
    ]

    print(f"Users: {len(result):,}")

    return result


# ============================================================
# 3. User-Item Sequence Features
# ============================================================

def calculate_user_item_sequence_features(
        df,
        dataset_split,
        history_start,
        history_end,
        label_date
):

    print("\n" + "-" * 60)
    print(f"Processing: {dataset_split}")
    print("-" * 60)

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    history_start_dt = pd.Timestamp(history_start)
    history_end_dt = pd.Timestamp(history_end)
    label_date_dt = pd.Timestamp(label_date)

    # --------------------------------------------------------
    # Filter history window ONLY
    #
    # IMPORTANT:
    # history_end is inclusive.
    # label_date and later data are excluded.
    # --------------------------------------------------------

    mask = (
        (df["behavior_time"] >= history_start_dt)
        &
        (df["behavior_time"] < label_date_dt)
    )

    history_df = df.loc[mask].copy()

    print(
        "History events:",
        f"{len(history_df):,}"
    )

    if history_df.empty:
        print("No history data found.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    history_df = history_df.sort_values(
        ["user_id", "item_id", "behavior_time"]
    )

    # ========================================================
    # Create User-Item Groups
    # ========================================================

    # --------------------------------------------------------
    # Last behavior
    # --------------------------------------------------------

    last_event = (
        history_df
        .groupby(
            ["user_id", "item_id"],
            sort=False
        )
        .tail(1)
        [
            [
                "user_id",
                "item_id",
                "behavior_name",
                "behavior_time"
            ]
        ]
        .rename(
            columns={
                "behavior_name": "last_behavior_type",
                "behavior_time": "last_behavior_time"
            }
        )
    )

    # --------------------------------------------------------
    # Last behavior hour
    # --------------------------------------------------------

    last_event["last_behavior_hour"] = (
        last_event["last_behavior_time"].dt.hour
    )

    # --------------------------------------------------------
    # Days since last behavior
    # --------------------------------------------------------

    last_event["last_behavior_days_ago"] = (
        (
            label_date_dt
            - last_event["last_behavior_time"]
        )
        .dt.total_seconds()
        / (24 * 60 * 60)
    ).round(4)

    # ========================================================
    # Last 10 Behavior Sequence
    # ========================================================

    # --------------------------------------------------------
    # Keep only the latest 10 behaviors
    # --------------------------------------------------------

    last_10 = (
        history_df
        .groupby(
            ["user_id", "item_id"],
            sort=False
        )
        .tail(10)
    )

    # --------------------------------------------------------
    # Create sequence
    # --------------------------------------------------------

    sequence_features = (
        last_10
        .groupby(
            ["user_id", "item_id"],
            sort=False
        )["behavior_name"]
        .agg(lambda x: "→".join(x))
        .reset_index(
            name="last_10_behavior_sequence"
        )
    )

    # ========================================================
    # Behavior Transition Features
    # ========================================================

    # --------------------------------------------------------
    # Create previous behavior
    # --------------------------------------------------------

    history_df["previous_behavior"] = (
        history_df
        .groupby(
            ["user_id", "item_id"],
            sort=False
        )["behavior_name"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Create transition labels
    # --------------------------------------------------------

    history_df["transition"] = (
        history_df["previous_behavior"]
        + "_to_"
        + history_df["behavior_name"]
    )

    # --------------------------------------------------------
    # Count selected transitions
    # --------------------------------------------------------

    transition_counts = (
        history_df
        .groupby(
            [
                "user_id",
                "item_id",
                "transition"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Make sure all required transition columns exist
    # --------------------------------------------------------

    required_transitions = [
        "pv_to_cart",
        "cart_to_buy",
        "pv_to_buy",
        "fav_to_buy"
    ]

    for transition in required_transitions:
        if transition not in transition_counts.columns:
            transition_counts[transition] = 0

    # --------------------------------------------------------
    # Keep only required transition features
    # --------------------------------------------------------

    transition_counts = transition_counts[
        [
            "user_id",
            "item_id",
            "pv_to_cart",
            "cart_to_buy",
            "pv_to_buy",
            "fav_to_buy"
        ]
    ]

    # --------------------------------------------------------
    # Rename transition columns
    # --------------------------------------------------------

    transition_counts = transition_counts.rename(
        columns={
            "pv_to_cart": "pv_to_cart_count",
            "cart_to_buy": "cart_to_buy_count",
            "pv_to_buy": "pv_to_buy_count",
            "fav_to_buy": "fav_to_buy_count"
        }
    )

    # ========================================================
    # Combine Features
    # ========================================================

    result = (
        last_event[
            [
                "user_id",
                "item_id",
                "last_behavior_type",
                "last_behavior_hour",
                "last_behavior_days_ago"
            ]
        ]
        .merge(
            sequence_features,
            on=["user_id", "item_id"],
            how="left"
        )
        .merge(
            transition_counts,
            on=["user_id", "item_id"],
            how="left"
        )
    )

    # --------------------------------------------------------
    # Fill missing transition counts
    # --------------------------------------------------------

    transition_columns = [
        "pv_to_cart_count",
        "cart_to_buy_count",
        "pv_to_buy_count",
        "fav_to_buy_count"
    ]

    result[transition_columns] = (
        result[transition_columns]
        .fillna(0)
        .astype("int64")
    )

    # ========================================================
    # Add Temporal Metadata
    # ========================================================

    result.insert(
        0,
        "dataset_split",
        dataset_split
    )

    result.insert(
        3,
        "history_start",
        history_start
    )

    result.insert(
        4,
        "history_end",
        history_end
    )

    result.insert(
        5,
        "label_date",
        label_date
    )

    # ========================================================
    # Reorder Columns
    # ========================================================

    result = result[
        [
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
    ]

    print(
        "User-item pairs:",
        f"{len(result):,}"
    )

    return result


# ============================================================
# 4. Item Behavior Features
# ============================================================

def calculate_item_behavior_features(
        df,
        dataset_split,
        history_start,
        history_end,
        label_date
):

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_split}")
    print("=" * 60)

    history_start_dt = pd.Timestamp(history_start)
    label_date_dt = pd.Timestamp(label_date)

    # --------------------------------------------------------
    # Filter history window
    # --------------------------------------------------------

    mask = (
        (df["behavior_time"] >= history_start_dt)
        &
        (df["behavior_time"] < label_date_dt)
    )

    history_df = df.loc[
        mask,
        [
            "user_id",
            "item_id",
            "category_id",
            "behavior_name",
            "behavior_time"
        ]
    ].copy()

    print(
        "History events:",
        f"{len(history_df):,}"
    )

    if history_df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Create event date
    # --------------------------------------------------------

    history_df["event_date"] = (
        history_df["behavior_time"]
        .dt.date
    )

    # ========================================================
    # Basic Item Counts
    # ========================================================

    print("Calculating item behavior counts...")

    # Total behavior count
    item_total = (
        history_df
        .groupby("item_id")
        .size()
        .rename("item_total_count")
    )

    # Behavior type counts
    behavior_counts = pd.crosstab(
        history_df["item_id"],
        history_df["behavior_name"]
    )

    # Make sure all behavior columns exist
    for behavior in ["pv", "fav", "cart", "buy"]:
        if behavior not in behavior_counts.columns:
            behavior_counts[behavior] = 0

    behavior_counts = behavior_counts[
        ["pv", "fav", "cart", "buy"]
    ]

    behavior_counts = behavior_counts.rename(
        columns={
            "pv": "item_pv_count",
            "fav": "item_fav_count",
            "cart": "item_cart_count",
            "buy": "item_buy_count"
        }
    )

    # ========================================================
    # Unique Users
    # ========================================================

    print("Calculating unique users...")

    item_unique_users = (
        history_df
        .groupby("item_id")["user_id"]
        .nunique()
        .rename("item_unique_user_count")
    )

    # ========================================================
    # Unique Buyers
    # ========================================================

    print("Calculating unique buyers...")

    buy_df = history_df.loc[
        history_df["behavior_name"] == "buy",
        ["item_id", "user_id"]
    ]

    item_unique_buyers = (
        buy_df
        .groupby("item_id")["user_id"]
        .nunique()
        .rename("item_unique_buyer_count")
    )

    # ========================================================
    # Active Days
    # ========================================================

    print("Calculating active days...")

    item_active_days = (
        history_df
        .groupby("item_id")["event_date"]
        .nunique()
        .rename("item_active_day_count")
    )

    # ========================================================
    # Category
    # ========================================================

    item_category = (
        history_df
        .groupby("item_id")["category_id"]
        .first()
        .rename("category_id")
    )

    # ========================================================
    # Combine
    # ========================================================

    result = pd.concat(
        [
            item_category,
            item_total,
            behavior_counts,
            item_unique_users,
            item_unique_buyers,
            item_active_days
        ],
        axis=1
    ).reset_index()

    # --------------------------------------------------------
    # Fill missing counts
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

    result[count_columns] = (
        result[count_columns]
        .fillna(0)
        .astype("int64")
    )

    # ========================================================
    # Conversion Rates
    # ========================================================

    print("Calculating conversion rates...")

    result["item_fav_to_pv_rate"] = np.where(
        result["item_pv_count"] > 0,
        result["item_fav_count"]
        / result["item_pv_count"],
        0
    )

    result["item_cart_to_pv_rate"] = np.where(
        result["item_pv_count"] > 0,
        result["item_cart_count"]
        / result["item_pv_count"],
        0
    )

    result["item_buy_to_pv_rate"] = np.where(
        result["item_pv_count"] > 0,
        result["item_buy_count"]
        / result["item_pv_count"],
        0
    )

    result[
        [
            "item_fav_to_pv_rate",
            "item_cart_to_pv_rate",
            "item_buy_to_pv_rate"
        ]
    ] = (
        result[
            [
                "item_fav_to_pv_rate",
                "item_cart_to_pv_rate",
                "item_buy_to_pv_rate"
            ]
        ]
        .round(4)
    )

    # ========================================================
    # Add Temporal Metadata
    # ========================================================

    result.insert(
        0,
        "dataset_split",
        dataset_split
    )

    result.insert(
        3,
        "history_start",
        history_start
    )

    result.insert(
        4,
        "history_end",
        history_end
    )

    result.insert(
        5,
        "label_date",
        label_date
    )

    # ========================================================
    # Column Order
    # ========================================================

    result = result[
        [
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
            "item_buy_to_pv_rate"
        ]
    ]

    print(
        "Items:",
        f"{len(result):,}"
    )

    return result


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("=" * 70)
    print("USER / ITEM FEATURE ENGINEERING")
    print("=" * 70)

    # ========================================================
    # Load data ONLY ONCE
    # ========================================================

    print("\nLoading data...")

    df = pd.read_parquet(INPUT_FILE)

    print(
        "Data loaded:",
        f"{len(df):,}",
        "rows"
    )

    print(
        "Columns:",
        ", ".join(df.columns)
    )

    # ========================================================
    # Check required columns
    # ========================================================

    required_columns = [
        "time",
        "user_id",
        "item_id",
        "category_id",
        "behavior_name",
        "behavior_date"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # Prepare date / timestamp columns ONLY ONCE
    # ========================================================

    print("\nConverting dates and timestamps...")

    df["behavior_date"] = pd.to_datetime(
        df["behavior_date"],
        errors="coerce"
    )

    df["behavior_time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "behavior_date",
            "behavior_time"
        ]
    )

    print(
        "Valid rows after date conversion:",
        f"{len(df):,}"
    )

    # ========================================================
    # 1. USER BASIC FEATURES
    # ========================================================

    print("\n")
    print("#" * 70)
    print("1. USER BASIC BEHAVIOR FEATURES")
    print("#" * 70)

    user_basic_list = []

    for split in TIME_SPLITS:

        features = calculate_user_basic_features(
            df=df,
            dataset_split=split["dataset_split"],
            history_start=split["history_start"],
            history_end=split["history_end"],
            label_date=split["label_date"]
        )

        user_basic_list.append(features)

    user_basic_features = pd.concat(
        user_basic_list,
        ignore_index=True
    )

    user_basic_features.to_parquet(
        USER_FEATURES_FILE,
        index=False
    )

    print(
        "\nUser basic features saved:",
        USER_FEATURES_FILE
    )

    print(
        "Shape:",
        user_basic_features.shape
    )


    # ========================================================
    # 2. USER ACTIVITY FEATURES
    # ========================================================

    print("\n")
    print("#" * 70)
    print("2. USER ACTIVITY FEATURES")
    print("#" * 70)

    user_activity_list = []

    for split in TIME_SPLITS:

        features = calculate_user_activity_features(
            df=df,
            dataset_split=split["dataset_split"],
            history_start=split["history_start"],
            history_end=split["history_end"],
            label_date=split["label_date"]
        )

        user_activity_list.append(features)

    user_activity_features = pd.concat(
        user_activity_list,
        ignore_index=True
    )

    # ========================================================
    # Calculate Activity Level
    #
    # IMPORTANT:
    # P25 / P75 are calculated ONLY from TRAIN.
    # ========================================================

    train_values = user_activity_features.loc[
        user_activity_features["dataset_split"] == "train",
        "avg_daily_event_count"
    ]

    p25 = train_values.quantile(0.25)
    p75 = train_values.quantile(0.75)

    print("\n" + "=" * 60)
    print("ACTIVITY LEVEL THRESHOLDS")
    print("=" * 60)

    print(
        "Training P25:",
        round(p25, 4)
    )

    print(
        "Training P75:",
        round(p75, 4)
    )

    # --------------------------------------------------------
    # Apply fixed train thresholds to ALL splits
    # --------------------------------------------------------

    user_activity_features["activity_level"] = np.select(
        [
            user_activity_features[
                "avg_daily_event_count"
            ] <= p25,

            user_activity_features[
                "avg_daily_event_count"
            ] <= p75
        ],
        [
            "low",
            "medium"
        ],
        default="high"
    )

    user_activity_features.to_parquet(
        USER_ACTIVITY_FEATURES_FILE,
        index=False
    )

    print(
        "\nUser activity features saved:",
        USER_ACTIVITY_FEATURES_FILE
    )

    print(
        "Shape:",
        user_activity_features.shape
    )


    # ========================================================
    # 3. USER-ITEM SEQUENCE FEATURES
    # ========================================================

    print("\n")
    print("#" * 70)
    print("3. USER-ITEM SEQUENCE FEATURES")
    print("#" * 70)

    user_sequence_list = []

    for split in TIME_SPLITS:

        features = calculate_user_item_sequence_features(
            df=df,
            dataset_split=split["dataset_split"],
            history_start=split["history_start"],
            history_end=split["history_end"],
            label_date=split["label_date"]
        )

        if not features.empty:
            user_sequence_list.append(features)

    if not user_sequence_list:
        raise ValueError(
            "No user-item sequence features were generated."
        )

    user_sequence_features = pd.concat(
        user_sequence_list,
        ignore_index=True
    )

    user_sequence_features.to_parquet(
        USER_SEQUENCE_FEATURES_FILE,
        index=False
    )

    print(
        "\nUser sequence features saved:",
        USER_SEQUENCE_FEATURES_FILE
    )

    print(
        "Shape:",
        user_sequence_features.shape
    )


    # ========================================================
    # 4. ITEM BEHAVIOR FEATURES
    # ========================================================

    print("\n")
    print("#" * 70)
    print("4. ITEM BEHAVIOR FEATURES")
    print("#" * 70)

    item_behavior_list = []

    for split in TIME_SPLITS:

        features = calculate_item_behavior_features(
            df=df,
            dataset_split=split["dataset_split"],
            history_start=split["history_start"],
            history_end=split["history_end"],
            label_date=split["label_date"]
        )

        if not features.empty:
            item_behavior_list.append(features)

    if not item_behavior_list:
        raise ValueError(
            "No item behavior features were generated."
        )

    item_behavior_features = pd.concat(
        item_behavior_list,
        ignore_index=True
    )

    # ========================================================
    # Item Heat Level
    # ========================================================

    print("\n" + "=" * 60)
    print("Calculating item heat level...")
    print("=" * 60)

    train_values = item_behavior_features.loc[
        item_behavior_features["dataset_split"] == "train",
        "item_total_count"
    ]

    p25 = train_values.quantile(0.25)
    p75 = train_values.quantile(0.75)

    print(
        "Training P25:",
        round(p25, 4)
    )

    print(
        "Training P75:",
        round(p75, 4)
    )

    item_behavior_features["item_heat_level"] = np.select(
        [
            item_behavior_features[
                "item_total_count"
            ] <= p25,

            item_behavior_features[
                "item_total_count"
            ] <= p75
        ],
        [
            "low",
            "medium"
        ],
        default="high"
    )

    # ========================================================
    # Save
    # ========================================================

    item_behavior_features.to_parquet(
        ITEM_BEHAVIOR_FEATURES_FILE,
        index=False
    )

    print(
        "\nItem behavior features saved:",
        ITEM_BEHAVIOR_FEATURES_FILE
    )

    print(
        "Shape:",
        item_behavior_features.shape
    )


    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ALL FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print("\nOutput files:")

    print(
        "1. User basic:",
        USER_FEATURES_FILE
    )

    print(
        "2. User activity:",
        USER_ACTIVITY_FEATURES_FILE
    )

    print(
        "3. User sequence:",
        USER_SEQUENCE_FEATURES_FILE
    )

    print(
        "4. Item behavior:",
        ITEM_BEHAVIOR_FEATURES_FILE
    )

    print("\nShapes:")

    print(
        "User basic:",
        user_basic_features.shape
    )

    print(
        "User activity:",
        user_activity_features.shape
    )

    print(
        "User sequence:",
        user_sequence_features.shape
    )

    print(
        "Item behavior:",
        item_behavior_features.shape
    )

    print("\nRows by split:")

    print(
        user_basic_features[
            "dataset_split"
        ].value_counts()
    )

    print(
        user_activity_features[
            "dataset_split"
        ].value_counts()
    )

    print(
        user_sequence_features[
            "dataset_split"
        ].value_counts()
    )

    print(
        item_behavior_features[
            "dataset_split"
        ].value_counts()
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
