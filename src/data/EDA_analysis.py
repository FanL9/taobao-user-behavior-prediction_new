import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/processed/user_behavior_clean.parquet"
OUTPUT_DIR = "data/interim"

BEHAVIOR_TYPES = ["pv", "fav", "cart", "buy"]


# ============================================================
# Data Loading
# ============================================================

def load_data(file_path):
    """
    Load the cleaned user behavior dataset.

    Parameters
    ----------
    file_path : str
        Path to the cleaned Parquet file.

    Returns
    -------
    pandas.DataFrame
        Cleaned user behavior dataset.
    """
    df = pd.read_parquet(file_path)
    return df


# ============================================================
# 1. Behavior Distribution
# ============================================================

def calculate_behavior_distribution(df):
    """
    Calculate the distribution of the four behavior types.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        A table containing behavior type, behavior name,
        behavior count, and percentage.
    """

    behavior_counts = df["behavior_name"].value_counts()

    result = pd.DataFrame({
        "behavior_type": [1, 2, 3, 4],
        "behavior_name": ["pv", "fav", "cart", "buy"],
        "behavior_count": [
            behavior_counts.get("pv", 0),
            behavior_counts.get("fav", 0),
            behavior_counts.get("cart", 0),
            behavior_counts.get("buy", 0)
        ]
    })

    total_count = result["behavior_count"].sum()

    result["percentage"] = (
        result["behavior_count"] / total_count * 100
    ).round(2)

    return result


# ============================================================
# 2. User Behavior and Purchase Summary
# ============================================================

def calculate_user_purchase_summary(df):
    """
    Calculate user-level purchase statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Summary containing total behavior count, purchase count,
        purchase users, non-purchase users, and repeat purchase users.
    """

    total_behavior_count = len(df)

    purchase_mask = df["behavior_name"] == "buy"

    purchase_count = purchase_mask.sum()

    purchase_users = df.loc[
        purchase_mask, "user_id"
    ].nunique()

    total_users = df["user_id"].nunique()

    non_purchase_users = total_users - purchase_users

    user_purchase_counts = (
        df.loc[purchase_mask]
        .groupby("user_id")
        .size()
    )

    repeat_purchase_users = (
        (user_purchase_counts >= 2).sum()
    )

    result = pd.DataFrame({
        "total_behavior_count": [total_behavior_count],
        "purchase_count": [purchase_count],
        "purchase_users": [purchase_users],
        "non_purchase_users": [non_purchase_users],
        "repeat_purchase_users": [repeat_purchase_users]
    })

    return result


# ============================================================
# 3. Item Statistics
# ============================================================

def calculate_item_statistics(df):
    """
    Calculate behavior statistics for each item.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Item-level statistics containing PV, favorite,
        cart, and purchase counts.
    """

    result = (
        df.groupby(["item_id", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for behavior in BEHAVIOR_TYPES:
        if behavior not in result.columns:
            result[behavior] = 0

    result = result.rename(columns={
        "pv": "pv_count",
        "fav": "fav_count",
        "cart": "cart_count",
        "buy": "buy_count"
    })

    result = result[
        [
            "item_id",
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count"
        ]
    ]

    result = result.sort_values("item_id").reset_index(drop=True)

    return result


def get_top_10_items(item_statistics):
    """
    Select the top 10 items ranked by purchase count.

    Parameters
    ----------
    item_statistics : pandas.DataFrame
        Item-level behavior statistics.

    Returns
    -------
    pandas.DataFrame
        Top 10 items ranked by buy_count in descending order.
    """

    return (
        item_statistics
        .sort_values("buy_count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


# ============================================================
# 4. Category Statistics
# ============================================================

def calculate_category_statistics(df):
    """
    Calculate behavior and purchase statistics for each category.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Category-level statistics containing total behavior count,
        purchase count, and purchase percentage.
    """

    result = (
        df.groupby("category_id")
        .agg(
            behavior_count=("category_id", "size"),
            buy_count=(
                "behavior_name",
                lambda x: (x == "buy").sum()
            )
        )
        .reset_index()
    )

    total_buy_count = result["buy_count"].sum()

    result["buy_percentage"] = (
        result["buy_count"] / total_buy_count * 100
    ).round(2)

    return result[
        [
            "category_id",
            "behavior_count",
            "buy_count",
            "buy_percentage"
        ]
    ]


def get_top_10_categories(category_statistics):
    """
    Select the top 10 categories ranked by purchase count.

    Parameters
    ----------
    category_statistics : pandas.DataFrame
        Category-level behavior statistics.

    Returns
    -------
    pandas.DataFrame
        Top 10 categories ranked by buy_count in descending order.
    """

    return (
        category_statistics
        .sort_values("buy_count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


# ============================================================
# 5. Daily Behavior Distribution
# ============================================================

def calculate_daily_behavior(df):
    """
    Calculate daily behavior distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Daily counts of PV, favorite, cart, and purchase behaviors.
    """

    result = (
        df.groupby(["behavior_date", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for behavior in BEHAVIOR_TYPES:
        if behavior not in result.columns:
            result[behavior] = 0

    result = result.rename(columns={
        "pv": "pv_count",
        "fav": "fav_count",
        "cart": "cart_count",
        "buy": "buy_count"
    })

    result = result[
        [
            "behavior_date",
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count"
        ]
    ]

    return result.sort_values(
        "behavior_date"
    ).reset_index(drop=True)


# ============================================================
# 6. Hourly Behavior Distribution
# ============================================================

def calculate_hourly_behavior(df):
    """
    Calculate behavior distribution by hour of day.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Hourly counts of PV, favorite, cart, and purchase behaviors.
    """

    result = (
        df.groupby(["behavior_hour", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for behavior in BEHAVIOR_TYPES:
        if behavior not in result.columns:
            result[behavior] = 0

    result = result.rename(columns={
        "pv": "pv_count",
        "fav": "fav_count",
        "cart": "cart_count",
        "buy": "buy_count"
    })

    result = result[
        [
            "behavior_hour",
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count"
        ]
    ]

    return result.sort_values(
        "behavior_hour"
    ).reset_index(drop=True)


# ============================================================
# 7. Descriptive Conversion Funnel
# ============================================================

def calculate_descriptive_funnel(df):
    """
    Construct a descriptive conversion funnel based on
    the number of recorded behaviors.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned user behavior dataset.

    Returns
    -------
    pandas.DataFrame
        Funnel stages and their behavior counts relative
        to the total PV count.
    """

    behavior_counts = df["behavior_name"].value_counts()

    pv_count = behavior_counts.get("pv", 0)
    fav_count = behavior_counts.get("fav", 0)
    cart_count = behavior_counts.get("cart", 0)
    buy_count = behavior_counts.get("buy", 0)

    result = pd.DataFrame({
        "stage": [
            "PV",
            "Favorite",
            "Cart",
            "Purchase"
        ],
        "behavior_count": [
            pv_count,
            fav_count,
            cart_count,
            buy_count
        ]
    })

    if pv_count > 0:
        result["relative_to_pv_percentage"] = (
            result["behavior_count"] / pv_count * 100
        ).round(2)
    else:
        result["relative_to_pv_percentage"] = 0.0

    return result


# ============================================================
# Main EDA Pipeline
# ============================================================

def main():
    """
    Run the complete EDA pipeline and save all statistical results.
    """

    print("Loading data...")

    df = load_data(INPUT_FILE)

    print("Data loaded successfully.")
    print("Number of rows:", len(df))
    print("Number of users:", df["user_id"].nunique())

    # 1. Behavior distribution
    behavior_distribution = calculate_behavior_distribution(df)
    behavior_distribution.to_csv(
        f"{OUTPUT_DIR}\\behavior_distribution.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2. User purchase summary
    user_purchase_summary = calculate_user_purchase_summary(df)
    user_purchase_summary.to_csv(
        f"{OUTPUT_DIR}\\user_purchase_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 3. Item statistics
    item_statistics = calculate_item_statistics(df)
    item_statistics.to_csv(
        f"{OUTPUT_DIR}\\item_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top_10_item = get_top_10_items(item_statistics)
    top_10_item.to_csv(
        f"{OUTPUT_DIR}\\top_10_item.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 4. Category statistics
    category_statistics = calculate_category_statistics(df)
    category_statistics.to_csv(
        f"{OUTPUT_DIR}\\category_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top_10_category = get_top_10_categories(
        category_statistics
    )
    top_10_category.to_csv(
        f"{OUTPUT_DIR}\\top_10_category.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 5. Daily behavior
    daily_behavior = calculate_daily_behavior(df)
    daily_behavior.to_csv(
        f"{OUTPUT_DIR}\\daily_behavior.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Hourly behavior
    hourly_behavior = calculate_hourly_behavior(df)
    hourly_behavior.to_csv(
        f"{OUTPUT_DIR}\\hourly_behavior.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 7. Descriptive funnel
    descriptive_funnel = calculate_descriptive_funnel(df)
    descriptive_funnel.to_csv(
        f"{OUTPUT_DIR}\\descriptive_funnel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nEDA completed successfully.")

    print("\nBehavior Distribution:")
    print(behavior_distribution.to_string(index=False))

    print("\nUser Purchase Summary:")
    print(user_purchase_summary.to_string(index=False))

    print("\nTop 10 Items:")
    print(top_10_item.to_string(index=False))

    print("\nTop 10 Categories:")
    print(top_10_category.to_string(index=False))

    print("\nDescriptive Funnel:")
    print(descriptive_funnel.to_string(index=False))


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
