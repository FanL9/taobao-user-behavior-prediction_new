from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_OUTPUT_DIR = REPO_ROOT / "data" / "interim"

# Lightweight EDA outputs consumed directly by the dashboard.
EDA_FILES = {
    "behavior_distribution": "behavior_distribution.csv",
    "user_purchase_summary": "user_purchase_summary.csv",
    "top_10_item": "top_10_item.csv",
    "category_statistics": "category_statistics.csv",
    "top_10_category": "top_10_category.csv",
    "daily_behavior": "daily_behavior.csv",
    "hourly_behavior": "hourly_behavior.csv",
    "descriptive_funnel": "descriptive_funnel.csv",
}

# Large detailed output retained for downstream analysis, but intentionally
# not loaded during normal dashboard startup.
LARGE_EDA_FILES = {
    "item_statistics": "item_statistics.csv",
}


@st.cache_data(show_spinner=False)
def load_eda_outputs(
    output_dir: str | Path = EDA_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Load lightweight Stage 1 EDA aggregate outputs for the dashboard."""
    output_path = Path(output_dir)

    required_files = {
        **EDA_FILES,
        **LARGE_EDA_FILES,
    }

    missing_files = [
        filename
        for filename in required_files.values()
        if not (output_path / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing EDA output files: " + ", ".join(missing_files)
        )

    return {
        name: pd.read_csv(output_path / filename)
        for name, filename in EDA_FILES.items()
    }
