"""Read-only Stage 3 data-preparation summary for the existing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3_REPORT_ROOT = REPO_ROOT / "reports" / "stage3"
REQUIRED_REPORTS = {
    "labels": "label_statistics.json",
    "splits": "dataset_split_statistics.json",
    "preprocessing": "preprocessing_statistics.json",
    "selection": "feature_selection_report.json",
    "imbalance": "class_imbalance_report.json",
}


def read_stage3_dashboard_outputs(report_root: str | Path = STAGE3_REPORT_ROOT) -> dict[str, Any]:
    """Load precomputed Stage 3 reports without processing raw data."""

    root = Path(report_root).expanduser().resolve()
    paths = {name: root / filename for name, filename in REQUIRED_REPORTS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 3 dashboard reports: " + ", ".join(missing))
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}


@st.cache_data(show_spinner=False)
def load_stage3_dashboard_outputs(report_root: str = str(STAGE3_REPORT_ROOT)) -> dict[str, Any]:
    """Cache compact Stage 3 report payloads for dashboard rendering."""

    return read_stage3_dashboard_outputs(report_root)


def _split_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert stored split statistics into a dashboard table."""

    rows = []
    for split, values in payload["statistics"]["by_split"].items():
        rows.append({
            "Split": split.title(),
            "Samples": int(values["sample_count"]),
            "Positives": int(values["positive_count"]),
            "Negatives": int(values["negative_count"]),
            "Positive rate": float(values["positive_ratio"]),
        })
    return pd.DataFrame(rows)


def _imbalance_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert training-only sampling statistics into a compact table."""

    rows = []
    for strategy, values in payload["training_strategy_statistics"].items():
        rows.append({
            "Training strategy": strategy,
            "Samples": int(values["sample_count"]),
            "Positives": int(values["positive_count"]),
            "Positive rate": float(values["positive_ratio"]),
        })
    return pd.DataFrame(rows)


def render_stage3_dashboard(report_root: str | Path = STAGE3_REPORT_ROOT) -> None:
    """Render the Stage 3 preparation section from persisted reports only."""

    st.divider()
    st.markdown('<div id="stage3-data-preparation"></div>', unsafe_allow_html=True)
    st.header("Stage 3 · Dataset Preparation")
    st.caption("Read-only summary of labels, time splits, preprocessing, feature selection and class-imbalance handling.")
    try:
        data = load_stage3_dashboard_outputs(str(report_root))
    except FileNotFoundError as error:
        st.warning(str(error))
        return

    label_statistics = data["labels"]["statistics"]["overall"]
    selection_statistics = data["selection"]["feature_statistics"]
    cards = st.columns(4)
    cards[0].metric("Labeled samples", f"{int(label_statistics['sample_count']):,}")
    cards[1].metric("Positive rate", f"{float(label_statistics['positive_ratio']):.4%}")
    cards[2].metric("Selected features", int(selection_statistics["selected_feature_count"]))
    cards[3].metric("Removed features", int(selection_statistics["removed_feature_count"]))

    st.subheader("Time-ordered dataset splits")
    split_frame = _split_frame(data["splits"])
    st.dataframe(
        split_frame,
        hide_index=True,
        use_container_width=True,
        column_config={"Positive rate": st.column_config.NumberColumn(format="%.4f%%")},
    )
    time_checks = data["splits"]["time_window_checks"]
    st.caption(
        f"Split order: {' → '.join(time_checks['dataset_order'])}. "
        f"Random split used: {time_checks['random_split_used']}. "
        f"Feature/label overlap rows: {time_checks['feature_and_label_window_overlap_rows']}."
    )

    st.subheader("Feature preparation and screening")
    feature_cards = st.columns(3)
    feature_cards[0].metric("Candidate features", int(selection_statistics["candidate_feature_count"]))
    feature_cards[1].metric("Final model features", int(selection_statistics["selected_feature_count"]))
    feature_cards[2].metric("Selection fitted on", "Train")
    leakage_check = data["selection"]["data_leakage_check"]
    st.caption(
        leakage_check["status"].replace("_", " ").title()
        + ": feature screening used training data only; validation/test rules were not refitted."
    )

    st.subheader("Training-only class-imbalance strategies")
    imbalance_frame = _imbalance_frame(data["imbalance"])
    st.dataframe(
        imbalance_frame,
        hide_index=True,
        use_container_width=True,
        column_config={"Positive rate": st.column_config.NumberColumn(format="%.4f%%")},
    )
    st.caption("SMOTE and undersampling are training-only; validation and test retain their original time distribution.")
