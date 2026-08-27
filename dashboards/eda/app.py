import pandas as pd
import plotly.express as px
import plotly.express as px
import streamlit as st

from data_loader import EDA_OUTPUT_DIR, load_eda_outputs


st.set_page_config(
    page_title="Taobao User Behavior Intelligence",
    page_icon="📊",
    layout="wide",
)







# ---------------------------------------------------------------------------
# Dashboard chart card styling
# ---------------------------------------------------------------------------

st.html(
    """
<style>
/* ---------- Chart modules ---------- */

div[data-testid="stPlotlyChart"] {
    position: relative;

    margin-top: 1.15rem;
    margin-bottom: 1.2rem;

    padding: 0.65rem 0.85rem 0.85rem 0.85rem;

    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.065);

    background:
        radial-gradient(
            circle at 100% 0%,
            rgba(59,130,246,0.055),
            transparent 32%
        ),
        linear-gradient(
            145deg,
            rgba(255,255,255,0.028),
            rgba(255,255,255,0.012)
        );

    box-shadow:
        0 18px 45px rgba(0,0,0,0.13),
        inset 0 1px 0 rgba(255,255,255,0.025);

    overflow: hidden;
}

/* subtle accent line */

div[data-testid="stPlotlyChart"]::before {
    content: "";

    position: absolute;
    top: 0;
    left: 24px;

    width: 46px;
    height: 2px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #818cf8
        );

    box-shadow:
        0 0 12px rgba(96,165,250,0.32);

    z-index: 2;
}

/* remove visual collision with Streamlit chart wrapper */

div[data-testid="stPlotlyChart"] > div {
    border-radius: 14px;
}

/* Plotly canvas should inherit dashboard card background */

.js-plotly-plot,
.js-plotly-plot .plot-container,
.js-plotly-plot .svg-container {
    border-radius: 14px;
}

/* ---------- Analysis section headings ---------- */

div[data-testid="stMarkdownContainer"] h3 {
    margin-top: 0.15rem;
    margin-bottom: 0.15rem;

    color: #f1f5f9;

    font-size: 1.42rem;
    font-weight: 720;

    letter-spacing: -0.035em;
}

/* ---------- Plot captions ---------- */

div[data-testid="stCaptionContainer"] {
    color: #718096;
}

/* ---------- Section divider ---------- */

hr {
    margin-top: 3.15rem !important;
    margin-bottom: 2.45rem !important;

    border: none !important;
    height: 1px !important;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(148,163,184,0.13) 12%,
            rgba(148,163,184,0.13) 88%,
            transparent
        ) !important;
}
</style>
    """
)
# ---------------------------------------------------------------------------
# Dashboard visual system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 1440px;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(72, 149, 239, 0.10),
                transparent 35%
            ),
            linear-gradient(
                180deg,
                #0a0f1a 0%,
                #0d1422 55%,
                #0a101b 100%
            );
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }

    .sidebar-brand {
        margin: 0.25rem 0 1.8rem 0;
        padding: 1rem 1rem 1.05rem 1rem;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                rgba(77, 163, 255, 0.14),
                rgba(118, 91, 255, 0.05)
            );
        border: 1px solid rgba(111, 177, 255, 0.18);
        box-shadow:
            0 12px 30px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        color: #dbeafe;
        background:
            linear-gradient(
                135deg,
                rgba(59,130,246,0.85),
                rgba(99,102,241,0.85)
            );
        box-shadow: 0 8px 24px rgba(59,130,246,0.22);
    }

    .brand-title {
        color: #f8fafc;
        font-size: 0.88rem;
        font-weight: 750;
        line-height: 1.25;
        letter-spacing: 0.02em;
    }

    .brand-subtitle {
        margin-top: 0.35rem;
        color: #718096;
        font-size: 0.70rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .nav-section-title {
        margin: 1.15rem 0 0.55rem 0.35rem;
        color: #64748b;
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .stage-card {
        margin-bottom: 0.65rem;
        padding: 0.70rem 0.80rem;
        border-radius: 12px;
        background: rgba(59,130,246,0.07);
        border: 1px solid rgba(96,165,250,0.13);
    }

    .stage-number {
        display: inline-block;
        margin-right: 0.45rem;
        color: #60a5fa;
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .stage-name {
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.70rem;
        width: 100%;
        box-sizing: border-box;
        margin: 0.20rem 0;
        padding: 0.62rem 0.72rem;
        border-radius: 10px;
        border: 1px solid transparent;

        color: #94a3b8 !important;
        text-decoration: none !important;

        background: transparent;

        transition:
            background 150ms ease,
            border-color 150ms ease,
            transform 150ms ease,
            color 150ms ease;
    }

    .nav-item:hover {
        color: #f8fafc !important;
        background:
            linear-gradient(
                90deg,
                rgba(59,130,246,0.14),
                rgba(59,130,246,0.04)
            );
        border-color: rgba(96,165,250,0.15);
        transform: translateX(3px);
    }

    .nav-index {
        min-width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;

        border-radius: 8px;

        color: #93c5fd;
        background: rgba(59,130,246,0.10);

        font-size: 0.64rem;
        font-weight: 750;
        letter-spacing: 0.03em;
    }

    .nav-text {
        font-size: 0.76rem;
        font-weight: 560;
    }

    .future-stage {
        margin: 0.28rem 0;
        padding: 0.62rem 0.72rem;

        color: #475569;

        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.035);
        background: rgba(255,255,255,0.015);

        font-size: 0.70rem;
    }

    .future-stage strong {
        color: #64748b;
        font-weight: 650;
    }

    /* ---------- Hero ---------- */

    .dashboard-hero {
        margin-bottom: 1.6rem;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;

        margin-bottom: 0.65rem;
        padding: 0.34rem 0.62rem;

        border-radius: 999px;
        border: 1px solid rgba(96,165,250,0.20);

        color: #93c5fd;
        background: rgba(59,130,246,0.08);

        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;

        color: #f8fafc;

        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 780;
        line-height: 1.05;

        letter-spacing: -0.045em;
    }

    .hero-title-accent {
        background:
            linear-gradient(
                90deg,
                #f8fafc 0%,
                #bfdbfe 48%,
                #93c5fd 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 760px;
        margin-top: 0.85rem;

        color: #7f8da3;

        font-size: 0.92rem;
        line-height: 1.65;
    }

    .stage-pill {
        display: inline-flex;
        align-items: center;

        margin-top: 1rem;
        padding: 0.42rem 0.70rem;

        color: #cbd5e1;
        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 9px;

        font-size: 0.71rem;
        font-weight: 600;
    }

    .stage-dot {
        width: 7px;
        height: 7px;

        margin-right: 0.5rem;

        border-radius: 50%;
        background: #22c55e;

        box-shadow: 0 0 12px rgba(34,197,94,0.6);
    }

    /* ---------- Metric cards ---------- */

    div[data-testid="stMetric"] {
        padding: 1rem 1.05rem;

        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.065);

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.045),
                rgba(255,255,255,0.018)
            );

        box-shadow:
            0 14px 35px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div[data-testid="stMetricLabel"] {
        color: #8492a6;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        letter-spacing: -0.035em;
    }

    /* ---------- Dataframe ---------- */

    div[data-testid="stDataFrame"] {
        border-radius: 13px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.055);
    }

    /* ---------- Separators ---------- */

    hr {
        margin: 2.7rem 0 !important;
        border-color: rgba(255,255,255,0.065) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


current_section = st.query_params.get("section", "overview")

NAV_ITEMS = [
    ("overview", "⌂", "Overview", "overview"),
    ("behavior", "01", "Behavior Mix", "behavior-distribution"),
    ("users", "02", "User Insights", "user-purchase-overview"),
    ("items", "03", "Top Items", "top-10-purchased-items"),
    ("categories", "04", "Top Categories", "top-10-purchased-categories"),
    ("daily", "05", "Daily Trend", "daily-behavior-trend"),
    ("hourly", "06", "Hourly Trend", "hourly-behavior-trend"),
    ("funnel", "07", "Conversion Funnel", "conversion-funnel"),
]


def build_nav_item(
    section: str,
    index: str,
    label: str,
    anchor: str,
) -> str:
    """Build one persistent sidebar navigation item."""
    active_class = " active" if current_section == section else ""

    return f"""
    <a
        class="nav-item{active_class}"
        href="?section={section}#{anchor}"
        target="_self"
    >
        <span class="nav-index">{index}</span>
        <span class="nav-text">{label}</span>
    </a>
    """


navigation_html = "".join(
    build_nav_item(section, index, label, anchor)
    for section, index, label, anchor in NAV_ITEMS
)

with st.sidebar:
    st.html(f"""
    <div class="sidebar-brand">
        <div class="brand-mark">TB</div>

        <div class="brand-title">
            User Behavior<br>
            Intelligence
        </div>

        <div class="brand-subtitle">
            Analytics Workspace
        </div>
    </div>

    <div class="nav-section-title">
        Workspace
    </div>

    <div class="stage-card">
        <span class="stage-number">01</span>
        <span class="stage-name">
            Exploratory Analysis
        </span>
    </div>

    {navigation_html}

    <div class="nav-section-title">
        Roadmap
    </div>

    <div class="future-stage">
        <strong>02</strong>
        &nbsp; Feature Engineering
    </div>

    <div class="future-stage">
        <strong>03</strong>
        &nbsp; Modeling
    </div>

    <div class="future-stage">
        <strong>04</strong>
        &nbsp; Evaluation & Prediction
    </div>
    """)


st.html(
    """
<div id="overview"></div>

<div class="dashboard-hero">
    <div class="hero-eyebrow">
        TAOBAO ANALYTICS PLATFORM
    </div>

    <h1 class="hero-title">
        User Behavior
        <span class="hero-title-accent">Intelligence</span>
    </h1>

    <div class="hero-description">
        An analytics and prediction workspace for understanding user behavior,
        conversion patterns, temporal dynamics, feature engineering and
        downstream modeling.
    </div>

    <div class="stage-pill">
        <span class="stage-dot"></span>
        Stage 1 · Exploratory Data Analysis
    </div>
</div>
    """
)




try:
    eda = load_eda_outputs()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

behavior_distribution = eda["behavior_distribution"]
user_purchase_summary = eda["user_purchase_summary"]

total_behavior_count = int(behavior_distribution["behavior_count"].sum())
purchase_count = int(user_purchase_summary.iloc[0]["purchase_count"])
purchase_users = int(user_purchase_summary.iloc[0]["purchase_users"])

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Total Behaviors",
    f"{total_behavior_count:,}",
)
metric_2.metric(
    "Purchase Behaviors",
    f"{purchase_count:,}",
)
metric_3.metric(
    "Purchase Users",
    f"{purchase_users:,}",
)

st.html(
    f"""
    <style>
    .data-status-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;

        margin: 1rem 0 0.8rem 0;
        padding: 0.85rem 1rem;

        border-radius: 13px;
        border: 1px solid rgba(34, 197, 94, 0.16);

        background:
            linear-gradient(
                90deg,
                rgba(34, 197, 94, 0.085),
                rgba(34, 197, 94, 0.025)
            );

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.025);
    }}

    .data-status-left {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }}

    .data-status-indicator {{
        width: 9px;
        height: 9px;

        flex: 0 0 auto;

        border-radius: 50%;
        background: #22c55e;

        box-shadow:
            0 0 0 4px rgba(34,197,94,0.08),
            0 0 14px rgba(34,197,94,0.48);
    }}

    .data-status-title {{
        color: #e2e8f0;

        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.015em;
    }}

    .data-status-meta {{
        color: #64748b;

        font-size: 0.70rem;
        margin-top: 0.12rem;
    }}

    .data-status-badge {{
        flex: 0 0 auto;

        padding: 0.34rem 0.58rem;

        border-radius: 8px;
        border: 1px solid rgba(96,165,250,0.12);

        color: #93c5fd;
        background: rgba(59,130,246,0.055);

        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    @media (max-width: 700px) {{
        .data-status-bar {{
            align-items: flex-start;
            flex-direction: column;
        }}
    }}
    </style>

    <div class="data-status-bar">
        <div class="data-status-left">
            <span class="data-status-indicator"></span>

            <div>
                <div class="data-status-title">
                    Data Ready
                </div>

                <div class="data-status-meta">
                    {len(eda)} dashboard datasets loaded
                    · Issue #4 EDA aggregates
                </div>
            </div>
        </div>

        <div class="data-status-badge">
            Stage 1 · Live
        </div>
    </div>
    """
)

status_rows = [
    {
        "Dataset": name,
        "Rows": len(dataframe),
        "Columns": len(dataframe.columns),
    }
    for name, dataframe in eda.items()
]

with st.expander(
    "Data & System Status",
    expanded=False,
):
    detail_col_1, detail_col_2, detail_col_3 = st.columns(3)

    detail_col_1.metric(
        "Loaded Datasets",
        len(eda),
    )

    detail_col_2.metric(
        "Data Layer",
        "Aggregated",
    )

    detail_col_3.metric(
        "Source Stage",
        "Issue #4",
    )

    st.caption(
        "The dashboard consumes aggregated Stage 1 EDA outputs "
        "instead of recomputing statistics from the full clean dataset."
    )

    st.dataframe(
        status_rows,
        width="stretch",
        hide_index=True,
    )

    st.caption("Local EDA output directory")
    st.code(
        EDA_OUTPUT_DIR.as_posix(),
        language=None,
    )



st.divider()

st.markdown('<div id="behavior-distribution"></div>', unsafe_allow_html=True)
st.subheader("Behavior Distribution")
st.caption("Distribution of PV, Favorite, Cart, and Purchase behaviors.")

behavior_chart = behavior_distribution.copy()

behavior_chart["behavior_name"] = (
    behavior_chart["behavior_name"]
    .str.upper()
)

behavior_chart["share_label"] = (
    behavior_chart["percentage"]
    .map(lambda value: f"{value:.2f}%")
)

behavior_figure = px.bar(
    behavior_chart,
    x="behavior_name",
    y="behavior_count",
    text="share_label",
    category_orders={
        "behavior_name": ["PV", "FAV", "CART", "BUY"]
    },
    labels={
        "behavior_name": "Behavior Type",
        "behavior_count": "Behavior Count",
    },
)

behavior_figure.update_traces(
    textposition="outside",
    cliponaxis=False,
)

behavior_figure.update_layout(
    xaxis_title="Behavior Type",
    yaxis_title="Behavior Count",
    showlegend=False,
)

st.plotly_chart(
    behavior_figure,
    width="stretch", config={"displayModeBar": False},
)

st.divider()

st.markdown('<div id="user-purchase-overview"></div>', unsafe_allow_html=True)
st.subheader("User Purchase Overview")
st.caption("User segments based on purchase behavior.")

summary = user_purchase_summary.iloc[0]

purchase_users = int(summary["purchase_users"])
non_purchase_users = int(summary["non_purchase_users"])
repeat_purchase_users = int(summary["repeat_purchase_users"])
single_purchase_users = purchase_users - repeat_purchase_users

user_segments = pd.DataFrame(
    {
        "User Segment": [
            "No Purchase",
            "Single Purchase",
            "Repeat Purchase",
        ],
        "User Count": [
            non_purchase_users,
            single_purchase_users,
            repeat_purchase_users,
        ],
    }
)

user_segments["share"] = (
    user_segments["User Count"]
    / user_segments["User Count"].sum()
    * 100
)

user_segments["label"] = user_segments.apply(
    lambda row: (
        f'{int(row["User Count"]):,}'
        f' ({row["share"]:.2f}%)'
    ),
    axis=1,
)

user_figure = px.bar(
    user_segments,
    x="User Segment",
    y="User Count",
    text="label",
    category_orders={
        "User Segment": [
            "No Purchase",
            "Single Purchase",
            "Repeat Purchase",
        ]
    },
)

user_figure.update_traces(
    textposition="outside",
    cliponaxis=False,
)

user_figure.update_layout(
    xaxis_title="User Segment",
    yaxis_title="User Count",
    showlegend=False,
)

st.plotly_chart(
    user_figure,
    width="stretch", config={"displayModeBar": False},
)



st.divider()

st.markdown('<div id="top-10-purchased-items"></div>', unsafe_allow_html=True)
st.subheader("Top 10 Purchased Items")
st.caption("Top 10 items ranked by purchase count from Issue #4 EDA output.")

top_items = eda["top_10_item"].copy()

top_items["item_id"] = top_items["item_id"].astype(str)

top_items = top_items.sort_values(
    "buy_count",
    ascending=True,
)

item_figure = px.bar(
    top_items,
    x="buy_count",
    y="item_id",
    orientation="h",
    text="buy_count",
    labels={
        "item_id": "Item ID",
        "buy_count": "Purchase Count",
    },
)

item_figure.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
    cliponaxis=False,
)

item_figure.update_layout(
    xaxis_title="Purchase Count",
    yaxis_title="Item ID",
    showlegend=False,
)

item_figure.update_yaxes(
    type="category",
    categoryorder="array",
    categoryarray=top_items["item_id"].tolist(),
)

st.plotly_chart(
    item_figure,
    width="stretch", config={"displayModeBar": False},
)


st.divider()

st.markdown('<div id="top-10-purchased-categories"></div>', unsafe_allow_html=True)
st.subheader("Top 10 Purchased Categories")
st.caption(
    "Top 10 categories ranked by purchase count from Issue #4 EDA output."
)

top_categories = eda["top_10_category"].copy()

top_categories["category_id"] = (
    top_categories["category_id"]
    .astype(str)
)

top_categories = top_categories.sort_values(
    "buy_count",
    ascending=True,
)

top_categories["label"] = top_categories.apply(
    lambda row: (
        f'{int(row["buy_count"]):,}'
        f' ({row["buy_percentage"]:.2f}%)'
    ),
    axis=1,
)

category_figure = px.bar(
    top_categories,
    x="buy_count",
    y="category_id",
    orientation="h",
    text="label",
    labels={
        "category_id": "Category ID",
        "buy_count": "Purchase Count",
    },
)

category_figure.update_traces(
    textposition="outside",
    cliponaxis=False,
)

category_figure.update_yaxes(
    type="category",
    categoryorder="array",
    categoryarray=top_categories["category_id"].tolist(),
)

category_figure.update_layout(
    xaxis_title="Purchase Count",
    yaxis_title="Category ID",
    showlegend=False,
)

st.plotly_chart(
    category_figure,
    width="stretch", config={"displayModeBar": False},
)

st.divider()

st.markdown('<div id="daily-behavior-trend"></div>', unsafe_allow_html=True)
st.subheader("Daily Behavior Trend")
st.caption("Daily PV, Favorite, Cart, and Purchase behavior trends.")

behavior_focus_options = [
    "All Behaviors",
    "PV",
    "FAV",
    "CART",
    "BUY",
]

daily_focus = st.segmented_control(
    "Daily behavior focus",
    options=behavior_focus_options,
    default="All Behaviors",
    key="daily_behavior_focus",
    label_visibility="collapsed",
)

daily_behavior = eda["daily_behavior"].copy()

daily_behavior["behavior_date"] = pd.to_datetime(
    daily_behavior["behavior_date"]
)

daily_long = daily_behavior.melt(
    id_vars="behavior_date",
    value_vars=[
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
    ],
    var_name="behavior_type",
    value_name="behavior_count",
)

daily_long["behavior_type"] = daily_long["behavior_type"].map(
    {
        "pv_count": "PV",
        "fav_count": "FAV",
        "cart_count": "CART",
        "buy_count": "BUY",
    }
)

daily_plot = (
    daily_long
    if daily_focus == "All Behaviors"
    else daily_long[
        daily_long["behavior_type"] == daily_focus
    ]
)

behavior_colors = {
    "PV": "#7DD3FC",
    "FAV": "#A78BFA",
    "CART": "#F59E0B",
    "BUY": "#22C55E",
}

daily_figure = px.line(
    daily_plot,
    x="behavior_date",
    y="behavior_count",
    color="behavior_type",
    markers=True,
    color_discrete_map=behavior_colors,
    labels={
        "behavior_date": "Date",
        "behavior_count": "Behavior Count",
        "behavior_type": "Behavior Type",
    },
)

daily_figure.update_layout(
    xaxis_title="Date",
    yaxis_title="Behavior Count",
)

if daily_focus == "All Behaviors":
    daily_figure.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "title": {"text": ""},
        }
    )
else:
    daily_figure.update_layout(showlegend=False)

st.plotly_chart(
    daily_figure,
    width="stretch", config={"displayModeBar": False},
)


st.divider()

st.markdown('<div id="hourly-behavior-trend"></div>', unsafe_allow_html=True)
st.subheader("Hourly Behavior Trend")
st.caption("Behavior distribution across the 24 hours of a day.")

hourly_focus = st.segmented_control(
    "Hourly behavior focus",
    options=behavior_focus_options,
    default="All Behaviors",
    key="hourly_behavior_focus",
    label_visibility="collapsed",
)

hourly_behavior = eda["hourly_behavior"].copy()

hourly_long = hourly_behavior.melt(
    id_vars="behavior_hour",
    value_vars=[
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
    ],
    var_name="behavior_type",
    value_name="behavior_count",
)

hourly_long["behavior_type"] = hourly_long["behavior_type"].map(
    {
        "pv_count": "PV",
        "fav_count": "FAV",
        "cart_count": "CART",
        "buy_count": "BUY",
    }
)

hourly_plot = (
    hourly_long
    if hourly_focus == "All Behaviors"
    else hourly_long[
        hourly_long["behavior_type"] == hourly_focus
    ]
)

hourly_figure = px.line(
    hourly_plot,
    x="behavior_hour",
    y="behavior_count",
    color="behavior_type",
    markers=True,
    color_discrete_map=behavior_colors,
    labels={
        "behavior_hour": "Hour",
        "behavior_count": "Behavior Count",
        "behavior_type": "Behavior Type",
    },
)

hourly_figure.update_xaxes(
    tickmode="array",
    tickvals=list(range(24)),
    ticktext=[str(hour) for hour in range(24)],
    range=[-0.5, 24.0],
    automargin=True,
)

hourly_figure.update_layout(
    xaxis_title="Hour",
    yaxis_title="Behavior Count",
)

if hourly_focus == "All Behaviors":
    hourly_figure.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "title": {"text": ""},
        }
    )
else:
    hourly_figure.update_layout(showlegend=False)

st.plotly_chart(
    hourly_figure,
    width="stretch", config={"displayModeBar": False},
)

st.divider()

st.markdown('<div id="conversion-funnel"></div>', unsafe_allow_html=True)
st.subheader("Descriptive Conversion Funnel")
st.caption(
    "Behavior counts and relative percentages compared with PV."
)

funnel = eda["descriptive_funnel"].copy()

funnel["stage"] = pd.Categorical(
    funnel["stage"],
    categories=[
        "PV",
        "Favorite",
        "Cart",
        "Purchase",
    ],
    ordered=True,
)

funnel = funnel.sort_values("stage")

funnel["label"] = funnel.apply(
    lambda row: (
        f'{int(row["behavior_count"]):,}'
        f' ({row["relative_to_pv_percentage"]:.2f}%)'
    ),
    axis=1,
)

funnel_figure = px.funnel(
    funnel,
    y="stage",
    x="behavior_count",
    text="label",
    labels={
        "stage": "Stage",
        "behavior_count": "Behavior Count",
    },
)

funnel_figure.update_traces(
    textposition="inside",
    textinfo="text",
)

funnel_figure.update_layout(
    xaxis_title="Behavior Count",
    yaxis_title="Behavior Stage",
)

st.plotly_chart(
    funnel_figure,
    width="stretch", config={"displayModeBar": False},
)

st.caption(
    "Note: percentages are calculated relative to PV and do not represent "
    "strict sequential user conversion rates."
)










st.html(
    """
<style>
/* ---------- Active navigation state ---------- */

body:has(#overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#overview"],

body:has(#behavior-distribution:target)
section[data-testid="stSidebar"]
a.nav-item[href="#behavior-distribution"],

body:has(#user-purchase-overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#user-purchase-overview"],

body:has(#top-10-purchased-items:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-items"],

body:has(#top-10-purchased-categories:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-categories"],

body:has(#daily-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#daily-behavior-trend"],

body:has(#hourly-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#hourly-behavior-trend"],

body:has(#conversion-funnel:target)
section[data-testid="stSidebar"]
a.nav-item[href="#conversion-funnel"] {
    position: relative;

    color: #f8fafc !important;

    background:
        linear-gradient(
            90deg,
            rgba(59,130,246,0.18),
            rgba(59,130,246,0.055)
        );

    border-color: rgba(96,165,250,0.22);

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 8px 24px rgba(0,0,0,0.10);
}

body:has(#overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#overview"]::before,

body:has(#behavior-distribution:target)
section[data-testid="stSidebar"]
a.nav-item[href="#behavior-distribution"]::before,

body:has(#user-purchase-overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#user-purchase-overview"]::before,

body:has(#top-10-purchased-items:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-items"]::before,

body:has(#top-10-purchased-categories:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-categories"]::before,

body:has(#daily-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#daily-behavior-trend"]::before,

body:has(#hourly-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#hourly-behavior-trend"]::before,

body:has(#conversion-funnel:target)
section[data-testid="stSidebar"]
a.nav-item[href="#conversion-funnel"]::before {
    content: "";

    position: absolute;
    left: -1px;
    top: 20%;
    bottom: 20%;

    width: 3px;

    border-radius: 0 999px 999px 0;

    background:
        linear-gradient(
            180deg,
            #60a5fa,
            #818cf8
        );

    box-shadow:
        0 0 12px rgba(96,165,250,0.65);
}

body:has(#overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#overview"] .nav-index,

body:has(#behavior-distribution:target)
section[data-testid="stSidebar"]
a.nav-item[href="#behavior-distribution"] .nav-index,

body:has(#user-purchase-overview:target)
section[data-testid="stSidebar"]
a.nav-item[href="#user-purchase-overview"] .nav-index,

body:has(#top-10-purchased-items:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-items"] .nav-index,

body:has(#top-10-purchased-categories:target)
section[data-testid="stSidebar"]
a.nav-item[href="#top-10-purchased-categories"] .nav-index,

body:has(#daily-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#daily-behavior-trend"] .nav-index,

body:has(#hourly-behavior-trend:target)
section[data-testid="stSidebar"]
a.nav-item[href="#hourly-behavior-trend"] .nav-index,

body:has(#conversion-funnel:target)
section[data-testid="stSidebar"]
a.nav-item[href="#conversion-funnel"] .nav-index {
    color: #dbeafe;

    background:
        linear-gradient(
            135deg,
            rgba(59,130,246,0.30),
            rgba(99,102,241,0.22)
        );

    box-shadow:
        0 0 14px rgba(59,130,246,0.12);
}
</style>
    """
)

# Active navigation state


st.html(
    """
<style>
/* ---------- Chart card refinement ---------- */

div[data-testid="stPlotlyChart"] {
    padding: 0.35rem 0.45rem 0.5rem 0.45rem;

    border-color: rgba(148,163,184,0.075);

    background:
        radial-gradient(
            circle at 100% 0%,
            rgba(59,130,246,0.035),
            transparent 30%
        ),
        rgba(255,255,255,0.012);

    box-shadow:
        0 16px 38px rgba(0,0,0,0.10),
        inset 0 1px 0 rgba(255,255,255,0.02);
}

div[data-testid="stPlotlyChart"]::before {
    left: 20px;
    width: 42px;
    opacity: 0.82;
}

/* Give analysis sections slightly more breathing room */

div[data-testid="stPlotlyChart"] + div {
    margin-top: 0;
}
</style>
    """
)



st.html(
    """
<style>
/* ---------- Persistent active navigation ---------- */

.nav-item.active {
    position: relative;

    color: #f8fafc !important;

    background:
        linear-gradient(
            90deg,
            rgba(59,130,246,0.20),
            rgba(59,130,246,0.055)
        );

    border-color: rgba(96,165,250,0.24);

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.035),
        0 10px 26px rgba(0,0,0,0.11);
}

.nav-item.active::before {
    content: "";

    position: absolute;

    left: -1px;
    top: 20%;
    bottom: 20%;

    width: 3px;

    border-radius: 0 999px 999px 0;

    background:
        linear-gradient(
            180deg,
            #60a5fa,
            #818cf8
        );

    box-shadow:
        0 0 13px rgba(96,165,250,0.72);
}

.nav-item.active .nav-index {
    color: #eff6ff;

    background:
        linear-gradient(
            135deg,
            rgba(59,130,246,0.38),
            rgba(99,102,241,0.27)
        );

    box-shadow:
        0 0 15px rgba(59,130,246,0.16);
}

.nav-item.active .nav-text {
    color: #f8fafc;
    font-weight: 650;
}
</style>
    """
)
