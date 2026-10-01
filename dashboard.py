import os
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# FORCE STREAMLIT LIGHT THEME CONFIG
# =========================================================
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write(
        """
[theme]
base="light"
primaryColor="#4f46e5"
backgroundColor="#c0c9ef"
secondaryBackgroundColor="#e0e7ff"
textColor="#1e1b4b"
"""
    )

st.set_page_config(
    page_title="Help Desk Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CLEAN AESTHETIC LAVENDER THEME & GLASSMORPHISM
# =========================================================
st.markdown(
    """
    <style>
    /* Main Gradient Lavender Background */
    .stApp {
        background: linear-gradient(135deg, #a8b3e3 0%, #c0c9ef 50%, #b2bcfa 100%);
        color: #1e1b4b;
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Top Navigation Header Fix */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Outer Container Padding */
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Main Dashboard Title */
    .main-title {
        font-size: 38px;
        font-weight: 850;
        color: #1e1b4b;
        letter-spacing: -0.8px;
        margin-top: 0px;
        margin-bottom: 4px;
    }

    /* Executive Summary Plain Text (No Tab / Card Behind) */
    .exec-summary-text {
        font-size: 13.5px;
        color: #312e81;
        line-height: 1.5;
        margin-bottom: 22px;
    }

    .exec-summary-label {
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #1e1b4b;
    }

    /* Section Headings */
    .sheet-heading {
        font-size: 13px;
        font-weight: 800;
        color: #312e81;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 5px;
        margin-bottom: 14px;
    }

    /* Metric Stat Grid */
    .stat-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 27, 75, 0.04);
    }

    .stat-label {
        font-size: 12px;
        font-weight: 800;
        color: #3730a3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .stat-value {
        font-size: 34px;
        font-weight: 850;
        color: #1e1b4b;
        line-height: 1.1;
    }

    .stat-sub {
        font-size: 12px;
        font-weight: 600;
        color: #4f46e5;
        margin-top: 4px;
    }

    /* Full-Height Vertical Divider Line Fix */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .v-divider-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 280px;
    }

    .v-divider {
        border-left: 1.5px solid rgba(255, 255, 255, 0.75);
        height: 100%;
        width: 1px;
    }

    /* Horizontal Divider Line */
    .h-line {
        border-bottom: 1.5px solid rgba(255, 255, 255, 0.75);
        margin-top: 20px;
        margin-bottom: 20px;
        width: 100%;
    }

    /* Streamlit DataFrame Override (Off-White & Lavender Theme) */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 14px !important;
        padding: 8px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05) !important;
        width: 100% !important;
    }

    div[data-testid="stDataFrame"] * {
        color: #1e1b4b !important;
        font-family: "Segoe UI", sans-serif !important;
    }

    /* Hide Default Sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    df_ = pd.read_csv("outputs/predictions.csv", parse_dates=["created_at"])
    return df_


df = load_data()
d_plot = df.copy()
d_plot["high_priority"] = d_plot["pred_priority"].isin(["High", "Critical"])
ORDER = {"pred_priority": ["Low", "Medium", "High", "Critical"]}

LAVENDER_PALETTE = ["#818cf8", "#6366f1", "#4f46e5", "#4338ca", "#3730a3"]


def build_lavender_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.25)",
        font=dict(color="#1e1b4b", size=10, family="Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.5)",
            showgrid=True,
            title_font=dict(color="#312e81", size=10, weight="bold"),
            tickfont=dict(color="#1e1b4b", size=9),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.5)",
            showgrid=True,
            title_font=dict(color="#312e81", size=10, weight="bold"),
            tickfont=dict(color="#1e1b4b", size=9),
        ),
        title=dict(
            font=dict(color="#1e1b4b", size=12, family="Segoe UI"),
            x=0,
            xanchor="left",
        ),
    )
    return fig


# Metrics Computation
tickets_count = len(d_plot)
high_critical_pct = (
    float(d_plot["high_priority"].mean() * 100) if len(d_plot) else 0.0
)
avg_resolution = (
    float(d_plot["resolution_hours"].mean())
    if (
        "resolution_hours" in d_plot.columns
        and d_plot["resolution_hours"].notna().any()
    )
    else 0.0
)
anomaly_count = (
    int(d_plot["is_anomaly"].sum()) if "is_anomaly" in d_plot.columns else 0
)


# =========================================================
# TITLE & PLAIN TEXT EXECUTIVE SUMMARY
# =========================================================
st.markdown(
    '<div class="main-title">Help Desk Analysis Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="exec-summary-text">
        <span class="exec-summary-label">Executive Summary </span>  
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# KPI METRIC GRID
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total Tickets</div>
            <div class="stat-value">{tickets_count:,}</div>
            <div class="stat-sub">Overall Volume</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">High / Critical Share</div>
            <div class="stat-value">{high_critical_pct:.1f}%</div>
            <div class="stat-sub">Urgent Priority</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Avg Days Open (h)</div>
            <div class="stat-value">{avg_resolution:.1f}</div>
            <div class="stat-sub">Mean Resolution Time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Flagged Anomalies</div>
            <div class="stat-value">{anomaly_count:,}</div>
            <div class="stat-sub">Autoencoder Flagged</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown('<div class="h-line"></div>', unsafe_allow_html=True)


# =========================================================
# ROW 1: 4 CHARTS WITH FULL-HEIGHT VERTICAL DIVIDERS
# =========================================================
st.markdown(
    '<div class="sheet-heading">Volume & Classification Insights</div>',
    unsafe_allow_html=True,
)

r1_col1, r1_v1, r1_col2, r1_v2, r1_col3, r1_v3, r1_col4 = st.columns(
    [0.23, 0.01, 0.23, 0.01, 0.23, 0.01, 0.23]
)

with r1_col1:
    trend = (
        d_plot.set_index("created_at")
        .resample("ME")
        .size()
        .reset_index(name="tickets")
    )
    fig_trend = px.line(
        trend,
        x="created_at",
        y="tickets",
        markers=True,
        title="Monthly Volume",
        color_discrete_sequence=["#4f46e5"],
    )
    st.plotly_chart(build_lavender_chart(fig_trend), use_container_width=True)

with r1_v1:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r1_col2:
    dept_counts = d_plot["department"].value_counts().reset_index()
    dept_counts.columns = ["department", "count"]
    fig_dept = px.bar(
        dept_counts,
        y="department",
        x="count",
        orientation="h",
        title="By Department",
        color_discrete_sequence=["#6366f1"],
    )
    st.plotly_chart(build_lavender_chart(fig_dept), use_container_width=True)

with r1_v2:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r1_col3:
    cat_counts = d_plot["pred_category"].value_counts().reset_index()
    cat_counts.columns = ["pred_category", "count"]
    fig_cat = px.bar(
        cat_counts,
        y="pred_category",
        x="count",
        orientation="h",
        title="By Category",
        color_discrete_sequence=["#4338ca"],
    )
    st.plotly_chart(build_lavender_chart(fig_cat), use_container_width=True)

with r1_v3:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r1_col4:
    fig_pri = px.histogram(
        d_plot,
        x="pred_priority",
        color="pred_category",
        category_orders=ORDER,
        title="Priority Distribution",
        color_discrete_sequence=LAVENDER_PALETTE,
    )
    st.plotly_chart(build_lavender_chart(fig_pri), use_container_width=True)


st.markdown('<div class="h-line"></div>', unsafe_allow_html=True)


# =========================================================
# ROW 2: 4 CHARTS WITH FULL-HEIGHT VERTICAL DIVIDERS
# =========================================================
st.markdown(
    '<div class="sheet-heading">Resolution Patterns & Anomaly Detection</div>',
    unsafe_allow_html=True,
)

r2_col1, r2_v1, r2_col2, r2_v2, r2_col3, r2_v3, r2_col4 = st.columns(
    [0.23, 0.01, 0.23, 0.01, 0.23, 0.01, 0.23]
)

with r2_col1:
    fig_box_res = px.box(
        d_plot,
        x="pred_priority",
        y="resolution_hours",
        category_orders=ORDER,
        title="Resolution Time (h)",
        color_discrete_sequence=["#4f46e5"],
    )
    st.plotly_chart(build_lavender_chart(fig_box_res), use_container_width=True)

with r2_v1:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r2_col2:
    top = (
        d_plot[d_plot["issue"] != "Miscellaneous"]["issue"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top.columns = ["issue", "count"]
    barh = px.bar(
        top,
        x="count",
        y="issue",
        orientation="h",
        title="Top Recurring Issues",
        color_discrete_sequence=["#6366f1"],
    )
    st.plotly_chart(build_lavender_chart(barh), use_container_width=True)

with r2_v2:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r2_col3:
    resolution_patterns = (
        d_plot.groupby(["pred_category", "resolution_action"])
        .size()
        .reset_index(name="n")
    )
    sun = px.sunburst(
        resolution_patterns,
        path=["pred_category", "resolution_action"],
        values="n",
        title="Resolution Action Breakdown",
        color_discrete_sequence=LAVENDER_PALETTE,
    )
    st.plotly_chart(build_lavender_chart(sun), use_container_width=True)

with r2_v3:
    st.markdown(
        '<div class="v-divider-container"><div class="v-divider"></div></div>',
        unsafe_allow_html=True,
    )

with r2_col4:
    anomaly_label = d_plot["is_anomaly"].map({0: "normal", 1: "unusual"})
    an_fig = px.histogram(
        d_plot,
        x="anomaly_score",
        color=anomaly_label,
        nbins=50,
        log_y=True,
        title="Anomaly Distribution",
        color_discrete_sequence=["#818cf8", "#e11d48"],
    )
    st.plotly_chart(build_lavender_chart(an_fig), use_container_width=True)


st.markdown('<div class="h-line"></div>', unsafe_allow_html=True)


# =========================================================
# OFF-WHITE & LAVENDER THEMED ANOMALY DATA TABLE
# =========================================================
st.markdown(
    "<h4 style='color: #1e1b4b; font-size: 14px; font-weight: 800; margin-top: 10px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.8px;'>Top Unusual Tickets Data</h4>",
    unsafe_allow_html=True,
)

cols = [
    "ticket_id",
    "department",
    "description",
    "pred_category",
    "pred_priority",
    "anomaly_score",
]
top_anoms = (
    d_plot[d_plot["is_anomaly"] == 1]
    .sort_values("anomaly_score", ascending=False)
    .loc[:, [c for c in cols if c in d_plot.columns]]
    .head(50)
)

st.dataframe(top_anoms, use_container_width=True, height=300)