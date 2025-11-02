# app.py
# ---------------------------------------------------------
# 🔎 Review Insights — Customer Feedback Analyzer
# Classify overall sentiment and score aspects (positive/negative %)
# ---------------------------------------------------------

from typing import List, Dict
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

# --- Dark/Light Mode Color Palettes ---
POS_LIGHT = "#2CB67D"   # green
NEG_LIGHT = "#EF5B5B"   # red
NEU_LIGHT = "#A0AEC0"   # grey

POS_DARK  = "#38E2A0"
NEG_DARK  = "#FF7B7B"
NEU_DARK  = "#C3CAD9"

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🔎 Review Insights", page_icon="🔎", layout="wide")
# --- Dashboard Header ---

DEFAULT_ASPECTS = [
    "price", "quality", "delivery speed", "customer service", "packaging", "usability"
]

# Only one sleek custom style
STYLE_KEY = "Sleek (custom)"
POS_COLOR = "#2CB67D"   # green
NEG_COLOR = "#EF5B5B"   # red
NEU_COLOR = "#A0AEC0"   # grey


# =========================
# MODEL LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_pipelines():
    """Load and cache Hugging Face pipelines once per session."""
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    return sentiment, zero_shot


# =========================
# HELPER FUNCTIONS
# =========================
def percent(x: float) -> float:
    return float(np.round(100.0 * float(x), 2))

def tidy_overall_label(lbl: str) -> str:
    u = lbl.upper()
    if "POS" in u:   return "POSITIVE"
    if "NEG" in u:   return "NEGATIVE"
    return "NEUTRAL"

def analyze_single_review(
    text: str,
    aspects: List[str],
    sentiment_pipeline,
    zeroshot_pipeline
) -> Dict:
    """Return overall sentiment + aspect positive/negative % for one review."""
    # Overall
    o = sentiment_pipeline(text)[0]
    overall_label = tidy_overall_label(o["label"])
    overall_conf  = percent(o["score"])

    # Aspects via zero-shot: "positive X" vs "negative X"
    rows = []
    for a in aspects:
        candidates = [f"positive {a}", f"negative {a}"]
        zs = zeroshot_pipeline(text, candidates, multi_label=False)
        labels = [s.lower() for s in zs["labels"]]
        scores = zs["scores"]

        pos_score = 0.0
        neg_score = 0.0
        for lab, sc in zip(labels, scores):
            if lab.startswith("positive "): pos_score = float(sc)
            if lab.startswith("negative "): neg_score = float(sc)

        denom = max(pos_score + neg_score, 1e-9)
        pos_pct = percent(pos_score / denom)
        neg_pct = percent(neg_score / denom)

        rows.append({
            "aspect": a,
            "positive_%": pos_pct,
            "negative_%": neg_pct,
            "top_label": "positive" if pos_pct >= neg_pct else "negative"
        })

    return {
        "overall_label": overall_label,
        "overall_confidence_%": overall_conf,
        "aspects": pd.DataFrame(rows)
    }


# =========================
# VISUAL STYLE + PLOTS
# =========================
def _apply_sleek_style(dark_mode: bool = False):
    """Apply the sleek modern dashboard look (light/dark)."""
    import matplotlib as mpl
    plt.rcParams.update(plt.rcParamsDefault)

    if dark_mode:
        mpl.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.color": "#3A3F4B",
            "grid.alpha": 0.25,
            "axes.edgecolor": "#3A3F4B",
            "axes.linewidth": 0.8,
            "figure.autolayout": True,
            "figure.facecolor": "#0E1117",
            "axes.facecolor": "#0E1117",
            "text.color": "#E6E6E6",
            "axes.labelcolor": "#E6E6E6",
            "xtick.color": "#E6E6E6",
            "ytick.color": "#E6E6E6",
        })
    else:
        mpl.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.color": "#8b8b8b",
            "grid.alpha": 0.18,
            "axes.edgecolor": "#DDDDDD",
            "axes.linewidth": 0.8,
            "figure.autolayout": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        })


def plot_sentiment_donut(overall_label: str, overall_conf: float, dark_mode: bool):
    _apply_sleek_style()
    labels = ["Positive", "Neutral", "Negative"]
    vals = [0, 0, 0]
    if overall_label == "POSITIVE":
        vals = [overall_conf, 100 - overall_conf, 0]
    elif overall_label == "NEGATIVE":
        vals = [0, 100 - overall_conf, overall_conf]
    else:
        vals = [0, overall_conf, 100 - overall_conf]
    colors = [POS_COLOR, NEU_COLOR, NEG_COLOR]

    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(
        vals, labels=None, autopct="%1.0f%%",
        startangle=90, pctdistance=0.78, colors=colors
    )
    centre = plt.Circle((0, 0), 0.58, fc="white")
    ax.add_artist(centre)

    ax.text(0, 0.05, overall_label.title(), ha="center", va="center",
            fontsize=12, weight="bold")
    ax.text(0, -0.12, f"{overall_conf:.0f}%", ha="center", va="center",
            fontsize=11, color="#555")

    ax.legend(
        [wedges[0], wedges[1], wedges[2]],
        labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    ax.axis("equal")
    st.pyplot(fig, use_container_width=True)


def plot_aspect_bars(df_aspects: pd.DataFrame, orientation: str, show_values: bool, dark_mode: bool):
    _apply_sleek_style()
    if df_aspects.empty:
        st.info("No aspects to plot.")
        return

    if orientation == "horizontal":
        y = np.arange(len(df_aspects))
        h = 0.38
        fig, ax = plt.subplots()
        ax.barh(y - h/2, df_aspects["positive_%"], h, label="Positive", color=POS_COLOR, alpha=0.9)
        ax.barh(y + h/2, df_aspects["negative_%"], h, label="Negative", color=NEG_COLOR, alpha=0.9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_yticks(y)
        ax.set_yticklabels(df_aspects["aspect"])
        ax.set_xlabel("%")
        ax.set_title("Aspect-level sentiment")
        ax.legend(loc="lower right", frameon=False)
        if show_values:
            for i, v in enumerate(df_aspects["positive_%"]):
                ax.text(v + 1, i - h/2, f"{v:.0f}%", va="center", fontsize=9, color="#333")
            for i, v in enumerate(df_aspects["negative_%"]):
                ax.text(v + 1, i + h/2, f"{v:.0f}%", va="center", fontsize=9, color="#333")
        st.pyplot(fig, use_container_width=True)

    else:
        x = np.arange(len(df_aspects))
        w = 0.40
        fig, ax = plt.subplots()
        p1 = ax.bar(x - w/2, df_aspects["positive_%"], w, label="Positive", color=POS_COLOR, alpha=0.9)
        p2 = ax.bar(x + w/2, df_aspects["negative_%"], w, label="Negative", color=NEG_COLOR, alpha=0.9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(df_aspects["aspect"], rotation=12, ha="right")
        ax.set_ylabel("%")
        ax.set_title("Aspect-level sentiment")
        ax.legend(frameon=False)
        if show_values:
            for rects in (p1, p2):
                for r in rects:
                    ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1,
                            f"{r.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color="#333")
        st.pyplot(fig, use_container_width=True)


def plot_aspect_radar(df_aspects: pd.DataFrame, dark_mode: bool):
    _apply_sleek_style()
    if df_aspects.empty:
        return

    labels = df_aspects["aspect"].tolist()
    pos = df_aspects["positive_%"].to_numpy()
    neg = df_aspects["negative_%"].to_numpy()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    pos = np.concatenate((pos, [pos[0]]))
    neg = np.concatenate((neg, [neg[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.2)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#777", fontsize=9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.plot(angles, pos, color=POS_COLOR, linewidth=2, label="Positive")
    ax.fill(angles, pos, color=POS_COLOR, alpha=0.15)
    ax.plot(angles, neg, color=NEG_COLOR, linewidth=2, label="Negative")
    ax.fill(angles, neg, color=NEG_COLOR, alpha=0.10)
    ax.set_title("Aspect radar (Positive vs Negative)", pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)
    st.pyplot(fig, use_container_width=True)


def make_download_link(df: pd.DataFrame, filename: str) -> str:
    """Return an HTML link that downloads the dataframe as CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)
# --- Responsive Header Banner ---
def render_header(dark: bool):
    bg   = "#0E1117" if dark else "#F5F7FA"   # background color
    fg   = "#E6E6E6" if dark else "#222222"   # title text color
    sub  = "#A9B2C3" if dark else "#666666"   # subtitle color
    edge = "#2A2F3A" if dark else "#DDDDDD"   # subtle border
    shadow = "none" if dark else "0 2px 8px rgba(0,0,0,0.05)"

    st.markdown(f"""
        <style>
            .header {{
                background-color: {bg};
                color: {fg};
                padding: 0.9rem 1.5rem;
                border-radius: 12px;
                border: 1px solid {edge};
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.2rem;
                box-shadow: {shadow};
            }}
            .header h1 {{
                font-size: 1.6rem;
                margin: 0;
                color: {fg};
            }}
            .header .tagline {{
                color: {sub};
                font-size: 0.95rem;
                font-weight: 400;
            }}
        </style>
        <div class="header">
            <h1>Welcome to Insightlytics !</h1>
            <div class="tagline">AI dashboard for understanding customer feedback</div>
        </div>
    """, unsafe_allow_html=True)

# Call it so it renders
render_header(dark_mode)
def _apply_page_css(dark: bool):
    if dark:
        st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: #E6E6E6; }
            div[data-testid="stSidebar"] { background-color: #111418; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #FFFFFF; color: #111111; }
            div[data-testid="stSidebar"] { background-color: #F7F9FB; }
        </style>
        """, unsafe_allow_html=True)

_apply_page_css(dark_mode)
try:
    sentiment_pipe, zeroshot_pipe = load_pipelines()
    st.sidebar.success("Models loaded.")
except Exception as e:
    st.sidebar.error(f"Failed to load models: {e}")
    st.stop()

bar_orientation = st.sidebar.radio("Bar orientation", ["vertical", "horizontal"], index=1)
show_values = st.sidebar.checkbox("Show % labels on bars", value=True)
show_radar = st.sidebar.checkbox("Show radar (Positive vs Negative)", value=True)

st.sidebar.markdown("### 🏷️ Aspects to score")
custom_aspects = st.sidebar.text_input("Comma-separated aspects", ", ".join(DEFAULT_ASPECTS))
ASPECTS = [a.strip() for a in custom_aspects.split(",") if a.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Batch mode")
example_btn = st.sidebar.button("Load example dataset")
upload_file = st.sidebar.file_uploader("Or upload CSV with a 'review' column", type=["csv"])

st.sidebar.markdown("---")
extra_opts = st.sidebar.checkbox("Show extras (keywords, length)", value=True)


# =========================
# MAIN — Single review
# =========================
st.title("🔎 Review Insights — Customer Feedback Analyzer")
st.caption("Paste a review to get overall sentiment and aspect-level scores (positive vs negative).")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    default_text = (
        "The shoes look great, the quality is excellent, and customer service was helpful. "
        "Delivery took longer than expected, though."
    )
    review_text = st.text_area("Paste a single review:", value=default_text, height=160)
    analyze_btn = st.button("Analyze review", type="primary")

with col2:
    st.markdown("**Tips**")
    st.markdown("- Edit aspects in the sidebar (e.g., *speed, return policy, sizing*).")
    st.markdown("- Use *Batch mode* to analyze many reviews and export a CSV.")

if analyze_btn and review_text.strip():
    with st.spinner("Analyzing review..."):
        out = analyze_single_review(review_text, ASPECTS, sentiment_pipe, zeroshot_pipe)
    k1, k2 = st.columns([1, 1])
    with k1:
        st.subheader("🧭 Overall sentiment")
        st.metric("Prediction", out["overall_label"])
        st.metric("Confidence", f"{out['overall_confidence_%']:.0f}%")
    with k2:
        plot_sentiment_donut(out["overall_label"], out["overall_confidence_%"])
    st.subheader("🧩 Aspect-level signals")
    st.dataframe(out["aspects"], use_container_width=True)
    plot_aspect_bars(out["aspects"], bar_orientation, show_values)
    if show_radar:
        plot_aspect_radar(out["aspects"])
    if extra_opts:
        st.subheader("🔍 Extras")
        st.write(f"Word count: **{len(review_text.split())}**")
        tokens = [t.strip(".,!?:;()[]\"'\n ").lower() for t in review_text.split()]
        stop = set("""
            a an the and or of to for in on with at from as by is was were are be been it its it's this that these those
            i you he she they we my our your his her their very really just not no yes too so if when then than because
            but also more most less least
        """.split())
        counts = {}
        for t in tokens:
            if t and t.isalpha() and t not in stop:
                counts[t] = counts.get(t, 0) + 1
        top_kw = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_kw:
            st.write("Top keywords:")
            st.write(", ".join([k for k, _ in top_kw]))


# =========================
# BATCH MODE
# =========================
if example_btn:
    st.session_state["batch_df"] = pd.DataFrame({
        "review": [
            "Great quality and fast delivery. Support answered my questions quickly.",
            "Product arrived late and packaging was damaged. Customer service didn't help.",
            "Fair price for the value. Build quality is solid but instructions are unclear.",
            "Amazing customer service! Delivery was slow but they kept me updated.",
            "Terrible quality, broke in two days. Fast refund though."
        ]
    })

if upload_file is not None:
    try:
        df_up = pd.read_csv(upload_file)
        if "review" not in df_up.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            st.session_state["batch_df"] = df_up
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if "batch_df" in st.session_state:
    st.subheader("📚 Batch analysis")
    df = st.session_state["batch_df"].copy()
    st.dataframe(df, use_container_width=True, height=220)
    run_batch = st.button("Run batch analysis", type="primary")
    if run_batch:
        rows = []
        with st.spinner("Analyzing all reviews..."):
            for i, r in df.iterrows():
                text = str(r["review"]).strip()
                if not text:
                    continue
                out = analyze_single_review(text, ASPECTS, sentiment_pipe, zeroshot_pipe)
                row = {"review_index": i, "overall_label": out["overall_label"],
                       "overall_confidence_%": out["overall_confidence_%"]}
                for _, ar in out["aspects"].iterrows():
                    a = ar["aspect"]
                    row[f"{a}__positive_%"] = ar["positive_%"]
                    row[f"{a}__negative_%"] = ar["negative_%"]
                    row[f"{a}__top_label"] = ar["top_label"]
                rows.append(row)
        res = pd.DataFrame(rows)
        st.dataframe(res, use_container_width=True)
        st.markdown(make_download_link(res, "review_insights_results.csv"), unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
---
**Notes**
- Overall sentiment uses `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- Aspects use zero-shot NLI (`facebook/bart-large-mnli`).
- Change aspects in the sidebar. Batch mode accepts a CSV with a `review` column.
""")
