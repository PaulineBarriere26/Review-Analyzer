# app.py
# ---------------------------------------------------------
# 🔎 Review Insights — Customer Feedback Analyzer
# Classify overall sentiment and score aspects (positive/negative %)
# ---------------------------------------------------------

from typing import List, Dict
import base64
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🔎 Review Insights", page_icon="🔎", layout="wide")

DEFAULT_ASPECTS = [
    "price", "quality", "delivery speed", "customer service", "packaging", "usability"
]

# Simple built-in matplotlib styles (no extra deps)
STYLE_MAP = {
    "Streamlit default": None,
    "ggplot": "ggplot",
    "seaborn (builtin)": "seaborn-v0_8",
    "grayscale": "grayscale",
}

# =========================
# MODEL LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_pipelines():
    """Load and cache HF pipelines once per session."""
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
# HELPERS
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
    """Return overall sentiment + aspect positive/negative %."""
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

def _apply_style(style_key: str):
    """Apply a global matplotlib style."""
    plt.rcParams.update(plt.rcParamsDefault)  # reset
    style = STYLE_MAP.get(style_key)
    if style:
        plt.style.use(style)
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.autolayout": True,
    })

def _annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height() if p.get_height() != 0 else p.get_width()
        # Handle horizontal bars too
        if isinstance(p, plt.Rectangle):
            if ax.yaxis.get_label_position() in ("left", "right") and p.get_width() != 0:
                # Horizontal bars
                ax.annotate(f"{p.get_width():.0f}%",
                            (p.get_x()+p.get_width(), p.get_y()+p.get_height()/2),
                            ha="left", va="center", fontsize=9, xytext=(3,0),
                            textcoords="offset points")
            else:
                # Vertical bars
                ax.annotate(f"{p.get_height():.0f}%",
                            (p.get_x()+p.get_width()/2, p.get_height()),
                            ha="center", va="bottom", fontsize=9, xytext=(0,3),
                            textcoords="offset points")

def plot_sentiment_pie(overall_label: str, overall_conf: float, style_key: str):
    _apply_style(style_key)
    labels = ["Positive", "Neutral", "Negative"]
    vals = [0, 0, 0]
    if overall_label == "POSITIVE":
        vals = [overall_conf, 100 - overall_conf, 0]
    elif overall_label == "NEGATIVE":
        vals = [0, 100 - overall_conf, overall_conf]
    else:
        vals = [0, overall_conf, 100 - overall_conf]

    fig, ax = plt.subplots()
    wedges, texts, autopcts = ax.pie(
        vals, labels=labels, autopct="%1.0f%%", startangle=90, pctdistance=0.75
    )
    # Donut center
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.set_title("Overall sentiment")
    ax.axis("equal")
    st.pyplot(fig, use_container_width=True)

def plot_aspect_bars(
    df_aspects: pd.DataFrame,
    style_key: str,
    orientation: str = "vertical",
    show_values: bool = True
):
    _apply_style(style_key)
    if df_aspects.empty:
        st.info("No aspects to plot.")
        return

    if orientation == "horizontal":
        y = np.arange(len(df_aspects))
        height = 0.4
        fig, ax = plt.subplots()
        ax.barh(y - height/2, df_aspects["positive_%"], height, label="Positive")
        ax.barh(y + height/2, df_aspects["negative_%"], height, label="Negative")
        ax.set_yticks(y)
        ax.set_yticklabels(df_aspects["aspect"])
        ax.set_xlabel("%")
        ax.set_title("Aspect-level sentiment")
        ax.legend(loc="lower right")
        if show_values:
            _annotate_bars(ax)
        st.pyplot(fig, use_container_width=True)
    else:
        x = np.arange(len(df_aspects))
        width = 0.40
        fig, ax = plt.subplots()
        ax.bar(x - width/2, df_aspects["positive_%"], width, label="Positive")
        ax.bar(x + width/2, df_aspects["negative_%"], width, label="Negative")
        ax.set_xticks(x)
        ax.set_xticklabels(df_aspects["aspect"], rotation=15, ha="right")
        ax.set_ylabel("%")
        ax.set_title("Aspect-level sentiment")
        ax.legend()
        if show_values:
            _annotate_bars(ax)
        st.pyplot(fig, use_container_width=True)

def plot_aspect_radar(df_aspects: pd.DataFrame, style_key: str):
    _apply_style(style_key)
    if df_aspects.empty:
        return
    labels = df_aspects["aspect"].tolist()
    pos = df_aspects["positive_%"].to_numpy()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    pos = np.concatenate((pos, [pos[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, pos, linewidth=2)
    ax.fill(angles, pos, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Aspect radar (Positive %)")
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

# Load models (cached)
try:
    sentiment_pipe, zeroshot_pipe = load_pipelines()
    st.sidebar.success("Models loaded.")
except Exception as e:
    st.sidebar.error(f"Failed to load models: {e}")
    st.stop()

# Graph options
st.sidebar.markdown("### 🎨 Graph style")
style_key = st.sidebar.selectbox("Matplotlib style", list(STYLE_MAP.keys()), index=1)
bar_orientation = st.sidebar.radio("Bar orientation", ["vertical", "horizontal"], index=1)
show_values = st.sidebar.checkbox("Show % labels on bars", value=True)
show_radar = st.sidebar.checkbox("Show radar (Positive %)", value=True)

# Aspects
st.sidebar.markdown("### 🏷️ Aspects to score")
custom_aspects = st.sidebar.text_input(
    "Comma-separated aspects", ", ".join(DEFAULT_ASPECTS)
)
ASPECTS = [a.strip() for a in custom_aspects.split(",") if a.strip()]

# Batch controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Batch mode")
example_btn = st.sidebar.button("Load example dataset")
upload_file = st.sidebar.file_uploader("Or upload CSV with a 'review' column", type=["csv"])

# Extras
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

    # KPIs + Pie
    k1, k2 = st.columns([1, 1])
    with k1:
        st.subheader("🧭 Overall sentiment")
        st.metric("Prediction", out["overall_label"])
        st.metric("Confidence", f"{out['overall_confidence_%']:.0f}%")
    with k2:
        plot_sentiment_pie(out["overall_label"], out["overall_confidence_%"], style_key)

    # Table + Bars (+ Radar)
    st.subheader("🧩 Aspect-level signals")
    st.dataframe(out["aspects"], use_container_width=True)
    plot_aspect_bars(out["aspects"], style_key, bar_orientation, show_values)
    if show_radar:
        plot_aspect_radar(out["aspects"], style_key)

    if extra_opts:
        st.subheader("🔍 Extras")
        # Word count
        st.write(f"Word count: **{len(review_text.split())}**")
        # Quick keywords (very simple)
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
                    rows.append({
                        "review_index": i,
                        "overall_label": "",
                        "overall_confidence_%": np.nan,
                        **{f"{a}__positive_%": np.nan for a in ASPECTS},
                        **{f"{a}__negative_%": np.nan for a in ASPECTS},
                        **{f"{a}__top_label": "" for a in ASPECTS},
                    })
                    continue

                out = analyze_single_review(text, ASPECTS, sentiment_pipe, zeroshot_pipe)
                row = {
                    "review_index": i,
                    "overall_label": out["overall_label"],
                    "overall_confidence_%": out["overall_confidence_%"],
                }
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
- Overall sentiment uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (POS/NEU/NEG).
- Aspects use zero-shot NLI (`facebook/bart-large-mnli`) with labels like *positive delivery speed* vs *negative delivery speed*.
- Change aspects in the sidebar. Batch mode accepts a CSV with a `review` column.
""")
