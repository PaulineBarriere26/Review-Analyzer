# app.py
# ---------------------------------------------
# 🔎 Insightlytics — Customer Feedback Analyzer
# A Streamlit app to classify review sentiment and score aspect-level signals
# ---------------------------------------------

import io
import json
import base64
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Light-weight, widely-available HF pipelines
from transformers import pipeline

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
st.set_page_config(page_title="🔎 Insightlytics", page_icon="🔎", layout="wide")

DEFAULT_ASPECTS = [
    "price", "quality", "delivery speed", "customer service", "packaging", "usability"
]

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipelines():
    """Load and cache models one time per session.
    We pick models that are reasonably small for quick cold starts on Streamlit Cloud.
    """
    # General sentiment (POS/NEG/NEU)
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # Aspect scoring via zero-shot NLI (works for any custom label)
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    return sentiment, zero_shot


def percent(x: float) -> float:
    return float(np.round(100.0 * float(x), 2))


def analyze_single_review(text: str, aspects: List[str], sentiment, zero_shot) -> Dict:
    """Return overall sentiment and per-aspect positive/negative percentages.

    Aspect method: we create candidate labels of the form
      - "positive {aspect}" and "negative {aspect}"
    and ask a zero-shot model to score the text against both.
    We normalize the two scores to 100% and report them as the aspect sentiment split.
    """
    # Overall sentiment
    overall_pred = sentiment(text)[0]
    overall_label = overall_pred["label"].upper()
    overall_score = percent(overall_pred["score"])

    # Tidy overall (force to POS/NEG/NEU)
    if "POS" in overall_label:
        overall_label = "POSITIVE"
    elif "NEG" in overall_label:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    # Aspect sentiment via zero-shot
    aspect_rows = []
    for a in aspects:
        cand = [f"positive {a}", f"negative {a}"]
        zs = zero_shot(text, cand, multi_label=False)
        # Extract proba for the two labels
        labels = [lbl.lower() for lbl in zs["labels"]]
        scores = zs["scores"]
        pos_score, neg_score = 0.0, 0.0
        for lbl, sc in zip(labels, scores):
            if lbl.startswith("positive "):
                pos_score = float(sc)
            if lbl.startswith("negative "):
                neg_score = float(sc)
        # Normalize to make them sum to 1 (avoid rounding artifacts)
        s = max(pos_score + neg_score, 1e-9)
        pos_pct = percent(pos_score / s)
        neg_pct = percent(neg_score / s)
        aspect_rows.append({
            "aspect": a,
            "positive_%": pos_pct,
            "negative_%": neg_pct,
            "top_label": "positive" if pos_pct >= neg_pct else "negative"
        })

    df_aspects = pd.DataFrame(aspect_rows)

    return {
        "overall_label": overall_label,
        "overall_confidence_%": overall_score,
        "aspects": df_aspects
    }


def plot_sentiment_pie(overall_label: str, overall_conf: float):
    # Simple 3-slice pie emphasizing predicted class
    labels = ["Positive", "Neutral", "Negative"]
    vals = [0, 0, 0]
    if overall_label == "POSITIVE":
        vals = [overall_conf, 100 - overall_conf, 0]
    elif overall_label == "NEGATIVE":
        vals = [0, 100 - overall_conf, overall_conf]
    else:  # NEUTRAL
        vals = [0, overall_conf, 100 - overall_conf]

    fig, ax = plt.subplots()
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.axis('equal')
    st.pyplot(fig, use_container_width=True)


def plot_aspect_bars(df_aspects: pd.DataFrame):
    # Grouped bars (positive vs negative) per aspect
    x = np.arange(len(df_aspects))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, df_aspects["positive_%"], width, label="Positive")
    ax.bar(x + width/2, df_aspects["negative_%"], width, label="Negative")
    ax.set_xticks(x)
    ax.set_xticklabels(df_aspects["aspect"], rotation=15, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Aspect-level sentiment")
    ax.legend()
    st.pyplot(fig, use_container_width=True)


def make_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'


# ---------------------------------------------
# UI — Sidebar
# ---------------------------------------------
st.sidebar.title("⚙️ Settings")
try:
    sentiment, zero_shot = load_pipelines()
    st.sidebar.success("Models loaded.")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    st.stop()

st.sidebar.markdown("### 🏷️ Aspects to score")
custom_aspects = st.sidebar.text_input(
    "Comma-separated aspects",
    ", ".join(DEFAULT_ASPECTS)
)
ASPECTS = [a.strip() for a in custom_aspects.split(",") if a.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Batch mode")
example_btn = st.sidebar.button("Load example dataset")
upload_file = st.sidebar.file_uploader("Or upload CSV with a 'review' column", type=["csv"]) 

st.sidebar.markdown("---")
extra_opts = st.sidebar.checkbox("Show extras (top keywords, length, etc.)", value=True)

# ---------------------------------------------
# UI — Main
# ---------------------------------------------
st.title("🔎 Review Insights — Customer Feedback Analyzer")
st.caption("Paste reviews to get overall sentiment and aspect-level scores (positive vs negative).")

col1, col2 = st.columns([2,1], gap="large")

with col1:
    default_text = (
        "The shoes look great, the quality is excellent, and customer service was helpful. "
        "Delivery took longer than expected, though."
    )
    review_text = st.text_area("Paste a single review:", value=default_text, height=160)
    analyze_btn = st.button("Analyze review", type="primary")

with col2:
    st.markdown("**Tips**")
    st.markdown("- Add or rename aspects in the sidebar (e.g., *speed, return policy, sizing*).")
    st.markdown("- Use *Batch mode* to analyze many reviews and export a CSV.")

# ---------------------------------------------
# SINGLE REVIEW ANALYSIS
# ---------------------------------------------
if analyze_btn and review_text.strip():
    with st.spinner("Analyzing review..."):
        out = analyze_single_review(review_text, ASPECTS, sentiment, zero_shot)

    st.subheader("🧭 Overall sentiment")
    st.metric("Prediction", out["overall_label"], help="Model's top class")
    st.metric("Confidence", f"{out['overall_confidence_%']:.0f}%", help="Confidence of the predicted class")

    plot_sentiment_pie(out["overall_label"], out["overall_confidence_%"])

    st.subheader("🧩 Aspect-level signals")
    st.dataframe(out["aspects"], use_container_width=True)
    plot_aspect_bars(out["aspects"])

    if extra_opts:
        st.subheader("🔍 Extras")
        n_tokens = len(review_text.split())
        st.write(f"Word count: **{n_tokens}**")
        # Simple tf-idf style keywords for this single text (fallback heuristic)
        # We'll just show the most frequent non-trivial words as a tiny helper.
        tokens = [t.strip(".,!?:;()[]\"'\n ").lower() for t in review_text.split()]
        stop = set("""
            a an the and or of to for in on with at from as by is was were are be been it its it's this that these those
            i you he she they we my our your his her their
            very really just not no yes too so if when then than because but also more most less least
        """.split())
        counts = {}
        for t in tokens:
            if t and t.isalpha() and t not in stop:
                counts[t] = counts.get(t, 0) + 1
        top_kw = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_kw:
            st.write("Top keywords:")
            st.write(", ".join([k for k, _ in top_kw]))

# ---------------------------------------------
# BATCH MODE
# ---------------------------------------------
if example_btn:
    df = pd.DataFrame({
        "review": [
            "Great quality and fast delivery. Support answered my questions quickly.",
            "Product arrived late and packaging was damaged. Customer service didn't help.",
            "Fair price for the value. Build quality is solid but instructions are unclear.",
            "Amazing customer service! Delivery was slow but they kept me updated.",
            "Terrible quality, broke in two days. Fast refund though."
        ]
    })
    st.session_state["batch_df"] = df

if upload_file is not None:
    try:
        df = pd.read_csv(upload_file)
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            st.session_state["batch_df"] = df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if "batch_df" in st.session_state:
    st.subheader("📚 Batch analysis")
    df = st.session_state["batch_df"].copy()

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
                out = analyze_single_review(text, ASPECTS, sentiment, zero_shot)
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

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("""
---
**Notes**
- Overall sentiment uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (POS/NEU/NEG).
- Aspect signals use zero-shot NLI (`facebook/bart-large-mnli`) with labels like _positive delivery speed_ vs _negative delivery speed_.
- You can change aspects in the sidebar. Batch mode accepts a CSV with a `review` column.
""")
