# app.py (Streamlit)
# 🔎 Review Insights — Customer Feedback Analyzer
# See README.md below for usage instructions.

import io
import json
import base64
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline

ASPECTS = [a.strip() for a in custom_aspects.split(",") if a.strip()]

st.title("🔎 Review Insights — Customer Feedback Analyzer")
st.caption("Paste reviews to get overall sentiment and aspect-level scores.")

default_text = "The shoes look great, the quality is excellent, and customer service was helpful. Delivery took longer than expected, though."
review_text = st.text_area("Paste a single review:", value=default_text, height=160)
analyze_btn = st.button("Analyze review", type="primary")

if analyze_btn and review_text.strip():
    with st.spinner("Analyzing review..."):
        out = analyze_single_review(review_text, ASPECTS, sentiment, zero_shot)
    st.subheader("🧭 Overall sentiment")
    st.metric("Prediction", out["overall_label"])
    st.metric("Confidence", f"{out['overall_confidence_%']:.0f}%")
    plot_sentiment_pie(out["overall_label"], out["overall_confidence_%"])
    st.subheader("🧩 Aspect-level signals")
    st.dataframe(out["aspects"], use_container_width=True)
    plot_aspect_bars(out["aspects"])

# ---------------- README ----------------

'''
# README.md

## 🔎 Review Insights — Customer Feedback Analyzer

A Streamlit web app that lets companies instantly analyze customer reviews for sentiment and key aspects.

### 🌟 Features
- Detect **positive / neutral / negative** sentiment with confidence.
- Evaluate **aspect-level polarity** (e.g., quality, service, delivery, etc.).
- Upload CSV of reviews for **batch analysis** and export results.
- Visualization: pie chart + grouped bar chart.

### 🧠 Models
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)

### ⚙️ Requirements
```bash
streamlit>=1.36
transformers>=4.44
torch>=2.1
pandas>=2.0
numpy>=1.26
matplotlib>=3.7
```

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### ☁️ Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/), click **New App**, and point to `app.py`.
3. Done — your hosted Review Insights dashboard is live.

### 📂 Input Format for Batch Mode
CSV file with one column:
```csv
review
"The product arrived on time and works great!"
"Customer support was slow to reply."
```

### 📜 License
MIT — free for commercial and academic use.
'''
