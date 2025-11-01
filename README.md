# 🔎 Review Insights — Customer Feedback Analyzer

A **Streamlit web app** that lets companies instantly analyze customer reviews for sentiment and key aspects like **quality**, **delivery**, or **customer service**.

---

## 🌟 Features

- **Overall Sentiment Analysis** — Detect whether a review is *positive*, *neutral*, or *negative*, with confidence percentage.  
- **Aspect-Level Insights** — Evaluate how each aspect (e.g. *quality*, *price*, *support*) scores between positive and negative tone.  
- **Batch Mode** — Upload a CSV of reviews to analyze them all at once and **download results**.  
- **Visual Dashboard** — Includes a sentiment pie chart and grouped bar chart by aspect.  
- **Custom Aspects** — Add or rename aspects in the sidebar (e.g. “speed”, “return policy”).  

---

## 🧠 Models Used

| Task | Model | Source |
|------|--------|--------|
| Sentiment Classification | `cardiffnlp/twitter-roberta-base-sentiment-latest` | [Hugging Face](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Aspect Analysis (Zero-Shot) | `facebook/bart-large-mnli` | [Hugging Face](https://huggingface.co/facebook/bart-large-mnli) |

---

## ⚙️ Requirements

Create a file called `requirements.txt` with the following content:

```txt
streamlit>=1.36
transformers>=4.44
torch>=2.1
pandas>=2.0
numpy>=1.26
matplotlib>=3.7
