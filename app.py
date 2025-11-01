# Sidebar input
st.sidebar.title("⚙️ Settings")
sentiment, zero_shot = load_pipelines()
st.sidebar.success("Models loaded.")

st.sidebar.markdown("### 🏷️ Aspects to score")
custom_aspects = st.sidebar.text_input(
    "Comma-separated aspects",
    ", ".join(DEFAULT_ASPECTS)
)
ASPECTS = [a.strip() for a in custom_aspects.split(",") if a.strip()]
