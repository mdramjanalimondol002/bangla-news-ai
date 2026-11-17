import json
from pathlib import Path

import streamlit as st

from app.classifier import BanglaNewsClassifier
from app.summarizer import BanglaNewsSummarizer

st.set_page_config(
    page_title="BanglaNews AI — Classifier & Summarizer",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource
def load_models():
    clf = BanglaNewsClassifier()
    clf.train_from_json("data/sample_news.json")
    summarizer = BanglaNewsSummarizer()
    return clf, summarizer


@st.cache_data
def load_sample_articles():
    records = json.loads(Path("data/sample_news.json").read_text(encoding="utf-8"))
    samples = {}
    for item in records:
        cat = item["category"]
        if cat not in samples:
            samples[cat] = item["text"]
    return samples


def main():
    clf, summarizer = load_models()
    samples = load_sample_articles()

    st.title("📰 BanglaNews AI: Bengali News Classifier & Summarizer")
    st.markdown(
        "Automated **topic classification** (`TF-IDF Word + Character N-grams + Logistic Regression`) "
        "and **abstractive summarization** (`Google Gemini` with offline extractive fallback) for Bengali news articles."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        summary_lang = st.selectbox(
            "Summary Output Language",
            options=["Bengali", "English"],
            index=0
        )
        st.divider()
        st.subheader("📌 Load Sample Article")
        selected_cat = st.selectbox("Choose a category:", list(samples.keys()))
        if st.button("Load Sample Text", use_container_width=True):
            st.session_state["article_input"] = samples[selected_cat]

        st.divider()
        st.caption("Categories: জাতীয়, আন্তর্জাতিক, খেলাধুলা, প্রযুক্তি, অর্থনীতি")

    default_text = st.session_state.get("article_input", samples["প্রযুক্তি"])
    article_text = st.text_area(
        "Paste or edit a Bengali news article below:",
        value=default_text,
        height=180
    )

    if st.button("🚀 Analyze & Summarize News", type="primary"):
        if not article_text.strip():
            st.warning("অনুগ্রহ করে বিশ্লেষণের জন্য একটি বাংলা সংবাদ অনুচ্ছেদ দিন।")
            return

        col1, col2 = st.columns([1, 1.3], gap="large")

        with col1:
            st.subheader("🏷️ Topic Classification")
            pred = clf.predict(article_text)
            st.metric(
                label="Predicted Category",
                value=pred["category"],
                delta=f"{pred['confidence']:.2f}% confidence"
            )
            st.markdown("**Category Probability Breakdown:**")
            for cat, score in pred["scores"].items():
                st.write(f"**{cat}** — `{score:.2f}%`")
                st.progress(min(int(round(score)), 100))

        with col2:
            st.subheader("📝 AI News Summary")
            result = summarizer.summarize(article_text, language=summary_lang)
            st.caption(f"Engine: `{result['mode']}`")
            st.markdown(result["summary"])


if __name__ == "__main__":
    main()