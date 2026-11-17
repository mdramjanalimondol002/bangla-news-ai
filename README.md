# BanglaNews AI — Automated Bengali News Classifier & Summarizer

An end-to-end Bengali Natural Language Processing (NLP) and Generative AI application built with **Python**, **Scikit-learn**, **Google Gemini (`google-genai`)**, and **Streamlit**. BanglaNews AI normalizes raw Bengali Unicode news articles, classifies them across 5 major news categories using a hybrid word + character n-gram TF-IDF classifier, and generates structured executive summaries in Bengali or English.

---

## Key Features
* **Bengali Unicode Normalization & Cleaning (`app/preprocessor.py`):** Applies Unicode NFC normalization to standardize compound Bengali characters (`য়`, `ড়`), removes URLs, HTML tags, punctuation, and Bengali/English numerals, and filters high-frequency Bengali stopwords.
* **Subword + Word N-Gram Topic Classifier (`app/classifier.py`):** Combines word-level TF-IDF (`ngram_range=(1, 2)`) with character boundary n-grams (`analyzer="char_wb"`, `ngram_range=(3, 5)`) and L2-regularized Logistic Regression to handle Bengali morphological inflections across 5 categories: **জাতীয় (National)**, **আন্তর্জাতিক (International)**, **খেলাধুলা (Sports)**, **প্রযুক্তি (Technology)**, and **অর্থনীতি (Economy)**.
* **Dual-Mode News Summarizer (`app/summarizer.py`):** Synthesizes a 2-sentence executive summary and 3 bullet-point key takeaways via **Google Gemini (`gemini-2.0-flash`)**, with an automatic offline extractive fallback when no API key is configured.
* **Interactive Web Dashboard (`main.py`):** Real-time Streamlit interface displaying predicted topic confidence scores, full category probability bars, and bilingual summary generation.

---

## Project Architecture

    bangla-news-ai/
    ├── app/
    │   ├── __init__.py
    │   ├── preprocessor.py   # Bengali Unicode NFC normalization & stopword filtering
    │   ├── classifier.py     # TF-IDF (word + char_wb n-grams) + Logistic Regression model
    │   └── summarizer.py     # Google Gemini LLM summarizer + extractive fallback
    ├── data/
    │   └── sample_news.json  # 100 curated multi-sentence Bengali news articles (20 per class)
    ├── evaluate.py           # 5-Fold Stratified Cross-Validation evaluation script
    ├── main.py               # Interactive Streamlit web application
    ├── requirements.txt      # Project dependencies
    ├── README.md             # Project documentation
    └── .gitignore

---

## Model Evaluation Benchmark

Evaluated on the curated dataset (`data/sample_news.json`, 100 multi-sentence Bengali news articles across 5 balanced categories) using **5-Fold Stratified Cross-Validation** (`evaluate.py`):

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **88.00%** |
| **Macro Precision** | **89.33%** |
| **Macro Recall** | **88.00%** |
| **Macro F1-Score** | **88.05%** |

### Per-Category F1 Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **খেলাধুলা (Sports)** | 100.0% | 100.0% | **1.00** | 20 |
| **প্রযুক্তি (Technology)** | 86.0% | 90.0% | **0.88** | 20 |
| **আন্তর্জাতিক (International)** | 94.0% | 80.0% | **0.86** | 20 |
| **অর্থনীতি (Economy)** | 73.0% | 95.0% | **0.83** | 20 |
| **জাতীয় (National)** | 94.0% | 75.0% | **0.83** | 20 |

Reproduce the benchmark locally:

    python evaluate.py

---

## Tech Stack
* **NLP & Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `FeatureUnion`, `LogisticRegression`), `numpy`, `unicodedata`
* **Generative AI:** Google Gemini API (`google-genai`), `python-dotenv`
* **Frontend UI:** `streamlit`

---

## Quick Start (Installation & Setup)

1. **Clone the repository:**

       git clone https://github.com/mdramjanalimondol002/bangla-news-ai.git
       cd bangla-news-ai

2. **Create and activate a virtual environment (Optional):**

       python -m venv .venv
       # On Windows:
       .venv\Scripts\activate
       # On macOS/Linux:
       source .venv/bin/activate

3. **Install dependencies:**

       pip install -r requirements.txt

4. **Configure Gemini API Key (Optional):**
   Create a `.env` file in the project root to enable abstractive LLM summaries (the app works offline via extractive fallback if omitted):

       GEMINI_API_KEY=your_api_key_here

5. **Launch the Streamlit App:**

       python -m streamlit run main.py

---

## How to Use the Application (Step-by-Step Guide)

1. **Start the Web Server:** Run `python -m streamlit run main.py` from the project root and open `http://localhost:8501` in your browser.
2. **Select Output Summary Language:** In the left sidebar under **Configuration**, choose either **Bengali** or **English** for the generated executive summary.
3. **Provide a News Article:** Paste any Bengali news article into the text area, or pick one of the 5 categories in the sidebar and click **Load Sample Text** to test with a built-in article.
4. **Run Classification & Summarization:** Click **🚀 Analyze & Summarize News** to view:
   * **Predicted Category & Confidence Score** on the left panel, along with probability bars across all 5 categories.
   * **Executive Summary & Key Takeaways** on the right panel, indicating whether `Gemini (gemini-2.0-flash)` or `Offline Extractive Fallback` generated the summary.