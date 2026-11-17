import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from app.preprocessor import tokenize_and_clean


class BanglaNewsClassifier:
    """Bengali news topic classifier combining word and subword character n-grams."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=5000,
                    sublinear_tf=True
                )),
            ])),
            ("clf", LogisticRegression(max_iter=1000, C=2.0))
        ])
        self.classes_ = []
        self.is_trained = False

    def train(self, texts: list[str], labels: list[str]):
        cleaned_texts = [tokenize_and_clean(t) for t in texts]
        self.pipeline.fit(cleaned_texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        self.is_trained = True

    def train_from_json(self, json_path: str | Path = "data/sample_news.json"):
        records = json.loads(Path(json_path).read_text(encoding="utf-8"))
        texts = [item["text"] for item in records]
        labels = [item["category"] for item in records]
        self.train(texts, labels)
        return len(records)

    def predict(self, text: str) -> dict:
        if not self.is_trained:
            self.train_from_json()

        cleaned = tokenize_and_clean(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        ranked = sorted(
            zip(self.classes_, probs),
            key=lambda x: x[1],
            reverse=True
        )
        top_category, top_conf = ranked[0]
        return {
            "category": top_category,
            "confidence": round(float(top_conf) * 100, 2),
            "scores": {cat: round(float(p) * 100, 2) for cat, p in ranked}
        }