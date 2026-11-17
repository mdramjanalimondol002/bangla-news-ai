"""Evaluate the Bengali news classifier using Stratified 5-Fold Cross-Validation.

Usage:
    python evaluate.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from app.classifier import BanglaNewsClassifier


def main():
    dataset_path = Path("data/sample_news.json")
    records = json.loads(dataset_path.read_text(encoding="utf-8"))

    texts = [r["text"] for r in records]
    labels = np.array([r["category"] for r in records])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(texts, labels):
        clf = BanglaNewsClassifier()
        clf.train([texts[i] for i in train_idx], labels[train_idx].tolist())
        for i in test_idx:
            pred = clf.predict(texts[i])["category"]
            y_true_all.append(labels[i])
            y_pred_all.append(pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )

    print(f"\nEvaluated on {len(records)} Bengali news articles (5-Fold Stratified CV)\n")
    print("| Metric | Score |")
    print("| :--- | :---: |")
    print(f"| Accuracy | {acc:.2%} |")
    print(f"| Macro Precision | {prec:.2%} |")
    print(f"| Macro Recall | {rec:.2%} |")
    print(f"| Macro F1-Score | {f1:.2%} |\n")
    print("Per-Category Breakdown:")
    print(classification_report(y_true_all, y_pred_all, zero_division=0))


if __name__ == "__main__":
    main()