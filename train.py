"""
train.py — Train and save the Game Stream Review Classifier.

Usage:
    python train.py --data data/reviews.csv

Expected CSV columns:
    - review  : raw review text
    - label   : "Positive" or "Negative"
"""

import os
import pickle
import argparse
import string
import logging

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "review_model.pkl")


# ── Text Preprocessing ─────────────────────────────────────────────────────────
def text_preprocessor(text: str) -> list[str]:
    """
    Remove punctuation, lowercase, and strip English stopwords.
    Returns a list of clean tokens.
    """
    # Remove punctuation
    clean = "".join(ch for ch in text if ch not in string.punctuation)
    # Tokenise and remove stopwords
    stop = set(stopwords.words("english"))
    tokens = [word for word in clean.split() if word.lower() not in stop]
    return tokens


# ── Pipeline ───────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("bow",   CountVectorizer(analyzer=text_preprocessor)),
        ("tfidf", TfidfTransformer()),
        ("clf",   MultinomialNB()),
    ])


# ── Train ──────────────────────────────────────────────────────────────────────
def train(data_path: str):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Validate columns
    assert "content" in df.columns and "is_positive" in df.columns, \
        "CSV must have 'content' and 'is_positive' columns."

    df.dropna(subset=["content", "is_positive"], inplace=True)
    logger.info(f"Dataset size: {len(df)} rows")
    logger.info(f"Label distribution:\n{df['is_positive'].value_counts()}")

    X = df["content"]
    y = df["is_positive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\nAccuracy: {acc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved → {MODEL_PATH}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Game Stream Review Classifier")
    parser.add_argument("--data", required=True, help="Path to reviews CSV file")
    args = parser.parse_args()
    train(args.data)