import os
import pickle
import string
import logging
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template, jsonify

nltk.download("stopwords", quiet=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Preprocessor (must match train.py exactly for pickle to work) ──────────────
def text_preprocessor(text: str) -> list:
    clean = "".join(ch for ch in text if ch not in string.punctuation)
    stop = set(stopwords.words("english"))
    return [word for word in clean.split() if word.lower() not in stop]

# ── App Init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Model Loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "review_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
    return model

model = load_model()

# ── Helpers ────────────────────────────────────────────────────────────────────
SENTIMENT_MAP = {
    "Positive": {"label": "Positive", "emoji": "😄", "color": "positive"},
    "Negative": {"label": "Negative", "emoji": "😞", "color": "negative"},
}

def predict_sentiment(text: str) -> dict:
    if model is None:
        return {"error": "Model not loaded. Please run train.py first."}
    raw = model.predict([text])[0]
    if str(raw).lower() in ("positive", "1", "1.0"):
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    confidence = None
    try:
        proba = model.predict_proba([text])[0]
        confidence = round(float(max(proba)) * 100, 1)
    except Exception:
        pass
    result = SENTIMENT_MAP[sentiment].copy()
    result["confidence"] = confidence
    return result

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        review_text = (data.get("review") or "").strip()
        if not review_text:
            return jsonify({"error": "Review text is empty."}), 400
        result = predict_sentiment(review_text)
        result["review"] = review_text
        return jsonify(result)
    review_text = (request.form.get("review") or "").strip()
    if not review_text:
        return render_template("index.html", error="Please enter a review.")
    result = predict_sentiment(review_text)
    return render_template("index.html", review=review_text, result=result)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
