
import json
from functools import lru_cache
from pathlib import Path

import joblib

# Reuse the exact preprocessing functions used by R1 training.
from ml.features.build_masked_tfidf import mask_text, preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "tfidf_masked"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "isolation_forest_masked"
)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained model, vectorizer, and validation threshold once."""

    model_path = MODEL_DIR / "model.joblib"
    vectorizer_path = FEATURE_DIR / "tfidf_vectorizer.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    for path in (model_path, vectorizer_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = float(metrics["anomaly_score_threshold"])

    if vectorizer.transform(["test"]).shape[1] != model.n_features_in_:
        raise RuntimeError(
            "Vectorizer feature count does not match the trained model."
        )

    return model, vectorizer, threshold


def predict_trace(trace_text: str) -> dict:
    """Classify one trace using the saved R1 model."""

    if not trace_text or not trace_text.strip():
        raise ValueError("Trace text must not be empty.")

    model, vectorizer, threshold = load_artifacts()

    # Match the training preprocessing pipeline.
    masked_text, _ = mask_text(trace_text)
    processed_text = preprocess_text(masked_text)

    if not processed_text.strip():
        raise ValueError("Trace is empty after preprocessing.")

    features = vectorizer.transform([processed_text])

    # Higher score means more anomalous in our experiment.
    anomaly_score = -float(model.decision_function(features)[0])

    prediction = (
        "Anomaly" if anomaly_score >= threshold else "Normal"
    )

    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "model": "R1_masked_tfidf_isolation_forest",
    }
