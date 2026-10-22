
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "features" / "tfidf_raw"
)

MODEL_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "model_results" / "isolation_forest"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "model_results" / "final_evaluation"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the already-fitted model and locked validation threshold.
    model = joblib.load(MODEL_DIR / "model.joblib")

    with (MODEL_DIR / "metrics.json").open(
        "r", encoding="utf-8"
    ) as file:
        validation_metrics = json.load(file)

    threshold = validation_metrics["anomaly_score_threshold"]

    X_test = load_npz(FEATURE_DIR / "X_test.npz")
    y_test = np.load(FEATURE_DIR / "y_test.npy")

    # Higher scores mean more anomalous.
    test_scores = -model.decision_function(X_test)

    # Use the validation threshold without changing it.
    y_pred = (test_scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results = {
        "model": "Isolation Forest",
        "threshold_source": "validation set",
        "threshold": float(threshold),
        "test_samples": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, test_scores)
        ),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    output_file = OUTPUT_DIR / "isolation_forest_test_metrics.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("=== FINAL TEST EVALUATION ===")
    print(f"Test samples: {len(y_test)}")
    print(f"Locked validation threshold: {threshold:.8f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC (Average Precision): {results['pr_auc_average_precision']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved results: {output_file}")
    print("Final test evaluation: PASS")


if __name__ == "__main__":
    main()
