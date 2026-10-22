
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "features" / "tfidf_raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data" / "processed"
    / "model_results" / "logistic_regression"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = load_npz(FEATURE_DIR / "X_train.npz")
    X_val = load_npz(FEATURE_DIR / "X_validation.npz")

    y_train = np.load(FEATURE_DIR / "y_train.npy")
    y_val = np.load(FEATURE_DIR / "y_validation.npy")

    # Supervised training: this model uses the training labels.
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Higher decision scores indicate a greater anomaly likelihood.
    val_scores = model.decision_function(X_val)

    # Choose the threshold using validation labels only.
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )

    if len(thresholds) == 0:
        raise RuntimeError("Could not determine a validation threshold.")

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    y_val_pred = (val_scores >= best_threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_val_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "experiment": "raw_tfidf_logistic_regression",
        "random_state": 42,
        "class_weight": "balanced",
        "fitted_on": "training features and labels",
        "threshold_selected_on": "validation set",
        "decision_score_threshold": best_threshold,
        "validation": {
            "samples": int(len(y_val)),
            "roc_auc": float(roc_auc_score(y_val, val_scores)),
            "pr_auc_average_precision": float(
                average_precision_score(y_val, val_scores)
            ),
            "precision": float(
                precision_score(y_val, y_val_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_val_pred, zero_division=0)
            ),
            "f1": float(f1_score(y_val, y_val_pred)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    joblib.dump(model, OUTPUT_DIR / "model.joblib")

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("=== LOGISTIC REGRESSION BASELINE ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Validation threshold: {best_threshold:.8f}")
    print(f"ROC-AUC: {metrics['validation']['roc_auc']:.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{metrics['validation']['pr_auc_average_precision']:.4f}"
    )
    print(f"Precision: {metrics['validation']['precision']:.4f}")
    print(f"Recall: {metrics['validation']['recall']:.4f}")
    print(f"F1-score: {metrics['validation']['f1']:.4f}")
    print(f"Confusion matrix counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved artifacts to: {OUTPUT_DIR}")
    print("Logistic Regression training: PASS")


if __name__ == "__main__":
    main()
