
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import IsolationForest
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
    PROJECT_ROOT / "data" / "processed"
    / "features" / "tfidf_masked"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "model_results" / "isolation_forest_masked"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = load_npz(FEATURE_DIR / "X_train.npz")
    X_val = load_npz(FEATURE_DIR / "X_validation.npz")
    y_val = np.load(FEATURE_DIR / "y_validation.npy")

    # Keep the same parameters as the new raw-text baseline.
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    # Unsupervised fitting: training labels are not used.
    model.fit(X_train)

    # Higher scores indicate greater anomaly likelihood.
    val_scores = -model.decision_function(X_val)

    # Select threshold using validation labels only.
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )

    if len(thresholds) == 0:
        raise RuntimeError("No validation thresholds were produced.")

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])
    y_pred = (val_scores >= best_threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    results = {
        "experiment": "R1_masked_tfidf_isolation_forest",
        "n_estimators": 200,
        "contamination": "auto",
        "random_state": 42,
        "threshold_selection": "validation F1",
        "anomaly_score_threshold": best_threshold,
        "validation": {
            "samples": int(len(y_val)),
            "roc_auc": float(roc_auc_score(y_val, val_scores)),
            "pr_auc_average_precision": float(
                average_precision_score(y_val, val_scores)
            ),
            "precision": float(
                precision_score(y_val, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_pred, zero_division=0)
            ),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    joblib.dump(model, OUTPUT_DIR / "model.joblib")

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("=== R1 MASKED TF-IDF: ISOLATION FOREST ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Validation threshold: {best_threshold:.8f}")
    print(f"ROC-AUC: {results['validation']['roc_auc']:.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{results['validation']['pr_auc_average_precision']:.4f}"
    )
    print(f"Precision: {results['validation']['precision']:.4f}")
    print(f"Recall: {results['validation']['recall']:.4f}")
    print(f"F1-score: {results['validation']['f1']:.4f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved artifacts: {OUTPUT_DIR}")
    print("R1 training: PASS")


if __name__ == "__main__":
    main()
