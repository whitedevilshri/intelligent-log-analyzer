
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "features" / "tfidf_raw"
)

LABEL_MAP = {"Normal": 0, "Anomaly": 1}


def load_split(filename: str):
    path = SPLIT_DIR / filename

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required = {"BlockId", "CombinedLogText", "Label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected columns in {path}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty split: {path}")

    if any(not row["CombinedLogText"].strip() for row in rows):
        raise ValueError(f"Empty trace text in {path}")

    unknown_labels = {row["Label"] for row in rows} - set(LABEL_MAP)
    if unknown_labels:
        raise ValueError(f"Unknown labels in {path}: {unknown_labels}")

    return (
        [row["CombinedLogText"] for row in rows],
        np.array([LABEL_MAP[row["Label"]] for row in rows], dtype=np.int8),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_texts, y_train = load_split("train.csv")
    validation_texts, y_validation = load_split("validation.csv")
    test_texts, y_test = load_split("test.csv")

    # Fit only on the training data to prevent vocabulary leakage.
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        lowercase=True,
        stop_words="english",
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(train_texts)

    # Do not fit or refit on validation or test data.
    X_validation = vectorizer.transform(validation_texts)
    X_test = vectorizer.transform(test_texts)

    save_npz(OUTPUT_DIR / "X_train.npz", X_train)
    save_npz(OUTPUT_DIR / "X_validation.npz", X_validation)
    save_npz(OUTPUT_DIR / "X_test.npz", X_test)

    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_validation.npy", y_validation)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")

    metadata = {
        "experiment": "raw_tfidf_baseline",
        "max_features": 10_000,
        "lowercase": True,
        "stop_words": "english",
        "fitted_on": "train.csv only",
        "label_mapping": LABEL_MAP,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "shapes": {
            "train": list(X_train.shape),
            "validation": list(X_validation.shape),
            "test": list(X_test.shape),
        },
    }

    with (OUTPUT_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("=== TF-IDF FEATURE REPORT ===")
    print(f"Training matrix:   {X_train.shape}")
    print(f"Validation matrix: {X_validation.shape}")
    print(f"Test matrix:       {X_test.shape}")
    print(f"Vocabulary size:   {len(vectorizer.vocabulary_)}")
    print(f"Training labels:   {np.bincount(y_train, minlength=2)}")
    print(f"Validation labels: {np.bincount(y_validation, minlength=2)}")
    print(f"Test labels:       {np.bincount(y_test, minlength=2)}")
    print(f"Saved files to:    {OUTPUT_DIR}")
    print("TF-IDF feature generation: PASS")


if __name__ == "__main__":
    main()
