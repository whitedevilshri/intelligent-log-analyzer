
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "features" / "tfidf_masked"
)

LABEL_MAP = {"Normal": 0, "Anomaly": 1}

# Keep the masking rules consistent with the earlier experiment.
BLOCK_ID_RE = re.compile(r"blk_-?\d+")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUMBER_RE = re.compile(r"\b\d+\b")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def mask_text(text: str) -> tuple[str, Counter]:
    counts = Counter()

    text, counts["block_ids"] = BLOCK_ID_RE.subn("BLOCK_ID", text)
    text, counts["ip_addresses"] = IPV4_RE.subn("IP_ADDR", text)
    text, counts["numbers"] = NUMBER_RE.subn("NUMBER", text)

    return text, counts


def preprocess_text(text: str) -> str:
    # Lowercase, then replace punctuation with spaces.
    text = text.lower()
    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation),
        )
    )

    tokens = word_tokenize(text)
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS
    ]
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens)


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

    texts = []
    mask_counts = Counter()
    examples = []

    for row in rows:
        original = row["CombinedLogText"]

        if not original.strip():
            raise ValueError(f"Empty trace text in {path}")

        masked, counts = mask_text(original)
        processed = preprocess_text(masked)

        if not processed.strip():
            raise ValueError(
                f"Trace became empty after preprocessing: {row['BlockId']}"
            )

        texts.append(processed)
        mask_counts.update(counts)

        if len(examples) < 2:
            examples.append((original, masked, processed))

    unknown_labels = {row["Label"] for row in rows} - set(LABEL_MAP)
    if unknown_labels:
        raise ValueError(f"Unexpected labels: {unknown_labels}")

    labels = np.array(
        [LABEL_MAP[row["Label"]] for row in rows],
        dtype=np.int8,
    )

    return texts, labels, mask_counts, examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_texts, y_train, train_masks, examples = load_split("train.csv")
    val_texts, y_val, val_masks, _ = load_split("validation.csv")
    test_texts, y_test, test_masks, _ = load_split("test.csv")

    print("=== PREPROCESSING EXAMPLES ===")
    for index, (original, masked, processed) in enumerate(examples, 1):
        print(f"\n--- Example {index} ---")
        print("ORIGINAL:", original[:500])
        print("MASKED:", masked[:500])
        print("PROCESSED:", processed[:500])

    # Fit on training text only to avoid vocabulary leakage.
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)

    # Reuse the fitted vectorizer without refitting.
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    for filename, matrix in [
        ("X_train.npz", X_train),
        ("X_validation.npz", X_val),
        ("X_test.npz", X_test),
    ]:
        save_npz(OUTPUT_DIR / filename, matrix)

    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_validation.npy", y_val)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")

    metadata = {
        "experiment": "R1_masked_tfidf",
        "masking": {
            "block_ids": "BLOCK_ID",
            "ipv4_addresses": "IP_ADDR",
            "numbers": "NUMBER",
        },
        "text_preprocessing": [
            "lowercase",
            "remove punctuation",
            "NLTK word tokenization",
            "English stopword removal",
            "WordNet lemmatization",
        ],
        "fitted_on": "train.csv only",
        "label_mapping": LABEL_MAP,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "matrix_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_val.shape),
            "test": list(X_test.shape),
        },
        "mask_counts": {
            "train": dict(train_masks),
            "validation": dict(val_masks),
            "test": dict(test_masks),
        },
    }

    with (OUTPUT_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n=== R1 MASKED TF-IDF REPORT ===")
    print(f"Training matrix:   {X_train.shape}")
    print(f"Validation matrix: {X_val.shape}")
    print(f"Test matrix:       {X_test.shape}")
    print(f"Vocabulary size:   {len(vectorizer.vocabulary_)}")
    print(f"Training labels:   {np.bincount(y_train, minlength=2)}")
    print(f"Validation labels: {np.bincount(y_val, minlength=2)}")
    print(f"Test labels:       {np.bincount(y_test, minlength=2)}")
    print(f"Mask counts (train): {dict(train_masks)}")
    print(f"Saved files to: {OUTPUT_DIR}")
    print("Feature generation completed.")


if __name__ == "__main__":
    main()
