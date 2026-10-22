
import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "HDFS_trace_labeled_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_STATE = 42


def main():
    with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required = {"BlockId", "CombinedLogText", "Label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Unexpected input CSV columns.")

        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")

    block_ids = [row["BlockId"] for row in rows]
    labels = [row["Label"] for row in rows]

    if len(block_ids) != len(set(block_ids)):
        raise ValueError("Duplicate BlockIds found.")

    if set(labels) != {"Normal", "Anomaly"}:
        raise ValueError(f"Unexpected labels: {set(labels)}")

    # Outer split: reserve 20% for the final test set.
    development, test = train_test_split(
        rows,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Inner split: validation comes only from development data.
    development_labels = [row["Label"] for row in development]

    train, validation = train_test_split(
        development,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=development_labels,
    )

    # Confirm that no BlockId appears in multiple splits.
    train_ids = {row["BlockId"] for row in train}
    validation_ids = {row["BlockId"] for row in validation}
    test_ids = {row["BlockId"] for row in test}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_rows in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        output_file = OUTPUT_DIR / f"{name}.csv"

        with output_file.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

        counts = Counter(row["Label"] for row in split_rows)

        print(f"\n{name.upper()}")
        print(f"Total: {len(split_rows)}")
        print(f"Normal: {counts['Normal']}")
        print(f"Anomaly: {counts['Anomaly']}")
        print(f"Saved: {output_file}")

    print("\nBlockId overlap checks: PASS")
    print("Dataset splitting: PASS")


if __name__ == "__main__":
    main()
