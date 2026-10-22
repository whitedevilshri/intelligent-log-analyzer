
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

project_root = Path(__file__).resolve().parent.parent
dataset_path = (
    project_root
    / "data"
    / "processed"
    / "HDFS_trace_labeled_dataset.csv"
)

required_columns = {"BlockId", "CombinedLogText", "Label"}

if not dataset_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

with dataset_path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames or not required_columns.issubset(
        reader.fieldnames
    ):
        raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

    rows = list(reader)

labels = Counter(row["Label"] for row in rows)
block_ids = [row["BlockId"] for row in rows]

empty_cells = sum(
    1
    for row in rows
    for column in required_columns
    if not row[column] or not row[column].strip()
)

trace_lengths = [
    len(row["CombinedLogText"].splitlines())
    for row in rows
]

checks = {
    "Correct row count": len(rows) == 7940,
    "Unique BlockIds": len(set(block_ids)) == len(rows),
    "Expected label distribution": labels == {
        "Normal": 7627,
        "Anomaly": 313,
    },
    "No empty required fields": empty_cells == 0,
    "Every trace has log messages": all(
        length > 0 for length in trace_lengths
    ),
}

print("=== SAVED DATASET VALIDATION ===")
print(f"File: {dataset_path}")
print(f"Rows: {len(rows)}")
print(f"Unique BlockIds: {len(set(block_ids))}")
print(f"Label counts: {dict(labels)}")
print(f"Empty required fields: {empty_cells}")
print(
    "Trace lengths: "
    f"min={min(trace_lengths)}, "
    f"median={median(trace_lengths)}, "
    f"mean={mean(trace_lengths):.4f}, "
    f"max={max(trace_lengths)}"
)

print("\n=== VALIDATION CHECKS ===")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit("Dataset validation failed.")

print("\nALL CHECKS PASSED")
