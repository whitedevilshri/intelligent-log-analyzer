"""Evaluate root-cause diagnostic coverage on validation.csv only.

Run this script from the python-backend directory using that project's venv.
It measures evidence coverage, NOT root-cause accuracy (there are no ground-truth
root-cause labels).
"""

import csv
from collections import defaultdict
from pathlib import Path

from app.services.root_cause_correlator import analyze_block

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"


def pct(part: int, total: int) -> str:
    return f"{(part / total * 100):.2f}%" if total else "n/a"


def main() -> None:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VALIDATION_PATH}")

    metrics = defaultdict(lambda: {
        "total": 0,
        "anchor": 0,
        "diagnostic_findings": 0,
        "failure_events": 0,
        "precursor_events": 0,
        "root_cause_candidates": 0,
        "errors": 0,
    })
    status_counts = defaultdict(int)
    processed = 0

    with VALIDATION_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("validation.csv has no header row")
        column_map = {name.strip().lower(): name for name in reader.fieldnames}
        block_col = column_map.get("blockid")
        label_col = column_map.get("label")
        if not block_col or not label_col:
            raise ValueError(
                f"Expected BlockId and Label columns; found: {reader.fieldnames}"
            )

        for row in reader:
            block_id = (row.get(block_col) or "").strip()
            label = (row.get(label_col) or "").strip().lower()
            if not block_id or label not in {"normal", "anomaly"}:
                continue

            bucket = metrics[label]
            bucket["total"] += 1
            try:
                report = analyze_block(block_id)
            except ValueError as exc:
                bucket["errors"] += 1
                status_counts[f"ERROR: {exc}"] += 1
                continue

            processed += 1
            status_counts[str(report.get("root_cause_status", "unknown"))] += 1
            if report.get("anchor_event"):
                bucket["anchor"] += 1
            if report.get("diagnostic_findings"):
                bucket["diagnostic_findings"] += 1
            failure_events = (report.get("failure_evidence") or {}).get("failure_events") or []
            if failure_events:
                bucket["failure_events"] += 1
            if report.get("precursor_events"):
                bucket["precursor_events"] += 1
            bucket["root_cause_candidates"] += len(report.get("root_cause_candidates") or [])

            if processed % 200 == 0:
                print(f"Processed {processed} validation traces...")

    print("\nROOT-CAUSE CORRELATOR — VALIDATION COVERAGE")
    print(f"Validation file: {VALIDATION_PATH}")
    print(f"Reports completed: {processed}")
    print("\nCoverage by ground-truth anomaly label (not root-cause accuracy):")

    for label in ("anomaly", "normal"):
        bucket = metrics[label]
        total = bucket["total"]
        print(f"\n{label.upper()} traces: {total}")
        for key, title in (
            ("anchor", "Traces with anchor"),
            ("diagnostic_findings", "Traces with diagnostic findings"),
            ("failure_events", "Traces with failure evidence"),
            ("precursor_events", "Traces with precursor events"),
        ):
            print(f"  {title}: {bucket[key]}/{total} ({pct(bucket[key], total)})")
        print(f"  Root-cause candidates returned (total): {bucket['root_cause_candidates']}")
        print(f"  Block analysis errors: {bucket['errors']}")

    print("\nRoot-cause status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nInterpretation note: these are coverage diagnostics only. Without ground-truth\n"
          "root-cause labels, they do not establish root-cause accuracy or causality.")


if __name__ == "__main__":
    main()
