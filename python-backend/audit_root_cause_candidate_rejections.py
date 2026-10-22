"""Audit why the root-cause correlator rejects potential candidates.

Uses validation.csv to inspect development behavior and train.csv-derived event
profiles for association statistics. It does not read test.csv, alter the model,
or change correlator thresholds. The output is diagnostic, not root-cause accuracy.

Run from python-backend using that project's virtual environment:
  python audit_root_cause_candidate_rejections.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from app.services.root_cause_correlator import (
    MAX_TIME_GAP_SECONDS,
    MIN_ANOMALY_SUPPORT,
    MIN_PREVALENCE_DIFFERENCE,
    analyze_block,
    load_training_event_profile,
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"
DETAILS_PATH = BACKEND_DIR / "root_cause_candidate_rejection_details.csv"

DEFAULT_STATS = {
    "anomaly_blocks": 0,
    "normal_blocks": 0,
    "anomaly_rate": 0.0,
    "normal_rate": 0.0,
    "prevalence_difference": 0.0,
}


def load_validation_rows():
    if not VALIDATION_PATH.is_file():
        raise FileNotFoundError(f"Validation split not found: {VALIDATION_PATH}")

    with VALIDATION_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("validation.csv has no header row")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        block_col = normalized.get("blockid")
        label_col = normalized.get("label")
        if not block_col or not label_col:
            raise ValueError(
                f"Expected BlockId and Label columns; found {reader.fieldnames}"
            )

        rows = []
        for row in reader:
            block_id = (row.get(block_col) or "").strip()
            label = (row.get(label_col) or "").strip().lower()
            if block_id and label in {"normal", "anomaly"}:
                rows.append((block_id, label))
        return rows


def rejection_reason(context, anchor, stats):
    """Apply the same ordered rejection checks used by candidate generation."""
    if context.get("event_id") == anchor.get("event_id"):
        return "same_event_as_anchor"

    gap = context.get("seconds_before_anchor")
    if gap is not None and (gap < 0 or gap > MAX_TIME_GAP_SECONDS):
        return "outside_time_window"

    if stats["anomaly_blocks"] < MIN_ANOMALY_SUPPORT:
        return "below_minimum_anomaly_support"

    if stats["prevalence_difference"] < MIN_PREVALENCE_DIFFERENCE:
        return "below_minimum_prevalence_difference"

    return "eligible_under_current_rules"


def pct(n, total):
    return f"{n / total * 100:.2f}%" if total else "n/a"


def main():
    validation_rows = load_validation_rows()
    profile = load_training_event_profile()

    label_totals = Counter(label for _, label in validation_rows)
    anchored_anomaly_blocks = []
    reports_completed = 0
    errors = []
    status_counts = Counter()

    for index, (block_id, label) in enumerate(validation_rows, start=1):
        try:
            report = analyze_block(block_id)
        except Exception as exc:  # Make failures visible; do not hide them.
            errors.append((block_id, label, type(exc).__name__, str(exc)))
            continue

        reports_completed += 1
        status_counts[str(report.get("root_cause_status", "unknown"))] += 1

        if label == "anomaly" and report.get("anchor_event"):
            anchored_anomaly_blocks.append((block_id, report))

        if index % 200 == 0:
            print(f"Processed {index}/{len(validation_rows)} validation traces...")

    reason_counts = Counter()
    per_event = defaultdict(lambda: {
        "instances": 0,
        "blocks": set(),
        "reasons": Counter(),
        "anomaly_support": None,
        "normal_support": None,
        "prevalence_difference": None,
    })
    detail_rows = []
    potentially_eligible_contexts = []

    for block_id, report in anchored_anomaly_blocks:
        anchor = report["anchor_event"]
        for context in report.get("preceding_context", []):
            event_id = context.get("event_id")
            stats = profile.get(event_id, DEFAULT_STATS)
            reason = rejection_reason(context, anchor, stats)
            reason_counts[reason] += 1

            event_bucket = per_event[event_id]
            event_bucket["instances"] += 1
            event_bucket["blocks"].add(block_id)
            event_bucket["reasons"][reason] += 1
            event_bucket["anomaly_support"] = stats["anomaly_blocks"]
            event_bucket["normal_support"] = stats["normal_blocks"]
            event_bucket["prevalence_difference"] = stats["prevalence_difference"]

            if reason == "eligible_under_current_rules":
                potentially_eligible_contexts.append((block_id, context, stats))

            detail_rows.append({
                "block_id": block_id,
                "anchor_line_id": anchor.get("line_id"),
                "anchor_event_id": anchor.get("event_id"),
                "context_line_id": context.get("line_id"),
                "context_event_id": event_id,
                "seconds_before_anchor": context.get("seconds_before_anchor"),
                "training_anomaly_blocks": stats["anomaly_blocks"],
                "training_normal_blocks": stats["normal_blocks"],
                "training_anomaly_rate": round(stats["anomaly_rate"], 6),
                "training_normal_rate": round(stats["normal_rate"], 6),
                "prevalence_difference": round(stats["prevalence_difference"], 6),
                "minimum_anomaly_support": MIN_ANOMALY_SUPPORT,
                "minimum_prevalence_difference": MIN_PREVALENCE_DIFFERENCE,
                "rejection_reason": reason,
                "message": context.get("message", ""),
            })

    with DETAILS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "block_id", "anchor_line_id", "anchor_event_id", "context_line_id",
            "context_event_id", "seconds_before_anchor", "training_anomaly_blocks",
            "training_normal_blocks", "training_anomaly_rate", "training_normal_rate",
            "prevalence_difference", "minimum_anomaly_support",
            "minimum_prevalence_difference", "rejection_reason", "message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\nROOT-CAUSE CANDIDATE REJECTION AUDIT")
    print(f"Validation split: {VALIDATION_PATH}")
    print(f"Reports completed: {reports_completed}/{len(validation_rows)}")
    print(f"Ground-truth anomaly traces: {label_totals['anomaly']}")
    print(f"Anomaly traces with anchor: {len(anchored_anomaly_blocks)} "
          f"({pct(len(anchored_anomaly_blocks), label_totals['anomaly'])})")
    print(f"Errors: {len(errors)}")
    print(f"Current candidate thresholds: anomaly_support >= {MIN_ANOMALY_SUPPORT}; "
          f"prevalence_difference >= {MIN_PREVALENCE_DIFFERENCE:.4f}; "
          f"time gap <= {MAX_TIME_GAP_SECONDS}s")

    print("\nRejection reasons for preceding-context event instances in anchored ANOMALY traces:")
    total_contexts = sum(reason_counts.values())
    if not total_contexts:
        print("  No preceding-context events were available to audit.")
    else:
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}/{total_contexts} ({pct(count, total_contexts)})")

    print("\nPer-event breakdown (events with most context instances):")
    if not per_event:
        print("  No events to report.")
    else:
        for event_id, data in sorted(
            per_event.items(), key=lambda pair: (-pair[1]["instances"], str(pair[0]))
        )[:20]:
            print(
                f"  {event_id}: instances={data['instances']}, "
                f"anchored_blocks={len(data['blocks'])}, "
                f"train_anomaly_blocks={data['anomaly_support']}, "
                f"train_normal_blocks={data['normal_support']}, "
                f"prevalence_diff={data['prevalence_difference']:.4f}, "
                f"rejections={dict(data['reasons'])}"
            )

    print("\nEligible-context check:")
    print(f"  Context events passing all current candidate filters: "
          f"{len(potentially_eligible_contexts)}")
    print("  These should align with candidate-generation rules. If nonzero but the API "
          "returned no candidates, inspect ordering/data consistency.")

    print(f"\nDetailed audit CSV written to: {DETAILS_PATH}")
    print("Interpretation: rejection counts describe heuristic filters, not root-cause accuracy.")

    if errors:
        print("\nFirst errors:")
        for block_id, label, error_type, message in errors[:10]:
            print(f"  {block_id} ({label}): {error_type}: {message}")


if __name__ == "__main__":
    main()
