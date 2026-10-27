"""Generate CSV, JSON, and HTML anomaly reports from the existing PostgreSQL data.

This script is read-only with respect to PostgreSQL. It does not retrain the
R1 model, alter rows, or execute ingestion/labeling/loading operations.
Run from any directory with the project's python-backend environment active.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PYTHON_BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PYTHON_BACKEND_ROOT.parent
if str(PYTHON_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_BACKEND_ROOT))

from app.db.database import get_connection  # noqa: E402
from app.services.root_cause_correlator import analyze_block  # noqa: E402
from ml.reporting.anomaly_html_report import render_html_report  # noqa: E402

MODEL_NAME = "R1_masked_tfidf_isolation_forest"
MODEL_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "isolation_forest_masked"
    / "metrics.json"
)
HELD_OUT_TEST_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "final_evaluation"
    / "isolation_forest_masked_test_metrics.json"
)
OUTPUT_DIR = PROJECT_ROOT / "reports"

FULL_REPORT_PATH = OUTPUT_DIR / "final_anomaly_report.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "predicted_anomalies_diagnostics.csv"
SUMMARY_PATH = OUTPUT_DIR / "anomaly_report_summary.json"
HTML_REPORT_PATH = OUTPUT_DIR / "anomaly_report.html"

FULL_REPORT_FIELDS = [
    "trace_id",
    "block_id",
    "actual_label",
    "predicted_label",
    "anomaly_score",
    "threshold",
    "model_name",
    "is_correct",
    "prediction_created_at",
    "combined_log_text",
]

DIAGNOSTIC_FIELDS = [
    "trace_id",
    "block_id",
    "combined_log_text",
    "actual_label",
    "predicted_label",
    "anomaly_score",
    "threshold",
    "is_correct",
    "root_cause_status",
    "diagnostic_summary",
    "anchor_selection_json",
    "anchor_event_json",
    "diagnostic_findings_json",
    "failure_evidence_json",
    "processing_evidence_json",
    "precursor_events_json",
    "preceding_context_json",
    "event_ordering",
    "method_notes_json",
    "root_cause_candidates_json",
    "candidate_evaluations_json",
    "candidate_screening_summary_json",
    "diagnostic_error",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nested_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_labeled_predictions() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch labeled traces and exactly-one R1 prediction per trace."""
    with get_connection() as conn:
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total_traces,
                COUNT(*) FILTER (WHERE label IS NOT NULL) AS labeled_traces,
                COUNT(*) FILTER (WHERE label IS NULL) AS unlabeled_traces
            FROM traces
            """
        ).fetchone()

        prediction_totals = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT trace_id)
            FROM anomaly_predictions
            WHERE model_name = %s
            """,
            (MODEL_NAME,),
        ).fetchone()

        duplicates = conn.execute(
            """
            SELECT trace_id, COUNT(*)
            FROM anomaly_predictions
            WHERE model_name = %s
            GROUP BY trace_id
            HAVING COUNT(*) > 1
            LIMIT 10
            """,
            (MODEL_NAME,),
        ).fetchall()

        rows = conn.execute(
            """
            SELECT
                t.id AS trace_id,
                t.block_id,
                t.label AS actual_label,
                t.combined_log_text,
                p.prediction AS predicted_label,
                p.anomaly_score,
                p.threshold,
                p.model_name,
                p.created_at AS prediction_created_at,
                p.id AS prediction_id
            FROM traces AS t
            LEFT JOIN anomaly_predictions AS p
                ON p.trace_id = t.id AND p.model_name = %s
            WHERE t.label IS NOT NULL
            ORDER BY t.block_id
            """,
            (MODEL_NAME,),
        ).fetchall()

    all_traces, labeled_traces, unlabeled_traces = map(int, totals)
    prediction_count, distinct_prediction_traces = map(int, prediction_totals)

    if duplicates:
        raise ValueError(
            "More than one R1 prediction exists for at least one trace: "
            f"{duplicates}"
        )
    if len(rows) != labeled_traces:
        raise ValueError(
            f"Expected {labeled_traces} labeled rows, query returned {len(rows)}."
        )
    if any(row[9] is None for row in rows):
        missing = [row[1] for row in rows if row[9] is None][:10]
        raise ValueError(f"Labeled traces missing an R1 prediction: {missing}")
    if any(row[4] not in {"Normal", "Anomaly"} for row in rows):
        raise ValueError("Unexpected or missing prediction label in database.")
    if any(row[2] not in {"Normal", "Anomaly"} for row in rows):
        raise ValueError("Unexpected actual label in labeled trace records.")

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "trace_id": int(row[0]),
                "block_id": str(row[1]),
                "actual_label": str(row[2]),
                "combined_log_text": str(row[3]),
                "predicted_label": str(row[4]),
                "anomaly_score": float(row[5]),
                "threshold": float(row[6]),
                "model_name": str(row[7]),
                "prediction_created_at": row[8].isoformat()
                if row[8] is not None
                else "",
                "prediction_id": int(row[9]),
            }
        )

    stats = {
        "all_traces": all_traces,
        "labeled_traces": labeled_traces,
        "unlabeled_traces": unlabeled_traces,
        "r1_prediction_rows_all_traces": prediction_count,
        "r1_distinct_predicted_traces_all_traces": distinct_prediction_traces,
    }
    return records, stats


def validate_threshold(records: list[dict[str, Any]]) -> float:
    if not records:
        raise ValueError("No labeled traces with predictions are available.")

    thresholds = {record["threshold"] for record in records}
    if len(thresholds) != 1:
        raise ValueError(
            "Labeled R1 predictions contain inconsistent thresholds: "
            f"{sorted(thresholds)}"
        )
    database_threshold = next(iter(thresholds))

    metrics = load_json(MODEL_METRICS_PATH)
    if metrics is None:
        raise FileNotFoundError(
            f"Required R1 threshold metadata not found: {MODEL_METRICS_PATH}"
        )
    if metrics.get("experiment") != MODEL_NAME:
        raise ValueError(
            "The saved R1 metrics file refers to a different experiment: "
            f"{metrics.get('experiment')!r}"
        )
    expected_threshold = float(metrics["anomaly_score_threshold"])
    if abs(database_threshold - expected_threshold) > 1e-9:
        raise ValueError(
            "Database prediction threshold does not match the saved R1 "
            f"validation threshold ({database_threshold} != {expected_threshold})."
        )
    return database_threshold


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    # UTF-8 with BOM improves compatibility with Excel on Windows.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_diagnostic_row(
    record: dict[str, Any], diagnostic: dict[str, Any] | None, error: str = ""
) -> dict[str, Any]:
    source = diagnostic or {}
    return {
        "trace_id": record["trace_id"],
        "block_id": record["block_id"],
        "combined_log_text": record["combined_log_text"],
        "actual_label": record["actual_label"],
        "predicted_label": record["predicted_label"],
        "anomaly_score": record["anomaly_score"],
        "threshold": record["threshold"],
        "is_correct": record["actual_label"] == record["predicted_label"],
        "root_cause_status": source.get("root_cause_status", "diagnostic_error" if error else "unknown"),
        "diagnostic_summary": source.get("summary", ""),
        "anchor_selection_json": nested_json(source.get("anchor_selection", {})),
        "anchor_event_json": nested_json(source.get("anchor_event")),
        "diagnostic_findings_json": nested_json(source.get("diagnostic_findings", [])),
        "failure_evidence_json": nested_json(source.get("failure_evidence", {})),
        "processing_evidence_json": nested_json(source.get("processing_evidence", {})),
        "precursor_events_json": nested_json(source.get("precursor_events", [])),
        "preceding_context_json": nested_json(source.get("preceding_context", [])),
        "event_ordering": source.get("event_ordering", "not_recorded"),
        "method_notes_json": nested_json(source.get("method_notes", [])),
        "root_cause_candidates_json": nested_json(source.get("root_cause_candidates", [])),
        "candidate_evaluations_json": nested_json(source.get("candidate_evaluations", [])),
        "candidate_screening_summary_json": nested_json(source.get("candidate_screening_summary", {})),
        "diagnostic_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate read-only R1 anomaly reports from PostgreSQL. "
            "Database rows and model artifacts are never modified."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite all generated report files if they already exist",
    )
    args = parser.parse_args()

    output_paths = [FULL_REPORT_PATH, DIAGNOSTICS_PATH, SUMMARY_PATH, HTML_REPORT_PATH]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing report file(s):", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        print("Use --force only when replacement is intended.", file=sys.stderr)
        return 2

    records, database_stats = load_labeled_predictions()
    threshold = validate_threshold(records)

    anomalies = [record for record in records if record["predicted_label"] == "Anomaly"]
    confusion = Counter()
    by_actual = Counter()
    by_prediction = Counter()
    full_rows: list[dict[str, Any]] = []

    for record in records:
        actual = record["actual_label"]
        predicted = record["predicted_label"]
        by_actual[actual] += 1
        by_prediction[predicted] += 1
        if actual == "Anomaly" and predicted == "Anomaly":
            confusion["true_positive"] += 1
        elif actual == "Anomaly" and predicted == "Normal":
            confusion["false_negative"] += 1
        elif actual == "Normal" and predicted == "Anomaly":
            confusion["false_positive"] += 1
        else:
            confusion["true_negative"] += 1

        full_rows.append(
            {
                "trace_id": record["trace_id"],
                "block_id": record["block_id"],
                "actual_label": actual,
                "predicted_label": predicted,
                "anomaly_score": record["anomaly_score"],
                "threshold": record["threshold"],
                "model_name": record["model_name"],
                "is_correct": actual == predicted,
                "prediction_created_at": record["prediction_created_at"],
                "combined_log_text": record["combined_log_text"],
            }
        )

    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_statuses: Counter[str] = Counter()
    diagnostic_errors: list[dict[str, str]] = []
    candidate_count_total = 0

    print(f"Validated {len(records)} labeled traces and R1 predictions.")
    print(f"Threshold matches saved validation metadata: {threshold:.10f}")
    print(f"Generating diagnostics for {len(anomalies)} predicted anomalies...")

    for index, record in enumerate(anomalies, start=1):
        try:
            report = analyze_block(record["block_id"])
            if not isinstance(report, dict):
                raise TypeError("analyze_block returned a non-dictionary result")
            diag_row = make_diagnostic_row(record, report)
            status = str(diag_row["root_cause_status"])
            diagnostic_statuses[status] += 1
            candidate_count_total += len(report.get("root_cause_candidates", []))
        except Exception as exc:  # preserve one output row for every flagged trace
            error = f"{type(exc).__name__}: {exc}"
            diag_row = make_diagnostic_row(record, None, error)
            diagnostic_statuses["diagnostic_error"] += 1
            diagnostic_errors.append(
                {"block_id": record["block_id"], "error": error}
            )
        diagnostic_rows.append(diag_row)

        if index % 25 == 0 or index == len(anomalies):
            print(f"  Diagnostics processed: {index}/{len(anomalies)}")

    test_metrics = load_json(HELD_OUT_TEST_METRICS_PATH)
    if test_metrics is not None and test_metrics.get("experiment") != MODEL_NAME:
        raise ValueError(
            "Held-out test metrics file refers to a different experiment: "
            f"{test_metrics.get('experiment')!r}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The complete report is generated only after the database data and
    # threshold have passed validation. Diagnostics errors are recorded per row.
    write_csv(FULL_REPORT_PATH, FULL_REPORT_FIELDS, full_rows)
    write_csv(DIAGNOSTICS_PATH, DIAGNOSTIC_FIELDS, diagnostic_rows)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "anomaly_score_definition": "negative Isolation Forest decision_function; higher means more anomalous",
        "threshold": threshold,
        "threshold_source": "validation set metrics file",
        "database_counts": database_stats,
        "report_scope": {
            "labeled_trace_count": len(records),
            "unlabeled_trace_count_excluded": database_stats["unlabeled_traces"],
            "predicted_anomaly_count": len(anomalies),
            "predicted_normal_count": len(records) - len(anomalies),
            "anomaly_diagnostic_rows": len(diagnostic_rows),
            "anomaly_diagnostic_errors": len(diagnostic_errors),
        },
        "actual_label_counts": dict(by_actual),
        "predicted_label_counts": dict(by_prediction),
        "full_labeled_dataset_confusion_matrix_descriptive_only": {
            "true_positive": confusion["true_positive"],
            "false_negative": confusion["false_negative"],
            "false_positive": confusion["false_positive"],
            "true_negative": confusion["true_negative"],
        },
        "full_dataset_metrics_note": (
            "These counts describe stored predictions over all labeled traces; "
            "they are not an independent held-out evaluation. See the separate "
            "held-out test metrics for official evaluation results."
        ),
        "diagnostic_status_counts": dict(diagnostic_statuses),
        "root_cause_candidates_reported_total": candidate_count_total,
        "diagnostic_errors": diagnostic_errors,
        "held_out_test_evaluation": test_metrics,
        "held_out_test_metrics_file_available": test_metrics is not None,
        "html_report_file": HTML_REPORT_PATH.name,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    html_content = render_html_report(summary, diagnostic_rows)
    with HTML_REPORT_PATH.open("w", encoding="utf-8", newline="\n") as file:
        file.write(html_content)

    print("\n=== ANOMALY REPORT GENERATION COMPLETE ===")
    print(f"Labeled traces in full report: {len(full_rows)}")
    print(f"Predicted anomalies with diagnostic rows: {len(diagnostic_rows)}")
    print(f"Diagnostic errors recorded: {len(diagnostic_errors)}")
    print("Descriptive confusion matrix:", dict(confusion))
    print(f"Full report: {FULL_REPORT_PATH}")
    print(f"Diagnostics: {DIAGNOSTICS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"HTML report: {HTML_REPORT_PATH}")
    print("PostgreSQL data and R1 model artifacts were not modified.")

    return 0 if not diagnostic_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
