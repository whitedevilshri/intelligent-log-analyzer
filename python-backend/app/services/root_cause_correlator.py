import csv
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.db.database import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"

# Minimum evidence needed to rank an event as a candidate precursor.
MIN_ANOMALY_SUPPORT = 20
MIN_PREVALENCE_DIFFERENCE = 0.02

# Explore at most 10 preceding events within a 5-minute window.
MAX_PRECEDING_EVENTS = 10
MAX_TIME_GAP_SECONDS = 300

# Events used to describe activity following an E7 exception.
FOLLOW_UP_EVENT_IDS = {"E9", "E26", "E11"}

# Expected processing markers used for diagnostic context only.
REFERENCE_MARKER_IDS = {"E9", "E26", "E11"}


# Parse timestamps from the HDFS structured dataset.
def parse_timestamp(log_date: str, log_time: str):
    try:
        return datetime.strptime(log_date + log_time, "%m%d%y%H%M%S")
    except (ValueError, TypeError):
        return None


def classify_exception(message: str) -> str:
    """Classify a failure signature found in a log message."""
    message_lower = message.lower()

    if "could not read from stream" in message_lower:
        return "stream_read_failure"
    if "connection reset by peer" in message_lower:
        return "connection_reset"
    if "eofexception" in message_lower:
        return "unexpected_end_of_stream"
    if "sockettimeoutexception" in message_lower:
        return "socket_timeout"
    if "interruptedioexception" in message_lower:
        return "interrupted_io"
    if "interrupted" in message_lower:
        return "interrupted"
    if "exception" in message_lower or "error" in message_lower:
        return "other_or_unclassified_exception"

    return "other_or_unclassified_exception"


@lru_cache(maxsize=1)
def load_training_event_profile():
    """Calculate per-event prevalence using training labels only.

    Rates are the percentage of traces in each class that contain
    the event at least once. Validation and test labels are not used.
    """
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_PATH}")

    with TRAIN_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"BlockId", "Label"}
    if not rows or not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            "Training split must contain BlockId and Label columns."
        )

    labels = {row["BlockId"]: row["Label"] for row in rows}
    anomaly_total = sum(label == "Anomaly" for label in labels.values())
    normal_total = sum(label == "Normal" for label in labels.values())

    if anomaly_total == 0 or normal_total == 0:
        raise ValueError("Training split must contain both classes.")

    with get_connection() as conn:
        event_rows = conn.execute(
            """
            SELECT DISTINCT b.block_id, e.event_id
            FROM log_event_blocks b
            JOIN structured_log_events e ON e.line_id = b.line_id
            WHERE b.block_id = ANY(%s)
            """,
            (list(labels.keys()),),
        ).fetchall()

    # Count each BlockId at most once per event type.
    event_blocks = defaultdict(lambda: {"Anomaly": set(), "Normal": set()})

    for block_id, event_id in event_rows:
        label = labels.get(block_id)
        if label in {"Anomaly", "Normal"}:
            event_blocks[event_id][label].add(block_id)

    profile = {}
    for event_id, groups in event_blocks.items():
        anomaly_count = len(groups["Anomaly"])
        normal_count = len(groups["Normal"])
        anomaly_rate = anomaly_count / anomaly_total
        normal_rate = normal_count / normal_total

        profile[event_id] = {
            "anomaly_blocks": anomaly_count,
            "normal_blocks": normal_count,
            "anomaly_rate": anomaly_rate,
            "normal_rate": normal_rate,
            "prevalence_difference": anomaly_rate - normal_rate,
        }

    return profile


def analyze_block(block_id: str) -> dict:
    """Produce an explainable heuristic report for a dataset BlockId.

    Failure evidence, processing-marker observations, precursor events,
    and root-cause candidates are kept separate. None proves causation.
    """
    if not block_id or not block_id.strip():
        raise ValueError("BlockId must not be empty.")

    profile = load_training_event_profile()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                e.line_id,
                e.log_date,
                e.log_time,
                e.component,
                e.level,
                e.event_id,
                e.event_template,
                e.content
            FROM structured_log_events e
            JOIN log_event_blocks b ON b.line_id = e.line_id
            WHERE b.block_id = %s
            ORDER BY e.line_id
            """,
            (block_id,),
        ).fetchall()

    if not rows:
        raise ValueError(f"No structured log events found for BlockId: {block_id}")

    events = []
    for row in rows:
        (
            line_id,
            log_date,
            log_time,
            component,
            level,
            event_id,
            template,
            content,
        ) = row

        events.append(
            {
                "line_id": line_id,
                "log_date": log_date,
                "log_time": log_time,
                "timestamp": parse_timestamp(log_date, log_time),
                "component": component,
                "level": level,
                "event_id": event_id,
                "template": template,
                "message": content,
            }
        )

    # ---------------------------------------------------------
    # 1. Describe expected processing markers in this trace.
    # ---------------------------------------------------------

    # Describe expected processing markers across the trace
    # and specifically after the final E7 exception.
    reference_markers = {"E9", "E26", "E11"}

    observed_event_ids = {
        event["event_id"] for event in events
    }

    observed_markers = sorted(
        reference_markers & observed_event_ids
    )

    missing_markers = sorted(
        reference_markers - observed_event_ids
    )

    e7_events_for_processing = [
        event for event in events
        if event["event_id"] == "E7"
    ]

    last_e7_line_id = (
        e7_events_for_processing[-1]["line_id"]
        if e7_events_for_processing
        else None
    )

    if last_e7_line_id is not None:
        observed_after_last_e7 = sorted({
            event["event_id"]
            for event in events
            if (
                event["line_id"] > last_e7_line_id
                and event["event_id"] in reference_markers
            )
        })

        missing_after_last_e7 = sorted(
            reference_markers - set(observed_after_last_e7)
        )

        after_last_e7_status = (
            "all_reference_markers_observed_after_last_E7"
            if not missing_after_last_e7
            else "reference_markers_missing_after_last_E7"
        )
    else:
        observed_after_last_e7 = []
        missing_after_last_e7 = []
        after_last_e7_status = "no_E7_event_in_trace"

    processing_evidence = {
        "reference_markers": sorted(reference_markers),
        "observed_markers_anywhere": observed_markers,
        "missing_markers_anywhere": missing_markers,
        "last_e7_line_id": last_e7_line_id,
        "observed_after_last_e7": observed_after_last_e7,
        "missing_after_last_e7": missing_after_last_e7,
        "sequence_status": after_last_e7_status,
        "interpretation": (
            "Marker presence or absence is a dataset-derived signal. "
            "Events after an exception do not prove recovery, and "
            "missing events do not prove the underlying cause."
        ),
    }


    # ---------------------------------------------------------
    # 2. Describe observed exception and timeout evidence.
    # ---------------------------------------------------------
    e7_events = [event for event in events if event["event_id"] == "E7"]

    failure_markers = (
        "exception",
        "timeout",
        "timed out",
        "interrupted",
        "connection reset",
        "could not read from stream",
        "eofexception",
    )

    failure_events = []
    for event in events:
        message_lower = event["message"].lower()
        if not any(marker in message_lower for marker in failure_markers):
            continue

        failure_events.append(
            {
                "line_id": event["line_id"],
                "log_time": event["log_time"],
                "event_id": event["event_id"],
                "component": event["component"],
                "signature": classify_exception(event["message"]),
                "message": event["message"],
            }
        )

    if e7_events:
        # The final E7 in LineId order is used for follow-up inspection.
        # This is sequence-based, not proof of causal order.
        last_e7 = e7_events[-1]
        follow_up_events = [
            event
            for event in events
            if (
                event["line_id"] > last_e7["line_id"]
                and event["event_id"] in FOLLOW_UP_EVENT_IDS
            )
        ]
        follow_up_event_ids = sorted(
            {event["event_id"] for event in follow_up_events}
        )
        sequence_observation = (
            "selected_follow_up_events_observed"
            if follow_up_events
            else "no_selected_follow_up_events_observed"
        )
    else:
        last_e7 = None
        follow_up_events = []
        follow_up_event_ids = []
        sequence_observation = "no_E7_event_in_trace"

    failure_evidence = {
        "failure_events": failure_events,
        "last_e7_line_id": last_e7["line_id"] if last_e7 else None,
        "follow_up_event_ids": follow_up_event_ids,
        "follow_up_events": [
            {
                "line_id": event["line_id"],
                "log_time": event["log_time"],
                "event_id": event["event_id"],
                "component": event["component"],
                "message": event["message"],
            }
            for event in follow_up_events
        ],
        "sequence_observation": sequence_observation,
        "interpretation": (
            "Observed log evidence only. Follow-up events do not prove "
            "recovery, and missing events do not prove the underlying cause."
        ),
    }

    # Build diagnostic findings separately from root-cause candidates.
    diagnostic_findings = []

    if failure_evidence["failure_events"]:
        diagnostic_findings.append({
            "code": "observed_exceptions",
            "message": (
                "Exception or timeout events were observed "
                "in this trace."
            ),
            "line_ids": [
                event["line_id"]
                for event in failure_evidence["failure_events"]
            ],
            "signatures": sorted({
                event["signature"]
                for event in failure_evidence["failure_events"]
            }),
        })

    if processing_evidence["missing_markers_anywhere"]:
        diagnostic_findings.append({
            "code": "missing_reference_markers",
            "message": (
                "One or more reference processing events "
                "were not found in this trace."
            ),
            "missing_markers": (
                processing_evidence["missing_markers_anywhere"]
            ),
            "interpretation": (
                "This is a dataset-derived warning, not proof "
                "of a failed operation or its underlying cause."
            ),
        })

    if failure_evidence["follow_up_event_ids"]:
        diagnostic_findings.append({
            "code": "post_exception_activity_observed",
            "message": (
                "Selected processing events were recorded "
                "after the final E7 exception."
            ),
            "event_ids": failure_evidence["follow_up_event_ids"],
            "interpretation": (
                "Subsequent activity does not by itself prove "
                "that the operation recovered successfully."
            ),
        })


    def event_stats(event):
        return profile.get(
            event["event_id"],
            {
                "anomaly_blocks": 0,
                "normal_blocks": 0,
                "anomaly_rate": 0.0,
                "normal_rate": 0.0,
                "prevalence_difference": 0.0,
            },
        )

    # ---------------------------------------------------------
    # 3. Select a discriminative anomaly anchor.
    # ---------------------------------------------------------
    eligible_indices = [
        index
        for index, event in enumerate(events)
        if (
            event_stats(event)["anomaly_blocks"] >= MIN_ANOMALY_SUPPORT
            and event_stats(event)["prevalence_difference"]
            >= MIN_PREVALENCE_DIFFERENCE
        )
    ]

    if not eligible_indices:
        return {
            "block_id": block_id,
            "root_cause_status": "insufficient_evidence",
            "summary": (
                "No event in this trace meets the minimum training-evidence "
                "requirements for an anomaly anchor."
            ),
            "anchor_event": None,
            "diagnostic_findings": diagnostic_findings,
            "failure_evidence": failure_evidence,
            "processing_evidence": processing_evidence,
            "precursor_events": [],
            "root_cause_candidates": [],
            "preceding_context": [],
            "method_notes": [
                "Event association is calculated using train.csv only.",
                "Common events are not selected as anomaly anchors.",
                "Observed failure signatures are not proof of an underlying cause.",
                "Processing-marker presence or absence is diagnostic evidence only.",
                "No root cause is inferred when the evidence is insufficient.",
            ],
        }

    # Pick the strongest eligible event according to training prevalence.
    anchor_index = max(
        eligible_indices,
        key=lambda index: (
            event_stats(events[index])["prevalence_difference"],
            event_stats(events[index])["anomaly_blocks"],
            index,
        ),
    )

    anchor = events[anchor_index]
    anchor_stats = event_stats(anchor)
    anchor_time = anchor["timestamp"]

    # ---------------------------------------------------------
    # 4. Collect preceding events from the same BlockId.
    # ---------------------------------------------------------
    start = max(0, anchor_index - MAX_PRECEDING_EVENTS)
    preceding = events[start:anchor_index]

    # Keep earlier error-like events visible even if they don't
    # meet historical support requirements for candidate ranking.
    precursor_events = []
    failure_text_markers = (
        "exception",
        "timeout",
        "timed out",
        "interrupted",
        "connection reset",
        "could not read from stream",
        "eofexception",
    )

    for event in preceding:
        # Repeated instances of the anchor event are already captured
        # as failure evidence, not independent upstream precursors.
        if event["event_id"] == anchor["event_id"]:
            continue

        message_lower = event["message"].lower()
        if not any(marker in message_lower for marker in failure_text_markers):
            continue

        timestamp = event["timestamp"]
        gap_seconds = None

        if timestamp is not None and anchor_time is not None:
            gap_seconds = (anchor_time - timestamp).total_seconds()
            if gap_seconds < 0 or gap_seconds > MAX_TIME_GAP_SECONDS:
                continue

        stats = event_stats(event)
        precursor_events.append(
            {
                "line_id": event["line_id"],
                "event_id": event["event_id"],
                "component": event["component"],
                "signature": classify_exception(event["message"]),
                "message": event["message"],
                "seconds_before_anchor": gap_seconds,
                "training_anomaly_blocks": stats["anomaly_blocks"],
                "training_normal_blocks": stats["normal_blocks"],
                "meets_candidate_support": (
                    stats["anomaly_blocks"] >= MIN_ANOMALY_SUPPORT
                    and stats["prevalence_difference"]
                    >= MIN_PREVALENCE_DIFFERENCE
                ),
            }
        )

    precursor_events.sort(
        key=lambda item: (
            item["seconds_before_anchor"]
            if item["seconds_before_anchor"] is not None
            else float("inf"),
            item["line_id"],
        )
    )

    candidates = []

    for event in preceding:
        # Repeated instances of the anchor event are not candidates.
        if event["event_id"] == anchor["event_id"]:
            continue

        stats = event_stats(event)
        timestamp = event["timestamp"]
        gap_seconds = None

        if timestamp is not None and anchor_time is not None:
            gap_seconds = (anchor_time - timestamp).total_seconds()
            if gap_seconds < 0 or gap_seconds > MAX_TIME_GAP_SECONDS:
                continue

        # Require stronger support to avoid promoting rare patterns.
        if (
            stats["anomaly_blocks"] < MIN_ANOMALY_SUPPORT
            or stats["prevalence_difference"] < MIN_PREVALENCE_DIFFERENCE
        ):
            continue

        support_factor = min(1.0, stats["anomaly_blocks"] / 10.0)
        temporal_factor = (
            1.0 / (1.0 + gap_seconds / 30.0)
            if gap_seconds is not None
            else 0.5
        )
        ranking_score = (
            stats["prevalence_difference"]
            * support_factor
            * temporal_factor
        )

        candidates.append(
            {
                "line_id": event["line_id"],
                "event_id": event["event_id"],
                "component": event["component"],
                "template": event["template"],
                "message": event["message"],
                "seconds_before_anchor": gap_seconds,
                "training_anomaly_blocks": stats["anomaly_blocks"],
                "training_normal_blocks": stats["normal_blocks"],
                "prevalence_difference": round(
                    stats["prevalence_difference"], 4
                ),
                "ranking_score": round(ranking_score, 6),
            }
        )

    candidates.sort(
        key=lambda candidate: candidate["ranking_score"],
        reverse=True,
    )

    candidate_line_ids = {candidate["line_id"] for candidate in candidates}

    context_events = []
    for event in preceding:
        stats = event_stats(event)
        gap_seconds = None

        if event["timestamp"] is not None and anchor_time is not None:
            gap_seconds = (anchor_time - event["timestamp"]).total_seconds()

        context_events.append(
            {
                "line_id": event["line_id"],
                "event_id": event["event_id"],
                "component": event["component"],
                "message": event["message"],
                "seconds_before_anchor": gap_seconds,
                "training_prevalence_difference": round(
                    stats["prevalence_difference"], 4
                ),
                "role": (
                    "ranked_candidate"
                    if event["line_id"] in candidate_line_ids
                    else "context_only"
                ),
            }
        )

    if candidates:
        status = "candidate_found"
        summary = (
            "Potential precursor events were ranked using training association "
            "and temporal proximity. They are not proven causes."
        )
    else:
        status = "insufficient_evidence"
        summary = (
            "No preceding event met the minimum evidence requirements. "
            "The root cause remains undetermined from this trace."
        )

    return {
        "block_id": block_id,
        "root_cause_status": status,
        "summary": summary,
        "anchor_event": {
            "line_id": anchor["line_id"],
            "event_id": anchor["event_id"],
            "component": anchor["component"],
            "template": anchor["template"],
            "message": anchor["message"],
            "training_anomaly_blocks": anchor_stats["anomaly_blocks"],
            "training_normal_blocks": anchor_stats["normal_blocks"],
            "training_anomaly_rate": round(anchor_stats["anomaly_rate"], 4),
            "training_normal_rate": round(anchor_stats["normal_rate"], 4),
        },
        "failure_evidence": failure_evidence,
        "diagnostic_findings": diagnostic_findings,
        "processing_evidence": processing_evidence,
        "precursor_events": precursor_events,
        "root_cause_candidates": candidates,
        "preceding_context": context_events,
        "method_notes": [
            "Training labels from train.csv only are used to calculate event association.",
            "Only preceding events from the same BlockId are considered for candidate ranking.",
            "Common events and one-off training associations are not promoted to root-cause candidates.",
            "Failure signatures and follow-up events describe observed log evidence, not proof of causation or recovery.",
            "Processing-marker presence or absence is diagnostic evidence only.",
            "Ranking scores are heuristics, not probabilities or proof of causation.",
            "This initial version does not yet correlate events across different BlockIds or components.",
        ],
    }
