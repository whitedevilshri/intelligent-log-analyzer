"""Rule-based cross-BlockId log association MVP.

Matches the same specific failure signature across different blocks within a
bounded time window. Results are associations for investigation, never proof
of a shared incident or root cause.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from app.db.database import get_connection
from app.services.root_cause_correlator import classify_exception, parse_timestamp

FAILURE_MARKERS = (
    "exception",
    "timeout",
    "timed out",
    "interrupted",
    "connection reset",
    "could not read from stream",
    "eofexception",
)
GENERIC_SIGNATURE = "other_or_unclassified_exception"


def _failure_signature(event_id: Any, message: Any) -> str | None:
    text = str(message or "")
    lowered = text.lower()
    if str(event_id or "") != "E7" and not any(marker in lowered for marker in FAILURE_MARKERS):
        return None
    signature = classify_exception(text)
    # Generic failures are too broad for a defensible cross-block association.
    if not signature or signature == GENERIC_SIGNATURE:
        return None
    return signature


def _normalise_event(event: dict[str, Any], default_block_id: str | None = None) -> dict[str, Any]:
    item = dict(event)
    item["block_id"] = str(item.get("block_id", default_block_id or ""))
    item["line_id"] = item.get("line_id")
    item["event_id"] = str(item.get("event_id") or "")
    item["component"] = str(item.get("component") or "")
    item["message"] = str(item.get("message") or "")
    ts = item.get("timestamp")
    if not isinstance(ts, datetime):
        ts = parse_timestamp(str(item.get("log_date") or ""), str(item.get("log_time") or ""))
    item["timestamp"] = ts
    item["failure_signature"] = item.get("failure_signature") or _failure_signature(
        item["event_id"], item["message"]
    )
    return item


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    timestamp = event.get("timestamp")
    return {
        "block_id": event.get("block_id"),
        "line_id": event.get("line_id"),
        "event_id": event.get("event_id"),
        "component": event.get("component"),
        "signature": event.get("failure_signature"),
        "timestamp": timestamp.isoformat(sep=" ") if isinstance(timestamp, datetime) else None,
        "message": event.get("message"),
    }


def build_cross_block_report(
    block_id: str,
    target_events: list[dict[str, Any]],
    other_events: list[dict[str, Any]],
    *,
    time_window_seconds: int = 300,
    max_candidates: int = 25,
) -> dict[str, Any]:
    """Build a report from event dictionaries; separated for offline testing."""
    if not block_id or not block_id.strip():
        raise ValueError("BlockId must not be empty.")
    if not 1 <= int(time_window_seconds) <= 3600:
        raise ValueError("time_window_seconds must be between 1 and 3600.")
    if not 1 <= int(max_candidates) <= 100:
        raise ValueError("max_candidates must be between 1 and 100.")

    target = [_normalise_event(event, block_id) for event in target_events]
    target_failure_observations = [event for event in target if event["failure_signature"]]
    target_failures = [event for event in target_failure_observations if event["timestamp"] is not None]

    base = {
        "block_id": block_id,
        "analysis_type": "cross_block_failure_signature_association",
        "time_window_seconds": int(time_window_seconds),
        "target_failure_events": [_public_event(event) for event in target_failures],
        "total_matching_blocks": 0,
        "candidates": [],
        "method_notes": [
            "Only specific, classified failure signatures are matched.",
            "Related events must belong to a different BlockId and fall within the configured absolute time window.",
            "Events with unparseable timestamps are excluded from matching.",
            "Associations are investigation clues, not proof of a shared incident or root cause.",
        ],
    }

    if not target_failure_observations:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "No specific failure signature was found in the target block.",
        }
    if not target_failures:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "Target failure events did not have parseable timestamps.",
        }

    candidates_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    window = int(time_window_seconds)
    for raw_event in other_events:
        candidate = _normalise_event(raw_event)
        candidate_block = candidate["block_id"]
        if not candidate_block or candidate_block == block_id:
            continue
        if candidate["timestamp"] is None or not candidate["failure_signature"]:
            continue

        for target_event in target_failures:
            if candidate["failure_signature"] != target_event["failure_signature"]:
                continue
            delta = (candidate["timestamp"] - target_event["timestamp"]).total_seconds()
            if abs(delta) > window:
                continue
            candidates_by_block[candidate_block].append({
                "target_event": _public_event(target_event),
                "related_event": _public_event(candidate),
                "time_delta_seconds": round(delta, 3),
                "absolute_time_gap_seconds": round(abs(delta), 3),
                "match_reason": "same_specific_failure_signature_within_time_window",
            })

    candidate_rows = []
    for related_block, pairs in candidates_by_block.items():
        pairs.sort(key=lambda pair: (pair["absolute_time_gap_seconds"], pair["related_event"]["line_id"] or 0))
        signatures = sorted({pair["related_event"]["signature"] for pair in pairs})
        candidate_rows.append({
            "related_block_id": related_block,
            "matched_event_pair_count": len(pairs),
            "nearest_time_gap_seconds": pairs[0]["absolute_time_gap_seconds"],
            "matched_signatures": signatures,
            "matched_event_pairs": pairs[:10],
            "interpretation": "Potential temporal association only; not a confirmed cause.",
        })

    candidate_rows.sort(key=lambda item: (
        item["nearest_time_gap_seconds"],
        -item["matched_event_pair_count"],
        item["related_block_id"],
    ))
    base["total_matching_blocks"] = len(candidate_rows)
    base["candidates"] = candidate_rows[:int(max_candidates)]
    if candidate_rows:
        return {**base, "status": "correlations_found", "reason": "Matching specific failure signatures were observed in other blocks within the time window."}
    return {**base, "status": "no_matching_cross_block_events", "reason": "No other block had the same specific failure signature within the time window."}


def _from_target_row(row: tuple[Any, ...], block_id: str) -> dict[str, Any]:
    line_id, log_date, log_time, component, level, event_id, template, content = row
    return {
        "block_id": block_id,
        "line_id": line_id,
        "log_date": str(log_date or ""),
        "log_time": str(log_time or ""),
        "component": component,
        "level": level,
        "event_id": event_id,
        "template": template,
        "message": str(content or ""),
    }


def _from_related_row(row: tuple[Any, ...]) -> dict[str, Any]:
    block_id, line_id, log_date, log_time, component, level, event_id, template, content = row
    return {
        "block_id": str(block_id),
        "line_id": line_id,
        "log_date": str(log_date or ""),
        "log_time": str(log_time or ""),
        "component": component,
        "level": level,
        "event_id": event_id,
        "template": template,
        "message": str(content or ""),
    }


def correlate_block(
    block_id: str,
    *,
    time_window_seconds: int = 300,
    max_candidates: int = 25,
) -> dict[str, Any]:
    """Query stored HDFS events and return cross-block signature associations."""
    if not block_id or not block_id.strip():
        raise ValueError("BlockId must not be empty.")
    if not 1 <= int(time_window_seconds) <= 3600:
        raise ValueError("time_window_seconds must be between 1 and 3600.")
    if not 1 <= int(max_candidates) <= 100:
        raise ValueError("max_candidates must be between 1 and 100.")

    with get_connection() as conn:
        target_rows = conn.execute(
            """
            SELECT e.line_id, e.log_date, e.log_time, e.component, e.level,
                   e.event_id, e.event_template, e.content
            FROM structured_log_events e
            JOIN log_event_blocks b ON b.line_id = e.line_id
            WHERE b.block_id = %s
            ORDER BY e.line_id
            """,
            (block_id,),
        ).fetchall()

        if not target_rows:
            raise ValueError(f"No structured log events found for BlockId: {block_id}")

        target_events = [_from_target_row(row, block_id) for row in target_rows]
        prepared_targets = [_normalise_event(event, block_id) for event in target_events]
        valid_failure_times = [
            event["timestamp"] for event in prepared_targets
            if event["failure_signature"] and event["timestamp"] is not None
        ]
        if not valid_failure_times:
            return build_cross_block_report(
                block_id, target_events, [],
                time_window_seconds=time_window_seconds,
                max_candidates=max_candidates,
            )

        # Include adjacent calendar dates so a time window spanning midnight is covered.
        log_dates = sorted({
            (timestamp + timedelta(days=offset)).strftime("%m%d%y")
            for timestamp in valid_failure_times
            for offset in (-1, 0, 1)
        })
        related_rows = conn.execute(
            """
            SELECT b.block_id, e.line_id, e.log_date, e.log_time, e.component,
                   e.level, e.event_id, e.event_template, e.content
            FROM structured_log_events e
            JOIN log_event_blocks b ON b.line_id = e.line_id
            WHERE e.log_date = ANY(%s)
              AND b.block_id <> %s
              AND (
                  e.event_id = 'E7'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%exception%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%timeout%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%timed out%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%interrupted%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%connection reset%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%could not read from stream%%'
                  OR LOWER(COALESCE(e.content, '')) LIKE '%%eofexception%%'
              )
            ORDER BY e.log_date, e.log_time, e.line_id
            """,
            (log_dates, block_id),
        ).fetchall()

    related_events = [_from_related_row(row) for row in related_rows]
    return build_cross_block_report(
        block_id,
        target_events,
        related_events,
        time_window_seconds=time_window_seconds,
        max_candidates=max_candidates,
    )
