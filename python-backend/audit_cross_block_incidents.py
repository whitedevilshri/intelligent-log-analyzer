"""Offline cross-block incident-window audit for Intelligent Log Analyzer.

Reads only BlockIds listed in train.csv and validation.csv. It does not read
 test.csv, modify PostgreSQL, change the R1 model, or change the API.

Run from python-backend with its virtual environment:
    python audit_cross_block_incidents.py

Outputs are written to python-backend/incident_audit_reports/:
  - incident_episodes.csv: E7 episodes under 30/60/120-second gap rules
  - incident_window_events.csv: cross-block events before episodes and in
    eligible E7-free control windows
  - incident_pattern_summary.csv: descriptive counts for event IDs,
    component/event pairs, exception signatures, and IP references
  - incident_audit_summary.txt: run metadata and limitations

This is exploratory diagnostics, not root-cause accuracy or proof of causality.
"""

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.db.database import get_connection

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"
OUTPUT_DIR = BACKEND_DIR / "incident_audit_reports"

INCIDENT_GAPS_SECONDS = (30, 60, 120)
LOOKBACK_SECONDS = 120
# Control windows are 120 seconds long and end 3, 6, and 9 minutes before onset.
CONTROL_OFFSETS_SECONDS = (180, 360, 540)
E7_EVENT_ID = "E7"
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def load_split(path: Path, split_name: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} split not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} has no header row")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        block_col = columns.get("blockid")
        label_col = columns.get("label")
        if not block_col or not label_col:
            raise ValueError(
                f"{path.name} must have BlockId and Label columns; "
                f"found {reader.fieldnames}"
            )

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            block_id = (row.get(block_col) or "").strip()
            label = (row.get(label_col) or "").strip().title()
            if not block_id or label not in {"Normal", "Anomaly"}:
                continue
            if block_id in rows:
                raise ValueError(f"Duplicate BlockId {block_id!r} in {path.name}")
            rows[block_id] = {"split": split_name, "label": label}
    return rows


def parse_timestamp(log_date: Any, log_time: Any) -> datetime | None:
    """Parse HDFS dates/times represented as strings or numeric DB values."""
    if log_date is None or log_time is None:
        return None

    date_digits = re.sub(r"\D", "", str(log_date).strip())
    time_digits = re.sub(r"\D", "", str(log_time).strip())

    # The source dataset uses MMDDYY, while a database may expose YYYY-MM-DD.
    if len(date_digits) == 5:
        date_digits = date_digits.zfill(6)
    if len(time_digits) < 6:
        time_digits = time_digits.zfill(6)
    if len(time_digits) > 6:
        time_digits = time_digits[:6]

    try:
        if len(date_digits) == 6:
            return datetime.strptime(date_digits + time_digits, "%m%d%y%H%M%S")
        if len(date_digits) == 8:
            return datetime.strptime(date_digits + time_digits, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return None


def classify_exception(text: str) -> str | None:
    message = (text or "").lower()
    if "could not read from stream" in message:
        return "stream_read_failure"
    if "connection reset" in message or "connection reset by peer" in message:
        return "connection_reset"
    if "eofexception" in message or "end of file" in message:
        return "unexpected_end_of_stream"
    if "sockettimeoutexception" in message or "socket timeout" in message:
        return "socket_timeout"
    if "interruptedioexception" in message:
        return "interrupted_io"
    if any(token in message for token in (
        "exception", " error", "error:", "timed out", "timeout", "failed",
        "failure", "interrupted", "connection refused", "could not",
    )):
        return "other_failure_or_error"
    return None


def extract_ips(text: str) -> list[str]:
    # Keep plausible IPv4 strings only; source log content is not treated as trusted.
    found = []
    for value in IPV4_RE.findall(text or ""):
        octets = value.split(".")
        if all(0 <= int(part) <= 255 for part in octets):
            found.append(value)
    return sorted(set(found))


def load_events(block_metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    block_ids = list(block_metadata)
    if not block_ids:
        raise ValueError("No BlockIds found in train.csv and validation.csv")

    query = """
        SELECT
            e.line_id,
            e.log_date,
            e.log_time,
            e.component,
            e.level,
            e.event_id,
            e.event_template,
            e.content,
            b.block_id
        FROM structured_log_events AS e
        JOIN log_event_blocks AS b ON b.line_id = e.line_id
        WHERE b.block_id = ANY(%s)
        ORDER BY e.line_id
    """

    with get_connection() as conn:
        rows = conn.execute(query, (block_ids,)).fetchall()

    events: list[dict[str, Any]] = []
    unparseable_timestamps = 0
    blocks_with_events: set[str] = set()
    for row in rows:
        (
            line_id, log_date, log_time, component, level, event_id,
            event_template, content, block_id,
        ) = row
        metadata = block_metadata.get(str(block_id))
        if metadata is None:
            # Defensive guard: SQL is filtered to the current split IDs.
            continue
        timestamp = parse_timestamp(log_date, log_time)
        if timestamp is None:
            unparseable_timestamps += 1
            continue
        content_text = str(content or "")
        template_text = str(event_template or "")
        combined_text = f"{content_text} {template_text}"
        events.append({
            "line_id": int(line_id) if line_id is not None else None,
            "timestamp": timestamp,
            "log_date": str(log_date),
            "log_time": str(log_time),
            "component": str(component or "").strip(),
            "level": str(level or "").strip(),
            "event_id": str(event_id or "").strip(),
            "event_template": template_text,
            "content": content_text,
            "block_id": str(block_id),
            "split": metadata["split"],
            "label": metadata["label"],
            "exception_signature": classify_exception(combined_text),
            "ips": extract_ips(combined_text),
        })
        blocks_with_events.add(str(block_id))

    events.sort(key=lambda event: (event["timestamp"], event["line_id"] or -1))
    print(f"Database rows in train/validation BlockIds: {len(rows):,}")
    print(f"Usable timestamped events: {len(events):,}")
    print(f"Rows skipped due to unparseable timestamps: {unparseable_timestamps:,}")
    print(f"Split BlockIds with timestamped events: {len(blocks_with_events):,}/"
          f"{len(block_ids):,}")
    return events


def cluster_e7_events(e7_events: list[dict[str, Any]], gap_seconds: int) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_time: datetime | None = None

    for event in e7_events:
        timestamp = event["timestamp"]
        if current and previous_time is not None:
            gap = (timestamp - previous_time).total_seconds()
            if gap > gap_seconds:
                episodes.append(current)
                current = []
        current.append(event)
        previous_time = timestamp

    if current:
        episodes.append(current)
    return episodes


def episode_metadata(episode_events: list[dict[str, Any]], gap_seconds: int, index: int) -> dict[str, Any]:
    blocks: dict[str, dict[str, str]] = {}
    for event in episode_events:
        blocks[event["block_id"]] = {"split": event["split"], "label": event["label"]}
    start = min(event["timestamp"] for event in episode_events)
    end = max(event["timestamp"] for event in episode_events)
    return {
        "gap_seconds": gap_seconds,
        "episode_id": f"G{gap_seconds}-E{index:02d}",
        "start": start,
        "end": end,
        "e7_event_count": len(episode_events),
        "blocks": blocks,
        "train_blocks": {b for b, m in blocks.items() if m["split"] == "train"},
        "validation_blocks": {b for b, m in blocks.items() if m["split"] == "validation"},
        "anomaly_blocks": {b for b, m in blocks.items() if m["label"] == "Anomaly"},
        "normal_blocks": {b for b, m in blocks.items() if m["label"] == "Normal"},
    }


def contains_e7(e7_events: list[dict[str, Any]], start: datetime, end: datetime) -> bool:
    return any(start <= event["timestamp"] < end for event in e7_events)


def make_window_events(
    events: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    excluded_blocks: set[str],
) -> list[dict[str, Any]]:
    """Return other-block events inside [start, end), ordered by timestamp."""
    return [
        event for event in events
        if start <= event["timestamp"] < end and event["block_id"] not in excluded_blocks
    ]


def event_patterns(event: dict[str, Any]) -> list[tuple[str, str]]:
    patterns: list[tuple[str, str]] = []
    event_id = event["event_id"]
    component = event["component"]
    if event_id:
        patterns.append(("event_id", event_id))
        patterns.append(("component_event", f"{component or '[unknown component]'}::{event_id}"))
    if event["exception_signature"]:
        patterns.append(("exception_signature", event["exception_signature"]))
    for ip in event["ips"]:
        patterns.append(("ip_reference", ip))
    return patterns


def add_window(
    windows: list[dict[str, Any]],
    gap_seconds: int,
    episode_id: str,
    episode_start: datetime,
    window_type: str,
    window_id: str,
    start: datetime,
    end: datetime,
    window_events: list[dict[str, Any]],
) -> dict[str, Any]:
    window = {
        "gap_seconds": gap_seconds,
        "episode_id": episode_id,
        "episode_start": episode_start,
        "window_type": window_type,
        "window_id": window_id,
        "start": start,
        "end": end,
        "events": window_events,
    }
    windows.append(window)
    return window


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, datetime):
                    value = value.isoformat(sep=" ")
                elif isinstance(value, (set, list, tuple)):
                    value = ";".join(str(item) for item in sorted(value))
                normalized[key] = value
            writer.writerow(normalized)


def main() -> None:
    train_meta = load_split(TRAIN_PATH, "train")
    validation_meta = load_split(VALIDATION_PATH, "validation")
    overlap = set(train_meta) & set(validation_meta)
    if overlap:
        raise ValueError(f"train.csv and validation.csv overlap on {len(overlap)} BlockIds")

    block_metadata = {**train_meta, **validation_meta}
    print("CROSS-BLOCK INCIDENT-WINDOW AUDIT")
    print(f"Training BlockIds: {len(train_meta):,}; validation BlockIds: {len(validation_meta):,}")
    print("Held-out test split: NOT READ")
    print("Production API/model/database writes: none")

    events = load_events(block_metadata)
    e7_events = [event for event in events if event["event_id"].strip().upper() == E7_EVENT_ID]
    print(f"E7 event rows in train/validation BlockIds: {len(e7_events):,}")
    if not e7_events:
        raise RuntimeError("No E7 events found in the selected train/validation BlockIds")

    all_episode_rows: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    episode_counts_by_gap: dict[int, int] = {}

    for gap_seconds in INCIDENT_GAPS_SECONDS:
        grouped = cluster_e7_events(e7_events, gap_seconds)
        episode_counts_by_gap[gap_seconds] = len(grouped)
        print(f"{gap_seconds}-second gap rule: {len(grouped)} episode(s)")

        for index, e7_group in enumerate(grouped, start=1):
            meta = episode_metadata(e7_group, gap_seconds, index)
            episode_id = meta["episode_id"]
            episode_blocks = set(meta["blocks"])
            target_start = meta["start"] - timedelta(seconds=LOOKBACK_SECONDS)
            target_end = meta["start"]
            target_events = make_window_events(events, target_start, target_end, episode_blocks)
            # A prior E7 in the lookback means this is not a clean precursor window:
            # it may be part of an earlier failure episode, not a precursor to this one.
            target_has_prior_e7 = contains_e7(e7_events, target_start, target_end)
            target_window = add_window(
                all_windows, gap_seconds, episode_id, meta["start"],
                "pre_incident", f"{episode_id}-PRE", target_start, target_end, target_events,
            )
            target_window["window_has_e7"] = target_has_prior_e7
            target_window["eligible_for_pattern_summary"] = not target_has_prior_e7

            controls_added = 0
            for control_index, offset_seconds in enumerate(CONTROL_OFFSETS_SECONDS, start=1):
                control_end = meta["start"] - timedelta(seconds=offset_seconds)
                control_start = control_end - timedelta(seconds=LOOKBACK_SECONDS)
                if contains_e7(e7_events, control_start, control_end):
                    continue
                control_events = make_window_events(events, control_start, control_end, episode_blocks)
                control_window = add_window(
                    all_windows, gap_seconds, episode_id, meta["start"],
                    "control", f"{episode_id}-C{control_index}",
                    control_start, control_end, control_events,
                )
                control_window["window_has_e7"] = False
                control_window["eligible_for_pattern_summary"] = True
                controls_added += 1

            episode_row = {
                "gap_seconds": gap_seconds,
                "episode_id": episode_id,
                "start": meta["start"],
                "end": meta["end"],
                "duration_seconds": int((meta["end"] - meta["start"]).total_seconds()),
                "e7_event_count": meta["e7_event_count"],
                "distinct_e7_blocks": len(meta["blocks"]),
                "train_e7_blocks": len(meta["train_blocks"]),
                "validation_e7_blocks": len(meta["validation_blocks"]),
                "anomaly_e7_blocks": len(meta["anomaly_blocks"]),
                "normal_e7_blocks": len(meta["normal_blocks"]),
                "train_and_validation_mixed": bool(meta["train_blocks"] and meta["validation_blocks"]),
                "pre_incident_other_block_events": len(target_events),
                "pre_incident_other_blocks": len({e["block_id"] for e in target_events}),
                "pre_window_contains_prior_e7": target_has_prior_e7,
                "pre_window_eligible_for_pattern_summary": not target_has_prior_e7,
                "eligible_control_windows": controls_added,
            }
            episode_summaries.append(episode_row)
            all_episode_rows.append({
                **episode_row,
                "e7_block_ids": ";".join(sorted(episode_blocks)),
            })

    detail_rows: list[dict[str, Any]] = []
    pattern_accumulator: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    window_pattern_presence: dict[tuple[int, str, str, str], set[str]] = defaultdict(set)
    all_window_count_by_gap_kind: Counter[tuple[int, str]] = Counter()
    eligible_window_count_by_gap_kind: Counter[tuple[int, str]] = Counter()

    for window in all_windows:
        gap = window["gap_seconds"]
        kind = window["window_type"]
        all_window_count_by_gap_kind[(gap, kind)] += 1
        if window.get("eligible_for_pattern_summary", False):
            eligible_window_count_by_gap_kind[(gap, kind)] += 1
        for event in window["events"]:
            detail_rows.append({
                "gap_seconds": gap,
                "episode_id": window["episode_id"],
                "window_type": kind,
                "window_id": window["window_id"],
                "window_start": window["start"],
                "window_end": window["end"],
                "window_contains_e7_in_included_splits": window.get("window_has_e7", ""),
                "eligible_for_pattern_summary": window.get("eligible_for_pattern_summary", False),
                "episode_start": window["episode_start"],
                "event_timestamp": event["timestamp"],
                "seconds_relative_to_episode_start": int((event["timestamp"] - window["episode_start"]).total_seconds()),
                "line_id": event["line_id"],
                "block_id": event["block_id"],
                "split": event["split"],
                "label": event["label"],
                "event_id": event["event_id"],
                "component": event["component"],
                "level": event["level"],
                "exception_signature": event["exception_signature"] or "",
                "ip_references": ";".join(event["ips"]),
                "event_template": event["event_template"],
                "content": event["content"],
            })
            # Keep contaminated pre-incident windows in the detail CSV, but do not
            # use them to describe precursor patterns.
            if not window.get("eligible_for_pattern_summary", False):
                continue
            for pattern_type, pattern_value in event_patterns(event):
                key = (gap, kind, pattern_type, pattern_value)
                if key not in pattern_accumulator:
                    pattern_accumulator[key] = {
                        "line_ids": 0,
                        "blocks": set(),
                        "train_blocks": set(),
                        "validation_blocks": set(),
                        "anomaly_blocks": set(),
                        "normal_blocks": set(),
                    }
                acc = pattern_accumulator[key]
                acc["line_ids"] += 1
                acc["blocks"].add(event["block_id"])
                if event["split"] == "train":
                    acc["train_blocks"].add(event["block_id"])
                else:
                    acc["validation_blocks"].add(event["block_id"])
                if event["label"] == "Anomaly":
                    acc["anomaly_blocks"].add(event["block_id"])
                else:
                    acc["normal_blocks"].add(event["block_id"])
                window_pattern_presence[key].add(window["window_id"])

    pattern_rows: list[dict[str, Any]] = []
    for (gap, kind, pattern_type, pattern_value), acc in sorted(pattern_accumulator.items()):
        pattern_rows.append({
            "gap_seconds": gap,
            "window_type": kind,
            "pattern_type": pattern_type,
            "pattern_value": pattern_value,
            "windows_with_pattern": len(window_pattern_presence[(gap, kind, pattern_type, pattern_value)]),
            "all_windows_of_type": all_window_count_by_gap_kind[(gap, kind)],
            "eligible_windows_of_type": eligible_window_count_by_gap_kind[(gap, kind)],
            "window_coverage_fraction": round(
                len(window_pattern_presence[(gap, kind, pattern_type, pattern_value)]) /
                max(1, eligible_window_count_by_gap_kind[(gap, kind)]), 4
            ),
            "event_rows": acc["line_ids"],
            "distinct_blocks": len(acc["blocks"]),
            "train_blocks": len(acc["train_blocks"]),
            "validation_blocks": len(acc["validation_blocks"]),
            "anomaly_blocks": len(acc["anomaly_blocks"]),
            "normal_blocks": len(acc["normal_blocks"]),
            "interpretation": "descriptive association only; not causal evidence",
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "incident_episodes.csv",
        [
            "gap_seconds", "episode_id", "start", "end", "duration_seconds",
            "e7_event_count", "distinct_e7_blocks", "train_e7_blocks",
            "validation_e7_blocks", "anomaly_e7_blocks", "normal_e7_blocks",
            "train_and_validation_mixed", "pre_incident_other_block_events",
            "pre_incident_other_blocks", "pre_window_contains_prior_e7",
            "pre_window_eligible_for_pattern_summary", "eligible_control_windows", "e7_block_ids",
        ],
        all_episode_rows,
    )
    write_csv(
        OUTPUT_DIR / "incident_window_events.csv",
        [
            "gap_seconds", "episode_id", "window_type", "window_id", "window_start",
            "window_end", "window_contains_e7_in_included_splits",
            "eligible_for_pattern_summary", "episode_start", "event_timestamp",
            "seconds_relative_to_episode_start", "line_id", "block_id", "split", "label",
            "event_id", "component", "level", "exception_signature", "ip_references",
            "event_template", "content",
        ],
        detail_rows,
    )
    write_csv(
        OUTPUT_DIR / "incident_pattern_summary.csv",
        [
            "gap_seconds", "window_type", "pattern_type", "pattern_value",
            "windows_with_pattern", "all_windows_of_type", "eligible_windows_of_type",
            "window_coverage_fraction",
            "event_rows", "distinct_blocks", "train_blocks", "validation_blocks",
            "anomaly_blocks", "normal_blocks", "interpretation",
        ],
        pattern_rows,
    )

    summary_lines = [
        "CROSS-BLOCK INCIDENT-WINDOW AUDIT",
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Training BlockIds read: {len(train_meta):,}",
        f"Validation BlockIds read: {len(validation_meta):,}",
        "Held-out test split: NOT READ",
        "PostgreSQL writes: none",
        "R1 model/API changes: none",
        f"E7 rows loaded: {len(e7_events):,}",
        f"Lookback window: {LOOKBACK_SECONDS} seconds before each episode start",
        "Controls: 120-second windows ending 180, 360, and 540 seconds before episode onset; any control window containing E7 is skipped.",
        "Cross-block rule: all events from BlockIds participating in the current E7 episode are excluded from its context/control windows.",
        "",
        "Episodes found by maximum consecutive E7 time gap:",
    ]
    for gap_seconds, count in episode_counts_by_gap.items():
        summary_lines.append(f"  {gap_seconds}s gap: {count} episode(s)")
    summary_lines.extend([
        "",
        "Important limitations:",
        "- The existing split is by BlockId, not by incident-time episode.",
        "- An episode containing both training and validation BlockIds is not independent validation evidence.",
        "- Control windows are E7-free only among train/validation BlockIds, because test BlockIds are excluded.",
        "- A pre-incident window containing another E7 in included splits is retained in details but excluded from pattern summaries.",
        "- Control-window comparisons are descriptive and are not a causal test.",
        "- HDFS labels describe Normal/Anomaly, not verified root causes.",
        "- No root-cause candidates are promoted by this script; the outputs are exploratory evidence only.",
        "",
        "Files written:",
        f"  {OUTPUT_DIR / 'incident_episodes.csv'}",
        f"  {OUTPUT_DIR / 'incident_window_events.csv'}",
        f"  {OUTPUT_DIR / 'incident_pattern_summary.csv'}",
    ])
    (OUTPUT_DIR / "incident_audit_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nAUDIT COMPLETE")
    print(f"Incident episodes CSV: {OUTPUT_DIR / 'incident_episodes.csv'}")
    print(f"Event details CSV:     {OUTPUT_DIR / 'incident_window_events.csv'} ({len(detail_rows):,} rows)")
    print(f"Pattern summary CSV:   {OUTPUT_DIR / 'incident_pattern_summary.csv'} ({len(pattern_rows):,} rows)")
    print(f"Run summary:           {OUTPUT_DIR / 'incident_audit_summary.txt'}")
    print("\nInterpretation: use this as descriptive incident evidence only. The script does not infer or prove root causes.")


if __name__ == "__main__":
    main()
