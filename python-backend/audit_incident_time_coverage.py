"""Read-only temporal coverage audit for Intelligent Log Analyzer.

Uses BlockIds in train.csv and validation.csv only; does not read test.csv,
write to PostgreSQL, alter the production API, or touch model artifacts.

Outputs under python-backend/incident_audit_reports_v3/:
- timeline_minute_counts.csv: observed log activity per minute
- window_coverage_inventory.csv: every proposed pre-incident/control window,
  including empty windows and clear reasons for exclusions
- audit_summary.txt: compact run summary and limitations
"""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.db.database import get_connection

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"
OUTPUT_DIR = BACKEND_DIR / "incident_audit_reports_v3"

GAP_RULES = (30, 60, 120)
LOOKBACK_SECONDS = 120
PREINCIDENT_CLEARANCE_SECONDS = 120
CONTROL_CLEARANCE_SECONDS = 120
CONTROL_OFFSETS_SECONDS = (180, 360, 540)
E7_ID = "E7"


def load_split(path: Path, split_name: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        block_col, label_col = columns.get("blockid"), columns.get("label")
        if not block_col or not label_col:
            raise ValueError(f"Expected BlockId and Label in {path.name}")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            block = (row.get(block_col) or "").strip()
            label = (row.get(label_col) or "").strip().title()
            if block and label in {"Normal", "Anomaly"}:
                if block in result:
                    raise ValueError(f"Duplicate BlockId {block} in {path.name}")
                result[block] = {"split": split_name, "label": label}
    return result


def parse_timestamp(log_date: Any, log_time: Any) -> datetime | None:
    if log_date is None or log_time is None:
        return None
    import re
    date_digits = re.sub(r"\D", "", str(log_date).strip())
    time_digits = re.sub(r"\D", "", str(log_time).strip())
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


def load_events(metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    block_ids = list(metadata)
    query = """
        SELECT e.line_id, e.log_date, e.log_time, e.event_id, e.component, b.block_id
        FROM structured_log_events e
        JOIN log_event_blocks b ON b.line_id = e.line_id
        WHERE b.block_id = ANY(%s)
        ORDER BY e.line_id
    """
    with get_connection() as conn:
        rows = conn.execute(query, (block_ids,)).fetchall()

    events: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        line_id, log_date, log_time, event_id, component, block_id = row
        ts = parse_timestamp(log_date, log_time)
        if ts is None:
            skipped += 1
            continue
        block = str(block_id)
        info = metadata.get(block)
        if not info:
            continue
        events.append({
            "line_id": int(line_id) if line_id is not None else None,
            "timestamp": ts,
            "event_id": str(event_id or "").strip().upper(),
            "component": str(component or "").strip(),
            "block_id": block,
            "split": info["split"],
            "label": info["label"],
        })
    events.sort(key=lambda e: (e["timestamp"], e["line_id"] or -1))
    print(f"Database rows for train/validation BlockIds: {len(rows):,}")
    print(f"Rows with parseable timestamps: {len(events):,}")
    print(f"Rows skipped for timestamp parsing: {skipped:,}")
    return events


def cluster_e7(events: list[dict[str, Any]], gap: int) -> list[list[dict[str, Any]]]:
    failures = [e for e in events if e["event_id"] == E7_ID]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prior: datetime | None = None
    for event in failures:
        ts = event["timestamp"]
        if current and prior and (ts - prior).total_seconds() > gap:
            groups.append(current)
            current = []
        current.append(event)
        prior = ts
    if current:
        groups.append(current)
    return groups


def e7_in(events: list[dict[str, Any]], start: datetime, end: datetime) -> bool:
    return any(e["event_id"] == E7_ID and start <= e["timestamp"] < end for e in events)


def event_rows_in(events: list[dict[str, Any]], start: datetime, end: datetime,
                  excluded_blocks: set[str]) -> list[dict[str, Any]]:
    return [e for e in events if start <= e["timestamp"] < end and e["block_id"] not in excluded_blocks]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, datetime):
                    value = value.isoformat(sep=" ")
                out[key] = value
            writer.writerow(out)


def main() -> None:
    train = load_split(TRAIN_PATH, "train")
    validation = load_split(VALIDATION_PATH, "validation")
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"Train/validation BlockId overlap: {len(overlap)}")
    metadata = {**train, **validation}

    print("INCIDENT TIME-COVERAGE AUDIT (READ ONLY)")
    print(f"Training BlockIds: {len(train):,}; validation BlockIds: {len(validation):,}")
    print("Held-out test split: NOT READ")
    print("PostgreSQL writes / API / model changes: none")
    events = load_events(metadata)
    if not events:
        raise RuntimeError("No timestamped train/validation events found")

    min_ts, max_ts = events[0]["timestamp"], events[-1]["timestamp"]
    e7_events = [e for e in events if e["event_id"] == E7_ID]
    print(f"Observed time range: {min_ts} through {max_ts}")
    print(f"E7 events: {len(e7_events)}")

    # Minute activity profile, including only split BlockIds by design.
    minute_acc: dict[datetime, dict[str, Any]] = {}
    for e in events:
        minute = e["timestamp"].replace(second=0, microsecond=0)
        acc = minute_acc.setdefault(minute, {
            "minute_start": minute, "event_rows": 0, "distinct_blocks": set(),
            "train_blocks": set(), "validation_blocks": set(), "e7_rows": 0,
            "anomaly_blocks": set(), "normal_blocks": set(),
        })
        acc["event_rows"] += 1
        acc["distinct_blocks"].add(e["block_id"])
        if e["split"] == "train":
            acc["train_blocks"].add(e["block_id"])
        else:
            acc["validation_blocks"].add(e["block_id"])
        if e["label"] == "Anomaly":
            acc["anomaly_blocks"].add(e["block_id"])
        else:
            acc["normal_blocks"].add(e["block_id"])
        if e["event_id"] == E7_ID:
            acc["e7_rows"] += 1
    minute_rows = []
    for minute in sorted(minute_acc):
        acc = minute_acc[minute]
        minute_rows.append({
            "minute_start": minute,
            "event_rows": acc["event_rows"],
            "distinct_blocks": len(acc["distinct_blocks"]),
            "train_blocks": len(acc["train_blocks"]),
            "validation_blocks": len(acc["validation_blocks"]),
            "e7_rows": acc["e7_rows"],
            "anomaly_blocks": len(acc["anomaly_blocks"]),
            "normal_blocks": len(acc["normal_blocks"]),
        })

    windows: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    for gap in GAP_RULES:
        episodes = cluster_e7(events, gap)
        print(f"{gap}-second grouping: {len(episodes)} episode(s)")
        accepted_controls: list[tuple[datetime, datetime]] = []
        for i, group in enumerate(episodes, 1):
            episode_id = f"G{gap}-E{i:02d}"
            onset = min(e["timestamp"] for e in group)
            episode_end = max(e["timestamp"] for e in group)
            episode_blocks = {e["block_id"] for e in group}
            pre_end = onset
            pre_start = pre_end - timedelta(seconds=LOOKBACK_SECONDS)
            # Require an E7-free buffer immediately before the pre-window. The
            # episode's own onset is exclusive at pre_end, so it is not counted.
            pre_clearance_start = pre_start - timedelta(seconds=PREINCIDENT_CLEARANCE_SECONDS)
            pre_contaminated = e7_in(events, pre_clearance_start, pre_end)
            pre_events = event_rows_in(events, pre_start, pre_end, episode_blocks)
            windows.append({
                "gap_seconds": gap, "episode_id": episode_id, "window_type": "pre_incident",
                "window_id": f"{episode_id}-PRE", "episode_start": onset,
                "episode_end": episode_end, "window_start": pre_start, "window_end": pre_end,
                "event_rows": len(pre_events),
                "distinct_blocks": len({e["block_id"] for e in pre_events}),
                "e7_rows_in_window": sum(e["event_id"] == E7_ID for e in pre_events),
                "prior_e7_in_clearance": pre_contaminated,
                "overlaps_accepted_control": False,
                "eligible_for_comparison": not pre_contaminated and bool(pre_events),
                "status": ("prior_e7_in_clearance" if pre_contaminated else
                           "empty_window_no_split_events" if not pre_events else "eligible"),
            })
            summary_counts[f"{gap}s_pre_{windows[-1]['status']}"] += 1

            for control_i, offset in enumerate(CONTROL_OFFSETS_SECONDS, 1):
                c_end = onset - timedelta(seconds=offset)
                c_start = c_end - timedelta(seconds=LOOKBACK_SECONDS)
                clearance_start = c_start - timedelta(seconds=CONTROL_CLEARANCE_SECONDS)
                clearance_end = c_end + timedelta(seconds=CONTROL_CLEARANCE_SECONDS)
                e7_buffered = e7_in(events, clearance_start, clearance_end)
                overlaps = any(c_start < old_end and c_end > old_start for old_start, old_end in accepted_controls)
                control_events = event_rows_in(events, c_start, c_end, episode_blocks)
                if e7_buffered:
                    status = "e7_in_clearance_buffer"
                elif overlaps:
                    status = "overlaps_accepted_control"
                elif not control_events:
                    status = "empty_window_no_split_events"
                else:
                    status = "eligible"
                    accepted_controls.append((c_start, c_end))
                windows.append({
                    "gap_seconds": gap, "episode_id": episode_id, "window_type": "control",
                    "window_id": f"{episode_id}-C{control_i}", "episode_start": onset,
                    "episode_end": episode_end, "window_start": c_start, "window_end": c_end,
                    "event_rows": len(control_events),
                    "distinct_blocks": len({e["block_id"] for e in control_events}),
                    "e7_rows_in_window": sum(e["event_id"] == E7_ID for e in control_events),
                    "prior_e7_in_clearance": e7_buffered,
                    "overlaps_accepted_control": overlaps,
                    "eligible_for_comparison": status == "eligible",
                    "status": status,
                })
                summary_counts[f"{gap}s_control_{status}"] += 1

    window_fields = [
        "gap_seconds", "episode_id", "window_type", "window_id", "episode_start", "episode_end",
        "window_start", "window_end", "event_rows", "distinct_blocks", "e7_rows_in_window",
        "prior_e7_in_clearance", "overlaps_accepted_control", "eligible_for_comparison", "status",
    ]
    minute_fields = [
        "minute_start", "event_rows", "distinct_blocks", "train_blocks", "validation_blocks",
        "e7_rows", "anomaly_blocks", "normal_blocks",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "timeline_minute_counts.csv", minute_fields, minute_rows)
    write_csv(OUTPUT_DIR / "window_coverage_inventory.csv", window_fields, windows)

    summary = [
        "INCIDENT TIME-COVERAGE AUDIT",
        f"Training BlockIds: {len(train):,}",
        f"Validation BlockIds: {len(validation):,}",
        "Held-out test split: NOT READ",
        "PostgreSQL writes: none",
        f"Timestamped events: {len(events):,}",
        f"Observed time range: {min_ts} through {max_ts}",
        f"E7 event rows: {len(e7_events):,}",
        "Pre-incident windows require a 120-second E7-clearance buffer before the lookback window.",
        "Control windows require a 120-second E7-clearance buffer, no overlap with an accepted control, and at least one split event.",
        "Coverage counts are based only on event records linked to train/validation BlockIds; empty windows are explicitly reported.",
        "This is an availability/coverage audit, not a root-cause test.",
        "",
        "Window statuses:",
    ] + [f"  {key}: {value}" for key, value in sorted(summary_counts.items())] + [
        "",
        f"Minute profile: {OUTPUT_DIR / 'timeline_minute_counts.csv'}",
        f"Window inventory: {OUTPUT_DIR / 'window_coverage_inventory.csv'}",
    ]
    (OUTPUT_DIR / "audit_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\nWINDOW STATUS COUNTS")
    for key, value in sorted(summary_counts.items()):
        print(f"{key}: {value}")
    print(f"Minute activity profile: {OUTPUT_DIR / 'timeline_minute_counts.csv'}")
    print(f"Window inventory: {OUTPUT_DIR / 'window_coverage_inventory.csv'}")
    print(f"Run summary: {OUTPUT_DIR / 'audit_summary.txt'}")


if __name__ == "__main__":
    main()
