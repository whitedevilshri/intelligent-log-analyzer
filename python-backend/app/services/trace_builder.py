from collections import defaultdict
from statistics import mean, median
from threading import Lock

from app.schemas.ingestion import StructuredLogRecord

_trace_lock = Lock()
_traces_by_block_id: dict[str, list[str]] = {}


def build_traces(
    records: list[StructuredLogRecord],
) -> dict[str, int | float]:

    if not records:
        raise ValueError("No ingested records are available.")

    grouped: dict[str, list[str]] = defaultdict(list)

    # Preserve the original log order.
    for record in sorted(records, key=lambda item: item.line_id):
        block_ids = set(record.block_ids)

        if len(block_ids) != 1:
            raise ValueError(
                f"Expected exactly one BlockId for LineId {record.line_id}"
            )

        block_id = next(iter(block_ids))
        grouped[block_id].append(record.content)

    traces = dict(grouped)
    trace_lengths = [len(logs) for logs in traces.values()]

    # Retain the constructed traces in memory for later processing.
    with _trace_lock:
        global _traces_by_block_id
        _traces_by_block_id = traces

    return {
        "records_grouped": sum(trace_lengths),
        "traces_built": len(traces),
        "min_logs_per_trace": min(trace_lengths),
        "median_logs_per_trace": median(trace_lengths),
        "mean_logs_per_trace": round(mean(trace_lengths), 4),
        "max_logs_per_trace": max(trace_lengths),
    }


def get_built_traces() -> dict[str, list[str]]:
    with _trace_lock:
        return {
            block_id: list(logs)
            for block_id, logs in _traces_by_block_id.items()
        }