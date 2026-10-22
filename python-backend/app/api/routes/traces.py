import csv
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.ingestion_store import get_all_records
from app.services.trace_builder import build_traces, get_built_traces

router = APIRouter(
    prefix="/api/v1/traces",
    tags=["traces"],
)


@router.post("/build")
def build_trace_endpoint() -> dict[str, str | int | float]:
    records = get_all_records()

    if not records:
        raise HTTPException(
            status_code=409,
            detail="No ingested records. Run Java ingestion first.",
        )

    try:
        stats = build_traces(records)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {
        "status": "built",
        **stats,
    }


@router.post("/label")
def label_trace_endpoint() -> dict[str, str | int]:
    traces = get_built_traces()

    if not traces:
        raise HTTPException(
            status_code=409,
            detail="No built traces. Run the trace-building endpoint first.",
        )

    project_root = Path(__file__).resolve().parents[4]

    labels_path = (
        project_root / "data" / "raw" / "anomaly_label.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "HDFS_trace_labeled_dataset.csv"
    )

    if not labels_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Label file not found: {labels_path}",
        )

    trace_ids = set(traces.keys())
    labels_by_block_id: dict[str, str] = {}

    try:
        with labels_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames or not {
                "BlockId",
                "Label",
            }.issubset(reader.fieldnames):
                raise HTTPException(
                    status_code=422,
                    detail="Label CSV must contain BlockId and Label columns.",
                )

            for row in reader:
                block_id = (row.get("BlockId") or "").strip()

                if block_id not in trace_ids:
                    continue

                label = (row.get("Label") or "").strip()

                if label not in {"Normal", "Anomaly"}:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unexpected label for {block_id}: {label}",
                    )

                if block_id in labels_by_block_id:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Duplicate label entry for {block_id}",
                    )

                labels_by_block_id[block_id] = label

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read label CSV: {exc}",
        ) from exc

    missing_ids = trace_ids - set(labels_by_block_id)

    if missing_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some traces have no matching label.",
                "missing_count": len(missing_ids),
                "examples": sorted(missing_ids)[:5],
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = Counter()

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["BlockId", "CombinedLogText", "Label"],
        )
        writer.writeheader()

        for block_id in traces:
            label = labels_by_block_id[block_id]

            writer.writerow({
                "BlockId": block_id,
                "CombinedLogText": "\n".join(traces[block_id]),
                "Label": label,
            })

            label_counts[label] += 1

    return {
        "status": "labeled",
        "traces_labeled": sum(label_counts.values()),
        "normal_traces": label_counts["Normal"],
        "anomalous_traces": label_counts["Anomaly"],
        "output_file": str(output_path),
    }