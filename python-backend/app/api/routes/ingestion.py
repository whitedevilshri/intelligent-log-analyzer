from fastapi import APIRouter, HTTPException, status

from app.schemas.ingestion import (
    IngestionBatchRequest,
    IngestionBatchResponse,
)
from app.services.ingestion_store import (
    DuplicateLineIdError,
    get_storage_stats,
    store_batch,
)

router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["ingestion"],
)


@router.post(
    "/batches",
    response_model=IngestionBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def receive_batch(
    payload: IngestionBatchRequest,
) -> IngestionBatchResponse:

    line_ids = [record.line_id for record in payload.records]

    if len(line_ids) != len(set(line_ids)):
        raise HTTPException(
            status_code=422,
            detail="Duplicate line_id values within the submitted batch.",
        )

    try:
        store_batch(payload.records)
    except DuplicateLineIdError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    block_ids = {
        block_id
        for record in payload.records
        for block_id in record.block_ids
    }

    return IngestionBatchResponse(
        status="accepted",
        batch_id=payload.batch_id,
        records_received=len(payload.records),
        unique_line_ids=len(set(line_ids)),
        unique_block_ids=len(block_ids),
    )


@router.get("/stats")
def ingestion_stats() -> dict[str, int]:
    return get_storage_stats()