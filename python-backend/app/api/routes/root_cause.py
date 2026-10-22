
from fastapi import APIRouter, HTTPException

from app.services.root_cause_correlator import analyze_block

router = APIRouter(
    prefix="/api/v1/root-cause",
    tags=["Root Cause Analysis"],
)


@router.get("/blocks/{block_id}")
def get_root_cause_report(block_id: str):
    """
    Return diagnostic evidence and potential root-cause candidates
    for a BlockId stored in PostgreSQL.
    """
    try:
        return analyze_block(block_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
