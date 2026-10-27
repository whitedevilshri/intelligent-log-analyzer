
from fastapi import APIRouter, HTTPException, Query

from app.services.root_cause_correlator import analyze_block
from app.services.cross_block_correlator import correlate_block

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

@router.get("/blocks/{block_id}/cross-block-correlations")
def get_cross_block_correlations(
    block_id: str,
    window_seconds: int = Query(default=300, ge=1, le=3600),
    limit: int = Query(default=25, ge=1, le=100),
):
    """Return possible cross-BlockId failure-signature associations.

    Associations are time-bounded investigation clues, not root-cause claims.
    """
    try:
        return correlate_block(
            block_id, time_window_seconds=window_seconds, max_candidates=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
