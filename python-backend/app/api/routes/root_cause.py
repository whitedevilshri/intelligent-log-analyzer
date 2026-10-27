
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.root_cause_correlator import analyze_block
from app.services.cross_block_correlator import correlate_block
from app.services.incident_grouper import build_incident_groups


class IncidentGroupingRequest(BaseModel):
    block_ids: list[str]
    window_seconds: int = Field(default=300, ge=1, le=3600)
    candidate_limit: int = Field(default=10, ge=1, le=100)


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
@router.post("/incidents")
def get_incident_groups(request: IncidentGroupingRequest):
    """Group potential incidents across a bounded list of BlockIds."""
    if not request.block_ids or len(request.block_ids) > 50:
        raise HTTPException(
            status_code=422,
            detail="Provide between 1 and 50 BlockIds per request.",
        )

    cleaned_ids = []
    for value in request.block_ids:
        if not value or not value.strip():
            raise HTTPException(
                status_code=422,
                detail="BlockIds must not be empty.",
            )
        cleaned_ids.append(value.strip())

    # Avoid repeating database work when the request contains duplicate IDs.
    unique_ids = list(dict.fromkeys(cleaned_ids))
    reports = []
    query_errors = []

    for block_id in unique_ids:
        try:
            report = correlate_block(
                block_id,
                time_window_seconds=request.window_seconds,
                max_candidates=request.candidate_limit,
            )
            reports.append(report)
        except ValueError as exc:
            # Keep successful results if an individual BlockId has no events.
            query_errors.append({
                "block_id": block_id,
                "error": str(exc),
            })

    result = build_incident_groups(
        reports,
        incident_window_seconds=request.window_seconds,
    )
    result.update({
        "requested_block_count": len(request.block_ids),
        "unique_block_count": len(unique_ids),
        "processed_block_count": len(reports),
        "query_errors": query_errors,
    })
    return result
