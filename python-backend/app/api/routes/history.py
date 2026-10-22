
from fastapi import APIRouter, Query

from app.db.database import get_connection

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Prediction History"],
)


@router.get("/history")
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
):
    query = """
        SELECT
            t.id AS trace_id,
            t.block_id,
            t.label AS actual_label,
            t.combined_log_text,
            p.id AS prediction_id,
            p.prediction,
            p.anomaly_score,
            p.threshold,
            p.model_name,
            p.created_at
        FROM anomaly_predictions p
        JOIN traces t ON t.id = p.trace_id
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT %s
    """

    with get_connection() as conn:
        rows = conn.execute(query, (limit,)).fetchall()

    items = [
        {
            "trace_id": row[0],
            "block_id": row[1],
            "actual_label": row[2],
            "trace_text": row[3],
            "prediction_id": row[4],
            "prediction": row[5],
            "anomaly_score": row[6],
            "threshold": row[7],
            "model": row[8],
            "created_at": row[9],
        }
        for row in rows
    ]

    return {
        "count": len(items),
        "items": items,
    }
