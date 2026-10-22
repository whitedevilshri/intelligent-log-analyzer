
from fastapi import APIRouter

from app.db.database import get_connection

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
)

MODEL_NAME = "R1_masked_tfidf_isolation_forest"


@router.get("/summary")
def get_dashboard_summary():
    with get_connection() as conn:
        row = conn.execute(
            """
            WITH trace_stats AS (
                SELECT
                    COUNT(*) AS total_traces,
                    COUNT(*) FILTER (
                        WHERE label IS NOT NULL
                    ) AS labeled_traces,
                    COUNT(*) FILTER (
                        WHERE label IS NULL
                    ) AS unlabeled_traces
                FROM traces
            ),
            prediction_stats AS (
                SELECT
                    COUNT(*) AS total_predictions,
                    COUNT(*) FILTER (
                        WHERE prediction = 'Anomaly'
                    ) AS predicted_anomalies,
                    COUNT(*) FILTER (
                        WHERE prediction = 'Normal'
                    ) AS predicted_normal
                FROM anomaly_predictions
                WHERE model_name = %s
            )
            SELECT
                total_traces,
                labeled_traces,
                unlabeled_traces,
                total_predictions,
                predicted_anomalies,
                predicted_normal
            FROM trace_stats
            CROSS JOIN prediction_stats
            """,
            (MODEL_NAME,),
        ).fetchone()

    return {
        "total_traces": row[0],
        "labeled_traces": row[1],
        "unlabeled_traces": row[2],
        "total_predictions": row[3],
        "predicted_anomalies": row[4],
        "predicted_normal": row[5],
        "model": MODEL_NAME,
    }
