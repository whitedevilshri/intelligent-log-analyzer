
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.database import get_connection
from app.services.anomaly_detector import predict_trace

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Predictions"],
)


class TracePredictionRequest(BaseModel):
    trace_text: str = Field(
        ...,
        min_length=1,
        description="Combined log text for one trace",
    )


@router.post("/trace")
def predict(request: TracePredictionRequest):
    # Step 1: Run the existing R1 model.
    try:
        result = predict_trace(request.trace_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Step 2: Generate a unique ID for this API-submitted trace.
    block_id = f"api_{uuid4().hex}"

    # Step 3: Save the trace and prediction in one transaction.
    try:
        with get_connection() as conn:
            trace_id = conn.execute(
                """
                INSERT INTO traces (
                    block_id, combined_log_text, label
                )
                VALUES (%s, %s, NULL)
                RETURNING id
                """,
                (block_id, request.trace_text),
            ).fetchone()[0]

            prediction_id = conn.execute(
                """
                INSERT INTO anomaly_predictions (
                    trace_id,
                    prediction,
                    anomaly_score,
                    threshold,
                    model_name
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    trace_id,
                    result["prediction"],
                    result["anomaly_score"],
                    result["threshold"],
                    result["model"],
                ),
            ).fetchone()[0]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save the prediction to the database.",
        ) from exc

    # Step 4: Return the prediction and database IDs.
    return {
        **result,
        "block_id": block_id,
        "trace_id": trace_id,
        "prediction_id": prediction_id,
        "saved": True,
    }
