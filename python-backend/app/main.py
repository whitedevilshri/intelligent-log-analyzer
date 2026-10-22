
from fastapi import FastAPI

from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.traces import router as traces_router
from app.api.routes.predictions import router as predictions_router
from app.api.routes.history import router as history_router
from app.api.routes.stats import router as stats_router
from app.api.routes.root_cause import router as root_cause_router

app = FastAPI(
    title="Intelligent Log Analyzer API",
    version="0.1.0",
    description="Backend for log ingestion and anomaly analysis",
)

app.include_router(ingestion_router)
app.include_router(traces_router)
app.include_router(predictions_router)
app.include_router(history_router)
app.include_router(stats_router)
app.include_router(root_cause_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "intelligent-log-analyzer-api",
    }
