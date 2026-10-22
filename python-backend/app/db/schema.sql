
CREATE TABLE IF NOT EXISTS traces (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    combined_log_text TEXT NOT NULL,
    label TEXT
        CHECK (label IN ('Normal', 'Anomaly')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomaly_predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trace_id BIGINT NOT NULL
        REFERENCES traces(id) ON DELETE CASCADE,
    prediction TEXT NOT NULL
        CHECK (prediction IN ('Normal', 'Anomaly')),
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_trace_id
    ON anomaly_predictions(trace_id);
