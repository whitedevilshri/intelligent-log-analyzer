
CREATE TABLE IF NOT EXISTS structured_log_events (
    line_id BIGINT PRIMARY KEY CHECK (line_id > 0),
    log_date TEXT NOT NULL,
    log_time TEXT NOT NULL,
    pid TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    content TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_template TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS log_event_blocks (
    line_id BIGINT NOT NULL
        REFERENCES structured_log_events(line_id) ON DELETE CASCADE,
    block_id TEXT NOT NULL,
    PRIMARY KEY (line_id, block_id)
);

CREATE INDEX IF NOT EXISTS idx_log_event_blocks_block_id
    ON log_event_blocks(block_id);

CREATE INDEX IF NOT EXISTS idx_structured_events_component
    ON structured_log_events(component);

CREATE INDEX IF NOT EXISTS idx_structured_events_event_id
    ON structured_log_events(event_id);
