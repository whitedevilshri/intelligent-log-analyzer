
from app.db.database import get_connection
from app.schemas.ingestion import StructuredLogRecord


class DuplicateLineIdError(ValueError):
    pass


def store_batch(records: list[StructuredLogRecord]) -> None:
    """Persist structured log events and their BlockId relationships."""

    line_ids = [record.line_id for record in records]

    if len(line_ids) != len(set(line_ids)):
        raise DuplicateLineIdError(
            "Duplicate LineId found within the batch."
        )

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Store each structured log event.
            # Re-ingesting a LineId updates its existing record.
            cur.executemany(
                """
                INSERT INTO structured_log_events (
                    line_id,
                    log_date,
                    log_time,
                    pid,
                    level,
                    component,
                    content,
                    event_id,
                    event_template
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (line_id) DO UPDATE SET
                    log_date = EXCLUDED.log_date,
                    log_time = EXCLUDED.log_time,
                    pid = EXCLUDED.pid,
                    level = EXCLUDED.level,
                    component = EXCLUDED.component,
                    content = EXCLUDED.content,
                    event_id = EXCLUDED.event_id,
                    event_template = EXCLUDED.event_template
                """,
                [
                    (
                        record.line_id,
                        record.date,
                        record.time,
                        record.pid,
                        record.level,
                        record.component,
                        record.content,
                        record.event_id,
                        record.event_template,
                    )
                    for record in records
                ],
            )

            # 2. Replace the BlockId relationships for these events.
            cur.executemany(
                """
                DELETE FROM log_event_blocks
                WHERE line_id = %s
                """,
                [(record.line_id,) for record in records],
            )

            # 3. Save each event-to-BlockId relationship.
            mappings = [
                (record.line_id, block_id)
                for record in records
                for block_id in record.block_ids
            ]

            cur.executemany(
                """
                INSERT INTO log_event_blocks (line_id, block_id)
                VALUES (%s, %s)
                ON CONFLICT (line_id, block_id) DO NOTHING
                """,
                mappings,
            )


def get_storage_stats() -> dict[str, int]:
    """Return ingestion statistics from PostgreSQL."""

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM structured_log_events),
                (SELECT COUNT(DISTINCT block_id) FROM log_event_blocks)
            """
        ).fetchone()

    total_events, unique_blocks = row

    return {
        "records_stored": total_events,
        "unique_line_ids": total_events,
        "unique_block_ids": unique_blocks,
        "trace_groups_expected": unique_blocks,
    }


def get_all_records() -> list[StructuredLogRecord]:
    """Load structured events and their BlockIds from PostgreSQL."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                e.line_id,
                e.log_date,
                e.log_time,
                e.pid,
                e.level,
                e.component,
                e.content,
                e.event_id,
                e.event_template,
                COALESCE(
                    array_agg(DISTINCT b.block_id)
                        FILTER (WHERE b.block_id IS NOT NULL),
                    ARRAY[]::TEXT[]
                ) AS block_ids
            FROM structured_log_events e
            LEFT JOIN log_event_blocks b
                ON b.line_id = e.line_id
            GROUP BY e.line_id
            ORDER BY e.line_id
            """
        ).fetchall()

    return [
        StructuredLogRecord(
            line_id=row[0],
            date=row[1],
            time=row[2],
            pid=row[3],
            level=row[4],
            component=row[5],
            content=row[6],
            event_id=row[7],
            event_template=row[8],
            block_ids=row[9],
        )
        for row in rows
    ]
