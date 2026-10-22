
import csv
from pathlib import Path

from app.db.database import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "HDFS_trace_labeled_dataset.csv"


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    records = []

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"BlockId", "CombinedLogText", "Label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain these columns: {sorted(required)}"
            )

        for line_number, row in enumerate(reader, start=2):
            block_id = (row.get("BlockId") or "").strip()
            log_text = (row.get("CombinedLogText") or "").strip()
            label = (row.get("Label") or "").strip()

            if not block_id or not log_text:
                raise ValueError(f"Missing BlockId or log text at CSV line {line_number}")

            if label not in {"Normal", "Anomaly"}:
                raise ValueError(f"Invalid label at CSV line {line_number}: {label}")

            records.append((block_id, log_text, label))

    if not records:
        raise ValueError("The dataset contains no records.")

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO traces (block_id, combined_log_text, label)
                VALUES (%s, %s, %s)
                ON CONFLICT (block_id) DO UPDATE SET
                    combined_log_text = EXCLUDED.combined_log_text,
                    label = EXCLUDED.label
                """,
                records,
            )

        conn.commit()

        total = conn.execute(
            "SELECT COUNT(*) FROM traces"
        ).fetchone()[0]

        counts = conn.execute(
            """
            SELECT label, COUNT(*)
            FROM traces
            GROUP BY label
            ORDER BY label
            """
        ).fetchall()

        print(f"Dataset records processed: {len(records)}")
        print(f"Total traces in database: {total}")
        print("Label counts:")

        for label, count in counts:
            print(f"  {label}: {count}")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
