
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Project root: intelligent-log-analyzer/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load local database credentials.
load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    """Create and return a PostgreSQL connection."""

    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            f"Missing database configuration: {', '.join(missing)}"
        )

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )
