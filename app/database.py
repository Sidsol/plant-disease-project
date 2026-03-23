"""
SQLite database for scan history and flagged (Human-in-the-Loop) reports.

Tables:
  - scan_history: Every prediction the API makes, with optional thumbnail + heatmap.
  - flagged_data: User-reported incorrect predictions, queued for retraining.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "plant_disease.db"


def _get_conn() -> sqlite3.Connection:
    """Open a new SQLite connection with WAL mode and foreign key enforcement.

    WAL (Write-Ahead Logging) allows concurrent readers while a single
    writer holds the lock, improving performance under the FastAPI
    async workload where multiple requests may read history simultaneously.
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager that provides a database connection with auto-commit.

    Yields:
        sqlite3.Connection: A connection that commits on clean exit
        and always closes on scope exit.

    Usage::

        with get_db() as conn:
            conn.execute("INSERT INTO ...")
    """
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id           TEXT PRIMARY KEY,
                timestamp    TEXT NOT NULL,
                model_name   TEXT NOT NULL,
                class_name   TEXT NOT NULL,
                plant        TEXT NOT NULL,
                condition    TEXT NOT NULL,
                healthy      INTEGER NOT NULL,
                confidence   REAL NOT NULL,
                top5_json    TEXT NOT NULL,
                thumbnail    TEXT,
                attention_map TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS flagged_data (
                id              TEXT PRIMARY KEY,
                scan_id         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                reason          TEXT,
                user_correction TEXT,
                image_base64    TEXT,
                ai_prediction   TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (scan_id) REFERENCES scan_history(id)
            );

            CREATE INDEX IF NOT EXISTS idx_scan_timestamp
                ON scan_history(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_flagged_status
                ON flagged_data(status);

            CREATE TABLE IF NOT EXISTS chat_history (
                id          TEXT PRIMARY KEY,
                scan_id     TEXT,
                session_id  TEXT,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_scan
                ON chat_history(scan_id);
            CREATE INDEX IF NOT EXISTS idx_chat_session
                ON chat_history(session_id);
            """
        )


# ---------------------------------------------------------------------------
# Scan history CRUD
# ---------------------------------------------------------------------------

def save_scan(
    model_name: str,
    class_name: str,
    plant: str,
    condition: str,
    healthy: bool,
    confidence: float,
    top5: list,
    metadata: dict,
    thumbnail: Optional[str] = None,
    attention_map: Optional[str] = None,
) -> str:
    """Insert a scan record and return its UUID."""
    scan_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scan_history
                (id, timestamp, model_name, class_name, plant, condition,
                 healthy, confidence, top5_json, thumbnail, attention_map,
                 metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id, ts, model_name, class_name, plant, condition,
                int(healthy), confidence,
                json.dumps(top5), thumbnail, attention_map,
                json.dumps(metadata),
            ),
        )
    return scan_id


def get_history(page: int = 1, limit: int = 10):
    """Return paginated scan history (newest first)."""
    offset = (page - 1) * limit
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, timestamp, model_name, class_name, plant, condition,
                   healthy, confidence, thumbnail, attention_map
            FROM scan_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    items = [dict(r) for r in rows]
    pages = max(1, -(-total // limit))  # ceil division
    return {"items": items, "total": total, "page": page, "limit": limit, "pages": pages}


def get_scan_by_id(scan_id: str) -> Optional[dict]:
    """Fetch a single scan record by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM scan_history WHERE id = ?", (scan_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Flagged data CRUD (Human-in-the-Loop)
# ---------------------------------------------------------------------------

def save_report(
    scan_id: str,
    reason: Optional[str] = None,
    user_correction: Optional[str] = None,
    image_base64: Optional[str] = None,
    ai_prediction: Optional[str] = None,
) -> str:
    """Flag a scan as incorrect for human review / retraining."""
    report_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO flagged_data
                (id, scan_id, timestamp, reason, user_correction,
                 image_base64, ai_prediction, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (report_id, scan_id, ts, reason, user_correction,
             image_base64, ai_prediction),
        )
    return report_id


def get_flagged(status: str = "pending", page: int = 1, limit: int = 20):
    """Return flagged items for review."""
    offset = (page - 1) * limit
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM flagged_data WHERE status = ?", (status,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT f.*, s.class_name AS original_class, s.confidence AS original_confidence
            FROM flagged_data f
            JOIN scan_history s ON f.scan_id = s.id
            WHERE f.status = ?
            ORDER BY f.timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (status, limit, offset),
        ).fetchall()
    items = [dict(r) for r in rows]
    pages = max(1, -(-total // limit))
    return {"items": items, "total": total, "page": page, "limit": limit, "pages": pages}


# ---------------------------------------------------------------------------
# Chat history CRUD
# ---------------------------------------------------------------------------

def save_chat_message(
    role: str,
    content: str,
    scan_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Save a chat message and return its UUID."""
    msg_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO chat_history (id, scan_id, session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, scan_id, session_id, role, content, ts),
        )
    return msg_id


def get_chat_history(
    scan_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Return chat messages for a given scan_id or session_id, oldest first."""
    with get_db() as conn:
        if scan_id:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM chat_history
                WHERE scan_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (scan_id, limit),
            ).fetchall()
        elif session_id:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            return []
    return [dict(r) for r in rows]
