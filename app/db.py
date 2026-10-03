"""Support tickets and office-plan quotes in SQLite."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "tickets.db"
FIELDS = ("product", "issue_type", "urgency", "summary", "status", "message")


@contextmanager
def _connect(path: Path):
    """Commit on success, always close. (`with sqlite3.connect()` alone commits but leaves the file open.)"""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY, product TEXT, issue_type TEXT NOT NULL, urgency TEXT NOT NULL,
                summary TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS quotes (
                quote_id TEXT PRIMARY KEY, tier TEXT NOT NULL, total INTEGER NOT NULL, plan TEXT NOT NULL,
                created_at TEXT NOT NULL)""")
            yield conn
    finally:
        conn.close()


def save_ticket(ticket: dict, path: Path = DB_PATH) -> str:
    ticket_id = "T-" + uuid.uuid4().hex[:8].upper()
    with _connect(path) as conn:
        conn.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (ticket_id, *(ticket[f] for f in FIELDS), datetime.now(timezone.utc).isoformat()))
    return ticket_id


def get_ticket(ticket_id: str, path: Path = DB_PATH) -> dict | None:
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    return dict(row) if row else None


def save_quote(plan: dict, path: Path = DB_PATH) -> str:
    quote_id = "Q-" + uuid.uuid4().hex[:8].upper()
    with _connect(path) as conn:
        conn.execute("INSERT INTO quotes VALUES (?, ?, ?, ?, ?)",
                     (quote_id, plan["tier"], plan["estimated_total"], json.dumps(plan, ensure_ascii=False),
                      datetime.now(timezone.utc).isoformat()))
    return quote_id


def get_quote(quote_id: str, path: Path = DB_PATH) -> dict | None:
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM quotes WHERE quote_id = ?", (quote_id,)).fetchone()
    return {**dict(row), "plan": json.loads(row["plan"])} if row else None
