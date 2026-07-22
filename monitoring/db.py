"""
Monitoring store for the Credit Risk Advisor.

SQLite database (monitoring/advisor.db, gitignored) with two tables,
mirroring the course's monitoring schema:

  conversations : one row per question answered by the app, with the
                  full answer, retrieval mode, token usage, cost, latency.
  feedback      : one row per feedback event on a conversation —
                  source='user' (thumbs +1/-1 score) or
                  source='judge' (online LLM relevance verdict).

SQLite keeps the app dependency-free locally and deployable on Streamlit
Community Cloud; the same two-table shape ports to Postgres if needed.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Overridable so docker-compose can point both services at a shared volume
# (mounting a volume over monitoring/ itself would shadow this code).
DB_FILE = Path(os.environ.get("ADVISOR_DB_PATH", ROOT / "monitoring" / "advisor.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the first schema shipped. Kept as a migration so an
# existing advisor.db (e.g. a reviewer's) gains them without being recreated.
# rewritten_query is NULL unless the query-rewriting toggle was used.
_ADDED_COLUMNS = {
    "rewritten_query": "TEXT",
    "rewrite_tokens": "INTEGER NOT NULL DEFAULT 0",
    "rewrite_cost": "REAL NOT NULL DEFAULT 0",
    "rewrite_time": "REAL NOT NULL DEFAULT 0",
}


def init_db() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                search_mode TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                response_time REAL NOT NULL,
                cost REAL NOT NULL,
                rewritten_query TEXT,
                rewrite_tokens INTEGER NOT NULL DEFAULT 0,
                rewrite_cost REAL NOT NULL DEFAULT 0,
                rewrite_time REAL NOT NULL DEFAULT 0
            )
        """)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        for name, decl in _ADDED_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {decl}")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                source TEXT NOT NULL,
                relevance TEXT,
                explanation TEXT,
                score INTEGER,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_conversation(question: str, result: dict, rewrite: dict | None = None) -> int:
    """Log one answered question.

    `result` is the dict from rag.llm.answer(). `rewrite`, when the
    query-rewriting toggle was used, is
    {"query": str, "tokens": int, "cost": float, "time": float} — the
    rewritten query actually sent to retrieval and what rewriting cost.
    `question` always stores the original user question.
    """
    init_db()
    rewrite = rewrite or {}
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO conversations (
                timestamp, question, answer, search_mode, model,
                prompt_tokens, completion_tokens, total_tokens,
                response_time, cost,
                rewritten_query, rewrite_tokens, rewrite_cost, rewrite_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                question,
                result["answer"],
                result["search_mode"],
                result["model"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["total_tokens"],
                result["response_time"],
                result["cost"],
                rewrite.get("query"),
                rewrite.get("tokens", 0),
                rewrite.get("cost", 0.0),
                rewrite.get("time", 0.0),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_feedback(
    conversation_id: int,
    source: str,
    relevance: str | None = None,
    explanation: str | None = None,
    score: int | None = None,
) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feedback (
                conversation_id, source, relevance, explanation, score, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, source, relevance, explanation, score, _now()),
        )
        conn.commit()
    finally:
        conn.close()
