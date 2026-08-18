"""Persistent history of iCARE-AI conversations.

Stored in a single SQLite file on the shared ``data`` volume. This deliberately
uses the standard-library ``sqlite3`` (no ORM, no extra dependency, no new
compose service) to match the app's existing file-based persistence style
(cf. ``admin.py``'s JSON settings file). The DB file sits next to those JSON
files under ``settings.data_folder`` so it persists across container restarts.

Conversations are upserted by the client after every completed turn (see the
``/api/chat/history`` endpoint in ``chat.py``), keyed by a client-generated
``conversation_id``. Recording per-turn means abandoned conversations are
captured as soon as they have at least one exchange, and the dual
summary/detailed answer variants are stored faithfully (the client owns the
richest view of the transcript, so it — not the streaming endpoint — persists).

Usage metrics (message counts, the span between first and last message, and
character volumes) are derived from the transcript on each upsert so the
list/summary views never have to recompute them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import settings

_DB_PATH = os.path.join(settings.data_folder, "ai_history.db")

# Serialize writes; SQLite handles concurrent readers fine but a single writer
# is simplest and more than enough for this app's traffic.
_write_lock = threading.Lock()
_initialized = False


class AccessError(Exception):
    """Raised when a user tries to upsert/read a conversation they do not own."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_schema() -> None:
    global _initialized
    if _initialized:
        return
    os.makedirs(settings.data_folder, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id                        TEXT PRIMARY KEY,
                user_id                   TEXT NOT NULL,
                arrival_path              TEXT,
                model                     TEXT,
                entry_context             TEXT,
                messages                  TEXT,
                started_at                TEXT,
                created_at                TEXT,
                updated_at                TEXT,
                duration_seconds          REAL,
                message_count             INTEGER,
                user_message_count        INTEGER,
                assistant_message_count   INTEGER,
                user_chars                INTEGER,
                assistant_chars           INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_path ON conversations(arrival_path)")
    _initialized = True


# ---- metrics -----------------------------------------------------------------


def _assistant_text(m: dict) -> str:
    """Canonical assistant content: the detailed variant is the fullest record,
    falling back to the summary variant, then the plain content field."""
    return m.get("detailed") or m.get("summary") or m.get("content") or ""


def _seconds_between(start_iso: Optional[str], end_iso: str) -> Optional[float]:
    if not start_iso:
        return None
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (ValueError, TypeError):
        return None
    return max(0.0, (end - start).total_seconds())


def _compute_metrics(messages: list[dict], started_at: Optional[str], now: str) -> dict:
    user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    asst_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    return {
        "message_count": len(messages),
        "user_message_count": len(user_msgs),
        "assistant_message_count": len(asst_msgs),
        "user_chars": sum(len(m.get("content") or "") for m in user_msgs),
        "assistant_chars": sum(len(_assistant_text(m)) for m in asst_msgs),
        # Span between the first message and the latest one (the conversation's
        # active lifetime), computed from the client-reported start.
        "duration_seconds": _seconds_between(started_at, now),
    }


def _first_user_message(messages: list[dict]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


# ---- write -------------------------------------------------------------------


def upsert_conversation(
    *,
    conv_id: str,
    user_id: str,
    arrival_path: str,
    model: str,
    entry_context: Any,
    messages: list[dict],
    started_at: Optional[str],
) -> None:
    """Insert or update a conversation, recomputing derived metrics.

    ``user_id`` and ``created_at``/``started_at`` are fixed on first insert and
    preserved on subsequent updates. Attempting to overwrite another user's
    conversation raises :class:`AccessError`.
    """
    _ensure_schema()
    now = _utcnow()
    entry_json = json.dumps(entry_context or {}, ensure_ascii=False)
    messages_json = json.dumps(messages or [], ensure_ascii=False)

    with _write_lock, _connect() as conn:
        existing = conn.execute(
            "SELECT user_id, created_at, started_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()

        if existing is not None and existing["user_id"] != user_id:
            raise AccessError(f"conversation {conv_id} belongs to another user")

        created_at = existing["created_at"] if existing else now
        # Prefer the earliest known start: the stored one, else the client's, else now.
        resolved_start = (existing["started_at"] if existing else None) or started_at or now
        metrics = _compute_metrics(messages, resolved_start, now)

        conn.execute(
            """
            INSERT INTO conversations (
                id, user_id, arrival_path, model, entry_context, messages,
                started_at, created_at, updated_at, duration_seconds,
                message_count, user_message_count, assistant_message_count,
                user_chars, assistant_chars
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                arrival_path            = excluded.arrival_path,
                model                   = excluded.model,
                entry_context           = excluded.entry_context,
                messages                = excluded.messages,
                started_at              = excluded.started_at,
                updated_at              = excluded.updated_at,
                duration_seconds        = excluded.duration_seconds,
                message_count           = excluded.message_count,
                user_message_count      = excluded.user_message_count,
                assistant_message_count = excluded.assistant_message_count,
                user_chars              = excluded.user_chars,
                assistant_chars         = excluded.assistant_chars
            """,
            (
                conv_id,
                user_id,
                arrival_path,
                model,
                entry_json,
                messages_json,
                resolved_start,
                created_at,
                now,
                metrics["duration_seconds"],
                metrics["message_count"],
                metrics["user_message_count"],
                metrics["assistant_message_count"],
                metrics["user_chars"],
                metrics["assistant_chars"],
            ),
        )


# ---- read --------------------------------------------------------------------

_SUMMARY_COLS = (
    "id, user_id, arrival_path, model, started_at, created_at, updated_at, "
    "duration_seconds, message_count, user_message_count, assistant_message_count, "
    "user_chars, assistant_chars, entry_context, messages"
)


def _row_to_summary(row: sqlite3.Row) -> dict:
    """List-view projection: metrics + a short preview, no full transcript."""
    try:
        messages = json.loads(row["messages"] or "[]")
    except (ValueError, TypeError):
        messages = []
    try:
        entry_context = json.loads(row["entry_context"] or "{}")
    except (ValueError, TypeError):
        entry_context = {}
    preview = _first_user_message(messages)
    if len(preview) > 160:
        preview = preview[:157] + "..."
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "arrival_path": row["arrival_path"],
        "model": row["model"],
        "started_at": row["started_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "duration_seconds": row["duration_seconds"],
        "message_count": row["message_count"],
        "user_message_count": row["user_message_count"],
        "assistant_message_count": row["assistant_message_count"],
        "user_chars": row["user_chars"],
        "assistant_chars": row["assistant_chars"],
        "preview": preview,
        "entry_context": entry_context,
    }


def list_conversations(
    *,
    viewer_id: str,
    is_admin: bool,
    scope: str = "own",
    path: Optional[str] = None,
    search: Optional[str] = None,
    min_messages: Optional[int] = None,
    max_messages: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Paginated list, newest activity first. ``scope='all'`` is admin-only and
    returns every user's conversations; otherwise only the viewer's own."""
    _ensure_schema()
    where: list[str] = []
    params: list[Any] = []

    if not (scope == "all" and is_admin):
        where.append("user_id = ?")
        params.append(viewer_id)
    if path:
        where.append("arrival_path = ?")
        params.append(path)
    if search:
        where.append("messages LIKE ?")
        params.append(f"%{search}%")
    if min_messages is not None:
        where.append("message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        where.append("message_count <= ?")
        params.append(max_messages)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM conversations {clause}", params
        ).fetchone()["c"]
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLS} FROM conversations {clause} "
            f"ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "scope": "all" if (scope == "all" and is_admin) else "own",
        "items": [_row_to_summary(r) for r in rows],
    }


def get_conversation(conv_id: str, *, viewer_id: str, is_admin: bool) -> Optional[dict]:
    """Full conversation incl. transcript. Owner or admin only; returns ``None``
    if not found, raises :class:`AccessError` if it belongs to someone else."""
    _ensure_schema()
    with _connect() as conn:
        row = conn.execute(
            f"SELECT {_SUMMARY_COLS} FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    if row is None:
        return None
    if row["user_id"] != viewer_id and not is_admin:
        raise AccessError(f"conversation {conv_id} belongs to another user")
    summary = _row_to_summary(row)
    try:
        summary["messages"] = json.loads(row["messages"] or "[]")
    except (ValueError, TypeError):
        summary["messages"] = []
    return summary


def usage_summary(*, viewer_id: str, is_admin: bool, scope: str = "own") -> dict:
    """Aggregate usage metrics for a dashboard. ``scope='all'`` is admin-only."""
    _ensure_schema()
    all_scope = scope == "all" and is_admin
    where = "" if all_scope else "WHERE user_id = ?"
    params: list[Any] = [] if all_scope else [viewer_id]

    with _connect() as conn:
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*)                       AS conversations,
                COALESCE(SUM(message_count), 0)          AS messages,
                COALESCE(SUM(user_message_count), 0)     AS user_messages,
                COALESCE(SUM(assistant_message_count), 0) AS assistant_messages,
                COUNT(DISTINCT user_id)        AS users,
                AVG(message_count)             AS avg_messages,
                AVG(duration_seconds)          AS avg_duration_seconds,
                COALESCE(SUM(user_chars), 0)   AS user_chars,
                COALESCE(SUM(assistant_chars), 0) AS assistant_chars
            FROM conversations {where}
            """,
            params,
        ).fetchone()

        by_path = conn.execute(
            f"SELECT arrival_path, COUNT(*) AS c FROM conversations {where} "
            f"GROUP BY arrival_path",
            params,
        ).fetchall()

        by_day = conn.execute(
            f"SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS c "
            f"FROM conversations {where} GROUP BY day ORDER BY day DESC LIMIT 30",
            params,
        ).fetchall()

        top_users = []
        if all_scope:
            top_users = [
                {"user_id": r["user_id"], "conversations": r["c"]}
                for r in conn.execute(
                    "SELECT user_id, COUNT(*) AS c FROM conversations "
                    "GROUP BY user_id ORDER BY c DESC LIMIT 10"
                ).fetchall()
            ]

    return {
        "scope": "all" if all_scope else "own",
        "conversations": totals["conversations"],
        "messages": totals["messages"],
        "user_messages": totals["user_messages"],
        "assistant_messages": totals["assistant_messages"],
        "users": totals["users"],
        "avg_messages": totals["avg_messages"],
        "avg_duration_seconds": totals["avg_duration_seconds"],
        "user_chars": totals["user_chars"],
        "assistant_chars": totals["assistant_chars"],
        "by_path": {r["arrival_path"] or "unknown": r["c"] for r in by_path},
        "by_day": [{"day": r["day"], "count": r["c"]} for r in by_day],
        "top_users": top_users,
    }
