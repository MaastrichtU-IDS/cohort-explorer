"""JSONL-based activity logger for analysis DCR wizard / creation events.

Each call to ``log_dcr_event`` appends one JSON object (one line) to
``<data_folder>/logs/analysis_dcr_events.jsonl`` using Python's
``TimedRotatingFileHandler`` (daily rotation, one year retention).

Inputs are assumed to be small and well-formed: the frontend caps free-text
fields (e.g. research question) and the backend only logs short scalars or
lists of identifiers. No size-guarding or truncation is performed here.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from src.config import settings

_LOGGER_NAME = "analysis_dcr_events"
_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """Return a process-wide singleton logger bound to a daily-rotating file."""
    global _logger
    if _logger is not None:
        return _logger

    logs_dir = os.path.join(settings.data_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "analysis_dcr_events.jsonl")

    lg = logging.getLogger(_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.propagate = False

    # Guard against re-adding handlers when this module is re-imported by
    # reload tooling (e.g. uvicorn --reload).
    if not lg.handlers:
        handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            backupCount=365,
            encoding="utf-8",
            utc=False,
        )
        handler.suffix = "%Y-%m-%d"
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)

    _logger = lg
    return lg


def log_dcr_event(
    event: str,
    user_email: str | None = None,
    session_id: str | None = None,
    **fields: Any,
) -> None:
    """Append one analysis DCR activity event to the JSONL log.

    Never raises; a logging failure must not break the DCR flow.
    """
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "user_email": user_email,
            "session_id": session_id,
            **fields,
        }
        _get_logger().info(json.dumps(record, default=str, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - logging must never raise
        logging.getLogger(__name__).warning(
            "Failed to log analysis DCR event %r: %s", event, exc
        )


def read_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Return events from the active and rotated JSONL files, newest first."""
    logs_dir = os.path.join(settings.data_folder, "logs")
    if not os.path.isdir(logs_dir):
        return []

    candidates = [
        os.path.join(logs_dir, name)
        for name in os.listdir(logs_dir)
        if name == "analysis_dcr_events.jsonl"
        or name.startswith("analysis_dcr_events.jsonl.")
    ]
    candidates.sort()

    events: list[dict[str, Any]] = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        events.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    events.reverse()
    if limit is not None:
        events = events[:limit]
    return events
