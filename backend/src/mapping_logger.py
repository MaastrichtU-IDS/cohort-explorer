"""Centralized structured mapping-activity logger.

Writes **JSONL** (one JSON object per line) to a single file so that:
 - the file can be tailed / streamed,
 - a future web endpoint can parse it line-by-line, and
 - both mapping processes share the same timeline.

Two detail levels
-----------------
* **MAIN** – high-level milestones (start/end of a run, per-target summary, …).
  Always carries a human-readable timestamp.
* **DETAIL** – per-variable or per-step granularity.

Every entry also carries:
  ``process``  – ``"cohort_var_linker"`` or ``"standard_code_mapping"``
  ``depth``    – 0 for top-level events, 1+ for nested sub-events
  ``ctx``      – optional dict of structured context data
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

# ── Constants ────────────────────────────────────────────────────────
LEVEL_MAIN = "MAIN"
LEVEL_DETAIL = "DETAIL"

PROCESS_CVL = "cohort_var_linker"
PROCESS_SCM = "standard_code_mapping"

_LOG_PATH: str | None = None


def _get_log_path() -> str:
    """Resolve (and lazily create) the JSONL log file path."""
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = os.path.join(settings.data_folder, "mapping_activity.jsonl")
        Path(_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so it exists for readers
        if not os.path.exists(_LOG_PATH):
            Path(_LOG_PATH).touch()
    return _LOG_PATH


def _write_entry(entry: dict) -> None:
    """Append a single JSON line to the log file (cross-process safe via fcntl.flock)."""
    path = _get_log_path()
    line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ── Public helpers ───────────────────────────────────────────────────

def log_main(
    process: str,
    event: str,
    msg: str,
    *,
    ctx: dict[str, Any] | None = None,
    depth: int = 0,
) -> None:
    """Log a **MAIN**-level event (milestone / phase boundary)."""
    _write_entry({
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": LEVEL_MAIN,
        "process": process,
        "event": event,
        "msg": msg,
        "ctx": ctx or {},
        "depth": depth,
    })


def log_detail(
    process: str,
    event: str,
    msg: str,
    *,
    ctx: dict[str, Any] | None = None,
    depth: int = 1,
) -> None:
    """Log a **DETAIL**-level event (per-variable, per-step, …)."""
    _write_entry({
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": LEVEL_DETAIL,
        "process": process,
        "event": event,
        "msg": msg,
        "ctx": ctx or {},
        "depth": depth,
    })


# ── Convenience class for scoping a mapping "run" ───────────────────

class MappingRun:
    """Lightweight context-manager that auto-logs start / end + elapsed time.

    Usage::

        with MappingRun(PROCESS_CVL, source="time-chf", targets=["gissi-hf"]):
            ...
    """

    def __init__(self, process: str, **run_ctx: Any):
        self.process = process
        self.run_ctx = run_ctx
        self._t0: float = 0.0

    def __enter__(self) -> "MappingRun":
        self._t0 = time.monotonic()
        log_main(self.process, "run_started",
                 f"Mapping run started", ctx=self.run_ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = round(time.monotonic() - self._t0, 2)
        if exc_type is not None:
            log_main(self.process, "run_failed",
                     f"Mapping run failed after {elapsed}s: {exc_val}",
                     ctx={**self.run_ctx, "elapsed_s": elapsed,
                          "error": str(exc_val)})
        else:
            log_main(self.process, "run_completed",
                     f"Mapping run completed in {elapsed}s",
                     ctx={**self.run_ctx, "elapsed_s": elapsed})
        return False  # don't suppress exceptions
