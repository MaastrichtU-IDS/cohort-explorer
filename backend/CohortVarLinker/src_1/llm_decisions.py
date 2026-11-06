"""Durable record of every LLM mapping decision.

Each time the cross-study mapping asks the LLM about one (source variable,
target variable) pair, the full input (the pair as the model saw it, the system
prompt, the prompt) and the full output (raw reply, parsed verdict, confidence)
are stored in one SQLite database, with the mapping run they belong to. A
content-based `pair_key` identifies the same pair across runs, so verdicts for
one pair can be compared between runs and models (consistency check).

The store is passive: llm_call.LLMConceptMatcher.assess() calls
record_decision() for every pair it assesses (cache hits included, flagged),
main.generate_mapping_csv() brackets its work with begin_run()/end_run() and
names the pair of studies being mapped with set_pair(). Nothing here may ever
break a mapping run: every public function swallows and logs its own errors.

Database: LLM_DECISIONS_DB (env) or <mapping output dir>/../llm_decisions.sqlite,
i.e. on the volume-mounted CohortVarLinker/data folder, so it survives rebuilds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from .config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_db_path: Optional[str] = None
# Context of the mapping run in progress (one run at a time: main.py serialises
# runs behind its lock), read by record_decision().
_ctx: dict[str, Any] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started       TEXT NOT NULL,
    finished      TEXT,
    status        TEXT,
    source_study  TEXT,
    target_studies TEXT,
    embed_model   TEXT,
    mapping_mode  TEXT,
    llm_model     TEXT,
    requested_by  TEXT,
    n_decisions   INTEGER DEFAULT 0,
    n_cached      INTEGER DEFAULT 0,
    n_failed      INTEGER DEFAULT 0,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    created_at    TEXT NOT NULL,
    model         TEXT,
    mapping_mode  TEXT,
    source_study  TEXT,
    target_study  TEXT,
    pair_key      TEXT NOT NULL,
    src_desc      TEXT, src_concepts TEXT, src_cats TEXT, src_unit TEXT,
    tgt_desc      TEXT, tgt_concepts TEXT, tgt_cats TEXT, tgt_unit TEXT,
    evidence      TEXT,
    system_prompt TEXT,
    prompt        TEXT,
    raw_response  TEXT,
    matched       INTEGER,            -- 1 / 0 / NULL (pending)
    ctx_type      TEXT,
    confidence    REAL,
    status        TEXT,               -- COMPLETE / COMPATIBLE / PARTIAL / IMPOSSIBLE / PENDING / PARSE_ERROR
    reason        TEXT,
    transform     TEXT,
    transform_direction TEXT,
    verdict_json  TEXT,
    cached        INTEGER DEFAULT 0,
    attempts      INTEGER,
    duration_ms   INTEGER,
    round_idx     INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS ix_decisions_pair ON decisions(pair_key);
CREATE INDEX IF NOT EXISTS ix_decisions_status ON decisions(status);
CREATE INDEX IF NOT EXISTS ix_decisions_studies ON decisions(source_study, target_study);
CREATE TABLE IF NOT EXISTS pairs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    source_study  TEXT,
    target_study  TEXT,
    started       TEXT NOT NULL,
    finished      TEXT,
    status        TEXT,               -- running / done / failed / crashed
    compute_seconds REAL,
    n_decisions   INTEGER,            -- pairs of variables the LLM assessed
    n_calls       INTEGER,            -- HTTP calls actually made (cache hits excluded, retries included)
    n_cached      INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_pairs_studies ON pairs(source_study, target_study);
"""

# A run (or a pair) still "running" after this long did not finish: the worker
# died or was restarted, so end_run() was never called.
STALE_RUN_HOURS = 3
STALE_STATUS = "crashed"
STALE_ERROR = f"did not finish: no completion recorded within {STALE_RUN_HOURS} hours (worker crashed or restarted)"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def configure(db_path: str) -> None:
    global _db_path
    _db_path = db_path


def db_path() -> str:
    if _db_path:
        return _db_path
    env = os.getenv("LLM_DECISIONS_DB")
    if env:
        return env
    return os.path.join(os.path.dirname(settings.output_dir), "llm_decisions.sqlite")


def _connect() -> sqlite3.Connection:
    path = db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe(fn):
    """Decorator: a failure in the decision store is logged, never raised."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive by design
            logger.warning("[llm_decisions] %s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Run context (written by main.py / the backend, read when recording)
# ---------------------------------------------------------------------------

def set_context(**fields: Any) -> None:
    """Merge fields (e.g. requested_by) into the context of the next/current run."""
    _ctx.update({k: v for k, v in fields.items() if v is not None})


@_safe
def begin_run(source_study: str, target_studies: list[str], embed_model: str,
              mapping_mode: str, llm_model: Optional[str]) -> str:
    mark_stale_runs()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    _ctx.update(run_id=run_id, source_study=source_study, target_study=None,
                mapping_mode=mapping_mode, llm_model=llm_model)
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, started, status, source_study, target_studies, embed_model, "
            "mapping_mode, llm_model, requested_by) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, _now(), "running", source_study, json.dumps(list(target_studies)), embed_model,
             mapping_mode, llm_model, _ctx.get("requested_by")))
    return run_id


def set_pair(source_study: Optional[str], target_study: Optional[str]) -> None:
    _ctx["source_study"] = source_study
    _ctx["target_study"] = target_study


@_safe
def begin_pair(source_study: str, target_study: str) -> Optional[int]:
    """One (source, target) mapping starts: sets the context for record_decision
    and opens a row in `pairs` whose timing and LLM counts end_pair() fills."""
    set_pair(source_study, target_study)
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO pairs (run_id, source_study, target_study, started, status) VALUES (?,?,?,?,?)",
            (_ctx.get("run_id"), source_study, target_study, _now(), "running"))
        pair_id = cur.lastrowid
    _ctx["pair_id"] = pair_id
    _ctx["pair_t0"] = time.time()
    return pair_id


@_safe
def end_pair(status: str = "done", error: Optional[str] = None) -> None:
    pair_id = _ctx.pop("pair_id", None)
    t0 = _ctx.pop("pair_t0", None)
    source, target, run_id = _ctx.get("source_study"), _ctx.get("target_study"), _ctx.get("run_id")
    set_pair(None, None)
    if pair_id is None:
        return
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN cached=0 THEN COALESCE(attempts, 1) ELSE 0 END),0) calls, "
            "COALESCE(SUM(cached),0) c FROM decisions WHERE run_id IS ? AND source_study IS ? AND target_study IS ?",
            (run_id, source, target)).fetchone()
        conn.execute(
            "UPDATE pairs SET finished=?, status=?, compute_seconds=?, n_decisions=?, n_calls=?, n_cached=?, error=? WHERE id=?",
            (_now(), status, round(time.time() - t0, 1) if t0 else None, row["n"], row["calls"], row["c"], error, pair_id))


@_safe
def end_run(run_id: Optional[str], status: str = "done", error: Optional[str] = None) -> None:
    if not run_id:
        return
    if _ctx.get("pair_id") is not None:
        end_pair(status=status if status != "done" else "failed", error=error or "run ended before the pair finished")
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cached),0) AS c, "
            "COALESCE(SUM(CASE WHEN matched IS NULL THEN 1 ELSE 0 END),0) AS f "
            "FROM decisions WHERE run_id=?", (run_id,)).fetchone()
        conn.execute(
            "UPDATE runs SET finished=?, status=?, error=?, n_decisions=?, n_cached=?, n_failed=? WHERE run_id=?",
            (_now(), status, error, row["n"], row["c"], row["f"], run_id))
    for k in ("run_id", "target_study"):
        _ctx.pop(k, None)
    _ctx.pop("requested_by", None)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _norm(x: Any) -> str:
    return " ".join(str(x or "").lower().split())


def pair_key(mode: str, g: dict, t: dict) -> str:
    """Content hash of a (source, target) pair as the model sees it, independent
    of run, study context and prompt wording: the identity of the question."""
    parts = [mode, _norm(g.get("src_desc")), _norm(g.get("src_concepts")), _norm(g.get("src_cats")),
             _norm(g.get("src_unit")), _norm(t.get("desc")), _norm(t.get("tgt_concepts")),
             _norm(t.get("tgt_cats")), _norm(t.get("tgt_unit"))]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


@_safe
def record_decision(*, g: dict, t: dict, prompt: str, system_prompt: str, model: str, mode: str,
                    verdict: Optional[tuple], meta: Optional[dict] = None, round_idx: int = 0) -> None:
    """Store one assessed pair. `verdict` is llm_call's (matched, ctx_type, conf,
    reason_json) tuple; `meta` comes from _eval_pair: raw, cached, attempts,
    duration_ms, error."""
    meta = meta or {}
    matched, ctx_type, conf, reason_json = verdict if verdict else (None, "PENDING", 0.0, "")
    d: dict[str, Any] = {}
    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
        if not isinstance(d, dict):
            d = {"status": str(d)}
    except Exception:
        d = {"status": reason_json}
    transform = d.get("transform")
    if isinstance(transform, (dict, list)):
        transform = json.dumps(transform, ensure_ascii=False)
    row = (
        _ctx.get("run_id"), _now(), model, mode, _ctx.get("source_study"), _ctx.get("target_study"),
        pair_key(mode, g, t),
        g.get("src_desc"), str(g.get("src_concepts") or ""), g.get("src_cats"), g.get("src_unit"),
        t.get("desc"), str(t.get("tgt_concepts") or ""), t.get("tgt_cats"), t.get("tgt_unit"),
        t.get("evidence"), system_prompt, prompt, meta.get("raw"),
        None if matched is None else int(bool(matched)), ctx_type, float(conf or 0.0),
        str(d.get("status") or ctx_type or ""), d.get("reason"), transform or None,
        d.get("transform_direction") or None, reason_json if isinstance(reason_json, str) else json.dumps(reason_json),
        int(bool(meta.get("cached"))), meta.get("attempts"), meta.get("duration_ms"), round_idx, meta.get("error"),
    )
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO decisions (run_id, created_at, model, mapping_mode, source_study, target_study, pair_key, "
            "src_desc, src_concepts, src_cats, src_unit, tgt_desc, tgt_concepts, tgt_cats, tgt_unit, evidence, "
            "system_prompt, prompt, raw_response, matched, ctx_type, confidence, status, reason, transform, "
            "transform_direction, verdict_json, cached, attempts, duration_ms, round_idx, error) "
            "VALUES (" + ",".join("?" * 32) + ")", row)


@_safe
def mark_stale_runs(max_hours: float = STALE_RUN_HOURS) -> int:
    """Runs and pairs still 'running' but started more than max_hours ago are
    marked crashed. Called before listing and before every new run, so the
    store never shows a dead run as in progress for long."""
    cutoff = (datetime.now() - timedelta(hours=max_hours)).isoformat(timespec="seconds")
    with _lock, _connect() as conn:
        n = conn.execute("UPDATE runs SET status=?, error=? WHERE status='running' AND started < ?",
                         (STALE_STATUS, STALE_ERROR, cutoff)).rowcount
        conn.execute("UPDATE pairs SET status=?, error=? WHERE status='running' AND started < ?",
                     (STALE_STATUS, STALE_ERROR, cutoff))
        # Counts for a crashed run, so its row shows what was done before it died.
        for r in conn.execute("SELECT run_id FROM runs WHERE status=? AND n_decisions=0", (STALE_STATUS,)).fetchall():
            c = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(cached),0) AS c, "
                "COALESCE(SUM(CASE WHEN matched IS NULL THEN 1 ELSE 0 END),0) AS f FROM decisions WHERE run_id=?",
                (r["run_id"],)).fetchone()
            conn.execute("UPDATE runs SET n_decisions=?, n_cached=?, n_failed=? WHERE run_id=?",
                         (c["n"], c["c"], c["f"], r["run_id"]))
    return n or 0


def pair_stats(source_study: str, target_study: str) -> Optional[dict]:
    """Compute time and LLM counts of the most recent finished mapping of this
    pair. From the `pairs` table when the mapping ran with it; for runs that
    predate it, the LLM counts come from the decisions and the time is the span
    of their timestamps (the LLM phase only, flagged approximate)."""
    src, tgt = (source_study or "").lower(), (target_study or "").lower()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT finished, compute_seconds, n_decisions, n_calls, n_cached, status, run_id FROM pairs "
                "WHERE LOWER(source_study)=? AND LOWER(target_study)=? AND status='done' ORDER BY id DESC LIMIT 1",
                (src, tgt)).fetchone()
            if row:
                run = conn.execute("SELECT llm_model FROM runs WHERE run_id=?", (row["run_id"],)).fetchone()
                return {"finished": row["finished"], "compute_seconds": row["compute_seconds"], "approximate": False,
                        "llm_calls": row["n_calls"], "llm_pairs": row["n_decisions"], "llm_cached": row["n_cached"],
                        "model": run["llm_model"] if run else None, "run_id": row["run_id"]}
            # Fallback: the latest run that has decisions for this pair.
            # Only runs that finished: a crashed run's decisions never produced a file.
            d = conn.execute(
                "SELECT d.run_id, MIN(d.created_at) t0, MAX(d.created_at) t1, COUNT(*) n, "
                "COALESCE(SUM(CASE WHEN d.cached=0 THEN COALESCE(d.attempts,1) ELSE 0 END),0) calls, "
                "COALESCE(SUM(d.cached),0) c, MIN(d.model) model "
                "FROM decisions d JOIN runs r ON r.run_id = d.run_id "
                "WHERE LOWER(d.source_study)=? AND LOWER(d.target_study)=? AND r.status='done' "
                "GROUP BY d.run_id ORDER BY t1 DESC LIMIT 1",
                (src, tgt)).fetchone()
            if not d:
                return None
            span = (datetime.fromisoformat(d["t1"]) - datetime.fromisoformat(d["t0"])).total_seconds()
            return {"finished": d["t1"], "compute_seconds": round(span, 1), "approximate": True,
                    "llm_calls": d["calls"], "llm_pairs": d["n"], "llm_cached": d["c"], "model": d["model"], "run_id": d["run_id"]}
    except Exception as exc:
        logger.warning("[llm_decisions] pair_stats failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Queries (for the admin API)
# ---------------------------------------------------------------------------

_LIST_COLUMNS = ("id, run_id, created_at, model, mapping_mode, source_study, target_study, pair_key, "
                 "src_desc, src_concepts, src_cats, src_unit, tgt_desc, tgt_concepts, tgt_cats, tgt_unit, "
                 "matched, ctx_type, confidence, status, reason, transform, transform_direction, cached, "
                 "attempts, duration_ms, round_idx, error")


def list_runs(limit: int = 200) -> list[dict]:
    mark_stale_runs()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["target_studies"] = json.loads(d.get("target_studies") or "[]")
        except ValueError:
            d["target_studies"] = []
        out.append(d)
    return out


def stats() -> dict:
    mark_stale_runs()
    with _connect() as conn:
        tot = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(cached),0) c FROM decisions").fetchone()
        by_status = conn.execute("SELECT status, COUNT(*) n FROM decisions GROUP BY status ORDER BY n DESC").fetchall()
        runs = conn.execute("SELECT COUNT(*) n FROM runs").fetchone()
        pairs = conn.execute("SELECT COUNT(DISTINCT pair_key) n FROM decisions").fetchone()
        models = conn.execute("SELECT model, COUNT(*) n FROM decisions GROUP BY model ORDER BY n DESC").fetchall()
    return {"decisions": tot["n"], "cached": tot["c"], "runs": runs["n"], "distinct_pairs": pairs["n"],
            "by_status": {r["status"]: r["n"] for r in by_status},
            "by_model": {r["model"]: r["n"] for r in models}, "db_path": db_path()}


def list_decisions(run_id: Optional[str] = None, status: Optional[str] = None, source: Optional[str] = None,
                   target: Optional[str] = None, model: Optional[str] = None, q: Optional[str] = None,
                   cached: Optional[bool] = None, limit: int = 200, offset: int = 0) -> dict:
    where, args = [], []
    if run_id:
        where.append("run_id=?"); args.append(run_id)
    if status:
        where.append("status=?"); args.append(status.upper())
    if source:
        where.append("LOWER(source_study)=?"); args.append(source.lower())
    if target:
        where.append("LOWER(target_study)=?"); args.append(target.lower())
    if model:
        where.append("model=?"); args.append(model)
    if cached is not None:
        where.append("cached=?"); args.append(int(cached))
    if q:
        like = f"%{q}%"
        where.append("(src_desc LIKE ? OR tgt_desc LIKE ? OR src_concepts LIKE ? OR tgt_concepts LIKE ? "
                     "OR reason LIKE ? OR pair_key = ?)")
        args += [like, like, like, like, like, q]
    sql_where = (" WHERE " + " AND ".join(where)) if where else ""
    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) n FROM decisions{sql_where}", args).fetchone()["n"]
        rows = conn.execute(
            f"SELECT {_LIST_COLUMNS} FROM decisions{sql_where} ORDER BY id DESC LIMIT ? OFFSET ?",
            args + [limit, offset]).fetchall()
    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


def get_decision(decision_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
    return dict(row) if row else None


def pair_history(key: str) -> list[dict]:
    """Every decision ever recorded for one pair, oldest first."""
    with _connect() as conn:
        rows = conn.execute(f"SELECT {_LIST_COLUMNS} FROM decisions WHERE pair_key=? ORDER BY id", (key,)).fetchall()
    return [dict(r) for r in rows]


def consistency(min_decisions: int = 2, only_conflicts: bool = True, limit: int = 500) -> dict:
    """Pairs asked more than once (across runs, models or retries) and whether the
    verdicts agree. A conflict is more than one distinct status for the same
    pair_key, PENDING/PARSE_ERROR left aside (those are failures, not opinions)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT pair_key, COUNT(*) n, COUNT(DISTINCT run_id) n_runs, COUNT(DISTINCT model) n_models, "
            "COUNT(DISTINCT CASE WHEN status IN ('COMPLETE','COMPATIBLE','PARTIAL','IMPOSSIBLE') THEN status END) n_status, "
            "GROUP_CONCAT(DISTINCT status) statuses, MIN(src_desc) src_desc, MIN(tgt_desc) tgt_desc, "
            "MIN(source_study) source_study, MIN(target_study) target_study, "
            "MIN(confidence) min_conf, MAX(confidence) max_conf, MIN(created_at) first_seen, MAX(created_at) last_seen "
            "FROM decisions GROUP BY pair_key HAVING COUNT(*) >= ? "
            + ("AND n_status > 1 " if only_conflicts else "")
            + "ORDER BY n_status DESC, n DESC LIMIT ?", (min_decisions, limit)).fetchall()
        summary = conn.execute(
            "SELECT COUNT(*) pairs, SUM(CASE WHEN n_status > 1 THEN 1 ELSE 0 END) conflicts FROM ("
            "SELECT pair_key, COUNT(DISTINCT CASE WHEN status IN ('COMPLETE','COMPATIBLE','PARTIAL','IMPOSSIBLE') "
            "THEN status END) n_status FROM decisions GROUP BY pair_key HAVING COUNT(*) >= ?)", (min_decisions,)).fetchone()
    items = []
    for r in rows:
        d = dict(r)
        d["statuses"] = sorted(set((d.get("statuses") or "").split(",")) - {""})
        d["conflict"] = d["n_status"] > 1
        items.append(d)
    return {"repeated_pairs": summary["pairs"] or 0, "conflicting_pairs": summary["conflicts"] or 0, "items": items}
