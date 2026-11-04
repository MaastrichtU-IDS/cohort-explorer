"""Admin settings endpoints — accessible only to users in the ADMINS list."""

import json
import logging
import os
import time
from typing import Any, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_403_FORBIDDEN

from src.auth import get_current_user
from src.config import settings
from src.cohort_cache import (
    get_cohorts_from_cache,
    add_cohort_to_cache,
    save_cache_to_disk,
)

router = APIRouter(prefix="/admin", tags=["admin"])

TIMECHF_TEST_EMAIL = "wei.wei@maastrichtuniversity.nl"
TIMECHF_COHORT_ID = "TIME-CHF"

# Persisted app settings toggled from the admin page (shared across workers via
# a small JSON file in the data folder).
APP_SETTINGS_FILE = os.path.join(settings.data_folder, "app_settings.json")
APP_SETTINGS_DEFAULTS = {
    # Whether the "iCARE-AI" nav button is shown to users.
    "ai_nav_enabled": False,
    # Whether My DCRs offers the "AI - Explain the Code" button, and, as a
    # second switch, whether it is offered for stopped (deactivated) DCRs.
    "code_explain_enabled": True,
    "code_explain_stopped_enabled": True,
    # Off = the traditional wizard opens directly.
}


def _load_app_settings() -> dict:
    result = dict(APP_SETTINGS_DEFAULTS)
    try:
        with open(APP_SETTINGS_FILE) as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            result.update({k: stored[k] for k in APP_SETTINGS_DEFAULTS if k in stored})
    except Exception:
        pass
    return result


def _save_app_settings(values: dict) -> None:
    os.makedirs(os.path.dirname(APP_SETTINGS_FILE), exist_ok=True)
    with open(APP_SETTINGS_FILE, "w") as fh:
        json.dump(values, fh, indent=2)


def _require_admin(user: dict[str, str]) -> str:
    """Return user email if admin, otherwise raise 403."""
    email = user["email"].lower()
    if email not in settings.admins_list:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Admin access required")
    return email


# ------------------------------------------------------------------
# GET /admin/check — is the current user an admin?
# ------------------------------------------------------------------
@router.get("/check")
def check_admin(user: Any = Depends(get_current_user)) -> dict:
    email = user["email"].lower()
    return {"is_admin": email in settings.admins_list}


# ------------------------------------------------------------------
# GET /admin/settings — retrieve current admin settings
# ------------------------------------------------------------------
@router.get("/settings")
def get_admin_settings(user: Any = Depends(get_current_user)) -> dict:
    _require_admin(user)

    cohorts = get_cohorts_from_cache("")
    timechf = cohorts.get(TIMECHF_COHORT_ID)

    timechf_testing = False
    if timechf:
        timechf_testing = TIMECHF_TEST_EMAIL in [
            e.lower() for e in (timechf.cohort_email or [])
        ]

    return {
        "timechf_testing_enabled": timechf_testing,
        "ai_nav_enabled": _load_app_settings()["ai_nav_enabled"],
        "code_explain_enabled": _load_app_settings()["code_explain_enabled"],
        "code_explain_stopped_enabled": _load_app_settings()["code_explain_stopped_enabled"],
    }


# ------------------------------------------------------------------
# GET /admin/public-settings — app settings anyone may read, logged in or
# not (used by the nav bar to decide whether to show the iCARE-AI button,
# which is visible to signed-out visitors too)
# ------------------------------------------------------------------
@router.get("/public-settings")
def get_public_settings() -> dict:
    values = _load_app_settings()
    return {
        "ai_nav_enabled": values["ai_nav_enabled"],
        "code_explain_enabled": values["code_explain_enabled"],
        "code_explain_stopped_enabled": values["code_explain_stopped_enabled"],
    }


def _toggle_setting(key: str, user: Any) -> dict:
    admin_email = _require_admin(user)
    values = _load_app_settings()
    values[key] = not values[key]
    _save_app_settings(values)
    logging.info("Admin %s set %s=%s", admin_email, key, values[key])
    return {k: values[k] for k in ("code_explain_enabled", "code_explain_stopped_enabled")}


# POST /admin/toggle-code-explain — AI code explanation on My DCRs on/off
@router.post("/toggle-code-explain")
def toggle_code_explain(user: Any = Depends(get_current_user)) -> dict:
    return _toggle_setting("code_explain_enabled", user)


# POST /admin/toggle-code-explain-stopped — the same, for stopped DCRs only
@router.post("/toggle-code-explain-stopped")
def toggle_code_explain_stopped(user: Any = Depends(get_current_user)) -> dict:
    return _toggle_setting("code_explain_stopped_enabled", user)


# ------------------------------------------------------------------
# POST /admin/toggle-ai-nav — show/hide the iCARE-AI nav button
# ------------------------------------------------------------------
@router.post("/toggle-ai-nav")
def toggle_ai_nav(user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)
    values = _load_app_settings()
    values["ai_nav_enabled"] = not values["ai_nav_enabled"]
    _save_app_settings(values)
    logging.info("Admin %s set ai_nav_enabled=%s", admin_email, values["ai_nav_enabled"])
    return {"ai_nav_enabled": values["ai_nav_enabled"]}


# ------------------------------------------------------------------


# ------------------------------------------------------------------
# POST /admin/toggle-timechf-testing — flip the switch
# ------------------------------------------------------------------
@router.post("/toggle-timechf-testing")
def toggle_timechf_testing(user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)

    # Operate on the *live* cache (pass empty email to skip can_edit logic)
    cohorts = get_cohorts_from_cache("")
    timechf = cohorts.get(TIMECHF_COHORT_ID)
    if not timechf:
        raise HTTPException(status_code=404, detail=f"Cohort {TIMECHF_COHORT_ID} not found in cache")

    emails_lower = [e.lower() for e in (timechf.cohort_email or [])]
    currently_enabled = TIMECHF_TEST_EMAIL in emails_lower

    if currently_enabled:
        # Remove
        timechf.cohort_email = [
            e for e in timechf.cohort_email if e.lower() != TIMECHF_TEST_EMAIL
        ]
        timechf.data_provider_email = ";".join(
            e for e in (timechf.data_provider_email or "").split(";") if e.strip() and e.strip().lower() != TIMECHF_TEST_EMAIL
        ) or None
        logging.info(
            "Admin %s DISABLED TIME-CHF testing capacity (removed %s)",
            admin_email, TIMECHF_TEST_EMAIL,
        )
    else:
        # Add (as a data provider too, so the test account is invited into DCRs)
        timechf.cohort_email.append(TIMECHF_TEST_EMAIL)
        timechf.data_provider_email = ";".join(
            [e for e in (timechf.data_provider_email or "").split(";") if e.strip()] + [TIMECHF_TEST_EMAIL]
        )
        logging.info(
            "Admin %s ENABLED TIME-CHF testing capacity (added %s)",
            admin_email, TIMECHF_TEST_EMAIL,
        )

    # Persist to the shared cache file so all workers pick it up
    add_cohort_to_cache(timechf, save_to_disk=True)

    return {
        "timechf_testing_enabled": not currently_enabled,
    }


# ------------------------------------------------------------------
# POST /admin/sparql — run a read-only SPARQL query against the triplestore
# ------------------------------------------------------------------
# The query text is sent to Oxigraph's /query endpoint, which only accepts
# queries (SELECT, ASK, CONSTRUCT, DESCRIBE); updates belong to /update and are
# refused there with HTTP 400, so nothing an admin types here can change data.
SPARQL_ADMIN_MAX_ROWS = 5000
SPARQL_ADMIN_TIMEOUT_S = 120


class SparqlQueryRequest(BaseModel):
    query: str
    max_rows: Optional[int] = None


@router.post("/sparql")
def run_admin_sparql(body: SparqlQueryRequest, user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="Empty query")
    max_rows = min(body.max_rows or SPARQL_ADMIN_MAX_ROWS, SPARQL_ADMIN_MAX_ROWS)

    logging.info("Admin %s runs SPARQL (%d chars): %s", admin_email, len(query), query[:200].replace("\n", " "))
    start = time.time()
    try:
        resp = requests.post(
            settings.query_endpoint,
            data=query.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-query",
                # SELECT/ASK come back as JSON, CONSTRUCT/DESCRIBE as Turtle
                "Accept": "application/sparql-results+json, text/turtle;q=0.9",
            },
            timeout=SPARQL_ADMIN_TIMEOUT_S,
        )
    except requests.Timeout:
        raise HTTPException(status_code=504, detail=f"The triplestore did not answer within {SPARQL_ADMIN_TIMEOUT_S} s")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the triplestore: {exc}")
    duration = round(time.time() - start, 3)

    if resp.status_code >= 400:
        # Oxigraph answers syntax errors and refused updates with a plain-text message
        raise HTTPException(status_code=400, detail=resp.text.strip()[:2000] or f"Triplestore returned HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        data = resp.json()
        if "boolean" in data:
            return {"kind": "boolean", "value": bool(data["boolean"]), "duration_s": duration}
        variables = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])
        rows = []
        for b in bindings[:max_rows]:
            row = []
            for v in variables:
                cell = b.get(v)
                if cell is None:
                    row.append(None)
                else:
                    value = cell.get("value", "")
                    if cell.get("type") == "literal":
                        if cell.get("xml:lang"):
                            value = f'"{value}"@{cell["xml:lang"]}'
                        elif cell.get("datatype"):
                            dt = cell["datatype"].rsplit("#", 1)[-1]
                            if dt not in ("string",):
                                value = f"{value} ({dt})"
                    row.append(value)
            rows.append(row)
        return {
            "kind": "bindings",
            "vars": variables,
            "rows": rows,
            "total": len(bindings),
            "truncated": len(bindings) > max_rows,
            "duration_s": duration,
        }
    return {"kind": "graph", "text": resp.text[:2_000_000], "truncated": len(resp.text) > 2_000_000, "duration_s": duration}


# ------------------------------------------------------------------
# GET /admin/omop-concept — what the upload validator knows about a concept
# ------------------------------------------------------------------
# The Athena/OMOP validation of an uploaded dictionary (validate_cde.py) asks
# the local OMOP graph first and only falls back to Athena when the id is not
# in the graph. This endpoint shows both sides for one (code, id) pair, so a
# "concept_id and concept_code do not match" complaint can be checked against
# the graph's own record of the id, the code index, and what Athena says.
ATHENA_CONCEPT_URL = "https://athena.ohdsi.org/api/v1/concepts"


def _file_info(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                "size_mb": round(st.st_size / 1e6, 1)}
    except OSError:
        return {"path": path, "exists": False}


@router.get("/omop-concept")
def lookup_omop_concept(
    code: Optional[str] = None,
    omop_id: Optional[str] = None,
    user: Any = Depends(get_current_user),
) -> dict:
    """Look up a prefixed concept code (e.g. UCUM:ng/mL) and/or an OMOP id in the
    local OMOP graph and in Athena, and give the validator's verdict for the pair."""
    admin_email = _require_admin(user)
    from CohortVarLinker.validate_cde import (  # noqa: WPS433 - lazy, heavy import
        get_omop_graph, split_code_prefixed, to_int_or_none, fetch_athena, rows_to_simple, _ATHENA_HEADERS,
    )
    from CohortVarLinker.src.config import settings as cvl_settings
    from CohortVarLinker.src.omop_graph_nx import _code_key

    code = (code or "").strip()
    vocab, bare_code = split_code_prefixed(code) if code else (None, None)
    vocabs = [v for v in (vocab if isinstance(vocab, list) else [vocab]) if v]
    cid = to_int_or_none(omop_id)
    if not bare_code and cid is None:
        raise HTTPException(status_code=422, detail="Give a prefixed concept code (e.g. UCUM:ng/mL), an OMOP id, or both")
    logging.info("Admin %s looks up OMOP concept code=%r id=%r", admin_email, code, cid)

    out: dict[str, Any] = {
        "input": {"code": code or None, "vocabulary": vocabs or None, "bare_code": bare_code, "omop_id": cid},
        "graph": {"pickle": _file_info(cvl_settings.omop_graph_pickle_path),
                  "source_csv": _file_info(cvl_settings.concepts_file_path)},
    }

    # --- local graph ---
    t0 = time.time()
    g = get_omop_graph()
    meta = g.graph.graph.get("meta")
    code_index = g.graph.graph.get("code_index") or {}
    out["graph"]["concepts"] = int(len(meta)) if meta is not None else 0
    out["graph"]["load_s"] = round(time.time() - t0, 1)

    def meta_row(concept_id: int) -> Optional[dict]:
        if meta is None or concept_id not in meta.index:
            return None
        r = meta.loc[concept_id]
        return {"concept_id": int(concept_id),
                "vocabulary": str(r.get("concept_vocabulary", "")),
                "code": str(r.get("concept_code", "")),
                "name": str(r.get("concept_name", "")),
                "class": str(r.get("concept_class", "")) if "concept_class" in meta.columns else None,
                "domain": str(r.get("concept_domain", "")) if "concept_domain" in meta.columns else None}

    out["graph"]["by_id"] = meta_row(cid) if cid is not None else None

    by_code: list[dict] = []
    if bare_code and vocabs and meta is not None:
        for v in vocabs:
            exact = code_index.get(_code_key(v, bare_code))
            if exact is not None:
                row = meta_row(int(exact))
                if row:
                    by_code.append(dict(row, match="exact"))
        if not by_code:
            # Case-insensitive fallback: tells a case difference (ng/ml vs ng/mL) apart from a wrong code.
            vocab_mask = meta["concept_vocabulary"].astype(str).str.lower().isin([v.lower() for v in vocabs])
            codes = meta.loc[vocab_mask, "concept_code"].astype(str)
            hits = codes[codes.str.lower() == bare_code.lower()]
            for concept_id in list(hits.index)[:10]:
                row = meta_row(int(concept_id))
                if row:
                    by_code.append(dict(row, match="case-insensitive"))
    out["graph"]["by_code"] = by_code

    if cid is not None and bare_code and vocabs:
        flag, status = g.concept_exists(cid, bare_code, vocabs)
        out["verdict"] = {"status": status, "pass": bool(flag),
                          "meaning": {"correct": "the graph agrees: this id carries this code",
                                      "incorrect": "the graph has this id with another code or vocabulary - the upload fails with 'concept_id and concept_code do not match'",
                                      "not found": "the id is not in the graph - the upload validator asks Athena instead"}.get(status, status)}
    else:
        out["verdict"] = None

    # --- Athena ---
    athena: dict[str, Any] = {"by_id": None, "by_code": [], "error": None}
    try:
        if cid is not None:
            r = requests.get(f"{ATHENA_CONCEPT_URL}/{cid}", headers=_ATHENA_HEADERS, timeout=20)
            if r.ok:
                d = r.json()
                athena["by_id"] = {"concept_id": d.get("id"), "code": d.get("code"), "name": d.get("name"),
                                   "vocabulary": d.get("vocabularyId"), "domain": d.get("domainId"),
                                   "class": d.get("conceptClassId"), "standard": d.get("standardConcept"),
                                   "invalid_reason": d.get("invalidReason")}
            else:
                athena["error"] = f"Athena returned HTTP {r.status_code} for id {cid}"
        if bare_code and vocabs:
            athena["by_code"] = [c for c in rows_to_simple(fetch_athena(bare_code, vocabs)) if c["code"] == bare_code] \
                or rows_to_simple(fetch_athena(bare_code, vocabs))
    except requests.RequestException as exc:
        athena["error"] = f"Could not reach Athena: {exc}"
    out["athena"] = athena
    return out
