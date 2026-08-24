"""No-code DCR analysis API.

Backs the /nocode-dcr wizard for domain experts who do not program:

  - variable search across the selected cohorts (name, label, standard concept
    name, code / OMOP id, and "equivalent names": variables in other cohorts
    that share a concept code)
  - evidence-ranked match suggestions for a chosen variable in the other
    cohorts: shared codes, text similarity (language-neutral token matching),
    rows of cached cross-mapping files, and optional AI suggestions via the
    platform's local LLM
  - value-level suggestions for categorical variables (category codes, label
    similarity, the cached mapping's value_mapping)
  - persistence of user mapping specs (reusable across DCRs)
  - running a no-code DCR node via the service account and serving the
    aggregate results (figures/tables) back to the wizard

The enclave script itself is generated in nocode_scripts.py.
"""

from __future__ import annotations

import ast
import csv
import difflib
import glob
import io
import json
import logging
import os
import re
import unicodedata
import uuid
import zipfile
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings
from src.nocode_scripts import ANALYSIS_KINDS, describe_spec

router = APIRouter(prefix="/api/nocode", tags=["nocode"])
logger = logging.getLogger(__name__)

MAPPINGS_DIR = os.path.join(settings.data_folder, "nocode_mappings")
RESULTS_DIR = os.path.join(settings.data_folder, "nocode_results")
# One JSON per no-code room: its analysis nodes, data source and specs (the
# whole configuration), written when the room is created.
ROOMS_DIR = os.path.join(settings.data_folder, "nocode_rooms")


def record_nocode_room(dcr_id: str, info: dict[str, Any]) -> None:
    """Persist a no-code room's configuration (called by the DCR creation code)."""
    try:
        os.makedirs(ROOMS_DIR, exist_ok=True)
        with open(os.path.join(ROOMS_DIR, f"{os.path.basename(dcr_id)}.json"), "w", encoding="utf-8") as fh:
            json.dump({"dcr_id": dcr_id, **info}, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("could not record no-code room %s: %s", dcr_id, exc)


def load_nocode_room(dcr_id: str) -> Optional[dict[str, Any]]:
    p = os.path.join(ROOMS_DIR, f"{os.path.basename(dcr_id)}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _migrate_legacy_dirs() -> None:
    """The feature was first shipped as "guided": move its data folders to the
    new names once, so saved mappings and fetched results are not orphaned."""
    for old, new in (("guided_mappings", MAPPINGS_DIR), ("guided_results", RESULTS_DIR)):
        old_path = os.path.join(settings.data_folder, old)
        try:
            if os.path.isdir(old_path) and not os.path.exists(new):
                os.rename(old_path, new)
                logger.info("migrated %s -> %s", old_path, new)
        except Exception as exc:
            logger.warning("could not migrate %s: %s", old_path, exc)


_migrate_legacy_dirs()

# ---------------------------------------------------------------------------
# Normalization + similarity
# ---------------------------------------------------------------------------

def _ascii(text: str) -> str:
    text = str(text or "")
    text = text.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _tokens(text: str) -> list[str]:
    t = _ascii(text)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)
    toks = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", t) if w]
    return toks


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _ratio(a: str, b: str) -> float:
    a, b = " ".join(_tokens(a)), " ".join(_tokens(b))
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def text_similarity(v1: dict, v2: dict) -> float:
    """0..1 similarity between two variable summaries using name, label and
    standard concept name, after token normalization (case, accents, camelCase,
    separators). Language-neutral by design: cross-language matches come from
    shared codes, cached mapping files or the AI suggestions, not from a
    hand-made dictionary."""
    names = _jaccard(_tokens(v1.get("var_name")), _tokens(v2.get("var_name")))
    labels = max(_ratio(v1.get("var_label"), v2.get("var_label")),
                 _jaccard(_tokens(v1.get("var_label")), _tokens(v2.get("var_label"))))
    cn1, cn2 = _ascii(v1.get("concept_name") or "").lower(), _ascii(v2.get("concept_name") or "").lower()
    concept = 1.0 if cn1 and cn1 == cn2 else (_ratio(cn1, cn2) if cn1 and cn2 else 0.0)
    cross = max(_ratio(v1.get("var_label"), v2.get("var_name")), _ratio(v1.get("var_name"), v2.get("var_label")))
    # Abbreviations and concatenations ("NTBNP" vs "BNP1" / "NT pro-BNP"): compare
    # the squashed, digit-stripped forms of names and labels by containment.
    def squash(text: Any) -> str:
        return re.sub(r"[^a-z]", "", _ascii(text).lower())
    s1 = {squash(v1.get("var_name")), squash(v1.get("var_label")), squash(v1.get("concept_name"))} - {""}
    s2 = {squash(v2.get("var_name")), squash(v2.get("var_label")), squash(v2.get("concept_name"))} - {""}
    contain = 0.0
    for x in s1:
        for y in s2:
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            if len(short) >= 3 and short in long_:
                contain = max(contain, 0.5 + 0.5 * len(short) / len(long_))
    return round(max(0.45 * names + 0.55 * labels, concept, 0.5 * labels + 0.5 * concept, 0.8 * cross, contain), 3)


def _norm_code(code: Any) -> str:
    return _ascii(code).strip().lower() if code else ""


# ---------------------------------------------------------------------------
# EDA statistics (from the cohort's profiling output, when it exists)
# ---------------------------------------------------------------------------

_eda_cache: dict[str, tuple[float, dict]] = {}


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in ("nan", "n/a", "none", "—")):
            return None
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _load_eda(cohort_id: str) -> dict[str, dict]:
    """{lower-cased variable name -> normalized stats} for a cohort, or {} when
    no EDA output exists. Handles both EDA formats (v1 flat keys such as
    'std dev' / 'count missing'; v2 numeric keys under 'variables')."""
    # v2 outputs live in a differently NAMED file (eda_output_v2_<id>.json);
    # prefer it, fall back to the legacy v1 name.
    dcr_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
    path = os.path.join(dcr_dir, f"eda_output_v2_{cohort_id}.json")
    if not os.path.exists(path):
        path = os.path.join(dcr_dir, f"eda_output_{cohort_id}.json")
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    cached = _eda_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.warning("EDA output for %s unreadable: %s", cohort_id, exc)
        return {}
    entries = raw.get("variables") if isinstance(raw, dict) and isinstance(raw.get("variables"), dict) else raw
    out: dict[str, dict] = {}
    if isinstance(entries, dict):
        for name, e in entries.items():
            if not isinstance(e, dict):
                continue
            missing_pct = None
            if e.get("completeness") is not None:
                c = _num(e.get("completeness"))
                if c is not None:
                    missing_pct = round(100 - (c * 100 if c <= 1 else c), 1)
            elif e.get("count missing"):
                m = re.search(r"\((\d+\.?\d*)%\)", str(e.get("count missing")))
                if m:
                    missing_pct = float(m.group(1))
            stats = {
                "n": _num(e.get("n")) if e.get("n") is not None else _num(e.get("count of observations (ex. missing/empty)")),
                "missing_pct": missing_pct,
                "mean": _num(e.get("mean")),
                "std": _num(e.get("std")) if e.get("std") is not None else _num(e.get("std dev")),
                "median": _num(e.get("median")),
                "min": _num(e.get("min")),
                "max": _num(e.get("max")),
                "q1": _num(e.get("q1")),
                "q3": _num(e.get("q3")),
                "n_unique": _num(e.get("n_unique")),
                # categorical frequencies (EDA v2): [{value, label, count, pct}]
                "distribution": [
                    {"value": str(d.get("value", "")), "label": str(d.get("label", "") or d.get("value", "")),
                     "count": _num(d.get("count")), "pct": _num(d.get("pct"))}
                    for d in (e.get("distribution") or []) if isinstance(d, dict)
                ][:12] or None,
                "type": e.get("type") or (e.get("metadata") or {}).get("type") if isinstance(e.get("metadata"), dict) else e.get("type"),
            }
            if any(v is not None for k, v in stats.items() if k not in ("type", "distribution")) or stats["distribution"]:
                out[str(name).strip().lower()] = stats
    _eda_cache[path] = (mtime, out)
    return out


def _code_system(code: str) -> str:
    """Human name of a CURIE's vocabulary: 'snomed:184107009' -> 'SNOMED'."""
    prefix = str(code or "").split(":")[0].strip().lower()
    names = {"snomed": "SNOMED", "loinc": "LOINC", "rxnorm": "RxNorm", "atc": "ATC", "icd10": "ICD-10",
             "icd10cm": "ICD-10-CM", "icd9": "ICD-9", "ucum": "UCUM", "omop": "OMOP", "hpo": "HPO", "ncit": "NCIT",
             "mesh": "MeSH", "cpt": "CPT", "ndc": "NDC", "read": "Read"}
    return names.get(prefix, prefix.upper() or "code")


# ---------------------------------------------------------------------------
# Variable index
# ---------------------------------------------------------------------------

def _var_summary(cohort_id: str, v: Any) -> dict[str, Any]:
    cats = []
    for c in getattr(v, "categories", None) or []:
        cats.append({
            "value": str(getattr(c, "value", "") or ""),
            "label": str(getattr(c, "label", "") or ""),
            "concept_code": getattr(c, "concept_id", None),
            "omop_id": getattr(c, "mapped_id", None),
            "concept_name": getattr(c, "mapped_label", None),
        })
    var_type = (getattr(v, "var_type", "") or "").upper()
    # The kind is a fact of the dictionary, never a user choice: categorical only
    # when categories are declared; numeric for numeric VARTYPEs; everything else
    # (free text, identifiers, dates, undeclared strings) is "other" and is not
    # offered for analysis roles.
    if cats:
        kind = "categorical"
    elif var_type in ("INT", "INTEGER", "FLOAT", "DOUBLE", "NUMERIC", "NUMBER", "DECIMAL", "REAL", "LONG"):
        kind = "numeric"
    else:
        kind = "other"
    return {
        "cohort_id": cohort_id,
        "var_name": getattr(v, "var_name", ""),
        "var_label": getattr(v, "var_label", "") or "",
        "var_type": var_type,
        "kind": kind,
        "units": getattr(v, "units", None) or "",
        "unit_concept_name": getattr(v, "unit_concept_name", None) or "",
        "concept_code": getattr(v, "concept_code", None) or "",
        "concept_name": getattr(v, "concept_name", None) or "",
        "omop_id": str(getattr(v, "omop_id", None) or ""),
        "omop_domain": getattr(v, "omop_domain", None) or "",
        "visits": getattr(v, "visits", None) or "",
        "count": getattr(v, "count", None),
        "categories": cats,
    }


def _index(cohort_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    from src.cohort_cache import get_cohorts_from_cache

    all_cohorts = get_cohorts_from_cache("")
    out: dict[str, list[dict[str, Any]]] = {}
    for cid in cohort_ids:
        cohort = all_cohorts.get(cid)
        if not cohort:
            continue
        eda = _load_eda(cid)
        summaries = []
        for v in (getattr(cohort, "variables", {}) or {}).values():
            summ = _var_summary(cid, v)
            summ["eda"] = eda.get(summ["var_name"].strip().lower())
            summaries.append(summ)
        out[cid] = summaries
    return out


def _equivalents(index: dict[str, list[dict]], v: dict) -> list[dict[str, str]]:
    """Variables in OTHER cohorts sharing a concept code or OMOP id."""
    code, omop = _norm_code(v.get("concept_code")), _norm_code(v.get("omop_id"))
    if not code and not omop:
        return []
    eq = []
    for cid, vars_ in index.items():
        if cid == v["cohort_id"]:
            continue
        for w in vars_:
            if (code and _norm_code(w.get("concept_code")) == code) or (omop and _norm_code(w.get("omop_id")) == omop):
                eq.append({"cohort_id": cid, "var_name": w["var_name"], "var_label": w["var_label"]})
    return eq[:12]


def _find(index: dict[str, list[dict]], cohort_id: str, var_name: str) -> Optional[dict]:
    for w in index.get(cohort_id, []):
        if w["var_name"] == var_name:
            return w
    low = var_name.lower().strip()
    for w in index.get(cohort_id, []):
        if w["var_name"].lower().strip() == low:
            return w
    return None


# ---------------------------------------------------------------------------
# Cached cross-mapping files
# ---------------------------------------------------------------------------

def _mapping_output_dir() -> str:
    from src.chat import _linker_output_dir

    return _linker_output_dir()


def _safe_literal(text: str) -> Any:
    text = (text or "").strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _value_mapping_from_row(row: dict) -> dict[str, dict[str, str]]:
    """{'source': {raw->harmonized}, 'target': {raw->harmonized}} from the row's
    LLMEvidence.transform.value_mapping or transformation_rule.value_mapping."""
    for key in ("LLMEvidence", "transformation_rule", "Mapping Description"):
        obj = _safe_literal(row.get(key, ""))
        if not isinstance(obj, dict):
            continue
        vm = (obj.get("transform") or {}).get("value_mapping") if isinstance(obj.get("transform"), dict) else obj.get("value_mapping")
        if isinstance(vm, dict) and (vm.get("source") or vm.get("target")):
            return {"source": {str(k): str(v) for k, v in (vm.get("source") or {}).items()},
                    "target": {str(k): str(v) for k, v in (vm.get("target") or {}).items()}}
    return {}


_cache_rows: dict[str, tuple[float, list[dict]]] = {}


def load_cached_rows(filenames: list[str]) -> list[dict]:
    out_dir = _mapping_output_dir()
    rows: list[dict] = []
    for fname in filenames:
        path = os.path.join(out_dir, os.path.basename(fname))
        if not os.path.exists(path):
            continue
        mtime = os.path.getmtime(path)
        cached = _cache_rows.get(path)
        if cached and cached[0] == mtime:
            rows.extend(cached[1])
            continue
        parsed: list[dict] = []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    parsed.append({
                        "file": os.path.basename(path),
                        "source_study": (r.get("source_study") or "").strip().lower(),
                        "target_study": (r.get("target_study") or "").strip().lower(),
                        "source": (r.get("source") or "").strip(),
                        "target": (r.get("target") or "").strip(),
                        "status": (r.get("harmonization_status") or "").strip(),
                        "relation": (r.get("mapping_relation") or r.get("mapping type") or "").strip(),
                        "sim_score": r.get("sim_score"),
                        "harmonized_variable": (r.get("harmonized_variable") or "").strip(),
                        "value_mapping": _value_mapping_from_row(r),
                    })
        except Exception as exc:
            logger.warning("cached mapping %s unreadable: %s", fname, exc)
        _cache_rows[path] = (mtime, parsed)
        rows.extend(parsed)
    return rows


def _cache_hits(rows: list[dict], a: dict, b: dict) -> list[dict]:
    """Rows linking variable a (cohort A) and b (cohort B) in either direction."""
    ca, cb = a["cohort_id"].lower(), b["cohort_id"].lower()
    na, nb = a["var_name"].lower(), b["var_name"].lower()
    hits = []
    for r in rows:
        if r["source_study"] == ca and r["target_study"] == cb and r["source"].lower() == na and r["target"].lower() == nb:
            hits.append({**r, "direction": "a->b"})
        elif r["source_study"] == cb and r["target_study"] == ca and r["source"].lower() == nb and r["target"].lower() == na:
            hits.append({**r, "direction": "b->a"})
    return hits


# ---------------------------------------------------------------------------
# Endpoints: catalog, search, suggestions
# ---------------------------------------------------------------------------

@router.get("/kinds")
def kinds(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    return {"kinds": ANALYSIS_KINDS}


@router.get("/cached-mappings")
def cached_mappings(cohort_ids: str = "", user: Any = Depends(get_current_user)) -> dict[str, Any]:
    from src.chat import mapping_pair_status

    ids = [c.strip() for c in cohort_ids.split(",") if c.strip()]
    pairs = mapping_pair_status(ids)
    # Every cached file for the pairs (not just the newest), so the user can pick.
    out_dir = _mapping_output_dir()
    files = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            s, t = ids[i].lower(), ids[j].lower()
            for p in glob.glob(os.path.join(out_dir, f"{s}_{t}.csv")) + glob.glob(os.path.join(out_dir, f"{s}_{t}_*.csv")):
                files.append({"filename": os.path.basename(p), "source": ids[i], "target": ids[j],
                              "generated_at": datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
                              "size_kb": round(os.path.getsize(p) / 1024)})
    files.sort(key=lambda f: f["generated_at"], reverse=True)
    return {"pairs": pairs, "files": files}


@router.post("/variables")
def variables(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Every variable of the given cohorts (for the wizard's searchable
    dropdowns), each with its equivalents in the other cohorts."""
    cohort_ids = [str(c) for c in body.get("cohort_ids") or []]
    index = _index(cohort_ids)
    out = []
    for cid, vars_ in index.items():
        for v in vars_:
            r = dict(v)
            r["equivalents"] = _equivalents(index, v)
            out.append(r)
    out.sort(key=lambda r: (cohort_ids.index(r["cohort_id"]) if r["cohort_id"] in cohort_ids else 99, r["var_name"].lower()))
    return {"variables": out}


@router.post("/search")
def search(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    cohort_ids = [str(c) for c in body.get("cohort_ids") or []]
    query = str(body.get("query") or "").strip()
    mode = body.get("mode") or "any"
    limit = int(body.get("limit") or 60)
    index = _index(cohort_ids)
    q_tokens = _tokens(query)
    q_low = _ascii(query).lower()
    results = []
    for cid, vars_ in index.items():
        for v in vars_:
            hay = " ".join([v["var_name"], v["var_label"], v["concept_name"], v["concept_code"], v["omop_id"]])
            hay_tokens = _tokens(hay)
            hay_low = _ascii(hay).lower()
            if not query:
                score = 0.0
            elif mode == "exact":
                if q_low not in hay_low:
                    continue
                score = 1.0
            elif mode == "all":
                if not all(t in hay_tokens or t in hay_low for t in q_tokens):
                    continue
                score = 1.0
            else:  # any
                hits = [t for t in q_tokens if t in hay_tokens or t in hay_low]
                if not hits:
                    continue
                score = len(hits) / max(len(q_tokens), 1)
                if q_low and (q_low == v["var_name"].lower() or q_low in v["var_label"].lower()):
                    score += 0.5
            r = dict(v)
            r["score"] = round(score, 3)
            r["equivalents"] = _equivalents(index, v)
            results.append(r)
    results.sort(key=lambda r: (-r["score"], r["cohort_id"], r["var_name"].lower()))
    return {"results": results[:limit], "total": len(results)}


@router.post("/suggest")
def suggest(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Evidence-ranked candidates for `anchor` in each of `targets`."""
    anchor = body.get("anchor") or {}
    targets = [str(c) for c in body.get("targets") or []]
    cached_files = [str(f) for f in body.get("cached_files") or []]
    index = _index([anchor.get("cohort_id", "")] + targets)
    a = _find(index, anchor.get("cohort_id", ""), anchor.get("var_name", ""))
    if not a:
        raise HTTPException(status_code=404, detail="Anchor variable not found")
    rows = load_cached_rows(cached_files) if cached_files else []
    scored = _score_candidates(index, a, targets, rows)
    return {"anchor": a, "candidates": {cid: cands[:15] for cid, cands in scored.items()}}


def _score_candidates(index: dict[str, list[dict]], a: dict, targets: list[str], rows: list) -> dict[str, list[dict]]:
    """Evidence-ranked candidates for anchor `a` in each target cohort (all of
    them, sorted by score): standard codes, text similarity, computed mappings."""
    out: dict[str, list[dict]] = {}
    for cid in targets:
        if cid == a["cohort_id"]:
            continue
        cands = []
        for b in index.get(cid, []):
            if b["kind"] == "other":
                continue
            evidence = []
            score = 0.0
            if a["concept_code"] and _norm_code(a["concept_code"]) == _norm_code(b["concept_code"]):
                evidence.append({"type": "code", "system": _code_system(a["concept_code"]), "detail": a["concept_code"]})
                score = max(score, 1.0)
            elif a["omop_id"] and _norm_code(a["omop_id"]) == _norm_code(b["omop_id"]):
                evidence.append({"type": "code", "system": "OMOP ID", "detail": a["omop_id"]})
                score = max(score, 0.95)
            ts = text_similarity(a, b)
            if ts >= 0.45:
                evidence.append({"type": "text", "score": ts})
                score = max(score, 0.35 + 0.5 * ts)
            for h in _cache_hits(rows, a, b):
                evidence.append({"type": "cache", "file": h["file"], "status": h["status"],
                                 "relation": h["relation"], "value_mapping": h["value_mapping"],
                                 "harmonized_variable": h["harmonized_variable"]})
                bonus = {"Identical Match": 0.98, "Compatible Match": 0.9, "Partial Match": 0.7}.get(h["status"], 0.6)
                score = max(score, bonus)
            if a["kind"] != b["kind"]:
                # still show but flag the type mismatch
                if evidence:
                    evidence.append({"type": "warning", "detail": "type differs: %s vs %s" % (a["kind"], b["kind"])})
                    score *= 0.85
            if evidence:
                c = {k: b[k] for k in ("cohort_id", "var_name", "var_label", "kind", "var_type", "units",
                                        "concept_code", "concept_name", "omop_id", "visits", "categories")}
                c["score"] = round(score, 3)
                c["evidence"] = evidence
                cands.append(c)
        cands.sort(key=lambda c: -c["score"])
        out[cid] = cands
    return out


def _canonical_value(cat: dict) -> str:
    """Preferred harmonized label for a category: its standard concept name,
    else its label, else the raw value."""
    return (cat.get("concept_name") or cat.get("label") or cat.get("value") or "").strip()


@router.post("/suggest-values")
def suggest_values(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Propose a harmonized value set for a categorical harmonized variable.

    body.members = {cohort_id: var_name}; body.cached_files = [...]
    Returns clusters: [{harmonized, sources: {cohort: [raw values]}, evidence: [...]}]
    plus per-cohort values that could not be placed.
    """
    members: dict[str, str] = {str(k): str(v) for k, v in (body.get("members") or {}).items()}
    cached_files = [str(f) for f in body.get("cached_files") or []]
    index = _index(list(members.keys()))
    vars_ = {cid: _find(index, cid, vn) for cid, vn in members.items()}
    vars_ = {cid: v for cid, v in vars_.items() if v}
    rows = load_cached_rows(cached_files) if cached_files else []

    clusters: list[dict[str, Any]] = []   # {"harmonized", "sources": {cid: set()}, "evidence": []}

    def place(cid: str, raw: str, harmonized: str, ev: dict) -> None:
        key = _ascii(harmonized).strip().lower()
        for cl in clusters:
            if _ascii(cl["harmonized"]).strip().lower() == key:
                cl["sources"].setdefault(cid, [])
                if raw not in cl["sources"][cid]:
                    cl["sources"][cid].append(raw)
                if ev not in cl["evidence"]:
                    cl["evidence"].append(ev)
                return
        clusters.append({"harmonized": harmonized, "sources": {cid: [raw]}, "evidence": [ev]})

    # 1. category concept codes / OMOP ids shared across cohorts
    by_code: dict[str, list[tuple[str, dict]]] = {}
    for cid, v in vars_.items():
        for cat in v["categories"]:
            for code in (_norm_code(cat.get("concept_code")), _norm_code(cat.get("omop_id"))):
                if code:
                    by_code.setdefault(code, []).append((cid, cat))
    placed: set[tuple[str, str]] = set()
    for code, items in by_code.items():
        if len({cid for cid, _ in items}) < 2:
            continue
        name = next((c.get("concept_name") for _, c in items if c.get("concept_name")), None) or _canonical_value(items[0][1])
        for cid, cat in items:
            if (cid, cat["value"]) in placed:
                continue
            place(cid, cat["value"], name, {"type": "code", "system": "OMOP ID" if code.isdigit() else _code_system(code), "detail": code})
            placed.add((cid, cat["value"]))

    # 2. cached value_mapping (source raw -> harmonized label, target raw -> harmonized label)
    cids = list(vars_.keys())
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a, b = vars_[cids[i]], vars_[cids[j]]
            for h in _cache_hits(rows, a, b):
                vm = h.get("value_mapping") or {}
                src_c, tgt_c = (a, b) if h["direction"] == "a->b" else (b, a)
                for raw, harm in (vm.get("source") or {}).items():
                    if (src_c["cohort_id"], raw) not in placed and harm:
                        place(src_c["cohort_id"], raw, harm, {"type": "cache", "file": h["file"], "status": h["status"]})
                        placed.add((src_c["cohort_id"], raw))
                for raw, harm in (vm.get("target") or {}).items():
                    if (tgt_c["cohort_id"], raw) not in placed and harm:
                        place(tgt_c["cohort_id"], raw, harm, {"type": "cache", "file": h["file"], "status": h["status"]})
                        placed.add((tgt_c["cohort_id"], raw))

    # 3. label similarity / synonyms
    for cid, v in vars_.items():
        for cat in v["categories"]:
            if (cid, cat["value"]) in placed:
                continue
            canon = _canonical_value(cat)
            best, best_score = None, 0.0
            for cl in clusters:
                s = max(_ratio(cl["harmonized"], canon), _ratio(cl["harmonized"], cat.get("label", "")))
                if s > best_score:
                    best, best_score = cl, s
            if best is not None and best_score >= 0.8:
                place(cid, cat["value"], best["harmonized"], {"type": "text", "score": round(best_score, 2)})
            else:
                place(cid, cat["value"], canon, {"type": "label"})
            placed.add((cid, cat["value"]))

    # Clusters present in only one cohort are "unmatched" proposals, still editable.
    for cl in clusters:
        cl["cohorts_covered"] = sorted(cl["sources"].keys())
        cl["complete"] = len(cl["sources"]) == len(vars_)
    clusters.sort(key=lambda c: (-len(c["sources"]), c["harmonized"].lower()))
    return {"clusters": clusters, "members": {cid: {"var_name": v["var_name"], "categories": v["categories"]} for cid, v in vars_.items()}}


def _heuristic_name(members: list[dict]) -> tuple[str, str]:
    """Short harmonized name from the member variables: tokens shared by all
    names, else the anchor's name without visit digits; always suffixed
    _harmonized. Label = the shortest label."""
    names = [str(m.get("var_name") or "") for m in members if m.get("var_name")]
    token_sets = [set(_tokens(n)) - {"v", "visit", "m", "month", "baseline"} for n in names]
    token_sets = [t for t in token_sets if t]
    shared = set.intersection(*token_sets) if len(token_sets) > 1 else (token_sets[0] if token_sets else set())
    shared = [t for t in shared if not t.isdigit()]
    if shared:
        base = "_".join(sorted(shared, key=lambda t: names[0].lower().find(t)))
    else:
        base = re.sub(r"\d+$", "", _ascii(names[0]).lower()) if names else "variable"
        base = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "variable"
    name = (base[:24].rstrip("_") + "_pooled")
    labels = [str(m.get("var_label") or "") for m in members if m.get("var_label")]
    label = min(labels, key=len) if labels else base.replace("_", " ")
    return name, label


@router.post("/ai-name")
def ai_name(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """A short harmonized variable name (snake_case, suffixed _harmonized) and a
    short label for a set of mapped variables. Uses the local LLM when chat is
    configured, else a heuristic; always returns something usable."""
    members = [m for m in (body.get("variables") or []) if isinstance(m, dict)]
    name, label = _heuristic_name(members)
    if not settings.chat_enabled or not members:
        return {"name": name, "label": label, "source": "heuristic"}
    from src.chat import _get_openai_client

    prompt = (
        "These variables from different clinical cohorts have been mapped to one harmonized variable:\n"
        f"{json.dumps(members, ensure_ascii=False)}\n\n"
        "Propose ONE short name for the merged variable, in the style of the variable names themselves "
        "(abbreviations are fine), snake_case, at most 20 characters, and ending with '_pooled' "
        "(e.g. nyha_pooled, ntprobnp_pooled, lvef_pooled). Also propose a short human-readable label "
        "(at most 6 words). Return STRICT JSON only: {\"name\": \"...\", \"label\": \"...\"}"
    )
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[{"role": "system", "content": "Answer with JSON only, no prose, no code fences."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
        ai_name_ = re.sub(r"[^a-z0-9_]+", "_", str(parsed.get("name") or "").lower()).strip("_")
        if ai_name_:
            ai_name_ = re.sub(r"_(pooled|harmonized)$", "", ai_name_)[:24].rstrip("_") + "_pooled"
            name = ai_name_[:40]
        if parsed.get("label"):
            label = str(parsed["label"])[:80]
        return {"name": name, "label": label, "source": "ai", "model": settings.litellm_model}
    except Exception as exc:
        logger.warning("ai-name failed, using heuristic: %s", exc)
        return {"name": name, "label": label, "source": "heuristic"}


def _fmt_num(x: Any) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return str(int(f)) if f == int(f) and abs(f) < 1e15 else ("%.4g" % f)


def _describe_var(v: dict, evidence: Optional[list[dict]] = None) -> str:
    """One compact line about a variable for the local LLM: name, label, concept,
    unit, type with the observed range (numeric) or the categories (categorical),
    visit, and the matcher's evidence when given."""
    bits = [str(v.get("var_name") or "")]
    if v.get("var_label") and v["var_label"] != v["var_name"]:
        bits.append("label: %s" % v["var_label"])
    if v.get("concept_name"):
        bits.append("concept: %s" % v["concept_name"])
    if v.get("units"):
        bits.append("unit: %s" % v["units"])
    eda = v.get("eda") or {}
    kind = v.get("kind") or ""
    if kind == "categorical":
        cats = v.get("categories") or []
        shown = ["%s=%s" % (c.get("value"), c.get("label")) if c.get("label") and c.get("label") != c.get("value") else str(c.get("value"))
                 for c in cats[:10]]
        bits.append("categorical: %s%s" % (", ".join(shown), " (+%d more)" % (len(cats) - 10) if len(cats) > 10 else ""))
    elif kind == "numeric":
        rng = ""
        if eda.get("min") is not None and eda.get("max") is not None:
            rng = ", range %s to %s" % (_fmt_num(eda["min"]), _fmt_num(eda["max"]))
            if eda.get("mean") is not None:
                rng += ", mean %s" % _fmt_num(eda["mean"])
        bits.append("numeric%s" % rng)
    elif kind:
        bits.append(kind)
    if v.get("visits"):
        bits.append("visit: %s" % v["visits"])
    if evidence:
        ev = []
        for e in evidence:
            if e.get("type") == "code":
                ev.append("same %s" % (e.get("system") or "code"))
            elif e.get("type") == "text":
                ev.append("text similarity %.2f" % float(e.get("score") or 0))
            elif e.get("type") == "cache":
                ev.append("computed mapping: %s" % (e.get("status") or "match"))
            elif e.get("type") == "warning":
                ev.append(str(e.get("detail") or "warning"))
        if ev:
            bits.append("evidence: " + "; ".join(ev))
    return "- " + " | ".join(bits)


_AI_TOP_N = 15          # candidates shown when the matcher has strong suggestions
_AI_LIST_BUDGET = 48000  # characters available for the full listings, all cohorts together


def _strong(c: dict) -> bool:
    """A candidate worth trusting as a shortlist: a shared standard code, a
    high text similarity, or an identical/compatible computed mapping."""
    for e in c.get("evidence") or []:
        if e.get("type") == "code":
            return True
        if e.get("type") == "text" and float(e.get("score") or 0) >= 0.8:
            return True
        if e.get("type") == "cache" and e.get("status") in ("Identical Match", "Compatible Match"):
            return True
    return False


def _ai_match_listing(index: dict[str, list[dict]], a: dict, scored: dict[str, list[dict]]) -> tuple[str, dict[str, dict]]:
    """What the local LLM gets to see per cohort: the matcher's top candidates
    when at least one of them is strong (codes, high text similarity, computed
    mappings), otherwise the cohort's whole variable list (same-type variables
    first, cut to a character budget)."""
    modes: dict[str, dict] = {}
    sections: list[tuple[str, list[str]]] = []
    full_cohorts = [cid for cid, cands in scored.items() if not any(_strong(c) for c in cands[:_AI_TOP_N])]
    per_cohort_budget = _AI_LIST_BUDGET // max(1, len(full_cohorts))
    for cid, cands in scored.items():
        if cid not in full_cohorts:
            top = cands[:_AI_TOP_N]
            lines = [_describe_var(_find(index, cid, c["var_name"]) or c, c.get("evidence")) for c in top]
            modes[cid] = {"mode": "top", "listed": len(lines), "total": len([v for v in index.get(cid, []) if v.get("kind") != "other"])}
            sections.append(("%s: the platform's %d top-ranked suggestions (strong evidence present)" % (cid, len(lines)), lines))
        else:
            all_vars = [v for v in index.get(cid, []) if v.get("kind") != "other"]
            all_vars.sort(key=lambda v: (0 if v.get("kind") == a.get("kind") else 1, str(v.get("var_name")).lower()))
            lines, used = [], 0
            for v in all_vars:
                line = _describe_var(v)
                if used + len(line) > per_cohort_budget:
                    break
                lines.append(line)
                used += len(line) + 1
            modes[cid] = {"mode": "all", "listed": len(lines), "total": len(all_vars)}
            head = "%s: no strong suggestion, so its FULL variable list (%d variables" % (cid, len(all_vars))
            if len(lines) < len(all_vars):
                head += "; cut to the first %d, same-type variables first" % len(lines)
            sections.append((head + ")", lines))
    text = "\n\n".join("%s\n%s" % (head, "\n".join(lines)) for head, lines in sections)
    return text, modes


# ---------------------------------------------------------------------------
# DCR name: NoCode-<Type>-<Variables>-<Cohorts|Ncohorts>-<MonDD>
# ---------------------------------------------------------------------------

_DCR_NAME_MAX = 100
_TYPE_WORDS = {"stratified": "Stratify", "correlation": "Correlate", "crosstab": "Crosstab",
               "pooled": "Pool", "distribution": "Distribution"}


def _name_token(text: str, n: int) -> str:
    """Letters and digits only (dashes inside are kept), cut to n characters."""
    t = re.sub(r"[^A-Za-z0-9-]+", "", _ascii(str(text or "")))
    return t.strip("-")[:n].rstrip("-")


def _abbrev_var(name: str, n: int = 14) -> str:
    """Heuristic short form of a variable name: CamelCase of its tokens without
    the _pooled/_harmonized suffix, e.g. 'nt_pro_bnp_pooled' -> 'NtProBnp'."""
    base = re.sub(r"_(pooled|harmonized)$", "", str(name or ""))
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", _ascii(base)) if t]
    if not toks:
        return "var"
    camel = "".join(t if t.isupper() else t[:1].upper() + t[1:] for t in toks)
    return camel[:n]


def _assemble_dcr_name(type_word: str, var_words: list[str], joiner: str, cohorts: list[str],
                       cohort_short: dict[str, str], day: str) -> str:
    """Builds the name from the template and shortens it until it fits: first
    the variables, then the cohorts become a count, then a hard cut."""
    def build(var_n: int, named_cohorts: bool) -> str:
        vars_part = joiner.join(_name_token(v, var_n) or "var" for v in var_words)
        if named_cohorts and len(cohorts) <= 2:
            coh = "-".join(_name_token(cohort_short.get(c) or c, 12) or "cohort" for c in cohorts)
        else:
            coh = "%dcohorts" % len(cohorts)
        parts = ["NoCode", _name_token(type_word, 18) or "Analysis", vars_part, coh, day]
        name = "-".join(p for p in parts if p)
        return re.sub(r"-{2,}", "-", name)
    for var_n, named in ((14, True), (10, True), (8, True), (8, False), (6, False)):
        name = build(var_n, named)
        if len(name) <= _DCR_NAME_MAX:
            return name
    return name[:_DCR_NAME_MAX].rstrip("-")


@router.post("/ai-dcr-name")
def ai_dcr_name(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Name of the Data Clean Room following the template
    NoCode-<analysis type>-<variables in play>-<cohort names if 1 or 2, else count>-<MonDD>
    (the creator's email is appended by the creation code). The local LLM, when
    available, supplies conventional short forms of the variables and cohorts
    (NTproBNP, NYHA, LVEF); a heuristic does otherwise. At most 100 characters."""
    kind = str(body.get("kind") or "")
    roles = body.get("roles") or {}
    cohorts = [str(c) for c in body.get("cohorts") or []]
    variables = {str(v.get("harmonized_name")): v for v in (body.get("variables") or []) if isinstance(v, dict)}
    single = len(cohorts) == 1
    if kind == "compare":
        type_word = "PoolAndStratify" if roles.get("group") else "PoolAndCompare"
    else:
        type_word = _TYPE_WORDS.get(kind, kind.title() or "Analysis")
    if kind in ("correlation", "crosstab"):
        role_keys, joiner = ["x", "y"], "-vs-" if kind == "correlation" else "-by-"
    else:
        role_keys, joiner = ["variable"] + (["group"] if roles.get("group") else []), "-by-"
    in_play = [str(roles.get(k) or "") for k in role_keys if roles.get(k)]

    def source_name(hname: str) -> str:
        # single cohort: the source variable's own name; several: the harmonized name
        hv = variables.get(hname) or {}
        members = hv.get("members") or {}
        if single and members:
            m = next(iter(members.values()))
            if isinstance(m, dict) and m.get("var_name"):
                return str(m["var_name"])
        return hname

    day = datetime.now().strftime("%b%d")
    heuristic_words = [_abbrev_var(source_name(h)) for h in in_play]
    name = _assemble_dcr_name(type_word, heuristic_words, joiner, cohorts, {}, day)
    if not settings.chat_enabled or not in_play:
        return {"name": name, "source": "heuristic"}
    from src.chat import _get_openai_client

    described = []
    for h in in_play:
        hv = variables.get(h) or {}
        members = hv.get("members") or {}
        described.append({"name": source_name(h), "label": hv.get("label") or "",
                          "source_variables": [{"cohort": c, "name": m.get("var_name"), "label": m.get("var_label")}
                                               for c, m in members.items() if isinstance(m, dict)]})
    prompt = (
        "Clinical cohort variables and cohort names are to be shortened for the name of a data clean room.\n"
        f"VARIABLES: {json.dumps(described, ensure_ascii=False)}\n"
        f"COHORTS: {json.dumps(cohorts, ensure_ascii=False)}\n\n"
        "For each variable give the conventional short form a cardiologist would use (e.g. NTproBNP, NYHA, LVEF, "
        "BMI, Age, Sex, SBP, eGFR, HbA1c), letters and digits only, at most 12 characters. For each cohort give a "
        "short form of at most 10 characters (keep it if already short). Return STRICT JSON only: "
        "{\"variables\": {\"<name as given>\": \"<short>\"}, \"cohorts\": {\"<cohort as given>\": \"<short>\"}}"
    )
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[{"role": "system", "content": "Answer with JSON only, no prose, no code fences."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
        short_vars = {str(k): str(v) for k, v in (parsed.get("variables") or {}).items() if v}
        short_cohorts = {str(k): str(v) for k, v in (parsed.get("cohorts") or {}).items() if v}
        words = []
        for h, fallback in zip(in_play, heuristic_words):
            w = short_vars.get(source_name(h)) or short_vars.get(h) or fallback
            words.append(_name_token(w, 14) or fallback)
        ai = _assemble_dcr_name(type_word, words, joiner, cohorts, short_cohorts, day)
        return {"name": ai, "source": "ai", "model": settings.litellm_model}
    except Exception as exc:
        logger.warning("ai-dcr-name failed, using heuristic: %s", exc)
        return {"name": name, "source": "heuristic"}


@router.post("/ai-suggest")
def ai_suggest(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Ask the platform's local LLM for match or value-map suggestions.

    body.task = "match": body.anchor {cohort_id, var_name}, body.targets [cohort ids],
                         body.cached_files [computed mapping files to consult]
    body.task = "values": body.variables {cohort: {var_name, var_label, categories}}
    Returns the model's JSON; the UI labels everything from here as "AI suggestion".
    """
    from src.chat import _get_openai_client

    task = body.get("task") or "match"
    client = _get_openai_client()
    modes: dict[str, dict] = {}
    if task == "match":
        anchor_ref = body.get("anchor") or {}
        targets = [str(c) for c in body.get("targets") or []]
        cached_files = [str(f) for f in body.get("cached_files") or []]
        index = _index([anchor_ref.get("cohort_id", "")] + targets)
        a = _find(index, anchor_ref.get("cohort_id", ""), anchor_ref.get("var_name", ""))
        if not a:
            raise HTTPException(status_code=404, detail="Anchor variable not found")
        rows = load_cached_rows(cached_files) if cached_files else []
        scored = _score_candidates(index, a, targets, rows)
        listing, modes = _ai_match_listing(index, a, scored)
        prompt = (
            "You align variables across clinical cohort datasets (cardiovascular research cohorts). "
            "You are given one ANCHOR variable and, for each other cohort, EITHER the platform's top-ranked "
            "suggestions (with the evidence that ranked them) OR, when no strong suggestion exists, that "
            "cohort's full list of variables. Each line gives the name, label, standard concept name, unit, "
            "type with the observed range (numeric) or the categories (categorical), and visit. For each cohort, "
            "pick the variable that most likely measures the same clinical quantity as the anchor, or null if none does. "
            "Use the categories and ranges: a variable whose values are NYHA classes I to IV is not a yes/no flag, "
            "and a range of 5 to 35000 pg/mL fits NT-proBNP but not a percentage.\n\n"
            "How to recognise the same quantity:\n"
            "- Variable names are often abbreviations or concatenations: NTBNP, NT_proBNP, ntprobnp, BNP1 and "
            "'NT pro-BNP at visit month 1' all refer to NT-proBNP; LVEF, EF, ejectionfraction refer to ejection fraction.\n"
            "- Trailing digits or suffixes usually encode the visit (BNP1 = BNP at visit 1, Hb12 = haemoglobin at month 12), "
            "not a different quantity; prefer the candidate at the same or the nearest visit.\n"
            "- Labels and standard concept names may be in another language (Geschlecht = sex, Gewicht = weight, "
            "Kreatinin = creatinine) or use synonyms (gender/sex, weight/body mass).\n"
            "- Units must be compatible (pg/mL vs ng/L is the same quantity; mg/dL vs mmol/L can still be the same analyte).\n"
            "- A different TYPE (numeric vs categorical) is a strong sign it is NOT the same variable, unless one is a "
            "binned version of the other.\n"
            "- Do not match on a shared generic word alone (e.g. 'date', 'visit', 'score').\n\n"
            "Return STRICT JSON only: {\"matches\": {\"<cohort_id>\": {\"var_name\": \"<exact name from the list>\", "
            "\"confidence\": 0-1, \"reason\": \"<one short sentence>\"} or null}}\n\n"
            f"ANCHOR ({a.get('cohort_id')}):\n{_describe_var(a)}\n\n"
            f"{listing}"
        )
    else:
        prompt = (
            "You harmonize categorical values across clinical cohorts. For the given variables (one per cohort, "
            "with their category values and labels), propose a shared set of harmonized values and map every raw "
            "value of every cohort to one of them (or null if it has no counterpart). Return STRICT JSON: "
            "{\"harmonized_values\": [\"...\"], \"value_map\": {\"<cohort_id>\": {\"<raw value>\": \"<harmonized>\"}}, \"notes\": \"...\"}\n\n"
            f"VARIABLES: {json.dumps(body.get('variables'), ensure_ascii=False)[:12000]}"
        )
    try:
        resp = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[{"role": "system", "content": "Answer with JSON only, no prose, no code fences."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("ai-suggest failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"AI suggestion failed: {exc}")
    return {"task": task, "result": parsed, "model": settings.litellm_model, "modes": modes}


# ---------------------------------------------------------------------------
# Mapping specs (persisted, reusable)
# ---------------------------------------------------------------------------

def _mapping_path(mid: str) -> str:
    return os.path.join(MAPPINGS_DIR, f"{os.path.basename(mid)}.json")


@router.post("/mappings")
def save_mapping(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    os.makedirs(MAPPINGS_DIR, exist_ok=True)
    mid = str(body.get("id") or uuid.uuid4())
    spec = dict(body)
    spec["id"] = mid
    spec.setdefault("created_by", user.get("email"))
    spec.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    spec["updated_at"] = datetime.now().isoformat(timespec="seconds")
    spec["updated_by"] = user.get("email")
    with open(_mapping_path(mid), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    return {"ok": True, "id": mid}


@router.get("/mappings")
def list_mappings(cohort_ids: str = "", user: Any = Depends(get_current_user)) -> dict[str, Any]:
    ids = {c.strip() for c in cohort_ids.split(",") if c.strip()}
    out = []
    for p in glob.glob(os.path.join(MAPPINGS_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                spec = json.load(fh)
        except Exception:
            continue
        spec_cohorts = set(spec.get("cohorts") or [])
        if ids and not ids & spec_cohorts:
            continue
        out.append({"id": spec.get("id"), "name": spec.get("name"), "cohorts": sorted(spec_cohorts),
                    "variables": len(spec.get("variables") or []), "created_by": spec.get("created_by"),
                    "updated_at": spec.get("updated_at") or spec.get("created_at")})
    out.sort(key=lambda m: m.get("updated_at") or "", reverse=True)
    return {"mappings": out}


@router.get("/mappings/{mid}")
def get_mapping(mid: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    p = _mapping_path(mid)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Mapping not found")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@router.post("/describe")
def describe(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Plain-language recipe + the generated script (for the review step)."""
    from src.nocode_scripts import nocode_analysis_script

    return {"description": describe_spec(body), "script": nocode_analysis_script(body)}


# ---------------------------------------------------------------------------
# Results: run the node via the service account and serve the aggregates
# ---------------------------------------------------------------------------

def _results_dir(dcr_id: str, node: str) -> str:
    return os.path.join(RESULTS_DIR, os.path.basename(dcr_id), os.path.basename(node))


def _read_summary(dir_: str) -> Optional[dict]:
    p = os.path.join(dir_, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["files"] = sorted(
        os.path.relpath(os.path.join(r, f), dir_)
        for r, _, fs in os.walk(dir_) for f in fs
    )
    try:
        with open(os.path.join(dir_, "provenance.md"), encoding="utf-8") as fh:
            summary["provenance_md"] = fh.read()
    except Exception:
        pass
    return summary


@router.post("/run/{dcr_id}")
def run_node(dcr_id: str, body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Run one no-code DCR node and pull its output into the explorer."""
    import decentriq_platform as dq

    node_name = str(body.get("node_name") or "")
    # Only the analysis nodes of a no-code room can be run from here: the room's
    # record lists them. Rooms created before records existed used the
    # "nocode-"/"guided-" name prefixes.
    room = load_nocode_room(dcr_id)
    known = set((room or {}).get("nodes") or [])
    if node_name not in known and not node_name.startswith(("nocode-", "guided-")):
        raise HTTPException(status_code=400, detail="node_name must be an analysis node of a no-code DCR")
    try:
        client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
        dcr = client.retrieve_analytics_dcr(dcr_id)
        node = dcr.get_node(node_name)
        result = node.run_computation_and_get_results_as_zip()
    except Exception as exc:
        logger.warning("no-code run failed for %s/%s: %s", dcr_id, node_name, exc)
        raise HTTPException(status_code=502, detail=f"Running the analysis failed: {exc}")
    out_dir = _results_dir(dcr_id, node_name)
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(result, zipfile.ZipFile):
        result.extractall(out_dir)
    else:
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            zf.extractall(out_dir)
    with open(os.path.join(out_dir, "_fetched_at.txt"), "w") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds"))
    summary = _read_summary(out_dir) or {"items": [], "files": os.listdir(out_dir)}
    return {"ok": True, "dcr_id": dcr_id, "node_name": node_name, "summary": summary}


@router.get("/rooms/{dcr_id}")
def get_room(dcr_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    room = load_nocode_room(dcr_id)
    if room is None:
        raise HTTPException(status_code=404, detail="No record of this no-code DCR")
    return room


@router.get("/results/{dcr_id}/{node_name}")
def get_results(dcr_id: str, node_name: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    out_dir = _results_dir(dcr_id, node_name)
    summary = _read_summary(out_dir)
    if summary is None:
        raise HTTPException(status_code=404, detail="No results fetched yet for this analysis")
    try:
        with open(os.path.join(out_dir, "_fetched_at.txt")) as fh:
            summary["fetched_at"] = fh.read().strip()
    except Exception:
        pass
    return summary


@router.get("/results/{dcr_id}/{node_name}/file/{path:path}")
def get_result_file(dcr_id: str, node_name: str, path: str, user: Any = Depends(get_current_user)):
    base = os.path.realpath(_results_dir(dcr_id, node_name))
    full = os.path.realpath(os.path.join(base, path))
    if not full.startswith(base + os.sep) or not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    if not full.lower().endswith((".png", ".csv", ".json", ".md", ".txt")):
        raise HTTPException(status_code=403, detail="File type not served")
    return FileResponse(full)
