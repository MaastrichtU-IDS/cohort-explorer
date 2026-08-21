"""Guided (no-code) analysis API.

Backs the /guided-analysis wizard for domain experts who do not program:

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
  - running a guided-analysis node via the service account and serving the
    aggregate results (figures/tables) back to the wizard

The enclave script itself is generated in guided_scripts.py.
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
from src.guided_scripts import ANALYSIS_KINDS, describe_spec

router = APIRouter(prefix="/api/guided", tags=["guided"])
logger = logging.getLogger(__name__)

MAPPINGS_DIR = os.path.join(settings.data_folder, "guided_mappings")
RESULTS_DIR = os.path.join(settings.data_folder, "guided_results")

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
    return round(max(0.45 * names + 0.55 * labels, concept, 0.5 * labels + 0.5 * concept, 0.8 * cross), 3)


def _norm_code(code: Any) -> str:
    return _ascii(code).strip().lower() if code else ""


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
    kind = "categorical" if cats or var_type in ("STR", "STRING", "TEXT", "CAT", "CATEGORICAL", "BOOL", "BOOLEAN") else "numeric"
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
        out[cid] = [_var_summary(cid, v) for v in (getattr(cohort, "variables", {}) or {}).values()]
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
    out: dict[str, list[dict]] = {}
    for cid in targets:
        if cid == a["cohort_id"]:
            continue
        cands = []
        for b in index.get(cid, []):
            evidence = []
            score = 0.0
            if a["concept_code"] and _norm_code(a["concept_code"]) == _norm_code(b["concept_code"]):
                evidence.append({"type": "code", "detail": a["concept_code"]})
                score = max(score, 1.0)
            elif a["omop_id"] and _norm_code(a["omop_id"]) == _norm_code(b["omop_id"]):
                evidence.append({"type": "code", "detail": "OMOP %s" % a["omop_id"]})
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
        out[cid] = cands[:15]
    return {"anchor": a, "candidates": out}


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
            place(cid, cat["value"], name, {"type": "code", "detail": code})
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


@router.post("/ai-suggest")
def ai_suggest(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Ask the platform's local LLM for match or value-map suggestions.

    body.task = "match": body.anchor (variable summary), body.candidates {cohort: [summaries]}
    body.task = "values": body.variables {cohort: {var_name, var_label, categories}}
    Returns the model's JSON; the UI labels everything from here as "AI suggestion".
    """
    from src.chat import _get_openai_client

    task = body.get("task") or "match"
    client = _get_openai_client()
    if task == "match":
        prompt = (
            "You align variables across clinical cohort datasets. Given an ANCHOR variable and CANDIDATE "
            "variables from other cohorts (names, labels, standard concept names, units, categories), pick "
            "for each cohort the candidate most likely measuring the same thing, or null if none does. "
            "Return STRICT JSON: {\"matches\": {\"<cohort_id>\": {\"var_name\": ..., \"confidence\": 0-1, \"reason\": \"...\"}}}\n\n"
            f"ANCHOR: {json.dumps(body.get('anchor'), ensure_ascii=False)}\n\n"
            f"CANDIDATES: {json.dumps(body.get('candidates'), ensure_ascii=False)[:12000]}"
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
    return {"task": task, "result": parsed, "model": settings.litellm_model}


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
    from src.guided_scripts import guided_analysis_script

    return {"description": describe_spec(body), "script": guided_analysis_script(body)}


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
    """Run one guided-analysis node and pull its output into the explorer."""
    import decentriq_platform as dq

    node_name = str(body.get("node_name") or "")
    if not node_name.startswith("guided-"):
        raise HTTPException(status_code=400, detail="node_name must be a guided-analysis node")
    try:
        client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
        dcr = client.retrieve_analytics_dcr(dcr_id)
        node = dcr.get_node(node_name)
        result = node.run_computation_and_get_results_as_zip()
    except Exception as exc:
        logger.warning("guided run failed for %s/%s: %s", dcr_id, node_name, exc)
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
