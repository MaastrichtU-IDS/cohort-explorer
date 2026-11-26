"""Whether a dictionary's concept annotations are right, beyond whether code and
id agree: specification checks S10 to S14.

The concept check in CohortVarLinker (validate_cde.py) covers S10, that a
concept's code and OMOP id refer to the same concept, from the local OMOP graph,
falling back to an Athena search when the id is not in the graph. This module
adds, per annotated concept (variable, additional context, category, unit,
visit):

- S10, for ids the local graph does not hold: Athena's record of the id says
  which concept it is, or that no such id exists. The search fallback reports
  "no candidates" as N/A, which passed a code that does not exist.
- S11, wrong vocabulary or domain for the field: a unit must be UCUM and only a
  unit may be UCUM; a visit comes from SNOMED, LOINC, OMOP or CDISC; a unit
  concept must be in the Unit domain and no category may be; a variable
  concept's domain must agree with the row's DOMAIN.
- S12, non-standard or deprecated: from Athena's record of the id. The local
  graph cannot answer this: the vocabulary preprocessing keeps valid concepts
  only and drops the standard flag.
- S13, concept name differs from the vocabulary's name and synonyms, compared
  after case, spacing and punctuation are set aside.
- S14, declaration implausible for the concept, for LOINC variable concepts:
  VARTYPE against the LOINC scale (a quantitative test declared STR), and the
  unit's dimension against the LOINC property (a mass concentration in mmol/L).

The ICARE vocabulary is the project's own and is skipped throughout, as the
concept check does. A concept whose code and id already disagree is left to
S10: whatever else is said about it would be about the wrong concept.

Athena answers are kept in {data_folder}/data_quality/athena_concepts.json, so
a concept is asked about once a month at most.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pandas as pd
import requests

from src.config import settings

logger = logging.getLogger(__name__)

ATHENA_CONCEPT_URL = "https://athena.ohdsi.org/api/v1/concepts"
_ATHENA_MAX_AGE = 30 * 24 * 3600
_ATHENA_WORKERS = 4

# Field → (column of names, column of codes, column of OMOP ids, label for messages).
FIELDS = {
    "variable": ("VARIABLE CONCEPT NAME", "VARIABLE CONCEPT CODE", "VARIABLE OMOP ID", "Variable concept"),
    "additional_context": ("ADDITIONAL CONTEXT CONCEPT NAME", "ADDITIONAL CONTEXT CONCEPT CODE",
                           "ADDITIONAL CONTEXT OMOP ID", "Additional context concept"),
    "categorical": ("CATEGORICAL VALUE CONCEPT NAME", "CATEGORICAL VALUE CONCEPT CODE",
                    "CATEGORICAL VALUE OMOP ID", "Category concept"),
    "unit": ("UNIT CONCEPT NAME", "UNIT CONCEPT CODE", "UNIT OMOP ID", "Unit concept"),
    "visit": ("VISIT CONCEPT NAME", "VISIT CONCEPT CODE", "VISIT OMOP ID", "Visit concept"),
}
_LIST_FIELDS = {"additional_context", "categorical"}

# S11: vocabularies, by the prefix written in the dictionary (upper-cased).
_UCUM = {"UCUM"}
_VISIT_VOCABS = {"SNOMED", "SNOMEDCT", "SNOMED-CT", "LOINC", "OMOP", "CDISC"}
_CUSTOM = {"ICARE"}

# S11: the OMOP domain a variable concept should have, by the DOMAIN column
# (which names the OMOP table). Tables without one domain are not checked.
_TABLE_DOMAIN = {
    "measurement": "Measurement",
    "condition_occurrence": "Condition",
    "condition_era": "Condition",
    "observation": "Observation",
    "procedure_occurrence": "Procedure",
    "drug_exposure": "Drug",
    "drug_era": "Drug",
    "device_exposure": "Device",
    "specimen": "Specimen",
    "visit_occurrence": "Visit",
}

# S14: the physical dimension a LOINC property implies, as pint dimensionality.
# Properties without a fixed dimension (arbitrary concentration, presence,
# type, finding) are not checked.
_MASS_CONC = frozenset({("[mass]", 1.0), ("[length]", -3.0)})
_SUBST_CONC = frozenset({("[substance]", 1.0), ("[length]", -3.0)})
_CATALYTIC_CONC = frozenset({("[substance]", 1.0), ("[time]", -1.0), ("[length]", -3.0)})
_NUMBER_CONC = frozenset({("[length]", -3.0)})
_DIMENSIONLESS = frozenset()
_PROPERTY_DIMS: dict[str, tuple[frozenset, str]] = {
    "mcnc": (_MASS_CONC, "a mass concentration (such as mg/L)"),
    "mass concentration": (_MASS_CONC, "a mass concentration (such as mg/L)"),
    "scnc": (_SUBST_CONC, "a substance concentration (such as mmol/L)"),
    "substance concentration": (_SUBST_CONC, "a substance concentration (such as mmol/L)"),
    "ccnc": (_CATALYTIC_CONC, "a catalytic concentration (such as U/L)"),
    "catalytic concentration": (_CATALYTIC_CONC, "a catalytic concentration (such as U/L)"),
    "ncnc": (_NUMBER_CONC, "a number concentration (such as 10^3/uL)"),
    "number concentration": (_NUMBER_CONC, "a number concentration (such as 10^3/uL)"),
    "mfr": (_DIMENSIONLESS, "a fraction (such as %)"),
    "nfr": (_DIMENSIONLESS, "a fraction (such as %)"),
    "vfr": (_DIMENSIONLESS, "a fraction (such as %)"),
    "sfr": (_DIMENSIONLESS, "a fraction (such as %)"),
    "mass fraction": (_DIMENSIONLESS, "a fraction (such as %)"),
    "number fraction": (_DIMENSIONLESS, "a fraction (such as %)"),
    "volume fraction": (_DIMENSIONLESS, "a fraction (such as %)"),
    "time": (frozenset({("[time]", 1.0)}), "a time (such as ms or days)"),
    "len": (frozenset({("[length]", 1.0)}), "a length (such as mm)"),
    "length": (frozenset({("[length]", 1.0)}), "a length (such as mm)"),
    "mass": (frozenset({("[mass]", 1.0)}), "a mass (such as kg)"),
    "vol": (frozenset({("[length]", 3.0)}), "a volume (such as mL)"),
    "volume": (frozenset({("[length]", 3.0)}), "a volume (such as mL)"),
    "area": (frozenset({("[length]", 2.0)}), "an area (such as cm2)"),
    "pres": (frozenset({("[mass]", 1.0), ("[length]", -1.0), ("[time]", -2.0)}), "a pressure (such as mmHg)"),
    "pressure": (frozenset({("[mass]", 1.0), ("[length]", -1.0), ("[time]", -2.0)}), "a pressure (such as mmHg)"),
    "vel": (frozenset({("[length]", 1.0), ("[time]", -1.0)}), "a velocity (such as cm/s)"),
    "velocity": (frozenset({("[length]", 1.0), ("[time]", -1.0)}), "a velocity (such as cm/s)"),
    "nrat": (frozenset({("[time]", -1.0)}), "a rate (such as /min)"),
    "number rate": (frozenset({("[time]", -1.0)}), "a rate (such as /min)"),
}

# S14: what VARTYPE a LOINC scale allows.
_QUANTITATIVE = {"qn", "quantitative"}
_CATEGORICAL_SCALES = {"ord", "nom", "ordinal", "nominal"}


# ---------------------------------------------------------------------------
# Athena records of concept ids, cached on disk
# ---------------------------------------------------------------------------

_athena_lock = threading.Lock()


def _athena_cache_path() -> str:
    return os.path.join(settings.data_folder, "data_quality", "athena_concepts.json")


def _load_athena_cache() -> dict[str, Any]:
    try:
        with open(_athena_cache_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_athena_cache(cache: dict[str, Any]) -> None:
    path = _athena_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def _fetch_athena_concept(cid: int) -> dict[str, Any]:
    """Athena's record of one id: {"found": bool, ...} or {"error": "..."}."""
    from CohortVarLinker.validate_cde import _ATHENA_HEADERS
    try:
        r = requests.get(f"{ATHENA_CONCEPT_URL}/{cid}", headers=_ATHENA_HEADERS, timeout=20)
    except requests.RequestException as e:
        return {"error": f"could not reach Athena: {e}"}
    if r.status_code == 404:
        return {"found": False, "fetched": time.time()}
    if not r.ok:
        return {"error": f"Athena returned HTTP {r.status_code}"}
    try:
        d = r.json()
    except ValueError:
        return {"error": "Athena returned no JSON"}
    if not isinstance(d, dict) or d.get("id") is None:
        return {"found": False, "fetched": time.time()}
    return {
        "found": True, "fetched": time.time(),
        "code": d.get("code"), "name": d.get("name"), "vocabulary": d.get("vocabularyId"),
        "domain": d.get("domainId"), "class": d.get("conceptClassId"),
        "standard": d.get("standardConcept"), "invalid_reason": d.get("invalidReason"),
    }


def athena_records(ids: set[int]) -> tuple[dict[int, dict[str, Any]], int]:
    """Athena's record of every id, from the cache where fresh, and how many
    could not be fetched."""
    with _athena_lock:
        cache = _load_athena_cache()
    now = time.time()
    out: dict[int, dict[str, Any]] = {}
    todo = []
    for cid in ids:
        hit = cache.get(str(cid))
        if hit and "error" not in hit and now - hit.get("fetched", 0) < _ATHENA_MAX_AGE:
            out[cid] = hit
        else:
            todo.append(cid)
    failed = 0
    if todo:
        with ThreadPoolExecutor(max_workers=_ATHENA_WORKERS) as pool:
            for cid, rec in zip(todo, pool.map(_fetch_athena_concept, todo)):
                if "error" in rec:
                    failed += 1
                    continue
                out[cid] = rec
        with _athena_lock:
            cache = _load_athena_cache()
            for cid in todo:
                if cid in out:
                    cache[str(cid)] = out[cid]
            _save_athena_cache(cache)
    return out, failed


# ---------------------------------------------------------------------------
# Reading the dictionary
# ---------------------------------------------------------------------------

def _cell(row: pd.Series, col: str) -> str:
    v = row.get(col, "")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "na", "none") else s


def _split(s: str) -> list[str]:
    return [p.strip() for p in s.split("|")] if s else []


def _to_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _prefix(code: str) -> tuple[str, str]:
    """'loinc:8302-2' → ('LOINC', '8302-2'); no colon → ('', code)."""
    if ":" not in code:
        return "", code.strip()
    pre, post = code.split(":", 1)
    return pre.strip().upper(), post.strip()


def _annotations(row: pd.Series) -> list[dict[str, Any]]:
    """Every concept annotation of one dictionary row."""
    out = []
    for field, (name_col, code_col, id_col, label) in FIELDS.items():
        names, codes, ids = _cell(row, name_col), _cell(row, code_col), _cell(row, id_col)
        if not codes and not ids:
            continue
        if field in _LIST_FIELDS:
            n, c, i = _split(names), _split(codes), _split(ids)
            if not (len(c) == len(i) and (not n or len(n) == len(c))):
                continue  # a count mismatch is S9, reported by the syntax check
            triples = zip(n or [""] * len(c), c, i)
        else:
            if "|" in codes or "|" in ids:
                continue  # several codes in one cell is S1, reported by the syntax check
            triples = [(names, codes, ids)]
        for name, code, oid in triples:
            vocab, bare = _prefix(code)
            out.append({"field": field, "field_label": label, "name": name, "code": code,
                        "vocab": vocab, "bare_code": bare, "omop_id": _to_int(oid)})
    return out


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _norm_name(s: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).split())


def _graph_record(g: Any, cid: Optional[int]) -> Optional[dict[str, Any]]:
    if cid is None:
        return None
    meta = g.graph.graph.get("meta")
    if meta is None or cid not in meta.index:
        return None
    row = meta.loc[cid]
    def get(col: str) -> str:
        v = row.get(col, "") if col in meta.columns else ""
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
    synonyms = [s.strip() for s in re.split(r"[|;]", get("concept_synonym")) if s.strip()]
    return {"name": get("concept_name"), "code": get("concept_code"), "vocabulary": get("concept_vocabulary"),
            "domain": get("concept_domain"), "class": get("concept_class"), "synonyms": synonyms}


def _codes_agree(given_vocab: str, given_code: str, vocab: str, code: str) -> bool:
    """Whether a written code is the code of a record. Vocabulary names differ
    in spelling (SNOMEDCT, SNOMED), so only the code is compared, with case
    ignored except for UCUM, where case carries meaning."""
    if not code:
        return True
    if given_vocab in _UCUM or (vocab or "").upper() == "UCUM":
        return given_code == code
    return given_code.lower() == code.lower()


def _describe(a: dict[str, Any]) -> str:
    bits = [a["code"]] + ([f"OMOP {a['omop_id']}"] if a["omop_id"] is not None else [])
    name = f" '{a['name']}'" if a["name"] else ""
    return f"{a['field_label']}{name} ({', '.join(bits)})"


def _concept_phrase(a: dict[str, Any]) -> str:
    """"variable concept 'Exacerbation of heart failure' (omop:OMOP5166003, OMOP 1340348)"."""
    field = a["field_label"][0].lower() + a["field_label"][1:]
    name = f" '{a['name']}'" if a["name"] else ""
    omop = f", OMOP {a['omop_id']}" if a["omop_id"] is not None else ""
    return f"{field}{name} ({a['code']}{omop})"


def _issue(issue_type: str, variable: Optional[str], row: int, a: dict[str, Any], text: str) -> dict[str, Any]:
    return {"type": issue_type, "variable": variable, "row": row, "field": a["field"],
            "message": f"{_describe(a)}: {text}"}


def check_dictionary(dict_path: str) -> dict[str, Any]:
    """S10 to S14 for one dictionary. Returns the issues and a note of what
    could not be checked."""
    from CohortVarLinker.src.units import parse_unit
    from CohortVarLinker.validate_cde import get_omop_graph
    from src.upload import normalize_column_name

    df = pd.read_csv(dict_path, na_values=[""], keep_default_na=False, dtype=str).fillna("")
    df.columns = [normalize_column_name(c) for c in df.columns]
    g = get_omop_graph()

    rows: list[tuple[int, pd.Series, list[dict[str, Any]]]] = []
    ids: set[int] = set()
    for i, row in df.iterrows():
        anns = [a for a in _annotations(row) if a["vocab"] not in _CUSTOM]
        for a in anns:
            a["graph"] = _graph_record(g, a["omop_id"])
            if a["omop_id"] is not None:
                ids.add(a["omop_id"])
        rows.append((int(i) + 2, row, anns))  # +2: header line, 1-based like the syntax check

    athena, athena_failed = athena_records(ids)

    issues: list[dict[str, Any]] = []
    for row_no, row, anns in rows:
        variable = _cell(row, "VARIABLENAME") or None
        for a in anns:
            graph, ath = a["graph"], athena.get(a["omop_id"]) if a["omop_id"] is not None else None
            # The id's own record: the local graph, else Athena.
            rec = graph or (ath if ath and ath.get("found") else None)

            # S10 for ids the local graph does not hold.
            if graph is None and ath is not None:
                if not ath.get("found"):
                    issues.append(_issue("concepts.code_id_mismatch", variable, row_no, a,
                                         f"OMOP id {a['omop_id']} does not exist in Athena."))
                    continue
                if not _codes_agree(a["vocab"], a["bare_code"], ath.get("vocabulary", ""), ath.get("code", "")):
                    issues.append(_issue("concepts.code_id_mismatch", variable, row_no, a,
                                         f"OMOP id {a['omop_id']} is '{ath.get('name')}' with code "
                                         f"{ath.get('vocabulary')}:{ath.get('code')}, not {a['code']}."))
                    continue
            elif graph is not None and not _codes_agree(a["vocab"], a["bare_code"], graph["vocabulary"], graph["code"]):
                continue  # S10, reported by the concept check

            # S11: vocabulary for the field.
            if a["vocab"]:
                if a["field"] == "unit" and a["vocab"] not in _UCUM:
                    issues.append(_issue("concepts.wrong_vocabulary", variable, row_no, a,
                                         f"a unit concept must come from UCUM, not {a['vocab']}."))
                elif a["field"] != "unit" and a["vocab"] in _UCUM:
                    issues.append(_issue("concepts.wrong_vocabulary", variable, row_no, a,
                                         "a UCUM code is a unit; it belongs in the unit columns."))
                elif a["field"] == "visit" and a["vocab"] not in _VISIT_VOCABS:
                    issues.append(_issue("concepts.wrong_vocabulary", variable, row_no, a,
                                         f"a visit concept is expected from SNOMED, LOINC, OMOP or CDISC, not {a['vocab']}."))

            # S11: domain for the field. Plain wording: what the dictionary says,
            # and what the concept's domain actually is.
            domain = (rec or {}).get("domain") or (ath or {}).get("domain") or ""
            if domain:
                shown = domain[:1].upper() + domain[1:]
                concept = _concept_phrase(a)
                message = None
                if a["field"] == "unit" and domain.lower() != "unit":
                    message = f'The dictionary uses {concept} as a unit. Its domain is "{shown}"; a unit\'s domain is "Unit".'
                elif a["field"] == "categorical" and domain.lower() == "unit":
                    message = f'The dictionary uses {concept} as a category, but it is a unit (domain "Unit").'
                elif a["field"] == "variable":
                    expected = _TABLE_DOMAIN.get(_cell(row, "DOMAIN").lower())
                    if expected and domain.lower() != expected.lower():
                        message = f'The dictionary declares {concept} to be in domain "{expected}". Correct domain: "{shown}".'
                if message:
                    issues.append({"type": "concepts.wrong_domain", "variable": variable, "row": row_no,
                                   "field": a["field"], "message": message})

            # S12: standard and valid, from Athena.
            if ath and ath.get("found"):
                invalid = str(ath.get("invalid_reason") or "").strip()
                if invalid and invalid.lower() not in ("valid", "v"):
                    issues.append(_issue("concepts.deprecated", variable, row_no, a,
                                         f"Athena marks this concept {invalid.lower()} "
                                         f"({'deprecated' if invalid.upper().startswith('D') else 'replaced' if invalid.upper().startswith('U') else 'not valid'})."))
                standard = str(ath.get("standard") or "").strip()
                if standard and standard.lower() not in ("standard", "s"):
                    kind = "a classification concept" if standard.lower() in ("classification", "c") else "not a standard concept"
                    issues.append(_issue("concepts.non_standard", variable, row_no, a,
                                         f"Athena says this is {kind}; analyses on OMOP use standard concepts."))

            # S13: the name.
            vocab_name = (rec or {}).get("name") or ""
            if a["name"] and vocab_name:
                written = _norm_name(a["name"])
                known = {_norm_name(vocab_name)} | {_norm_name(s) for s in (rec or {}).get("synonyms", [])}
                if written and written not in known:
                    # Plainly: what the dictionary calls the code, and what the
                    # code's own vocabulary calls it.
                    vocabulary = ((rec or {}).get("vocabulary") or a["vocab"] or "the vocabulary").strip()
                    field = a["field_label"][0].lower() + a["field_label"][1:]
                    id_part = f" (OMOP {a['omop_id']})" if a["omop_id"] is not None else ""
                    issues.append({
                        "type": "concepts.name_mismatch", "variable": variable, "row": row_no, "field": a["field"],
                        "message": (f"The dictionary associates {field} {a['code']}{id_part} with the name "
                                    f"'{a['name']}'. {vocabulary} instead associates {a['code']} with the "
                                    f"name '{vocab_name}'."),
                    })

            # S14: declarations against what a LOINC variable concept implies.
            if a["field"] == "variable" and graph is not None and a["omop_id"] is not None:
                axes = g.get_loinc_axes(a["omop_id"]) or {}
                scale = str((axes.get("scale_type") or (None, ""))[1] or "").strip().lower()
                vartype = _cell(row, "VARTYPE").upper()
                if scale in _QUANTITATIVE and vartype in ("STR", "DATETIME"):
                    issues.append(_issue("concepts.vartype_implausible", variable, row_no, a,
                                         f"LOINC measures this quantitatively, but VARTYPE is {vartype}."))
                elif scale in _CATEGORICAL_SCALES and vartype in ("FLOAT", "DATETIME"):
                    issues.append(_issue("concepts.vartype_implausible", variable, row_no, a,
                                         f"LOINC records this on an {scale} scale (categories), but VARTYPE is {vartype}."))
                prop = str((axes.get("property") or (None, ""))[1] or "").strip().lower()
                expected = _PROPERTY_DIMS.get(prop)
                unit_code = _cell(row, "UNIT CONCEPT CODE")
                unit_text = unit_code.split(":", 1)[1] if unit_code.lower().startswith("ucum:") else _cell(row, "UNITS")
                if expected and unit_text:
                    parsed = parse_unit(unit_text)
                    if parsed is not None and parsed[1] != expected[0]:
                        issues.append(_issue("concepts.unit_implausible", variable, row_no, a,
                                             f"LOINC measures {expected[1]}, which '{unit_text}' is not."))

    return {"issues": issues, "athena_failed": athena_failed, "concepts_checked": len(ids)}
