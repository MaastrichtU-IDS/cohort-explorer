"""Standalone longitudinal-variable audit.

Extracts the *identification*, *labelling* and *ordering* logic that lives
inside the generated c4 script in ``eda_scripts.py`` so it can be run outside
the Decentriq enclave against the raw data-dictionary CSV files.

The result is a structured report (list of dicts) that an API endpoint can
render as HTML for quick visual verification.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd

from src.cohort_cache import get_cohorts_from_cache
from src.upload import normalize_column_name

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept in sync with the c4 script in eda_scripts.py
# ---------------------------------------------------------------------------

_PATIENT_OMOP_ID = "4086934"
_PATIENT_CONCEPT_CODE = "snomed:184107009"

_BASELINE_RE = re.compile(
    r"(baseline|screening|enrol{1,2}ment|day\s*0\b|week\s*0\b|visit\s*0\b)",
    re.I,
)
_END_RE = re.compile(
    r"(end\s*of\s*(study|trial)|\bfinal\b|\blast\b|\beos\b|study\s*end|follow[- ]?up\s*end)",
    re.I,
)
_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")

_VISIT_UNIT_DAYS = {
    "d": 1.0, "day": 1.0, "days": 1.0,
    "w": 7.0, "wk": 7.0, "wks": 7.0, "week": 7.0, "weeks": 7.0,
    "m": 30.44, "mo": 30.44, "mos": 30.44, "mon": 30.44,
    "month": 30.44, "months": 30.44,
    "y": 365.25, "yr": 365.25, "yrs": 365.25, "year": 365.25, "years": 365.25,
}

_LABEL_STOPWORDS = {
    "visit", "visits", "at", "on", "in", "of", "the", "for", "to", "from", "by",
    "baseline", "base", "screening", "enrolment", "enrollment", "entry",
    "follow", "followup", "fup", "fu", "up", "end", "study", "trial", "final",
    "last", "first", "time", "timepoint", "point", "v", "val", "value",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "d", "w", "wk", "wks", "mo", "mos", "mon", "y", "yr", "yrs",
}


# ---------------------------------------------------------------------------
# Helper functions — mirror the c4 script exactly
# ---------------------------------------------------------------------------

def _visit_label(m: dict) -> str | None:
    return m["visit_concept_name"] or m["visits"]


def _visit_offset_days(text: str) -> float | None:
    tokens = _TOKEN_RE.findall(text)
    for i, tok in enumerate(tokens):
        if not tok[0].isdigit():
            continue
        for neighbour in (i + 1, i - 1):
            if 0 <= neighbour < len(tokens) and tokens[neighbour] in _VISIT_UNIT_DAYS:
                return float(tok) * _VISIT_UNIT_DAYS[tokens[neighbour]]
    return None


def _visit_ordinal(text: str) -> float | None:
    tokens = _TOKEN_RE.findall(text)
    for tok in tokens:
        if tok[0].isdigit():
            return float(tok)
    return None


def _visit_sort_key(label: str | None) -> tuple:
    if not label:
        return (3.0, 0.0, "")
    s = str(label).strip().lower()
    if _BASELINE_RE.search(s):
        return (0.0, 0.0, s)
    if _END_RE.search(s):
        return (4.0, 0.0, s)
    offset = _visit_offset_days(s)
    if offset is not None:
        return (1.0, offset, s)
    ordinal = _visit_ordinal(s)
    if ordinal is not None:
        return (2.0, ordinal, s)
    return (3.0, 0.0, s)


def _concept_tokens(name: str) -> list[str]:
    tokens = []
    for raw in re.split(r"[^a-z0-9]+", str(name).lower()):
        if not raw or raw[0].isdigit():
            continue
        stripped = raw.rstrip("0123456789")
        token = stripped if len(stripped) >= 2 else raw
        if token not in _LABEL_STOPWORDS:
            tokens.append(token)
    return tokens


def _strip_visit_words(text: str) -> str:
    """Drop visit/time boilerplate from a single human-readable label, so that
    "Weight at baseline" names a family as "weight" rather than claiming the
    whole family was measured at baseline. Returns '' if nothing else is left.
    """
    return " ".join(_concept_tokens(text)).strip()


def _shared_tokens(names: list[str]) -> str:
    """The tokens a set of names has in common: a shared prefix if there is one,
    otherwise the tokens present in every name, in first-name order.
    """
    token_lists = [_concept_tokens(n) for n in names]
    token_lists = [t for t in token_lists if t]
    if not token_lists:
        return ""
    shared: list[str] = []
    for group in zip(*token_lists):
        if len(set(group)) == 1:
            shared.append(group[0])
        else:
            break
    if not shared:
        common = set(token_lists[0])
        for tokens in token_lists[1:]:
            common &= set(tokens)
        for tok in token_lists[0]:
            if tok in common and tok not in shared:
                shared.append(tok)
    return " ".join(shared).strip()


def _family_label(members: list[str], member_labels: list[str], fallback: str) -> str:
    """Preference order: what the member column names share, then what their
    human-readable labels share, then the first label with its visit words
    stripped. Every step goes through _concept_tokens, so no step can leak
    visit boilerplate into the family name.
    """
    for candidate in (
        _shared_tokens(members),
        _shared_tokens(member_labels),
        _strip_visit_words(member_labels[0] if member_labels else ""),
    ):
        if candidate:
            return candidate
    return fallback


def _context_key(m: dict) -> str:
    context = m["additional_context"]
    if not context:
        return "no-additional-context"
    parts = sorted(p.strip().lower() for p in str(context).split("|") if p.strip())
    return "|".join(parts) or "no-additional-context"


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _dict_val(row: pd.Series, col: str, columns: pd.Index) -> str | None:
    """Read a value from a CSV row, treating NaN/empty/'na' as absent."""
    if col not in columns:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "na" else None


def _read_var_meta(csv_path: str) -> dict[str, dict[str, Any]]:
    """Read a data-dictionary CSV and build the ``var_meta`` dict used by
    the longitudinal logic.

    Returns a mapping of ``var_name -> {var_type, var_label, omop_id,
    concept_code, additional_context, units, visits, visit_concept_name}``.
    """
    df = pd.read_csv(csv_path, na_values=[""], keep_default_na=False)
    df = df.dropna(how="all")
    df = df.fillna("")
    df.columns = [normalize_column_name(c) for c in df.columns]
    columns = df.columns

    var_meta: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        var_name = str(row.get("VARIABLENAME", "")).strip()
        if not var_name:
            continue
        var_name = var_name.lower()

        var_label = _dict_val(row, "VARIABLELABEL", columns) or var_name
        var_type = str(row.get("VARTYPE", "")).strip().lower()

        var_meta[var_name] = {
            "var_type": var_type,
            "var_label": var_label,
            "omop_id": _dict_val(row, "VARIABLE OMOP ID", columns),
            "concept_code": _dict_val(row, "VARIABLE CONCEPT CODE", columns),
            "concept_name": _dict_val(row, "VARIABLE CONCEPT NAME", columns),
            "additional_context": _dict_val(row, "ADDITIONAL CONTEXT CONCEPT NAME", columns),
            "units": _dict_val(row, "UNITS", columns),
            "visits": _dict_val(row, "VISITS", columns),
            "visit_concept_name": _dict_val(row, "VISIT CONCEPT NAME", columns),
        }
    return var_meta


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def _identify_families(var_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the same grouping / labelling / ordering logic as c4.

    Returns a list of family dicts, each containing:
        key, concept, context, var_type, var_label, units,
        members, visit_labels, visit_sort_keys
    """
    # Identify patient-id column (last fallback: the OMOP patient-id concept
    # written into the concept code itself, e.g. "omop:4086934")
    patient_id_col = None
    for v, m in var_meta.items():
        if (
            m["omop_id"] == _PATIENT_OMOP_ID
            or (m["concept_code"] and m["concept_code"].lower() == _PATIENT_CONCEPT_CODE)
            or (m["concept_code"] and m["concept_code"].lower().strip()
                in (f"omop:{_PATIENT_OMOP_ID}", _PATIENT_OMOP_ID))
        ):
            patient_id_col = v
            break

    # Group by (concept, context_key)
    groups: dict[tuple[str, str], list[str]] = {}
    for v, m in var_meta.items():
        if v == patient_id_col:
            continue
        concept = m["omop_id"] or m["concept_code"]
        if not concept or not _visit_label(m):
            continue
        groups.setdefault((str(concept), _context_key(m)), []).append(v)

    families: list[dict[str, Any]] = []
    for (concept, context), members in groups.items():
        key = f"{concept} / {context}"
        if len(members) < 2:
            continue
        labels = set(_visit_label(var_meta[v]) for v in members)
        if len(labels) < 2:
            continue

        # Determine dominant type
        type_counts: dict[str, int] = {}
        for v in members:
            t = var_meta[v]["var_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        fam_type = max(type_counts, key=type_counts.get)

        # Keep only members of the dominant type
        members = [v for v in members if var_meta[v]["var_type"] == fam_type]
        if len(members) < 2:
            continue

        # Sort by visit order
        visit_labels = [_visit_label(var_meta[v]) for v in members]
        sort_keys = [_visit_sort_key(lbl) for lbl in visit_labels]
        sorted_pairs = sorted(zip(members, visit_labels, sort_keys), key=lambda x: x[2])
        members = [p[0] for p in sorted_pairs]
        visit_labels = [p[1] for p in sorted_pairs]
        sort_keys = [p[2] for p in sorted_pairs]

        units = next((var_meta[v]["units"] for v in members if var_meta[v]["units"]), None)
        # In visit order, because `members` is already sorted.
        member_labels = [var_meta[v]["var_label"] for v in members]
        concept_name = next(
            (var_meta[v]["concept_name"] for v in members if var_meta[v]["concept_name"]), None
        )
        context_name = next(
            (var_meta[v]["additional_context"] for v in members if var_meta[v]["additional_context"]),
            None,
        )
        # Last resort for the family name. The raw member label is deliberately
        # not used: when every name and label is pure visit boilerplate
        # ("baseline", "end of study") it would name the family after one visit.
        fam_fallback = concept_name or f"concept {concept}"

        families.append({
            "key": key,
            "concept": str(concept),
            "context": context,
            "var_type": fam_type,
            "var_label": _family_label(members, member_labels, fam_fallback),
            "concept_name": concept_name,
            "context_name": context_name,
            "units": units,
            "members": members,
            "member_labels": member_labels,
            "visit_labels": visit_labels,
            "visit_sort_keys": [list(sk) for sk in sort_keys],
        })
    return families


def audit_all_cohorts() -> list[dict[str, Any]]:
    """Run the longitudinal audit on every cohort in the cache.

    Returns a list of ``{cohort_id, cohort_name, csv_path, families, error}`` dicts.
    """
    cohorts = get_cohorts_from_cache("")
    results: list[dict[str, Any]] = []

    for cohort_id, cohort in sorted(cohorts.items()):
        entry: dict[str, Any] = {
            "cohort_id": cohort_id,
            "cohort_name": cohort_id,
            "csv_path": None,
            "n_variables": len(cohort.variables) if cohort.variables else 0,
            "n_families": 0,
            "families": [],
            "error": None,
        }
        try:
            csv_path = cohort.metadata_filepath
            entry["csv_path"] = csv_path
            var_meta = _read_var_meta(csv_path)
            families = _identify_families(var_meta)
            entry["families"] = families
            entry["n_families"] = len(families)
            entry["n_variables"] = len(var_meta)
        except FileNotFoundError as e:
            entry["error"] = f"CSV not found: {e}"
        except Exception as e:
            log.exception("Longitudinal audit failed for %s", cohort_id)
            entry["error"] = str(e)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_audit_html(results: list[dict[str, Any]]) -> str:
    """Render the audit results as a self-contained HTML page."""
    import html as html_mod

    total_cohorts = len(results)
    total_families = sum(r["n_families"] for r in results)
    cohorts_with_families = sum(1 for r in results if r["n_families"] > 0)

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Longitudinal Variable Audit</title>")
    parts.append("<style>")
    parts.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 1.5rem; background: #f8f9fa; color: #1a1a1a; }
        h1 { border-bottom: 2px solid #0d6efd; padding-bottom: .5rem; }
        h2 { margin-top: 2rem; color: #0d6efd; }
        .summary { display: flex; gap: 1rem; margin: 1rem 0 2rem; flex-wrap: wrap; }
        .stat { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem;
                text-align: center; min-width: 140px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
        .stat .num { font-size: 2rem; font-weight: 700; color: #0d6efd; }
        .stat .lbl { font-size: .85rem; color: #6c757d; text-transform: uppercase; letter-spacing: .5px; }
        .cohort { background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
                  margin-bottom: 1.5rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
        .cohort-header { padding: .75rem 1.25rem; background: #f1f3f5; border-bottom: 1px solid #dee2e6;
                         font-weight: 600; display: flex; justify-content: space-between; align-items: center; }
        .cohort-header .badge { background: #0d6efd; color: #fff; border-radius: 12px; padding: 2px 10px;
                                font-size: .8rem; font-weight: 600; }
        .cohort-body { padding: 1rem 1.25rem; }
        .error { color: #dc3545; font-style: italic; padding: .5rem 0; }
        table { width: 100%; border-collapse: collapse; margin-top: .5rem; }
        th { text-align: left; padding: .5rem .75rem; border-bottom: 2px solid #dee2e6; font-size: .85rem;
             text-transform: uppercase; letter-spacing: .5px; color: #495057; }
        td { padding: .5rem .75rem; border-bottom: 1px solid #f1f3f5; font-size: .9rem; vertical-align: top; }
        tr:hover td { background: #f8f9fa; }
        .visits { display: flex; gap: .25rem; flex-wrap: wrap; }
        .visit { background: #e9ecef; border-radius: 4px; padding: 2px 8px; font-size: .8rem; }
        .visit-arrow { color: #adb5bd; align-self: center; font-size: .75rem; }
        .type-num { color: #0d6efd; font-weight: 600; }
        .type-cat { color: #6f42c1; font-weight: 600; }
        .label { font-weight: 600; }
        .context { color: #6c757d; font-size: .85rem; }
        .units { color: #198754; font-size: .8rem; }
        .sort-key { color: #adb5bd; font-size: .75rem; font-family: monospace; }
        .no-families { color: #6c757d; font-style: italic; padding: .5rem 0; }
    """)
    parts.append("</style></head><body>")

    # Header
    parts.append(f"<h1>Longitudinal Variable Audit</h1>")
    parts.append("<div class='summary'>")
    parts.append(f"<div class='stat'><div class='num'>{total_cohorts}</div><div class='lbl'>Cohorts</div></div>")
    parts.append(f"<div class='stat'><div class='num'>{total_families}</div><div class='lbl'>Families</div></div>")
    parts.append(f"<div class='stat'><div class='num'>{cohorts_with_families}</div><div class='lbl'>Cohorts with Families</div></div>")
    parts.append("</div>")

    for r in results:
        n_fam = r["n_families"]
        parts.append("<div class='cohort'>")
        parts.append(
            f"<div class='cohort-header'>"
            f"<span>{html_mod.escape(r['cohort_id'])}</span>"
            f"<span class='badge'>{n_fam} family{'s' if n_fam != 1 else ''}</span>"
            f"</div>"
        )
        parts.append("<div class='cohort-body'>")

        if r["error"]:
            parts.append(f"<div class='error'>{html_mod.escape(r['error'])}</div>")
        elif n_fam == 0:
            parts.append("<div class='no-families'>No longitudinal families identified.</div>")
        else:
            parts.append(f"<div style='font-size:.85rem;color:#6c757d;margin-bottom:.5rem;'>"
                          f"Variables: {r['n_variables']} &middot; CSV: {html_mod.escape(r['csv_path'] or 'N/A')}"
                          f"</div>")
            parts.append("<table><thead><tr>")
            parts.append("<th>Family Label</th>")
            parts.append("<th>Type</th>")
            parts.append("<th>Concept</th>")
            parts.append("<th>Context</th>")
            parts.append("<th>Units</th>")
            parts.append("<th>Members</th>")
            parts.append("<th>Ordered Visits</th>")
            parts.append("</tr></thead><tbody>")

            for fam in r["families"]:
                vtype = fam["var_type"]
                type_class = "type-num" if vtype in ("int", "float") else "type-cat"
                members = ", ".join(html_mod.escape(m) for m in fam["members"])
                member_labels = " &rarr; ".join(
                    html_mod.escape(str(l)) for l in fam.get("member_labels") or []
                )
                if member_labels:
                    members += (
                        f"<div style='font-size:.75rem;color:#6c757d;margin-top:.25rem;'>"
                        f"{member_labels}</div>"
                    )
                context = html_mod.escape(fam["context"]) if fam["context"] != "no-additional-context" else "<span class='context'>none</span>"
                units = f"<span class='units'>{html_mod.escape(fam['units'])}</span>" if fam["units"] else ""

                # Build ordered visits display
                visit_parts: list[str] = []
                for i, (vl, sk) in enumerate(zip(fam["visit_labels"], fam["visit_sort_keys"])):
                    if i > 0:
                        visit_parts.append("<span class='visit-arrow'>&rarr;</span>")
                    visit_parts.append(
                        f"<span class='visit' title='sort key: ({sk[0]:.0f}, {sk[1]:.1f}, &quot;{html_mod.escape(sk[2])}&quot;)'>"
                        f"{html_mod.escape(str(vl))}</span>"
                    )
                visits_html = "<div class='visits'>" + "".join(visit_parts) + "</div>"

                parts.append("<tr>")
                parts.append(f"<td class='label'>{html_mod.escape(fam['var_label'])}</td>")
                parts.append(f"<td class='{type_class}'>{html_mod.escape(vtype)}</td>")
                concept_html = f"<code>{html_mod.escape(fam['concept'])}</code>"
                if fam.get("concept_name"):
                    concept_html += (
                        f"<div style='font-size:.75rem;color:#6c757d;margin-top:.25rem;'>"
                        f"{html_mod.escape(str(fam['concept_name']))}</div>"
                    )
                parts.append(f"<td>{concept_html}</td>")
                parts.append(f"<td>{context}</td>")
                parts.append(f"<td>{units}</td>")
                parts.append(f"<td>{members}</td>")
                parts.append(f"<td>{visits_html}</td>")
                parts.append("</tr>")

            parts.append("</tbody></table>")

        parts.append("</div></div>")

    parts.append("</body></html>")
    return "\n".join(parts)
