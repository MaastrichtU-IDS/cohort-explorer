"""Server-side variable retrieval for iCARE-AI.

Mirrors the cohorts-page search (substring matching in permissive "OR" mode over
the same variable fields) to inject question-relevant variable details into the
model's context in a SINGLE round: the user's question is tokenized, terms that
are too broad (matching too many cohorts/variables across the catalog) are
automatically excluded, and the remaining terms retrieve full variable details
(label, type, units, concept, additional context).

Also provides catalog size estimates (thin catalog / concept index / full
detail) for the admin context diagnostics.
"""
import logging
import re
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A term is "too broad" when it matches at least this fraction of cohorts...
BROAD_COHORT_FRACTION = 0.5
# ...or at least this many variables across the catalog.
BROAD_VARIABLE_LIMIT = 120

# Caps for the injected retrieval section.
MAX_RETRIEVED_VARS_TOTAL = 40
MAX_RETRIEVED_VARS_PER_COHORT = 10

# Generic English words dropped before matching (domain-broad words like
# "function" are handled by the breadth filter, not this list).
QUERY_STOPWORDS = {
    "the", "and", "are", "for", "with", "have", "has", "had", "that", "this", "these", "those",
    "from", "about", "does", "how", "can", "you", "your", "what", "which", "who", "when",
    "where", "why", "there", "their", "them", "they", "was", "were", "will", "would", "could",
    "should", "into", "over", "under", "between", "across", "each", "any", "all", "some",
    "more", "most", "many", "much", "our", "out", "not", "but", "than", "then", "also", "its",
    "related", "available", "measured", "measure", "measures", "cohort", "cohorts", "variable",
    "variables", "data", "dataset", "datasets", "study", "studies", "catalog", "catalogue",
    "please", "give", "show", "list", "tell", "compare", "summarize", "suggest", "identify",
}


def _normalize(text: str) -> str:
    """Mirror the cohorts-page normalizeText: split camelCase, separators -> spaces, lowercase."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[_\-.,—]", " ", text)
    return text.lower()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "na", "n/a", "nan", "none", "null", "-", "--") else text


# The same variable fields the cohorts-page search looks at.
_SEARCHABLE_VAR_FIELDS = ("var_name", "var_label", "concept_name", "mapped_label", "omop_domain", "concept_code", "omop_id")
_SEARCHABLE_CAT_FIELDS = ("value", "label", "mapped_label")


def _variable_blob(var: Any) -> str:
    """Concatenated, normalized searchable text of one variable (incl. categories)."""
    bits = [_clean(getattr(var, f, "")) for f in _SEARCHABLE_VAR_FIELDS]
    for cat in getattr(var, "categories", None) or []:
        for f in _SEARCHABLE_CAT_FIELDS:
            bits.append(_clean(getattr(cat, f, "")))
    return _normalize(" ".join(b for b in bits if b))


def _variable_detail_line(var: Any) -> str:
    """One rich line per retrieved variable: name, label, type/units, concept, context."""
    name = _clean(getattr(var, "var_name", "")) or "?"
    bits = []
    label = _clean(getattr(var, "var_label", ""))
    if label and label.lower() != name.lower():
        bits.append(label)
    meta = [m for m in (_clean(getattr(var, "var_type", "")), _clean(getattr(var, "units", ""))) if m]
    if meta:
        bits.append(f"[{', '.join(meta)}]")
    concept = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
    domain = _clean(getattr(var, "omop_domain", ""))
    if concept or domain:
        bits.append(f"(concept: {concept or '?'}{f'; domain: {domain}' if domain else ''})")
    extra = _clean(getattr(var, "additional_context", ""))
    if extra:
        bits.append(f"note: {extra[:120]}")
    return f"{name} — {' '.join(bits)}" if bits else name


# ---- Cached search index -----------------------------------------------------
# One entry per variable: (cohort_id, blob, var object). Rebuilt when the cohort
# cache changes shape, at most every few minutes.

_index_lock = threading.Lock()
_index_cache: dict[str, Any] = {"key": None, "built_at": 0.0, "entries": []}
_INDEX_TTL_SECONDS = 300


def _get_index(all_cohorts: dict[str, Any]) -> list[tuple[str, str, Any]]:
    n_vars = sum(len(getattr(c, "variables", {}) or {}) for c in all_cohorts.values())
    key = f"{len(all_cohorts)}:{n_vars}"
    now = time.time()
    with _index_lock:
        if _index_cache["key"] == key and now - _index_cache["built_at"] < _INDEX_TTL_SECONDS:
            return _index_cache["entries"]
    entries: list[tuple[str, str, Any]] = []
    for cohort_id, cohort in all_cohorts.items():
        for var in (getattr(cohort, "variables", {}) or {}).values():
            entries.append((cohort_id, _variable_blob(var), var))
    with _index_lock:
        _index_cache.update({"key": key, "built_at": now, "entries": entries})
    logger.info("Chat retrieval index built: %d variables across %d cohorts", len(entries), len(all_cohorts))
    return entries


# ---- Query-time retrieval ----------------------------------------------------

def extract_query_terms(question: str) -> list[str]:
    """Tokenize the question into candidate search terms (before breadth filtering)."""
    words = re.split(r"[^a-zA-Z0-9]+", _normalize(question))
    seen = []
    for w in words:
        w = w.strip()
        if len(w) >= 3 and w not in QUERY_STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:12]


def retrieve_for_question(
    question: str,
    all_cohorts: dict[str, Any],
    restrict_to: Optional[list[str]] = None,
) -> Optional[str]:
    """Build the context section with variable details matching the question.

    Matching mirrors the cohorts-page search in OR mode; terms matching too many
    cohorts/variables are excluded (and the exclusion is stated in the section so
    the model can tell the user). Returns None when there is nothing to inject.
    """
    terms = extract_query_terms(question)
    if not terms or not all_cohorts:
        return None
    entries = _get_index(all_cohorts)
    if not entries:
        return None
    n_cohorts = len(all_cohorts)

    # Per-term breadth statistics over the whole catalog.
    included: list[str] = []
    excluded: list[dict[str, Any]] = []
    term_matches: dict[str, list[int]] = {}
    for term in terms:
        idxs = [i for i, (_, blob, _v) in enumerate(entries) if term in blob]
        if not idxs:
            continue
        cohorts_hit = {entries[i][0] for i in idxs}
        if len(cohorts_hit) >= BROAD_COHORT_FRACTION * n_cohorts or len(idxs) >= BROAD_VARIABLE_LIMIT:
            excluded.append({"term": term, "cohorts": len(cohorts_hit), "variables": len(idxs)})
        else:
            included.append(term)
            term_matches[term] = idxs

    matched_idxs: list[int] = sorted({i for idxs in term_matches.values() for i in idxs})
    restrict = set(restrict_to or [])
    if restrict:
        in_focus = [i for i in matched_idxs if entries[i][0] in restrict]
        # Prefer matches inside the focused cohorts; fall back to the catalog.
        if in_focus:
            matched_idxs = in_focus

    header_bits = []
    if included:
        header_bits.append(f"searched the catalog for: {', '.join(included)}")
    for e in excluded:
        header_bits.append(
            f"excluded '{e['term']}' as too broad (matches {e['variables']} variables across {e['cohorts']} cohorts)"
        )
    if not matched_idxs:
        if excluded:
            return "Variable search note: " + "; ".join(header_bits) + ". No specific variables matched the remaining terms."
        return None

    # Group by cohort with per-cohort and total caps.
    by_cohort: dict[str, list[str]] = {}
    total = 0
    truncated = False
    for i in matched_idxs:
        cohort_id, _blob, var = entries[i]
        lines = by_cohort.setdefault(cohort_id, [])
        if len(lines) >= MAX_RETRIEVED_VARS_PER_COHORT:
            truncated = True
            continue
        if total >= MAX_RETRIEVED_VARS_TOTAL:
            truncated = True
            break
        lines.append(_variable_detail_line(var))
        total += 1

    parts = [f"Variables matching the question ({'; '.join(header_bits)}):"]
    for cohort_id, lines in by_cohort.items():
        parts.append(f"### {cohort_id}:")
        parts.extend(f"  - {line}" for line in lines)
    if truncated:
        parts.append(
            f"  (list capped at {MAX_RETRIEVED_VARS_TOTAL} variables / {MAX_RETRIEVED_VARS_PER_COHORT} per cohort — "
            "suggest the user narrows the question or selects cohorts for the full picture)"
        )
    return "\n".join(parts)


# ---- Model-driven catalog search (the chat's "search tool") ------------------
# The chat's planning round proposes search terms; each term is run here with
# the same matching as the cohorts-page search (all words of the term must
# appear in the variable's searchable text). Results are structured so the UI
# can render them in a dedicated search-results panel, and formatted for the
# model with explicit totals (ALL matching cohorts; per-cohort counts).

SEARCH_VARS_SHOWN_PER_COHORT = 10
# Variable details are expanded only for this many cohorts per term (the user's
# selected cohorts first, then by match count) — ALL matching cohorts are still
# named with their counts.
SEARCH_COHORTS_DETAILED = 6
SEARCH_EQUIVALENTS_SHOWN = 6
SEARCH_CONTEXT_CHAR_CAP = 40000
# Standard-code expansion: variables sharing a standard code with a text match
# are pulled into the results too (that is how BB_3M or ALTROBB count as beta
# blockers via ATC:C07A). Codes carried by more than this many variables are
# considered too generic to expand.
SEARCH_CODE_EXPANSION_LIMIT = 400
SEARCH_CODES_PER_TERM = 10


def _var_public(var: Any) -> dict[str, Any]:
    """The variable fields the search panel shows (no values, aggregate-free)."""
    return {
        "var_name": _clean(getattr(var, "var_name", "")),
        "var_label": _clean(getattr(var, "var_label", "")),
        "concept_name": _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", "")),
        "omop_domain": _clean(getattr(var, "omop_domain", "")),
        "var_type": _clean(getattr(var, "var_type", "")),
        "units": _clean(getattr(var, "units", "")),
        "visits": _clean(getattr(var, "visits", "")),
        "categorical": bool(getattr(var, "categories", None)),
    }


def _norm_code(value: Any) -> str:
    text = _clean(value).lower()
    return text.split(":", 1)[-1].strip() if ":" in text else text


def _eda_names(cohort_id: str) -> set:
    """Names (lowercased) of the variables that have an EDA entry for this
    cohort — those get a clickable chart marker in the chat."""
    try:
        from src.nocode import _load_eda

        return set((_load_eda(cohort_id) or {}).keys())
    except Exception:
        return set()


def _equivalents_map(entries: list[tuple[str, str, Any]]) -> dict[str, list[tuple[str, str, Any]]]:
    """code/OMOP id -> [(cohort_id, var_name, var)]: variables sharing a standard code."""
    by_code: dict[str, list[tuple[str, str, Any]]] = {}
    for cohort_id, _blob, var in entries:
        for raw in (getattr(var, "concept_code", None), getattr(var, "omop_id", None)):
            code = _norm_code(raw)
            if code:
                by_code.setdefault(code, []).append((cohort_id, _clean(getattr(var, "var_name", "")), var))
    return by_code


def _word_match(blob: str, w: str) -> bool:
    """Substring match with light stemming, so 'blockers' and 'blocking' both
    find 'blocker' (and vice versa via the substring direction)."""
    if w in blob:
        return True
    for suf in ("es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4 and w[: len(w) - len(suf)] in blob:
            return True
    return False


def run_chat_searches(
    terms: list[str],
    all_cohorts: dict[str, Any],
    restrict_to: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Run each term through the catalog search. A variable matches a term when
    every word of the term appears in its searchable text. Returns, per term,
    ALL matching cohorts with their counts and up to SEARCH_VARS_SHOWN_PER_COHORT
    variables each, with cross-cohort equivalents (shared standard codes)."""
    entries = _get_index(all_cohorts)
    eq_map = _equivalents_map(entries)
    restrict = {str(c) for c in (restrict_to or [])}
    runs: list[dict[str, Any]] = []
    for raw_term in terms[:8]:
        words = [w for w in _normalize(str(raw_term)).split() if len(w) >= 2 and w not in QUERY_STOPWORDS]
        if not words:
            continue
        by_cohort: dict[str, list[Any]] = {}
        for cohort_id, blob, var in entries:
            if all(_word_match(blob, w) for w in words):
                by_cohort.setdefault(cohort_id, []).append(var)
        # Standard-code expansion: any code carried by a text match pulls in the
        # other variables sharing it, in every cohort (marked "via code").
        seen_pairs = {(cid, _clean(getattr(v, "var_name", ""))) for cid, vs in by_cohort.items() for v in vs}
        codes_used: dict[str, dict] = {}
        code_added: dict[str, list[tuple[Any, str]]] = {}
        for cid, vs in list(by_cohort.items()):
            for var in vs:
                for raw in (getattr(var, "concept_code", None), getattr(var, "omop_id", None)):
                    code = _norm_code(raw)
                    peers = eq_map.get(code, [])
                    if not code or code in codes_used or len(peers) > SEARCH_CODE_EXPANSION_LIMIT or len(codes_used) >= SEARCH_CODES_PER_TERM:
                        continue
                    name = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
                    display = _clean(raw) + (f" ({name})" if name else "")
                    expanded = False
                    for ocid, oname, ovar in peers:
                        if (ocid, oname) in seen_pairs:
                            continue
                        seen_pairs.add((ocid, oname))
                        code_added.setdefault(ocid, []).append((ovar, display))
                        expanded = True
                    if expanded:
                        # display (code + name) goes to the model's context; the
                        # bare concept name is what the search panel shows.
                        codes_used[code] = {"display": display, "name": name}
        cohorts_out = []
        all_cohort_ids = set(by_cohort) | set(code_added)
        totals = {cid: len(by_cohort.get(cid, [])) + len(code_added.get(cid, [])) for cid in all_cohort_ids}
        ranked = sorted(all_cohort_ids, key=lambda cid: (cid not in restrict if restrict else False, -totals[cid]))
        detailed_ids = set(ranked[:SEARCH_COHORTS_DETAILED])
        for cohort_id in sorted(all_cohort_ids, key=lambda cid: -totals[cid]):
            # text matches first, then the ones pulled in by a shared code
            candidates = [(v, None) for v in by_cohort.get(cohort_id, [])] + list(code_added.get(cohort_id, []))
            shown = []
            # Variables are included for EVERY cohort (the search panel shows them
            # on click); the "detailed" flag marks the top cohorts whose lists go
            # into the model's context (format_search_context keeps that cap).
            eda_names = _eda_names(cohort_id)
            for var, via_code in candidates[:SEARCH_VARS_SHOWN_PER_COHORT]:
                d = _var_public(var)
                if via_code:
                    d["via_code"] = True
                    d["matched_code"] = via_code
                if (d.get("var_name") or "").strip().lower() in eda_names:
                    d["has_eda"] = True
                eqs = []
                for raw in (getattr(var, "concept_code", None), getattr(var, "omop_id", None)):
                    code = _norm_code(raw)
                    for other_cohort, other_name, _ovar in eq_map.get(code, []):
                        if other_cohort != cohort_id and (other_cohort, other_name) not in eqs:
                            eqs.append((other_cohort, other_name))
                if eqs:
                    d["equivalents"] = [{"cohort_id": c, "var_name": n} for c, n in eqs[:SEARCH_EQUIVALENTS_SHOWN]]
                shown.append(d)
            cohorts_out.append({
                "cohort_id": cohort_id,
                "matches": totals[cohort_id],
                "text_matches": len(by_cohort.get(cohort_id, [])),
                "code_matches": len(code_added.get(cohort_id, [])),
                "in_selection": (not restrict) or cohort_id in restrict,
                "detailed": cohort_id in detailed_ids,
                # EDA / variable profiling exists for this cohort (any variable)
                "has_eda_profile": len(eda_names) > 0,
                "variables": shown,
            })
        runs.append({
            "term": str(raw_term),
            "total_matches": sum(c["matches"] for c in cohorts_out),
            "cohorts_matched": len(cohorts_out),
            "codes": [{"code": c, "display": v["display"], "name": v["name"]} for c, v in codes_used.items()],
            "cohorts": cohorts_out,
        })
    return runs


def format_search_context(runs: list[dict[str, Any]], concepts: Optional[list] = None,
                          intersection: Optional[list] = None) -> str:
    """The search results as the model sees them, totals spelled out. When the
    searches were grouped into concepts, the cross-concept INTERSECTION is
    stated up front - computed by the platform, never left to the model."""
    if not runs:
        return ""
    parts = [
        "CATALOG SEARCH RESULTS (from the platform's built-in search tool; the user sees these "
        "same results in a search panel above your answer):"
    ]
    named = [c for c in (concepts or []) if isinstance(c, dict) and c.get("cohorts")]
    if len(named) >= 2:
        labels = [c.get("name") or " / ".join((c.get("terms") or [])[:2]) for c in named]
        parts.append("CONCEPTS SEARCHED: " + "; ".join(
            f"{label} (terms: {', '.join(c.get('terms') or [])}; {len(c['cohorts'])} cohorts match)"
            for label, c in zip(labels, named)))
        if intersection:
            rows = []
            for row in intersection:
                counts = ", ".join(f"{k}: {v}" for k, v in (row.get("per_concept") or {}).items())
                rows.append(f"{row.get('cohort_id')} ({counts})")
            parts.append(
                "COHORTS MATCHING EVERY CONCEPT - computed by the platform, this IS the answer to "
                f"'which cohorts have all of these' ({len(intersection)} cohort(s)): " + "; ".join(rows))
        elif intersection is not None:
            parts.append("COHORTS MATCHING EVERY CONCEPT: none - no cohort matches all of the "
                         "concepts at once (each concept's own matches are below).")
    def _counts_line(coh):
        return ", ".join(
            f"{c['cohort_id']} ({c['matches']}{' incl. ' + str(c['code_matches']) + ' via code' if c.get('code_matches') else ''})"
            for c in coh
        )

    def _is_detailed(c):
        # Since the panel change, every cohort carries a (capped) variables list
        # and "detailed" marks the top cohorts whose list goes into the model's
        # context. Runs saved before that change lack the flag: fall back to
        # "has variables", which reproduces the old top-cohorts-only behavior.
        return c.get("detailed", bool(c.get("variables")))

    def _present_full(run):
        """The main presentation of one term: every matching cohort, details for the top ones."""
        coh = run.get("cohorts") or []
        if not coh:
            parts.append(f'## Search "{run.get("term")}": no matching variables in any cohort.')
            return
        parts.append(
            f'## Search "{run.get("term")}": {run.get("total_matches")} matching variables across '
            f"{len(coh)} cohort(s) — ALL matching cohorts with their counts: {_counts_line(coh)}"
        )
        if run.get("codes"):
            parts.append("   (results include variables matched via shared standard codes: "
                         + "; ".join(c["display"] for c in run["codes"]) + ")")
        prof = [c["cohort_id"] for c in coh if c.get("has_eda_profile")]
        if prof and len(prof) < len(coh):
            parts.append("   Summary statistics (variable distributions) on record for: " + ", ".join(prof)
                         + ". The other matching cohorts have no summary statistics yet.")
        elif prof:
            parts.append("   Summary statistics (variable distributions) are on record for ALL of these cohorts.")
        else:
            parts.append("   None of these cohorts has summary statistics on record.")
        _present_details([c for c in coh if _is_detailed(c)])

    def _present_expansion(run, seen):
        """An expansion term of the same concept: only what it ADDS is spelled out."""
        coh = run.get("cohorts") or []
        term = run.get("term")
        if not coh:
            parts.append(f'   Equivalent term "{term}": no matching variables.')
            return
        new = [c for c in coh if c["cohort_id"] not in seen]
        known = len(coh) - len(new)
        if new:
            parts.append(
                f'   Equivalent term "{term}": matches {len(coh)} cohort(s) — {known} already matched '
                f"earlier terms of this concept, {len(new)} NEW. Cohorts discovered ONLY through this "
                f"term expansion: {_counts_line(new)}"
            )
            _present_details([c for c in new if _is_detailed(c)], indent="   ")
        else:
            parts.append(
                f'   Equivalent term "{term}": matches {len(coh)} cohort(s), all of which already '
                "matched earlier terms of this concept — no new cohorts."
            )

    def _present_details(coh, indent=""):
        for c in coh:
            shown = c.get("variables") or []
            if not shown:
                continue
            parts.append(f"{indent}### {c['cohort_id']} — showing {len(shown)} of {c['matches']} matching variables:")
            for v in shown:
                bits = [v.get("var_name") or "?"]
                if v.get("var_label") and (v.get("var_label") or "").lower() != (v.get("var_name") or "").lower():
                    bits.append(v["var_label"])
                meta = [m for m in (v.get("var_type"), v.get("units"), v.get("omop_domain"),
                                     "categorical" if v.get("categorical") else "") if m]
                if meta:
                    bits.append("[" + ", ".join(meta) + "]")
                if v.get("concept_name"):
                    bits.append(f"(concept: {v['concept_name']})")
                if v.get("equivalents"):
                    eq = ", ".join(f"{e['cohort_id']}::{e['var_name']}" for e in v["equivalents"])
                    bits.append(f"EQUIVALENT BY STANDARD CODE to: {eq}")
                if v.get("via_code"):
                    bits.append(f"MATCHED VIA STANDARD CODE {v.get('matched_code')}")
                if v.get("has_eda"):
                    bits.append(f"CHART-MARKER: \U0001F4CA[{c['cohort_id']}::{v.get('var_name')}]")
                parts.append(indent + "  - " + " — ".join(bits[:2]) + (" " + " ".join(bits[2:]) if len(bits) > 2 else ""))
            if c["matches"] > len(shown):
                parts.append(f"{indent}  (+{c['matches'] - len(shown)} more matching variables in {c['cohort_id']} not listed here)")

    # Named concepts: the concept's first term gets the full presentation; the
    # remaining terms are its EXPANSIONS (synonyms, member drugs, related
    # measurements the model proposed) and only what each one newly discovers is
    # spelled out - a cohort found solely through an expansion is a win worth
    # marking, a cohort repeated from the main term is noise. Terms outside any
    # concept, and the flat single-concept fallback, keep the full presentation.
    runs_by_term = {r.get("term"): r for r in runs}
    presented = set()
    grouped = [c for c in (concepts or []) if isinstance(c, dict) and c.get("name") and c.get("terms")]
    for c in grouped:
        c_terms = [t for t in c["terms"] if t in runs_by_term]
        if not c_terms:
            continue
        parts.append(f"# CONCEPT: {c['name']}")
        seen: set = set()
        for i, t in enumerate(c_terms):
            run = runs_by_term[t]
            if i == 0:
                _present_full(run)
            else:
                _present_expansion(run, seen)
            seen.update(x["cohort_id"] for x in (run.get("cohorts") or []))
            presented.add(t)
    for run in runs:
        if run.get("term") not in presented:
            _present_full(run)
    text_so_far = "\n".join(parts)
    if len(text_so_far) > SEARCH_CONTEXT_CHAR_CAP:
        parts = [text_so_far[:SEARCH_CONTEXT_CHAR_CAP],
                 "(search results truncated for length — the cohort counts above are complete)"]
    parts.append(
        "HOW TO USE THESE RESULTS: base your answer on them, not on memory. When the user asks "
        "which cohorts have something, name ALL the matching cohorts listed above with their "
        "counts — never drop any. When you name cohorts, also say which of them have summary "
        "statistics on record (stated per search above): for those cohorts the catalog holds "
        "each variable's real distribution, so per-category patient counts can be given. Say "
        "'summary statistics', never 'EDA'. DO NOT extract patient counts, build 'available "
        "counts' tables, or tell the user to click markers / visit pages to find counts in "
        "THIS answer: when the question asks for counts or values, a separate follow-up "
        "answer grounded in the recorded summary statistics is generated automatically right "
        "after this one whenever those statistics can answer it — at most add ONE line saying "
        "the summary statistics are being checked and a follow-up with the concrete numbers "
        "may appear below. When listing variables, name at most 10-15 per cohort and "
        "ALWAYS state the full counts explicitly (e.g. \"TIME-CHF has 57 matching variables; "
        "here are 12\"), so the user knows there are more. The per-cohort variable lists are "
        "CAPPED and only the top cohorts are expanded - NEVER conclude that a cohort lacks "
        "something because its variables are not shown; the 'ALL matching cohorts' line of each "
        "search, and the COHORTS MATCHING EVERY CONCEPT line, are the complete truth. For "
        "multi-criteria questions, answer from the COHORTS MATCHING EVERY CONCEPT line exactly "
        "as given. A concept's complete cohort set is its main term's ALL-cohorts line PLUS the "
        "NEW cohorts of each of its expansion terms; when a cohort was found only through an "
        "expansion, say so - e.g. 'GISSI-HF (found via the metoprolol expansion)'. The variables "
        "listed are recorded COHORT-WIDE: for a 'patients with X' question, name the variable(s) "
        "that identify X, say plainly that the other variables are not restricted to X patients, "
        "and point to a Data Clean Room for the actual subsetting. NEVER shorten an answer by dropping "
        "cohorts: every cohort listed above must appear in your answer, even in the short/summary "
        "style — describe a few in detail if space is tight, then end with one line naming ALL the "
        "remaining matches and pointing at the search results for the details. Use the EQUIVALENT BY STANDARD CODE "
        "links to point out which variables correspond across cohorts. Variables with a "
        "CHART-MARKER have an EDA distribution graph: EVERY time you mention such a variable by "
        "name, put its exact marker (the \U0001F4CA[cohort::variable] token, copied verbatim) right "
        "after the name — it renders as a clickable chart icon that opens the graph. Never invent "
        "a marker for a variable that has none. The search HAS ALREADY BEEN RUN: never tell the "
        "user to run a search themselves or to \"confirm with a fresh search\" — answer from "
        "these results."
    )
    return "\n".join(parts)


# ---- Catalog size estimates (admin diagnostics) ------------------------------

def _estimate_tokens(text: str) -> int:
    return round(len(text) / 4)


def catalog_size_estimates(all_cohorts: dict[str, Any]) -> dict[str, Any]:
    """Token estimates for the candidate always-present context encodings."""
    n_vars = 0
    full_lines: list[str] = []
    concept_map: dict[str, set] = {}
    for cohort_id, cohort in all_cohorts.items():
        full_lines.append(f"### {cohort_id}")
        for var in (getattr(cohort, "variables", {}) or {}).values():
            n_vars += 1
            full_lines.append(f"- {_variable_detail_line(var)}")
            concept = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
            name = _clean(getattr(var, "var_name", ""))
            if concept:
                concept_map.setdefault(concept, set()).add(f"{cohort_id}:{name}")
    concept_lines = [
        f"- {concept}: {', '.join(sorted(refs))}" for concept, refs in sorted(concept_map.items())
    ]
    return {
        "n_cohorts": len(all_cohorts),
        "n_variables": n_vars,
        "n_distinct_concepts": len(concept_map),
        "full_detail_tokens": _estimate_tokens("\n".join(full_lines)),
        "concept_index_tokens": _estimate_tokens("\n".join(concept_lines)),
    }
