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
