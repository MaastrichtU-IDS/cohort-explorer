"""Server-side variable retrieval for iCARE-AI.

- The chat's catalog search: the planner's terms are run with the same
  matching as the cohorts-page search, results structured for the search
  panel and formatted for the model (every matching cohort, with counts).
- Related variables: every variable is scored by the words it shares with the
  question (IDF-weighted), so relevant labels the exact search missed still
  reach the model within the context budget.
- Equivalent variables: for the standard codes of the variables found, every
  catalog variable mapped to the same concept / OMOP id, per cohort.

Also provides catalog size estimates (thin catalog / concept index / full
detail) for the admin context diagnostics.
"""
import bisect
import logging
import math
import re
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Catalog data is budgeted in characters; ~3 chars per token is conservative
# for this catalog (codes, abbreviations, German labels tokenize densely).
CHARS_PER_TOKEN = 3
# Categories listed per variable before the list is cut ("+N more").
MAX_CATEGORIES_LISTED = 15


def budget_chars(share: float) -> int:
    """Character budget for one part of the chat context: `share` of the
    configured CHAT_CONTEXT_BUDGET_TOKENS."""
    from src.config import settings

    return int(settings.chat_context_budget_tokens * share * CHARS_PER_TOKEN)

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


def variable_values(var: Any) -> str:
    """Compact category list ('1=Yes; 0=No') or numeric range ('range 20–95')."""
    cats = getattr(var, "categories", None) or []
    if cats:
        shown = []
        for cat in cats[:MAX_CATEGORIES_LISTED]:
            value = _clean(getattr(cat, "value", ""))
            label = _clean(getattr(cat, "label", "")) or _clean(getattr(cat, "mapped_label", ""))
            shown.append(f"{value}={label}" if value and label and label != value else (value or label))
        more = f"; +{len(cats) - MAX_CATEGORIES_LISTED} more" if len(cats) > MAX_CATEGORIES_LISTED else ""
        return "; ".join(s for s in shown if s) + more
    lo, hi = _clean(getattr(var, "min", "")), _clean(getattr(var, "max", ""))
    if lo or hi:
        return f"range {lo or '?'}–{hi or '?'}"
    return ""


def _variable_detail_line(var: Any) -> str:
    """One rich line per variable: everything the catalog knows about it
    (label, type/units, concept + code, domain, values or range, visits,
    definition/formula, additional context)."""
    name = _clean(getattr(var, "var_name", "")) or "?"
    bits = []
    label = _clean(getattr(var, "var_label", ""))
    if label and label.lower() != name.lower():
        bits.append(label)
    meta = [m for m in (_clean(getattr(var, "var_type", "")), _clean(getattr(var, "units", ""))) if m]
    if meta:
        bits.append(f"[{', '.join(meta)}]")
    concept = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
    code = _clean(getattr(var, "concept_code", "")) or _clean(getattr(var, "omop_id", ""))
    domain = _clean(getattr(var, "omop_domain", ""))
    if concept or code or domain:
        inner = "; ".join(x for x in (
            f"concept: {concept}" if concept else "",
            f"code: {code}" if code else "",
            f"domain: {domain}" if domain else "",
        ) if x)
        bits.append(f"({inner})")
    values = variable_values(var)
    if values:
        bits.append(f"values: {values}")
    visits = _clean(getattr(var, "visits", ""))
    if visits:
        bits.append(f"visits: {visits}")
    for field, tag in (("definition", "definition"), ("formula", "formula"), ("additional_context", "note")):
        text = _clean(getattr(var, field, ""))
        if text:
            bits.append(f"{tag}: {text}")
    return f"{name} — {' | '.join(bits)}" if bits else name


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


# ---- Related variables (relevance-ranked labels) -----------------------------
# The catalog search (below) finds variables containing the planner's exact
# terms. Labels that describe the same thing in other words are missed, and
# listing every variable's label is far beyond the context budget. So each
# question also scores EVERY variable by the words it shares with the question
# and the search terms - rare words weigh more than common ones (IDF) - and
# the best-scoring variables not already in the search results are added as
# compact "name — label (concept)" lines, as many as the budget allows.

# A word matching more than this share of all variables says nothing.
RELATED_MAX_WORD_SHARE = 0.2
# Keep only variables scoring at least this fraction of the best score (low
# enough that a variable matching just one of several concepts still counts).
RELATED_MIN_RELATIVE_SCORE = 0.3
RELATED_MAX_VARIABLES = 200
RELATED_BUDGET_SHARE = 0.12

_word_index_lock = threading.Lock()
_word_index_cache: dict[str, Any] = {"key": None, "postings": {}, "vocab": []}


def _get_word_index(all_cohorts: dict[str, Any]) -> tuple[list[tuple[str, str, Any]], dict[str, list[int]], list[str]]:
    """(entries, word -> entry indexes, sorted vocabulary), rebuilt with the search index."""
    entries = _get_index(all_cohorts)
    key = (_index_cache["key"], id(entries))
    with _word_index_lock:
        if _word_index_cache["key"] == key:
            return entries, _word_index_cache["postings"], _word_index_cache["vocab"]
    postings: dict[str, list[int]] = {}
    for i, (_cid, blob, _var) in enumerate(entries):
        for w in set(re.split(r"[^a-z0-9]+", blob)):
            if len(w) >= 2:
                postings.setdefault(w, []).append(i)
    vocab = sorted(postings)
    with _word_index_lock:
        _word_index_cache.update({"key": key, "postings": postings, "vocab": vocab})
    return entries, postings, vocab


def _stems(token: str) -> set:
    stems = {token}
    for suf in ("es", "s", "ing", "ed", "ers", "er"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            stems.add(token[: len(token) - len(suf)])
    return stems


def _matching_words(token: str, vocab: list[str]) -> list[str]:
    """Vocabulary words for a query token: exact, or sharing its (lightly
    stemmed) prefix of 4+ letters ('blockers' -> 'blocker', 'blocking')."""
    stems = _stems(token)
    found = {token} if token in vocab else set()
    for stem in stems:
        if len(stem) < 4:
            continue
        i = bisect.bisect_left(vocab, stem)
        while i < len(vocab) and vocab[i].startswith(stem):
            found.add(vocab[i])
            i += 1
    return list(found)


def related_variables_section(
    question: str,
    terms: list[str],
    all_cohorts: dict[str, Any],
    restrict_to: Optional[list[str]] = None,
    exclude: Optional[set] = None,
) -> tuple[str, list[tuple[str, str]]]:
    """(context section, [(cohort_id, lowercased var_name)] included) of the variables whose names,
    labels, concepts and categories best match the question's words. `exclude`
    holds (cohort_id, lowercased var_name) pairs already in the search results."""
    tokens: list[str] = []
    for text in [question] + list(terms or []):
        for w in re.split(r"[^a-z0-9]+", _normalize(str(text))):
            if len(w) >= 3 and not w.isdigit() and w not in QUERY_STOPWORDS and w not in tokens:
                tokens.append(w)
    if not tokens or not all_cohorts:
        return "", []
    entries, postings, vocab = _get_word_index(all_cohorts)
    n = len(entries)
    if not n:
        return "", []
    scores: dict[int, float] = {}
    used: list[str] = []
    seen_stems: set = set()
    for token in tokens[:20]:
        # One weight per word, whatever its form ('blocker' / 'blockers').
        stem = min(_stems(token), key=len)
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        hit: set = set()
        for w in _matching_words(token, vocab):
            hit.update(postings.get(w, ()))
        if not hit or len(hit) > RELATED_MAX_WORD_SHARE * n:
            continue
        used.append(token)
        idf = math.log(n / len(hit))
        for i in hit:
            scores[i] = scores.get(i, 0.0) + idf
    if not scores:
        return "", []
    exclude = exclude or set()
    restrict = {str(c) for c in (restrict_to or [])}
    ranked = [i for i in sorted(scores, key=lambda i: -scores[i])
              if (entries[i][0], _clean(getattr(entries[i][2], "var_name", "")).lower()) not in exclude]
    if restrict and any(entries[i][0] in restrict for i in ranked):
        ranked = [i for i in ranked if entries[i][0] in restrict]
    if not ranked:
        return "", []
    floor = scores[ranked[0]] * RELATED_MIN_RELATIVE_SCORE
    ranked = [i for i in ranked if scores[i] >= floor][:RELATED_MAX_VARIABLES]

    cap = budget_chars(RELATED_BUDGET_SHARE)
    by_cohort: dict[str, list[str]] = {}
    included: list[tuple[str, str]] = []
    size = 0
    for i in ranked:
        cohort_id, _blob, var = entries[i]
        name = _clean(getattr(var, "var_name", "")) or "?"
        label = (_clean(getattr(var, "var_label", "")) or _clean(getattr(var, "definition", ""))
                 or _clean(getattr(var, "additional_context", "")))[:150]
        concept = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
        line = name + (f" — {label}" if label and label.lower() != name.lower() else "")
        if concept and concept.lower() not in (label.lower(), name.lower()):
            line += f" (concept: {concept})"
        if size + len(line) > cap:
            break
        by_cohort.setdefault(cohort_id, []).append(line)
        size += len(line) + 6
        included.append((cohort_id, name.lower()))
    if not included:
        return "", []
    parts = [
        f"Variables whose names, labels or concepts share words with the question ({', '.join(used)}), "
        "best matches first. They are NOT exact search matches and not a complete list: use them to "
        "spot relevant variables the search terms missed, cite them by name, but do not count them "
        "when stating how many cohorts or variables match a search."
    ]
    for cohort_id, lines in by_cohort.items():
        parts.append(f"#### {cohort_id}")
        parts.extend(f"  - {line}" for line in lines)
    return "\n".join(parts), included


# ---- Model-driven catalog search (the chat's "search tool") ------------------
# The chat's planning round proposes search terms; each term is run here with
# the same matching as the cohorts-page search (all words of the term must
# appear in the variable's searchable text). Results are structured so the UI
# can render them in a dedicated search-results panel, and formatted for the
# model with explicit totals (ALL matching cohorts; per-cohort counts).

SEARCH_VARS_SHOWN_PER_COHORT = 30
SEARCH_EQUIVALENTS_SHOWN = 20
# Share of CHAT_CONTEXT_BUDGET_TOKENS for the formatted search context. EVERY
# matching cohort is expanded with variable details; when a result set is so
# large that the text would exceed the budget, the per-cohort variable lists
# are pared down step by step (30 -> 20 -> 10 -> 5 -> 3 -> 1) until it fits -
# cohorts are never dropped.
SEARCH_CONTEXT_BUDGET_SHARE = 0.25
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
        "values": variable_values(var),
        "definition": (_clean(getattr(var, "definition", "")) or _clean(getattr(var, "additional_context", "")))[:200],
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
    if w.isdigit():
        # A bare number must not match inside a LARGER number (the '1' of
        # 'type 1' should hit 'type 1' and 'dm1', never omop id '201254').
        return re.search(rf"(?<!\d){re.escape(w)}(?!\d)", blob) is not None
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
        # Bare numbers are kept ("type 1 diabetes" must NOT collapse into
        # "type diabetes", which matches type 2 variables just as well).
        words = [w for w in _normalize(str(raw_term)).split()
                 if (len(w) >= 2 or w.isdigit()) and w not in QUERY_STOPWORDS]
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
        for cohort_id in sorted(all_cohort_ids, key=lambda cid: -totals[cid]):
            # text matches first, then the ones pulled in by a shared code
            candidates = [(v, None) for v in by_cohort.get(cohort_id, [])] + list(code_added.get(cohort_id, []))
            shown = []
            # Variables are included for EVERY cohort - the search panel shows
            # them on click, and format_search_context details every cohort to
            # the model (paring the per-cohort lists only if space runs out).
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
                "detailed": True,
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
    def _render_runs(max_vars: int) -> list[str]:
        """Render every run's presentation with at most max_vars variables per
        cohort. EVERY matching cohort is expanded; format_search_context calls
        this with a shrinking max_vars until the text fits the char budget, so
        completeness at the cohort level never depends on the result size."""
        out: list[str] = []

        def _counts_line(coh):
            return ", ".join(
                f"{c['cohort_id']} ({c['matches']}{' incl. ' + str(c['code_matches']) + ' via code' if c.get('code_matches') else ''})"
                for c in coh
            )

        def _is_detailed(c):
            # Every cohort is detailed nowadays; runs saved before the panel
            # change lack the flag AND have variables only for their top
            # cohorts - the fallback reproduces that old behavior faithfully.
            return c.get("detailed", bool(c.get("variables")))

        def _present_details(coh, indent=""):
            for c in coh:
                shown = (c.get("variables") or [])[:max_vars]
                if not shown:
                    continue
                out.append(f"{indent}#### {c['cohort_id']} — showing {len(shown)} of {c['matches']} matching variables:")
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
                    if v.get("values"):
                        bits.append(f"values: {v['values']}")
                    if v.get("visits"):
                        bits.append(f"visits: {v['visits']}")
                    if v.get("definition"):
                        bits.append(f"definition: {v['definition']}")
                    if v.get("via_code"):
                        bits.append(f"MATCHED VIA STANDARD CODE {v.get('matched_code')}")
                    if v.get("has_eda"):
                        bits.append(f"CHART-MARKER: \U0001F4CA[{c['cohort_id']}::{v.get('var_name')}]")
                    out.append(indent + "  - " + " — ".join(bits[:2]) + (" " + " ".join(bits[2:]) if len(bits) > 2 else ""))
                if c["matches"] > len(shown):
                    out.append(f"{indent}  (+{c['matches'] - len(shown)} more matching variables in {c['cohort_id']} not listed here)")

        def _present_name_lines(coh, indent=""):
            """Old saved payloads only: cohorts without stored details get their
            variable NAMES, so the model can cite without inventing meanings."""
            rows = [c for c in coh if c.get("variables")]
            if not rows:
                return
            out.append(indent + "   Variable NAMES ONLY for the remaining cohorts (labels/details "
                                "not shown - do not guess what a variable measures beyond its name):")
            for c in rows:
                names = ", ".join(str(v.get("var_name") or "?") for v in c["variables"])
                more = c["matches"] - len(c["variables"])
                out.append(f"{indent}     - {c['cohort_id']}: {names}" + (f" (+{more} more)" if more > 0 else ""))

        def _present_full(run):
            """The main presentation of one term: every matching cohort, expanded."""
            coh = run.get("cohorts") or []
            if not coh:
                out.append(f'### Search "{run.get("term")}": no matching variables in any cohort.')
                return
            out.append(
                f'### Search "{run.get("term")}": {run.get("total_matches")} matching variables across '
                f"{len(coh)} cohort(s) — ALL matching cohorts with their counts: {_counts_line(coh)}"
            )
            if run.get("codes"):
                out.append("   (results include variables matched via shared standard codes: "
                           + "; ".join(c["display"] for c in run["codes"]) + ")")
            prof = [c["cohort_id"] for c in coh if c.get("has_eda_profile")]
            if prof and len(prof) < len(coh):
                out.append("   Summary statistics (variable distributions) on record for: " + ", ".join(prof)
                           + ". The other matching cohorts have no summary statistics yet.")
            elif prof:
                out.append("   Summary statistics (variable distributions) are on record for ALL of these cohorts.")
            else:
                out.append("   None of these cohorts has summary statistics on record.")
            _present_details([c for c in coh if _is_detailed(c)])
            _present_name_lines([c for c in coh if not _is_detailed(c)])

        def _present_expansion(run, seen):
            """An expansion term of the same concept: only what it ADDS is spelled out."""
            coh = run.get("cohorts") or []
            term = run.get("term")
            if not coh:
                out.append(f'   Equivalent term "{term}": no matching variables.')
                return
            new = [c for c in coh if c["cohort_id"] not in seen]
            known = len(coh) - len(new)
            if new:
                out.append(
                    f'   Equivalent term "{term}": matches {len(coh)} cohort(s) — {known} already matched '
                    f"earlier terms of this concept, {len(new)} NEW. Cohorts discovered ONLY through this "
                    f"term expansion: {_counts_line(new)}"
                )
                _present_details([c for c in new if _is_detailed(c)], indent="   ")
                _present_name_lines([c for c in new if not _is_detailed(c)], indent="   ")
            else:
                out.append(
                    f'   Equivalent term "{term}": matches {len(coh)} cohort(s), all of which already '
                    "matched earlier terms of this concept — no new cohorts."
                )

        # Named concepts: the concept's first term gets the full presentation;
        # the remaining terms are its EXPANSIONS (synonyms, member drugs,
        # related measurements the model proposed) and only what each one newly
        # discovers is spelled out - a cohort found solely through an expansion
        # is a win worth marking, a cohort repeated from the main term is
        # noise. Terms outside any concept, and the flat single-concept
        # fallback, keep the full presentation.
        runs_by_term = {r.get("term"): r for r in runs}
        presented = set()
        grouped = [c for c in (concepts or []) if isinstance(c, dict) and c.get("name") and c.get("terms")]
        for c in grouped:
            c_terms = [t for t in c["terms"] if t in runs_by_term]
            if not c_terms:
                continue
            out.append(f"### CONCEPT: {c['name']}")
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
        return out

    # Fit-to-budget: try the full per-cohort variable lists first; if the text
    # would blow the budget, pare the lists down step by step - never cohorts.
    cap = budget_chars(SEARCH_CONTEXT_BUDGET_SHARE)
    header_len = len("\n".join(parts))
    rendered: list[str] = []
    for max_vars in (SEARCH_VARS_SHOWN_PER_COHORT, 20, 10, 5, 3, 1):
        rendered = _render_runs(max_vars)
        if header_len + len("\n".join(rendered)) <= cap:
            if max_vars < SEARCH_VARS_SHOWN_PER_COHORT:
                rendered.append(f"(the result set is large: variable lists were shortened to "
                                f"{max_vars} per cohort to fit - every matching cohort is still "
                                f"listed and the counts show the full numbers)")
            break
    parts.extend(rendered)
    text_so_far = "\n".join(parts)
    if len(text_so_far) > cap:
        parts = [text_so_far[:cap],
                 "(search results truncated for length — the cohort counts above are complete)"]
    return "\n".join(parts)


# ---- Equivalent variables across cohorts -------------------------------------
# Variables mapped to the same standard concept code or OMOP id capture the
# same thing. For every such code carried by a variable found for the question
# (search matches and related variables), one line names ALL the catalog's
# variables sharing it, per cohort: the basis for cross-cohort comparisons.

EQUIVALENTS_BUDGET_SHARE = 0.08
# Variable names listed per cohort in one cluster before "+N more".
EQUIVALENT_NAMES_PER_COHORT = 4

_eq_lock = threading.Lock()
_eq_cache: dict[str, Any] = {"key": None, "map": {}}


def _get_equivalents_map(all_cohorts: dict[str, Any]) -> dict[str, list[tuple[str, str, Any]]]:
    """code -> [(cohort_id, var_name, var)], rebuilt with the search index."""
    entries = _get_index(all_cohorts)
    key = (_index_cache["key"], id(entries))
    with _eq_lock:
        if _eq_cache["key"] == key:
            return _eq_cache["map"]
    eq_map = _equivalents_map(entries)
    with _eq_lock:
        _eq_cache.update({"key": key, "map": eq_map})
    return eq_map


def equivalents_section(pairs: list[tuple[str, str]], all_cohorts: dict[str, Any]) -> tuple[str, int]:
    """(context section, number of clusters) for the standard codes carried by
    the given (cohort_id, lowercased var_name) variables; only codes shared by
    two or more cohorts, the most widely shared first."""
    if not pairs or not all_cohorts:
        return "", 0
    eq_map = _get_equivalents_map(all_cohorts)
    wanted = {(c, n) for c, n in pairs}
    # Codes of the wanted variables, in the order the variables were found.
    codes: list[tuple[str, str]] = []  # (normalized code, display)
    seen_codes: set = set()
    by_pair: dict[tuple[str, str], Any] = {}
    for cohort_id, cohort in all_cohorts.items():
        for var in (getattr(cohort, "variables", {}) or {}).values():
            key = (cohort_id, _clean(getattr(var, "var_name", "")).lower())
            if key in wanted:
                by_pair[key] = var
    for pair in pairs:
        var = by_pair.get(tuple(pair))
        if var is None:
            continue
        for raw in (getattr(var, "concept_code", None), getattr(var, "omop_id", None)):
            code = _norm_code(raw)
            if code and code not in seen_codes:
                seen_codes.add(code)
                name = _clean(getattr(var, "concept_name", "")) or _clean(getattr(var, "mapped_label", ""))
                codes.append((code, _clean(raw) + (f" ({name})" if name else "")))

    clusters = []
    seen_members: set = set()
    for code, display in codes:
        members = eq_map.get(code, [])
        per_cohort: dict[str, list[str]] = {}
        for cohort_id, var_name, _var in members:
            if var_name and var_name not in per_cohort.setdefault(cohort_id, []):
                per_cohort[cohort_id].append(var_name)
        if len(per_cohort) < 2:
            continue
        # A concept code and an OMOP id often describe the same set: list it once.
        signature = frozenset((c, n) for c, names in per_cohort.items() for n in names)
        if signature in seen_members:
            continue
        seen_members.add(signature)
        clusters.append((display, per_cohort))
    if not clusters:
        return "", 0
    clusters.sort(key=lambda cl: -len(cl[1]))

    cap = budget_chars(EQUIVALENTS_BUDGET_SHARE)
    parts = [
        "Variables mapped to the same standard concept or OMOP id capture the same thing. Each line "
        "is one concept found among the variables above, with EVERY catalog variable mapped to it, "
        "per cohort (complete, not just search matches). Use these lines to compare cohorts and to "
        "name each cohort's variable for a concept."
    ]
    size = len(parts[0])
    shown = 0
    for display, per_cohort in clusters:
        cohorts_txt = " · ".join(
            f"{cid}: {', '.join(names[:EQUIVALENT_NAMES_PER_COHORT])}"
            + (f" +{len(names) - EQUIVALENT_NAMES_PER_COHORT} more" if len(names) > EQUIVALENT_NAMES_PER_COHORT else "")
            for cid, names in sorted(per_cohort.items())
        )
        line = f"- {display} — {len(per_cohort)} cohorts: {cohorts_txt}"
        if size + len(line) > cap:
            break
        parts.append(line)
        size += len(line) + 1
        shown += 1
    if shown < len(clusters):
        parts.append(f"(+{len(clusters) - shown} more shared concepts not listed for length)")
    return ("\n".join(parts), shown) if shown else ("", 0)


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
