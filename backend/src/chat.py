"""Experimental AI chat backend.

Proxies chat completions to a local LiteLLM server (OpenAI-compatible) and
augments the conversation with context built from the cohort cache: cohort
metadata and a sample of their variables. The frontend selects which cohorts
to focus on; this module turns those selections into a compact system prompt so
the model can answer questions grounded in the actual cohort catalog.

Environment (.env):
    LITELLM_BASE_URL  base_url of the local proxy (required to enable chat)
    LITELLM_API_KEY   api_key for the proxy
    LITELLM_MODEL     default model name (optional, defaults to gpt-3.5-turbo)
"""
import json
import logging
import os
import random
import re
import threading
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src import ai_history
from src.admin import _require_admin
from src.auth import get_current_user
from src.config import settings

router = APIRouter()

logger = logging.getLogger(__name__)

# Context sizing. Catalog data is budgeted from CHAT_CONTEXT_BUDGET_TOKENS (see
# config.py and chat_retrieval.budget_chars); these shares split it between the
# parts of one request. Each part degrades gracefully when it does not fit
# (fewer details per variable, never fewer cohorts).
COHORT_BLOCK_BUDGET_SHARE = 0.55
# Caps for client-supplied text (Glass Box overrides, follow-up blocks).
MAX_SYSTEM_PROMPT_CHARS = 16_000
MAX_CONTEXT_CHARS = 60_000
MAX_SUMMARIZE_CHARS = 24_000
MAX_EDA_CONTEXT_CHARS = 24_000

# How the platform works — shared so the assistant can guide users through the
# actual workflow, not just describe data.
PLATFORM_OVERVIEW = (
    "- The Explorer's main (explore) page lets analysts discover cardiovascular studies/cohorts "
    "and each cohort's variables (its metadata / data dictionary in the catalog).\n"
    "- The explore page has a SEARCH BOX with exactly two settings: WHERE to search (cohorts "
    "metadata, variables information, or all) and the MODE (OR search, AND search, or exact "
    "phrase). It searches variable names, labels, concept names and codes. There are NO other "
    "filters (no domain, data-type or visit filter), so never tell the user to apply one. When "
    "users need to find variables, point them to this search box, never to browser tricks like "
    "Ctrl+F.\n"
    "- To analyse data, an analyst creates an analysis DCR (Data Clean Room): a secure computing "
    "enclave. The data owners (cohort admins) upload their real data into it, and the analyst's "
    "script computes over that data WITHOUT ever seeing the raw records; only permitted outputs "
    "leave the enclave.\n"
    "- Cross-cohort variable mapping is done on the dedicated MAPPING PAGE: the analyst picks a "
    "source cohort and target cohort(s) and the platform generates a mapping file of likely "
    "variable correspondences (suggestions, not guarantees). This is the ONLY way to map "
    "variables across cohorts; there is no per-variable mapping control anywhere else."
)

# Standing rules: the first part of the single system message. The catalog
# data and the task for this particular reply follow it (see _assemble_payload).
SYSTEM_PROMPT = (
    "You are iCARE-AI, the assistant of the iCARE4CVD Cohort Explorer. You help researchers "
    "understand and compare cardiovascular research cohorts and their variables.\n\n"
    "This message has three parts: these standing RULES, then the CATALOG DATA for this "
    "request, then YOUR TASK FOR THIS REPLY. Where the task asks for something more specific "
    "than a rule (a length limit, a particular focus), the task wins.\n\n"
    "# THE PLATFORM\n\n"
    f"{PLATFORM_OVERVIEW}\n\n"
    "# RULES\n\n"
    "1. GROUNDING. Answer only from the catalog data below and the conversation. Never invent "
    "cohorts, variables, values, codes or statistics. If the data does not contain the answer, "
    "say so plainly. Never claim to have run an analysis or seen raw data.\n"
    "2. COHORTS WITHOUT VARIABLE METADATA. Only cohorts whose variables are in the catalog can "
    "be explored here: focus searches, comparisons and suggestions on them. Mention cohorts "
    "without variable metadata only when the user asks about them directly.\n"
    "3. INTERPRETING THE QUESTION. First decide what is being asked: a specific variable, a "
    "list of cohorts, an inventory of what is tracked ('medications of X patients' means ALL "
    "medication variables, not one drug class), or data about a subgroup ('patients with X'). "
    "When one reading contains the other (an inventory contains a single flag), answer the "
    "broader one and point out the narrower inside it. When the readings genuinely diverge, "
    "present each, sketch in a line what the data says under each, and end by asking which one "
    "is meant. Never silently pick the narrower or more familiar reading, and never narrow a "
    "condition to one drug class because it is clinically typical. Open with 'Interpreting "
    "this as ...' whenever you chose a reading. The users are analysts: a short clarifying "
    "question is a good outcome, not a failure to answer.\n"
    "4. COHORT NAMES. A word matching a cohort's name or the first part of it ('biostat' -> "
    "BIOSTAT-CHF, 'aachen' -> Aachen-HF, 'time' -> TIME-CHF, 'check' -> CHECK-HF) refers to "
    "that cohort, never to a general topic ('biostat' is not biostatistics). Answer about the "
    "cohort in the context of the conversation; if the match was only partial, end with one "
    "short question confirming the reading.\n"
    "5. PATIENT COUNTS AND SUBGROUPS. For cohorts marked 'summary statistics on record' the "
    "catalog holds each variable's real distribution, with per-category patient counts, so "
    "'how many patients have X' can be answered from the matching category of an X-status "
    "variable. Never claim counts cannot come from the catalog for those cohorts. What the "
    "catalog canNOT do is patient-level filtering ACROSS variables ('X patients below age "
    "50', 'X patients who also take Y'): that needs a Data Clean Room. Variables are recorded "
    "cohort-wide: for a 'patients with X' question, name the variable(s) that identify X and "
    "say that the other variables are not restricted to X patients.\n"
    "6. TERMINOLOGY. Say 'summary statistics', never 'EDA'. Never call variables 'uploaded': a "
    "cohort's variables are either in the catalog or not.\n"
    "7. NO UNASKED NEXT STEPS. Do not end with suggested next steps or offers (creating a DCR, "
    "generating mappings, contacting data owners, 'let me know if you need help') unless the "
    "user asked how to proceed. Stating a factual boundary is fine, e.g. that data without "
    "summary statistics can only be examined in a Data Clean Room after the data owners "
    "upload it there.\n"
    "8. FORMAT. Be concise: short paragraphs and bullet points. Refer to cohorts and variables "
    "by their exact names."
)

# Rules for reading the catalog search results; sent with the results in every
# reply that receives them.
SEARCH_RESULTS_RULES = (
    "How to read these search results:\n"
    "- The search HAS ALREADY BEEN RUN. Answer from these results, not from memory, and never "
    "tell the user to run a search themselves or to confirm with a fresh search.\n"
    "- Each search's 'ALL matching cohorts' line, and the COHORTS MATCHING EVERY CONCEPT line, "
    "are complete. The variable lists under each cohort are CAPPED: never conclude that a "
    "cohort lacks something because a variable is not listed, and never characterize "
    "variables that are not shown (you may cite listed names verbatim).\n"
    "- For a question with several criteria, answer from the COHORTS MATCHING EVERY CONCEPT "
    "line exactly as given - the platform computed it.\n"
    "- A concept's complete cohort set is its main term's ALL-cohorts line PLUS the NEW "
    "cohorts of each of its expansion terms. When a cohort was found only through an "
    "expansion, say so, e.g. 'GISSI-HF (found via the metoprolol expansion)'.\n"
    "- The EQUIVALENT VARIABLES section groups variables mapped to the same standard concept "
    "or OMOP id across cohorts: they capture the same thing. Use it to compare cohorts and to "
    "name each cohort's variable for a concept.\n"
    "- CHART-MARKER: every time you mention a variable that has one, put its exact marker (the "
    "\U0001F4CA[cohort::variable] token, copied verbatim) right after the name; it renders as "
    "a clickable chart. Never invent a marker for a variable that has none."
)

# Answer style of a regular reply. Every question is answered as a detailed
# variant first; the short variant is then condensed from it (SUMMARIZE task),
# and "summary" here is only the fallback when the detailed answer failed.
STYLE_INSTRUCTIONS = {
    "summary": (
        "Style: SHORT SUMMARY. Only the essential answer, in at most 4 sentences or 5 short "
        "bullet points. No preamble, no closing offer, and NO tables (compress tabular material "
        "into a line or two). Brevity never drops cohorts: when the question asks which cohorts "
        "have something, detail the most relevant two or three, then close with one line naming "
        "EVERY remaining match, e.g. 'Also with beta-blocker variables: TIM-HF, GISSI-HF, "
        "CHECK-HF, ... - see the search results for the variables in each.' A question that "
        "pins down an ambiguous request ('did you mean A or B?') is part of the essential answer."
    ),
    "detailed": (
        "Style: DETAILED. A thorough, well-structured answer with specifics (cohort names, "
        "variable names, values, caveats) where the data supports them."
    ),
}


def _get_openai_client() -> Any:
    """Build an OpenAI client pointed at the local LiteLLM proxy."""
    if not settings.chat_enabled:
        raise HTTPException(
            status_code=503,
            detail="AI chat is not configured. Set LITELLM_BASE_URL (and LITELLM_API_KEY) in the .env file.",
        )
    try:
        import openai
    except ImportError:
        raise HTTPException(status_code=500, detail="The 'openai' package is not installed on the server.")
    return openai.OpenAI(
        api_key=settings.litellm_api_key or "sk-no-key",
        base_url=settings.litellm_base_url,
    )


def _clean(value: Any) -> str:
    """Normalise a metadata value to a trimmed string, dropping NA-like values."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("", "na", "n/a", "nan", "none", "null", "-", "--"):
        return ""
    return text


def _has_eda_profile(cohort_id: str) -> bool:
    """Whether EDA / variable profiling output exists for this cohort (mtime-cached)."""
    try:
        from src.nocode import _load_eda

        return bool(_load_eda(str(cohort_id)))
    except Exception:
        return False


# Cohort metadata shown to the model. The long free-text fields (objective,
# outcomes, criteria) are always kept; when the block does not fit the budget
# they are shortened step by step (see build_context), never dropped.
_COHORT_SHORT_FIELDS = [
    ("Study type", "study_type"),
    ("Study design", "study_design"),
    ("Participants", "study_participants"),
    ("Population", "study_population"),
    ("Morbidity", "morbidity"),
    ("Location", "population_location"),
]
_COHORT_LONG_FIELDS = [
    ("Objective", "study_objective"),
    ("Primary outcome", "primary_outcome_spec"),
    ("Secondary outcome", "secondary_outcome_spec"),
    ("Interventions", "interventions"),
]
_INCLUSION_FIELDS = [
    ("sex", "sex_inclusion"),
    ("health status", "health_status_inclusion"),
    ("clinically relevant exposure", "clinically_relevant_exposure_inclusion"),
    ("age group", "age_group_inclusion"),
    ("BMI range", "bmi_range_inclusion"),
    ("ethnicity", "ethnicity_inclusion"),
    ("family status", "family_status_inclusion"),
    ("hospital patient", "hospital_patient_inclusion"),
    ("use of medication", "use_of_medication_inclusion"),
]
_EXCLUSION_FIELDS = [
    ("health status", "health_status_exclusion"),
    ("BMI range", "bmi_range_exclusion"),
    ("limited life expectancy", "limited_life_expectancy_exclusion"),
    ("need for surgery", "need_for_surgery_exclusion"),
    ("surgical procedure history", "surgical_procedure_history_exclusion"),
    ("clinically relevant exposure", "clinically_relevant_exposure_exclusion"),
]
# Length caps tried for the long fields, loosest first (None = full text).
_LONG_FIELD_CAPS = (None, 600, 300, 150, 80)
# Variable listings for SELECTED cohorts, richest first; used only as far as
# the budget left after the metadata allows. Without a selection no variables
# are listed: the search results bring in the ones matching the question.
_VARIABLE_DETAIL_LEVELS = ("labels", "names")


def _cut(text: str, cap: Optional[int]) -> str:
    return text if cap is None or len(text) <= cap else text[:cap].rstrip() + "…"


def _cohort_header(cohort: Any, cap: Optional[int] = None) -> list[str]:
    """The cohort's metadata as bullet lines; long fields cut at `cap` chars."""
    variables = getattr(cohort, "variables", {}) or {}
    if not variables:
        # Cohorts without variable metadata cannot be explored here: one line.
        bits = [b for b in (_clean(getattr(cohort, "study_type", "")),
                            _cut(_clean(getattr(cohort, "study_population", "")), 120)) if b]
        return [f"- {cohort.cohort_id}" + (f" ({'; '.join(bits)})" if bits else "")
                + " — no variable metadata in the catalog"]
    lines = [f"### Cohort: {cohort.cohort_id}"]
    for label, attr in _COHORT_SHORT_FIELDS:
        value = _clean(getattr(cohort, attr, ""))
        if value:
            lines.append(f"- {label}: {_cut(value, 150)}")
    for label, attr in _COHORT_LONG_FIELDS:
        value = _clean(getattr(cohort, attr, ""))
        if value:
            lines.append(f"- {label}: {_cut(value, cap)}")
    for title, fields in (("Inclusion criteria", _INCLUSION_FIELDS), ("Exclusion criteria", _EXCLUSION_FIELDS)):
        crit = [f"{label}: {_cut(_clean(getattr(cohort, attr, '')), cap)}"
                for label, attr in fields if _clean(getattr(cohort, attr, ""))]
        if crit:
            lines.append(f"- {title}: " + "; ".join(crit))
    male, female = getattr(cohort, "male_percentage", None), getattr(cohort, "female_percentage", None)
    if male is not None or female is not None:
        lines.append(f"- Sex: {male if male is not None else '?'}% male, {female if female is not None else '?'}% female")
    ages = getattr(cohort, "age_distribution", None) or {}
    if ages:
        lines.append("- Age distribution: " + ", ".join(f"{k}: {v}%" for k, v in ages.items()))
    lines.append(f"- Variable count: {len(variables)}")
    stats = "on record (variable distributions available)" if _has_eda_profile(cohort.cohort_id) else "not available"
    if getattr(cohort, "has_longitudinal", False):
        stats += "; longitudinal (per-visit) statistics also on record"
    lines.append(f"- Summary statistics: {stats}")
    return lines


def _variable_lines(cohort: Any, level: str) -> list[str]:
    """The cohort's variables: 'name — label' or names only."""
    variables = list((getattr(cohort, "variables", {}) or {}).values())
    if not variables:
        return []
    if level == "names":
        names = [_clean(getattr(v, "var_name", "")) for v in variables]
        return ["- Variable names: " + ", ".join(n for n in names if n)]
    items = []
    for v in variables:
        name = _clean(getattr(v, "var_name", "")) or "?"
        label = _clean(getattr(v, "var_label", "")) or _clean(getattr(v, "concept_name", ""))
        items.append(f"{name} — {label[:100]}" if label and label.lower() != name.lower() else name)
    return ["- Variables (name — label): " + "; ".join(items)]


def build_context(cohort_ids: list[str], focus: Optional[str] = None, info: Optional[dict] = None) -> str:
    """Assemble the cohort block of the chat context.

    Every cohort (the selected ones, or the whole catalog) gets its metadata:
    objective, outcomes, inclusion/exclusion criteria and a few short facts,
    plus its variable count. Long fields are shortened step by step until the
    block fits the budget; cohorts are never dropped. Selected cohorts
    additionally list their variables (name + label, or names only) when the
    remaining budget allows. Otherwise variables come in through the catalog
    search results and the related variables, both chosen by the question.
    When `info` is given it receives what was included (for the progress UI).
    """
    from src.chat_retrieval import budget_chars
    from src.cohort_cache import get_cohorts_from_cache

    all_cohorts = get_cohorts_from_cache("")
    selected = [cid for cid in cohort_ids if cid in all_cohorts]
    if selected:
        intro = (
            f"The user is focusing on {len(selected)} cohort(s): {', '.join(selected)}. "
            "Base your answer MAINLY on these cohorts and their variables; mention other "
            "cohorts only when directly relevant to the question."
        )
        cohorts = [all_cohorts[cid] for cid in selected]
    else:
        with_vars = [c for c in all_cohorts.values() if getattr(c, "variables", None)]
        n_profiled = sum(1 for c in all_cohorts.values() if _has_eda_profile(c.cohort_id))
        intro = (
            f"No specific cohort is selected. The catalog holds {len(all_cohorts)} cohorts "
            f"({len(with_vars)} with variable metadata; summary statistics on record for "
            f"{n_profiled}). All of them follow; variables are not listed here - the catalog "
            "search results and the related variables bring in the ones matching the question."
        )
        # Cohorts with variables first: they are the ones that can be explored.
        cohorts = sorted(all_cohorts.values(), key=lambda c: not getattr(c, "variables", None))
    if _clean(focus):
        intro += f"\nThe user is particularly interested in: {_clean(focus)}"

    budget = budget_chars(COHORT_BLOCK_BUDGET_SHARE)
    # 1. Metadata of every cohort, long fields shortened until it fits.
    headers: list[str] = []
    for cap in _LONG_FIELD_CAPS:
        headers = ["\n".join(_cohort_header(c, cap)) for c in cohorts]
        if len("\n\n".join(headers)) <= budget:
            break
    body = "\n\n".join(headers)
    # 2. Selected cohorts: their variables at the richest level that still fits.
    level = "none"
    if selected:
        for candidate in _VARIABLE_DETAIL_LEVELS:
            blocks = [h + ("\n" + "\n".join(v) if v else "")
                      for h, v in zip(headers, (_variable_lines(c, candidate) for c in cohorts))]
            if len("\n\n".join(blocks)) <= budget:
                body, level = "\n\n".join(blocks), candidate
                break
        if level == "none":
            intro += ("\n(The selected cohorts' variables are too many to list here; the catalog "
                      "search results and the related variables bring in the ones matching the question.)")
    if len(body) > budget:
        body = body[:budget] + "\n(cohort metadata truncated for length)"
    if info is not None:
        info["cohorts"] = len(cohorts)
        info["variables"] = sum(len(getattr(c, "variables", {}) or {}) for c in cohorts)
        info["detail"] = level
    return f"## Cohorts\n\n{intro}\n\n{body}"


# ---- Cross-cohort mapping files ----------------------------------------------
#
# Cache status of the mapping files generated from the mapping page
# (CohortVarLinker), used by /api/chat/mapping-status and the no-code tools.
# Mapping files are not part of the chat context for now: cross-cohort
# correspondences come from the variables sharing a standard concept (see
# chat_retrieval.equivalents_section).


def _linker_output_dir() -> str:
    """The CohortVarLinker mapping-output directory (cheap import: config only)."""
    import sys as _sys

    _sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../CohortVarLinker")))
    from CohortVarLinker.src.config import settings as linker_settings

    return linker_settings.output_dir


def _find_cached_mapping(source: str, target: str, output_dir: str) -> Optional[str]:
    """Most recent cached CSV for source→target (same matching rule as
    CohortVarLinker.main.find_cached_csv, re-implemented here to avoid
    importing that heavy module for a cache probe)."""
    import glob as _glob

    s, t = source.lower(), target.lower()
    matches = _glob.glob(os.path.join(output_dir, f"{s}_{t}.csv"))
    matches += _glob.glob(os.path.join(output_dir, f"{s}_{t}_*.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def mapping_pair_status(cohort_ids: list[str]) -> list[dict[str, Any]]:
    """Cache status for every unordered pair of the selected cohorts.

    A pair counts as cached if a mapping exists in either direction (mapping
    files are directional; the most recent direction wins for display).
    """
    pairs: list[dict[str, Any]] = []
    ids = [c for c in cohort_ids if _clean(c)]
    if len(ids) < 2:
        return pairs
    try:
        output_dir = _linker_output_dir()
    except Exception as exc:
        logger.warning("Mapping output dir unavailable: %s", exc)
        return pairs
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            cand = [(a, b, _find_cached_mapping(a, b, output_dir)),
                    (b, a, _find_cached_mapping(b, a, output_dir))]
            cand = [(s, t, p) for s, t, p in cand if p]
            if cand:
                src, tgt, path = max(cand, key=lambda c: os.path.getmtime(c[2]))
                pairs.append({
                    "source": src,
                    "target": tgt,
                    "cached": True,
                    "filename": os.path.basename(path),
                    "generated_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
                })
            else:
                pairs.append({"source": a, "target": b, "cached": False, "filename": None})
    return pairs


def _normalize_messages(raw: Any) -> list[dict[str, str]]:
    """Validate and coerce the incoming message list to OpenAI's format."""
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")
    allowed = {"user", "assistant", "system"}
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if role not in allowed or not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages provided.")
    return messages


# ---- Task of one reply --------------------------------------------------------
#
# Every request carries exactly one task, chosen from the request body: the
# regular answer (detailed, or the summary fallback), the summary condensed
# from the detailed answer, the clarifying reply for an ambiguous question, or
# the summary-statistics follow-up.

ANSWER_WITH_SEARCH_TASK = (
    "- Give each named cohort's match count, and say which of the cohorts have summary "
    "statistics on record.\n"
    "- Name at most 10-15 variables per cohort and ALWAYS state the full count (e.g. "
    "\"TIME-CHF has 57 matching variables; here are 12\"), so the user knows there are more.\n"
    "- Patient counts are NOT part of this reply. When the question asks for counts or values, "
    "a separate follow-up grounded in the summary statistics is generated right after this "
    "answer whenever those statistics can answer it. At most add ONE line saying the summary "
    "statistics are being checked and a follow-up with the concrete numbers may appear below. "
    "Do not build count tables, and do not send the user to charts or visit pages for counts."
)

SUMMARIZE_TASK = (
    "Write the SHORT SUMMARY variant of the detailed answer in the catalog data above. The "
    "user can already see that detailed answer; condense IT - do not answer afresh.\n"
    "- HARD LIMITS: at most 4 sentences or 5 short bullet points, roughly a QUARTER of the "
    "detailed answer's length or less; a summary nearly as long as the original is a failure.\n"
    "- NO tables (turn a table into a line or a couple of bullets), no headings, no preamble, "
    "no closing offer.\n"
    "- Stay perfectly consistent with the detailed answer: every cohort it names must still "
    "appear (one closing line naming the rest is enough); keep the same numbers, key caveats "
    "and \U0001F4CA chart markers for whatever you keep; end with the same clarifying question "
    "if the detailed answer ends with one.\n"
    "- Add NO claims that are not in the detailed answer."
)

EDA_FOLLOWUP_TASK = (
    "Add ONE short follow-up to the answer already given, grounded ONLY in the variable summary "
    "statistics in the catalog data above. Do NOT repeat the main answer and do NOT re-list "
    "cohorts.\n"
    "- Read each variable's 'values:' line and take the category that answers the question: "
    "for 'how many patients with X', the number is the count of the matching category (the "
    "'yes' / '1' / 'type 1' value), NOT n. n only counts records with any value (including "
    "'no'); presenting n as a patient count is WRONG. Spell the extraction out, e.g. "
    "\"CHECK-HF, Comorbiditeit_DM1: 'yes' 812 of 10,802 recorded (7.5%)\".\n"
    "- Give BOTH the count and the percentage for every category you report, even when only "
    "one was asked for.\n"
    "- Keep the '~' on approximate counts. When any reported count carries '~', add a single "
    "line saying those counts are approximate because they are calculated from the stored "
    "percentage statistics. That is the only standing note.\n"
    "- A variable whose statistics say per-category counts are NOT available cannot give a "
    "patient count: say so for that cohort rather than substituting n.\n"
    "- Name the variable and cohort for every number, and copy the variable's \U0001F4CA chart "
    "marker verbatim.\n"
    "- Where several readings are possible (e.g. a diabetes-type variable vs a cause-of-death "
    "variable), give the number under each and say which variable answers which reading.\n"
    "- No boilerplate: do not explain what percentages mean, do not say the numbers are per "
    "cohort / not pooled, and mention missing data only when the missing share could change "
    "the reading (above about 10%).\n"
    "- End with a short COVERAGE line using the figures at the top of the statistics block: "
    "this answer draws on the K cohorts with summary statistics on record, of which <however "
    "many appear above> had a relevant variable. For the cohorts without summary statistics, "
    "examining their data requires a Data Clean Room into which the data owners upload their "
    "data (a fact of availability, not advice or an offer).\n"
    "- If the statistics do not settle the question, say which variable comes closest and "
    "what is missing."
)


def _matching_cohorts(runs: Any) -> list[str]:
    """Every cohort the searches matched, most matches first."""
    best: dict[str, int] = {}
    for run in runs if isinstance(runs, list) else []:
        for c in (run.get("cohorts") or []) if isinstance(run, dict) else []:
            cid = str(c.get("cohort_id") or "")
            if cid:
                best[cid] = max(best.get(cid, 0), int(c.get("matches") or 0))
    return sorted(best, key=lambda cid: -best[cid])


def _completeness_block(cohorts: list[str], intersection: Any) -> str:
    """The must-name list for an answer based on search results. Models tend
    to drop cohorts from long lists; spelling the names out, with a check at
    the end, works far better than a general 'name every cohort' rule."""
    lines = [
        f"COMPLETENESS - THE MOST IMPORTANT REQUIREMENT OF THIS REPLY. The search matched these "
        f"{len(cohorts)} cohorts: {', '.join(cohorts)}.",
        f"Your answer MUST name every one of these {len(cohorts)} cohorts by its exact name. None "
        "may be dropped, and none may be folded into 'and others', 'etc.' or 'several more'. "
        "Describe the most relevant ones in detail if you like, then name ALL the rest in one "
        "closing line (e.g. 'Also matching: A (4), B (2), C (1)').",
    ]
    if isinstance(intersection, list):
        both = [str(r.get("cohort_id")) for r in intersection if isinstance(r, dict) and r.get("cohort_id")]
        if both:
            lines.append(f"Of these, the cohorts matching EVERY concept are: {', '.join(both)}. For a "
                         "question asking for all criteria at once, lead with them - but still name "
                         "the others, saying which criteria each one covers.")
        else:
            lines.append("No cohort matches every concept at once: say so, then name the cohorts "
                         "matching each concept.")
    lines.append(f"Before you finish, go through the list of {len(cohorts)} names above and check "
                 "that each one appears in your answer; add any that are missing.")
    return "\n".join(lines)


def _clarify_task(readings: list[str]) -> str:
    return (
        "The question is ambiguous between these readings: " + "; ".join(readings) + ".\n"
        "Do NOT give a full answer and do NOT enumerate whole result lists. For each reading, "
        "give one or two lines of preliminary findings from the search results (roughly how "
        "many cohorts / variables match, the two or three key cohorts), then end with ONE "
        "short question asking which reading is meant. Nothing else."
    )


def _answer_task(style: Any, has_search: bool, completeness: str) -> str:
    parts = ["Answer the user's latest message."]
    if completeness:
        parts.append(completeness)
    if has_search:
        parts.append(ANSWER_WITH_SEARCH_TASK)
    if isinstance(style, str) and style in STYLE_INSTRUCTIONS:
        parts.append(STYLE_INSTRUCTIONS[style])
    return "\n\n".join(parts)


def _assemble_payload(body: dict[str, Any]) -> tuple[list[dict[str, str]], str, float, dict[str, Any]]:
    """Turn the request body into (messages, model, temperature, info); `info`
    summarizes what went into the context, for the chat's progress display.

    The model receives ONE system message (several chat templates, Qwen's among
    them, reject any system message that is not the first) in three parts:
      1. RULES - SYSTEM_PROMPT, or the client's system_prompt override;
      2. CATALOG DATA - what this reply needs: cohorts, search results,
         related variables, cross-cohort equivalents, and for the follow-up
         turns the detailed answer or the summary statistics;
      3. YOUR TASK FOR THIS REPLY - exactly one task (see above).
    followed by the conversation's user/assistant turns.

    Optional overrides (used by the Glass Box / Atlas layouts, which construct
    their payloads client-side for full transparency):
      - system_prompt: replaces the default SYSTEM_PROMPT.
      - context: replaces the server-built catalog data entirely, so what the
        user previews in the UI is exactly what the model receives.
    """
    messages = _normalize_messages(body.get("messages"))
    cohort_ids = body.get("cohort_ids") or []
    if not isinstance(cohort_ids, list):
        cohort_ids = []
    cohort_ids = [str(c) for c in cohort_ids]
    focus = body.get("focus")
    model = _clean(body.get("model")) or settings.litellm_model
    try:
        temperature = float(body.get("temperature", 0.3))
    except (TypeError, ValueError):
        temperature = 0.3

    system_prompt = body.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        system_prompt = system_prompt.strip()[:MAX_SYSTEM_PROMPT_CHARS]
    else:
        system_prompt = SYSTEM_PROMPT

    # Which task this reply carries.
    summarize_text = body.get("summarize_text")
    summarize_text = summarize_text.strip() if isinstance(summarize_text, str) else ""
    eda_ctx = body.get("eda_context")
    eda_ctx = eda_ctx.strip() if isinstance(eda_ctx, str) else ""
    clarify = body.get("clarify_interpretations")
    if summarize_text:
        mode = "summarize"
    elif eda_ctx:
        mode = "eda"
    elif isinstance(clarify, list) and len(clarify) >= 2:
        mode = "clarify"
    else:
        mode = "answer"

    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    data: list[str] = []
    has_search = False
    info: dict[str, Any] = {"mode": mode}
    # Search results from the planning round (see /api/chat/plan-search): the
    # client runs the plan once per user turn and passes the structured results
    # into every reply of the turn, so the model and the user's search panel see
    # exactly the same thing. The summary reply gets them only for the list of
    # cohorts it must keep.
    search_results = body.get("search_results")
    runs_in, concepts_in, intersection_in = None, None, None
    if isinstance(search_results, dict):
        runs_in = search_results.get("runs")
        concepts_in = search_results.get("concepts")
        intersection_in = search_results.get("intersection")
    elif isinstance(search_results, list):
        runs_in = search_results
    matching = _matching_cohorts(runs_in)
    if matching:
        info["search_cohorts"] = len(matching)
    context_override = body.get("context")
    if isinstance(context_override, str) and context_override.strip():
        data.append(context_override.strip()[:MAX_CONTEXT_CHARS])
    elif mode != "summarize":
        # The summary condenses the detailed answer and the statistics
        # follow-up reads only the statistics block, so neither needs the
        # (large) cohort block.
        if mode != "eda":
            data.append(build_context(cohort_ids, focus, info))
        try:
            note = cohort_name_note(last_user) if last_user else ""
            if note:
                data.append(f"## Cohort name match\n\n{note}")
        except Exception as exc:
            logger.warning("Cohort-name note failed: %s", exc)
        if isinstance(runs_in, list) and runs_in:
            try:
                from src.chat_retrieval import format_search_context

                section = format_search_context(runs_in, concepts=concepts_in, intersection=intersection_in)
                if section:
                    data.append(f"## Catalog search results\n\n{section}\n\n{SEARCH_RESULTS_RULES}")
                    has_search = True
            except Exception as exc:
                logger.warning("Search-results formatting failed: %s", exc)
        # Related variables: labels of the variables best matching the
        # question's words (and the search terms), beyond the exact matches
        # already in the search results. See chat_retrieval.py.
        if mode in ("answer", "clarify") and last_user:
            try:
                from src.chat_retrieval import related_variables_section
                from src.cohort_cache import get_cohorts_from_cache

                terms = [str(r.get("term") or "") for r in runs_in or [] if isinstance(r, dict)]
                shown = {(str(c.get("cohort_id")), str(v.get("var_name") or "").strip().lower())
                         for r in runs_in or [] if isinstance(r, dict)
                         for c in r.get("cohorts") or [] for v in c.get("variables") or []}
                all_cohorts = get_cohorts_from_cache("")
                section, related_pairs = related_variables_section(
                    last_user, terms, all_cohorts, restrict_to=cohort_ids, exclude=shown)
                if section:
                    data.append(f"## Related variables\n\n{section}")
                    info["related_variables"] = len(related_pairs)
                # Cross-cohort equivalents of everything found above.
                from src.chat_retrieval import equivalents_section

                eq_section, n_clusters = equivalents_section(list(shown) + related_pairs, all_cohorts)
                if eq_section:
                    data.append(f"## Equivalent variables across cohorts\n\n{eq_section}")
                    info["equivalent_clusters"] = n_clusters
            except Exception as exc:
                logger.warning("Related-variable retrieval failed: %s", exc)

    if mode == "summarize":
        data.append("## Detailed answer to condense\n\n" + summarize_text[:MAX_SUMMARIZE_CHARS])
        task = SUMMARIZE_TASK
        if matching:
            task += (f"\n\nCOMPLETENESS - MOST IMPORTANT: the search behind the detailed answer "
                     f"matched these {len(matching)} cohorts: {', '.join(matching)}. Every one of them "
                     "that the detailed answer names MUST appear in your summary by its exact name; "
                     "a closing line 'Also matching: ...' naming the rest is enough. Never write "
                     "'and others' or 'etc.' instead of names. Check the list before you finish.")
    elif mode == "eda":
        data.append("## Variable summary statistics\n\n" + eda_ctx[:MAX_EDA_CONTEXT_CHARS])
        task = EDA_FOLLOWUP_TASK
    elif mode == "clarify":
        task = _clarify_task([str(c)[:200] for c in clarify[:4]])
    else:
        completeness = _completeness_block(matching, intersection_in) if has_search and matching else ""
        task = _answer_task(body.get("style"), has_search, completeness)

    # System messages sent by the client (none from the main chat today) become
    # part of the task rather than separate messages.
    extra = [m["content"] for m in messages if m["role"] == "system"]
    if extra:
        task += "\n\nAdditional instructions:\n" + "\n\n".join(extra)

    system = "\n\n".join([
        system_prompt,
        "# CATALOG DATA\n\n" + ("\n\n".join(data) if data else "(none for this reply)"),
        "# YOUR TASK FOR THIS REPLY\n\n" + task,
    ])
    full_messages = [{"role": "system", "content": system}] + [m for m in messages if m["role"] != "system"]
    from src.chat_retrieval import CHARS_PER_TOKEN

    info["approx_tokens"] = sum(len(m["content"]) for m in full_messages) // CHARS_PER_TOKEN
    return full_messages, model, temperature, info


@router.get("/api/chat/config")
def chat_config(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Report whether chat is enabled and which model is used by default."""
    return {
        "enabled": settings.chat_enabled,
        "model": settings.litellm_model,
    }


@router.post("/api/chat")
def chat(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Non-streaming chat completion grounded in the selected cohort context."""
    client = _get_openai_client()
    full_messages, model, temperature, _info = _assemble_payload(body)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        return {"content": content, "model": model}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Chat completion failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}")


def _cohort_first_segment(cohort_id: str) -> str:
    """First name segment of a cohort id, lowercased: 'BIOSTAT-CHF' -> 'biostat',
    'RotterdamStudyCohort2' -> 'rotterdam', 'TIME-CHF' -> 'time'."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(cohort_id))
    parts = re.split(r"[^A-Za-z0-9]+|\s+", text)
    return parts[0].lower() if parts and parts[0] else ""


def cohort_name_note(message: str) -> str:
    """A context note when the message mentions a cohort by name or by the
    first part of its name: the model must assume the cohort is meant (and
    hedge when the match was only partial)."""
    try:
        from src.cohort_cache import get_cohorts_from_cache

        all_ids = list(get_cohorts_from_cache("").keys())
    except Exception:
        return ""
    words = set(re.split(r"[^a-z0-9]+", str(message).lower()))
    norm_msg = re.sub(r"[^a-z0-9]+", "", str(message).lower())
    exact, partial = [], {}
    for cid in all_ids:
        norm_id = re.sub(r"[^a-z0-9]+", "", cid.lower())
        if norm_id and norm_id in norm_msg:
            exact.append(cid)
            continue
        seg = _cohort_first_segment(cid)
        if len(seg) >= 4 and seg in words:
            partial.setdefault(seg, []).append(cid)
    bits = []
    if exact:
        bits.append(f"the user explicitly names the cohort(s) {', '.join(exact)}")
    for seg, cids in partial.items():
        bits.append(f"'{seg}' matches the cohort name(s) {', '.join(cids)} — assume the cohort is meant, "
                    "answer about it, and end with one short question confirming that reading")
    if not bits:
        return ""
    return "COHORT NAME MATCH: " + "; ".join(bits) + ". Do not treat these words as general topics."


# Deterministic backstop: questions shaped like "which studies/cohorts have X"
# must always search, even if the planning model returns nothing.
_IDENTIFICATION_RE = re.compile(
    r"\b(which|what|find|list|identify|show|any|are there)\b.{0,80}\b(stud(?:y|ies)|cohorts?|registr(?:y|ies)|datasets?)\b"
    r"|\b(stud(?:y|ies)|cohorts?)\b.{0,50}\b(have|has|with|contain|include|measure|collect|record)\b",
    re.I | re.S,
)


SEARCH_PLANNER_PROMPT = (
    "You plan catalog searches for iCARE-AI, an assistant over a catalog of cardiovascular "
    "research cohorts and their variables. Given the user's question (and the conversation so "
    "far), decide whether answering involves IDENTIFYING COHORTS OR VARIABLES that meet some "
    "criteria (a topic, a measurement, a condition, a kind of data). If it does, the built-in "
    "search tool MUST be used. Questions like 'which studies have "
    "X', 'which cohorts measure Y', 'is Z available anywhere' ALWAYS need searches.\n"
    "GROUP YOUR TERMS BY CONCEPT: one concept per distinct criterion in the question, each with "
    "1 to 4 short terms (1-3 words) covering its RELATED TERMINOLOGY: synonyms, common "
    "abbreviations and closely related measurements (kidney function: creatinine, eGFR, urea; "
    "ejection fraction: LVEF, ejection fraction). Expand medication CLASSES into the class name, "
    "its abbreviation and typical member drugs. Beta blockers matter a lot here: for them ALWAYS "
    "propose both 'beta blocker' AND 'beta blocking' (plus e.g. bisoprolol, metoprolol, "
    "carvedilol). Other examples — RAS: ACE, ARB, sartan, ARNI; MRA: MRA, spironolactone, "
    "eplerenone. Prefer singular word forms ('beta blocker', not 'beta blockers'). "
    "German or other-language variable names are found via concept names, so English terms "
    "suffice. A question with several criteria ('beta blockers AND BNP AND an outcome') gets one "
    "concept per criterion; the platform then computes which cohorts match every concept.\n"
    "AMBIGUOUS QUESTIONS: when the question supports different readings - a single flag vs an "
    "inventory of a whole category ('medications of X patients'), a condition vs the drug class "
    "typical for it, one cohort vs all - add concepts covering EVERY plausible reading, so the "
    "answer can show each interpretation with real results and ask the user which they meant, "
    "AND declare the readings in \"interpretations\". Example: 'I want a variable to check the "
    "medication of patients with atrial fibrillation' is ambiguous - (a) variables recording the "
    "medication given specifically FOR atrial fibrillation (anticoagulants, antiarrhythmics), vs "
    "(b) general medication variables to cross-reference against an atrial-fibrillation status "
    "variable; declare both and search both. Litmus test: if a careful answer would have to open "
    "with 'Interpreting this as ...', the question IS ambiguous - declare the readings instead of "
    "silently picking one. Declaring interpretations is ENCOURAGED: these users are analysts who "
    "find a clarification round genuinely useful, so when torn between one reading and several, "
    "declare several. Do not manufacture ambiguity for a clearly phrased question. "
    "Treat a follow-up as a NEW question unless it clearly refines the previous one; do not "
    "just re-run the previous turn's terms.\n"
    "COHORT NAMES: the catalog's cohort names are listed below. A word matching a cohort name or "
    "its first part (biostat -> BIOSTAT-CHF, aachen -> Aachen-HF, time -> TIME-CHF) refers to THE "
    "COHORT, never to a topic — NEVER turn it into search terms (no 'biostatistics'). For a "
    "follow-up like 'what about <cohort>?', re-propose the search terms of the criteria already "
    "under discussion in the conversation, so the results cover that cohort too.\n"
    "If the question needs no catalog search (small talk, platform how-to, questions about "
    "already-shown results), return an empty list.\n"
    'Return STRICT JSON only: '
    '{"concepts": [{"name": "<short label>", "terms": ["term", ...]}, ...], '
    '"interpretations": ["<reading 1>", "<reading 2>", ...]} '
    'where "interpretations" is present ONLY when the question is genuinely ambiguous between '
    'readings (then list each reading as a short phrase, and make sure the concepts cover all '
    'of them); omit it or use [] for a clear question. Empty concepts when no search is needed.'
)


@router.post("/api/chat/plan-search")
def plan_search(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Planning round for the chat's search tool: the model proposes search
    terms for the user's question; the terms are run through the catalog search
    (see chat_retrieval.run_chat_searches) and the structured results are
    returned — the client shows them in the search panel and passes them back
    into the answer requests so both see the same thing."""
    from src.chat_retrieval import run_chat_searches
    from src.cohort_cache import get_cohorts_from_cache

    question = _clean(body.get("question"))
    if not question:
        return {"needed": False, "terms": [], "searches": []}
    cohort_ids = [str(c) for c in (body.get("cohort_ids") or []) if c]
    # The first turn of a conversation has no history; _normalize_messages
    # treats an empty list as an error (it guards the chat endpoints), so it
    # must not run here on empty input.
    try:
        history = _normalize_messages(body.get("history"))[-8:] if body.get("history") else []
    except HTTPException:
        history = []
    convo = "\n".join(f"{m['role']}: {m['content'][:1500]}" for m in history)
    try:
        from src.cohort_cache import get_cohorts_from_cache

        cohort_names = ", ".join(sorted(get_cohorts_from_cache("").keys()))
    except Exception:
        cohort_names = ""
    name_note = cohort_name_note(question)
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[
                {"role": "system", "content": SEARCH_PLANNER_PROMPT + (f"\nCOHORT NAMES IN THE CATALOG: {cohort_names}" if cohort_names else "")},
                {"role": "user", "content": (f"Conversation so far:\n{convo}\n\n" if convo else "")
                                            + (f"{name_note}\n\n" if name_note else "")
                                            + f"Question: {question}"},
            ],
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
        concepts = []
        for c in (parsed.get("concepts") or [])[:5]:
            if isinstance(c, dict):
                c_terms = [str(t).strip() for t in (c.get("terms") or []) if str(t).strip()][:4]
                if c_terms:
                    concepts.append({"name": str(c.get("name") or c_terms[0]).strip()[:60], "terms": c_terms})
        if not concepts:
            # older/simpler planner shape: a flat list, one unnamed concept (no
            # intersection implied)
            flat = [str(t).strip() for t in (parsed.get("searches") or []) if str(t).strip()][:6]
            if flat:
                concepts = [{"name": "", "terms": flat}]
        terms = [t for c in concepts for t in c["terms"]][:10]
        interpretations = [str(i).strip()[:120] for i in (parsed.get("interpretations") or [])
                           if str(i).strip()][:4]
    except Exception as exc:
        logger.warning("Search planning failed: %s", exc)
        concepts, terms, interpretations = [], [], []
    # Backstop: an identification-shaped question always searches, with terms
    # from the question itself if the planner offered none.
    if not terms and _IDENTIFICATION_RE.search(question):
        from src.chat_retrieval import extract_query_terms

        terms = extract_query_terms(question)[:6]
        concepts = [{"name": "", "terms": terms}] if terms else []
        interpretations = []
        logger.info("Search planner returned no terms; identification backstop used: %s", terms)
    if not terms:
        return {"needed": False, "terms": [], "searches": []}
    try:
        runs = run_chat_searches(terms, get_cohorts_from_cache(""), restrict_to=cohort_ids or None)
    except Exception as exc:
        logger.warning("Chat search execution failed: %s", exc)
        return {"needed": False, "terms": terms, "searches": [], "error": str(exc)}
    # Per concept: the union of its terms' matching cohorts (count = max over
    # the terms, so overlapping terms are not double-counted). Across concepts:
    # the INTERSECTION - computed here, so the model never does set logic itself.
    runs_by_term = {r["term"]: r for r in runs}
    concept_summaries = []
    for c in concepts:
        cohort_counts: dict[str, int] = {}
        for t in c["terms"]:
            for coh in (runs_by_term.get(t) or {}).get("cohorts") or []:
                cohort_counts[coh["cohort_id"]] = max(cohort_counts.get(coh["cohort_id"], 0), coh["matches"])
        concept_summaries.append({"name": c["name"], "terms": c["terms"], "cohorts": cohort_counts})
    named = [c for c in concept_summaries if c["cohorts"]]
    intersection = None
    if len(named) >= 2:
        common = set.intersection(*(set(c["cohorts"]) for c in named))
        intersection = sorted(
            ({"cohort_id": cid,
              "per_concept": {(c["name"] or " / ".join(c["terms"][:2])): c["cohorts"][cid] for c in named}}
             for cid in common),
            key=lambda row: -min(row["per_concept"].values()),
        )
    return {"needed": True, "terms": terms, "searches": runs,
            "concepts": concept_summaries, "intersection": intersection,
            "interpretations": interpretations if len(interpretations) >= 2 else [],
            "model": settings.litellm_model}


# ---- EDA follow-up -----------------------------------------------------------
#
# After a search-grounded answer, a second round can turn it into concrete
# numbers: the model picks, from the matched variables that HAVE an EDA profile,
# the few whose distribution actually bears on the question (e.g. a
# diabetes-TYPE variable with per-category patient counts for "how many type 1
# diabetics"), the server loads those profiles, and the client streams one
# extra follow-up bubble grounded in them (see EDA FOLLOW-UP MODE in
# _assemble_payload).

EDA_FOLLOWUP_SELECTION_PROMPT = (
    "You decide whether VARIABLE PROFILES can sharpen a catalog answer into concrete numbers, "
    "for iCARE-AI (an assistant over a catalog of cardiovascular research cohorts). The user's "
    "question was just answered from catalog search results that list matching variables per "
    "cohort. For the variables listed below, an EDA profile is on record: the variable's value "
    "type and its real distribution - per-category patient counts for categorical variables, "
    "n/mean/median/min/max for numeric ones.\n"
    "First judge what the user is actually after (e.g. 'how many type 1 diabetes patients' -> "
    "variables whose CATEGORIES distinguish diabetes types; 'patients who died of X' -> "
    "cause-of-death variables; a lab value's range -> the numeric profile). Then select ONLY "
    "the variables whose profile directly bears on that - the fewer and sharper the better, at "
    "most 8. Prefer variables from different cohorts over near-duplicates within one cohort. "
    "Each candidate is tagged: '[per-category counts available]' means its profile lists the "
    "patient count per value (what a 'how many patients with X' question needs); '[no per-value "
    "breakdown in profile]' means only n and summary statistics exist. For count questions "
    "STRONGLY prefer the candidates with per-category counts. "
    "DEFAULT TO needed=false. Return needed=true ONLY when the user's question itself asks for "
    "numbers the profiles hold: how many patients / what share have X, what values or range a "
    "measurement takes, how something is distributed, or a comparison of such numbers across "
    "cohorts. Questions about WHICH cohorts or variables exist or measure something ('which "
    "cohorts have beta-blocker variables', 'what kidney-function variables does TIME-CHF "
    "have', 'can I study X') are answered by the search results alone: needed=false, even "
    "when profiles exist for the matches. Numbers the user did not ask for are not needed.\n"
    'Return STRICT JSON only: {"needed": true|false, '
    '"focus": "<one short line: what to extract from the profiles>", '
    '"selections": [{"cohort_id": "<exact id>", "var_name": "<exact name>"}, ...]}'
)


def _format_eda_profiles(rows: list[dict]) -> str:
    """The selected variables' EDA profiles as model-ready text. Each row:
    {cohort_id, var_name, var_label, stats} with stats from nocode._load_eda."""
    parts = [
        "VARIABLE SUMMARY STATISTICS - real distributions from the catalog.",
        "READ CAREFULLY: n is the number of records with ANY value for the variable - it is "
        "NEVER the count of patients with a condition. The count of patients with a condition "
        "is the count of the matching category on a 'values:' line (e.g. the 'yes' / 'type 1' "
        "category). Counts marked '~' are derived from recorded percentages and approximate; "
        "an '<na>' category counts the records without a value.",
    ]
    for r in rows:
        s = r.get("stats") or {}
        head = f"### {r['cohort_id']} :: {r['var_name']}"
        if r.get("var_label"):
            head += f" — {r['var_label']}"
        bits = []
        if s.get("type"):
            bits.append(str(s["type"]))
        if s.get("n") is not None:
            bits.append(f"n={int(s['n'])}")
        if s.get("missing_pct") is not None:
            bits.append(f"missing {s['missing_pct']}%")
        if bits:
            head += " [" + ", ".join(bits) + "]"
        head += f" \U0001F4CA[{r['cohort_id']}::{r['var_name']}]"
        parts.append(head)
        dist = s.get("distribution") or []
        if dist:
            vals = "; ".join(
                f"'{d.get('label') or d.get('value')}': "
                + (("~" if d.get("approx") else "") + str(int(d["count"])) if d.get("count") is not None else "?")
                + (f" ({d['pct']}%)" if d.get("pct") is not None else "")
                for d in dist
            )
            parts.append(f"   values: {vals}")
        elif str(s.get("type") or "").lower().startswith("categor"):
            parts.append("   per-category counts NOT available for this variable - "
                         "the breakdown by value is unknown; do NOT infer it from n.")
        nums = [(k, s.get(k)) for k in ("mean", "std", "median", "min", "max", "q1", "q3")]
        nums = [(k, v) for k, v in nums if v is not None]
        if nums:
            parts.append("   " + ", ".join(f"{k}={v}" for k, v in nums))
        if s.get("n_unique") is not None:
            parts.append(f"   distinct values: {int(s['n_unique'])}")
    return "\n".join(parts)


@router.post("/api/chat/eda-followup")
def eda_followup(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Selection round for the EDA follow-up bubble: given the question and the
    search results, the model picks the profiled variables whose distributions
    answer the question; their profiles are returned as a context block the
    client passes back into one extra streamed answer."""
    from src.nocode import _load_eda

    question = _clean(body.get("question"))
    runs = ((body.get("search_results") or {}).get("runs")
            if isinstance(body.get("search_results"), dict) else None) or body.get("searches") or []
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in runs:
        for c in (r.get("cohorts") or []) if isinstance(r, dict) else []:
            for v in c.get("variables") or []:
                if not v.get("has_eda"):
                    continue
                key = (str(c.get("cohort_id")), str(v.get("var_name") or "").strip().lower())
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                candidates.append({"cohort_id": str(c.get("cohort_id")),
                                   "var_name": str(v.get("var_name")),
                                   "var_label": _clean(v.get("var_label") or ""),
                                   "term": str(r.get("term") or "")})
                if len(candidates) >= 80:
                    break
    if not question or not candidates:
        return {"needed": False}

    # One EDA load per cohort (mtime-cached), reused for the profile block below.
    eda_by_cohort: dict[str, dict] = {}

    def _stats_for(cid: str, vname: str):
        if cid not in eda_by_cohort:
            try:
                eda_by_cohort[cid] = _load_eda(cid) or {}
            except Exception:
                eda_by_cohort[cid] = {}
        return eda_by_cohort[cid].get(vname.strip().lower())

    for c in candidates:
        s = _stats_for(c["cohort_id"], c["var_name"]) or {}
        # Only v2 EDA outputs carry per-value counts; the tag lets the model
        # prefer variables it can actually count patients from.
        c["has_counts"] = bool(s.get("distribution"))

    listing = "\n".join(
        f"- {c['cohort_id']} :: {c['var_name']}"
        + (f" — {c['var_label']}" if c["var_label"] else "")
        + (f" (matched '{c['term']}')" if c["term"] else "")
        + (" [per-category counts available]" if c["has_counts"] else " [no per-value breakdown in profile]")
        for c in candidates
    )
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[
                {"role": "system", "content": EDA_FOLLOWUP_SELECTION_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nProfiled variables from the search results:\n{listing}"},
            ],
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
    except Exception as exc:
        logger.warning("EDA follow-up selection failed: %s", exc)
        return {"needed": False}
    if not parsed.get("needed"):
        return {"needed": False}
    by_key = {(c["cohort_id"], c["var_name"].strip().lower()): c for c in candidates}
    rows: list[dict] = []
    for sel in (parsed.get("selections") or [])[:8]:
        if not isinstance(sel, dict):
            continue
        cand = by_key.get((str(sel.get("cohort_id")), str(sel.get("var_name") or "").strip().lower()))
        if not cand:
            continue
        stats = _stats_for(cand["cohort_id"], cand["var_name"])
        if stats:
            rows.append({**cand, "stats": stats})
    if not rows:
        return {"needed": False}
    focus = _clean(parsed.get("focus") or "")[:200]
    context_block = _format_eda_profiles(rows)
    # Coverage disclaimer for "across all cohorts" questions.
    try:
        from src.cohort_cache import get_cohorts_from_cache

        all_c = get_cohorts_from_cache("")
        n_profiled = sum(1 for cid in all_c if _has_eda_profile(cid))
        context_block = (f"(Summary statistics are on record for {n_profiled} of the catalog's "
                         f"{len(all_c)} cohorts.)\n" + context_block)
    except Exception:
        pass
    if focus:
        context_block += f"\nFOCUS: {focus}"
    return {"needed": True, "focus": focus,
            "variables": [{"cohort_id": r["cohort_id"], "var_name": r["var_name"]} for r in rows],
            "context": context_block, "model": settings.litellm_model}


# ---- Conversation starters ---------------------------------------------------
#
# "Conversation starters" are model-generated questions shown on the chat
# landing page and — grouped under thematic keywords — in Guided Exploration.
# Generation is admin-driven from the manager page (/ai/starters): admins can
# generate new starters (optionally steered by a direction/theme prompt),
# regroup them under keywords, and prune the pool. Generated starters are
# APPENDED to a JSON file so the pool grows over time; the keyword grouping is
# rewritten on each pass so it always reflects the full pool.

CONVERSATION_STARTERS_FILE = os.path.join(settings.data_folder, "ai_conversation_starters.json")
STARTER_KEYWORDS_FILE = os.path.join(settings.data_folder, "ai_starter_keywords.json")

# Serializes pool-file writes (generate / delete may run concurrently).
_pool_lock = threading.Lock()

# Static fallback so the UI always has something to show before an admin has
# generated any starters (or when chat is not configured).
FALLBACK_STARTERS = [
    {"text": "Give me an overview of what this cohort catalog contains.", "kind": "basic"},
    {"text": "Which cohorts have the most variables available?", "kind": "basic"},
    {"text": "Which cohorts focus on diabetes or cardiovascular disease?", "kind": "basic"},
    {"text": "Which cohorts could be combined to study blood pressure over time?", "kind": "interesting"},
    {"text": "What variables related to heart failure are measured across multiple cohorts?", "kind": "interesting"},
    {"text": "Suggest a research question that two or more cohorts could answer together.", "kind": "interesting"},
]

STARTER_GENERATION_INSTRUCTIONS = (
    "You generate conversation starters for iCARE-AI, an assistant that helps researchers "
    "explore a catalog of cardiovascular research cohorts. Based ONLY on the catalog "
    "context provided, produce questions a user could ask the assistant.\n\n"
    "IMPORTANT: only reference cohorts that have variable metadata in the catalog — i.e. "
    "those listed with 1 or more variables in the context. Never build a question around a "
    "cohort with 0 variables, as there is no data to explore for it.\n\n"
    "Return STRICT JSON, no markdown fences and no commentary, exactly of the form:\n"
    '{"interesting": ["...", "..."], "basic": ["...", "..."]}\n\n'
    "- \"interesting\": 8 specific, research-oriented questions. Reference actual cohorts "
    "(with variable metadata), domains or variables from the context where possible; favour "
    "cross-cohort angles.\n"
    "- \"basic\": 6 simple orientation questions a first-time user might ask.\n"
    "- Each question must be a single sentence under 140 characters, ending with a question mark."
)


def _load_starter_pool() -> list[dict]:
    try:
        with open(CONVERSATION_STARTERS_FILE) as fh:
            data = json.load(fh)
        starters = data.get("starters", [])
        return [s for s in starters if isinstance(s, dict) and _clean(s.get("text"))]
    except Exception:
        return []


def _write_starter_pool(pool: list[dict]) -> None:
    os.makedirs(os.path.dirname(CONVERSATION_STARTERS_FILE), exist_ok=True)
    with open(CONVERSATION_STARTERS_FILE, "w") as fh:
        json.dump({"starters": pool}, fh, indent=2)


def _append_to_starter_pool(new_starters: list[dict], direction: Optional[str] = None) -> int:
    """Append starters to the pool file, deduplicating on text. Returns count added."""
    with _pool_lock:
        pool = _load_starter_pool()
        seen = {s["text"].strip().lower() for s in pool}
        added = 0
        now = datetime.now().isoformat(timespec="seconds")
        for s in new_starters:
            text = _clean(s.get("text"))
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            entry = {
                "text": text,
                "kind": s.get("kind", "interesting"),
                "generated_at": now,
                "model": settings.litellm_model,
            }
            if direction:
                entry["direction"] = direction
            pool.append(entry)
            added += 1
        if added:
            _write_starter_pool(pool)
    return added


def _parse_generated_starters(content: str) -> list[dict]:
    """Parse the model's JSON (tolerating code fences); fall back to line parsing."""
    text = content.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            starters = []
            for kind in ("interesting", "basic"):
                for item in data.get(kind, []):
                    if isinstance(item, str) and item.strip():
                        starters.append({"text": item.strip()[:200], "kind": kind})
            if starters:
                return starters
        except json.JSONDecodeError:
            pass
    # Fallback: any line that looks like a question.
    starters = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if line.endswith("?") and 10 < len(line) <= 200:
            starters.append({"text": line, "kind": "interesting"})
    return starters


def generate_conversation_starters(direction: Optional[str] = None) -> dict[str, Any]:
    """Ask the model for conversation starters and append them to the pool.

    Admin-driven (see the /api/chat/starters/generate endpoint). An optional
    `direction` steers the generation towards a specific theme. Returns a
    summary dict; after a successful append the keyword grouping is refreshed.
    """
    if not settings.chat_enabled:
        return {"parsed": 0, "added": 0, "error": "AI chat is not configured."}
    direction = _clean(direction) or None
    try:
        import openai

        client = openai.OpenAI(
            api_key=settings.litellm_api_key or "sk-no-key",
            base_url=settings.litellm_base_url,
        )
        instructions = STARTER_GENERATION_INSTRUCTIONS
        if direction:
            instructions += (
                f"\n\nIMPORTANT: generate the questions in this specific direction/theme: \"{direction}\". "
                "Every question, including the basic ones, must relate to it."
            )
        context = build_context([], None)
        response = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Cohort catalog context:\n\n{context}\n\nGenerate the questions now."},
            ],
            temperature=0.9,
        )
        content = response.choices[0].message.content or ""
        starters = _parse_generated_starters(content)
        added = _append_to_starter_pool(starters, direction=direction)
        logger.info(
            "Conversation-starter generation%s: parsed %d, appended %d new to %s",
            f" (direction: {direction})" if direction else "", len(starters), added,
            CONVERSATION_STARTERS_FILE,
        )
    except Exception as exc:
        logger.warning("Conversation-starter generation failed: %s", exc)
        return {"parsed": 0, "added": 0, "error": str(exc)}
    # Refresh the keyword grouping so it reflects the grown pool.
    group_starters_by_keyword()
    return {"parsed": len(starters), "added": added, "direction": direction}


# ---- Keyword grouping of the starter pool ------------------------------------
#
# A second pass over the pool: the model groups the conversation starters under
# short thematic keywords. The Guided Exploration flow shows these keywords as
# the next selection after "Formulate Research Questions".

KEYWORD_GROUPING_INSTRUCTIONS = (
    "You organize conversation starters for iCARE-AI, an assistant for exploring "
    "a catalog of cardiovascular research cohorts. Group the questions below under 6 to 12 "
    "short thematic keywords (1-3 words each, lowercase, e.g. \"heart failure\", "
    "\"medication use\", \"aging\"). A question may appear under multiple keywords; prefer "
    "keywords that group at least 2 questions.\n\n"
    "Return STRICT JSON only, no markdown fences and no commentary, exactly of the form:\n"
    '{"keywords": [{"keyword": "...", "questions": ["...", "..."]}]}'
)


def _load_starter_keywords_file() -> dict:
    try:
        with open(STARTER_KEYWORDS_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _load_starter_keywords() -> list[dict]:
    keywords = _load_starter_keywords_file().get("keywords", [])
    return [
        k for k in keywords
        if isinstance(k, dict) and _clean(k.get("keyword")) and isinstance(k.get("questions"), list)
    ]


def _parse_keyword_groups(content: str) -> list[dict]:
    text = content.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    groups = []
    for entry in data.get("keywords", []):
        if not isinstance(entry, dict):
            continue
        keyword = _clean(entry.get("keyword"))
        questions = [q.strip()[:200] for q in entry.get("questions", []) if isinstance(q, str) and q.strip()]
        if keyword and questions:
            groups.append({"keyword": keyword.lower()[:40], "questions": questions})
    return groups[:15]


def group_starters_by_keyword() -> dict[str, Any]:
    """Group the current starter pool under keywords and rewrite the keywords file."""
    if not settings.chat_enabled:
        return {"groups": 0, "error": "AI chat is not configured."}
    pool = _load_starter_pool()
    if len(pool) < 4:
        logger.info("Keyword grouping skipped: starter pool too small (%d)", len(pool))
        return {"groups": 0, "error": f"Starter pool too small ({len(pool)} starters)."}
    try:
        import openai

        client = openai.OpenAI(
            api_key=settings.litellm_api_key or "sk-no-key",
            base_url=settings.litellm_base_url,
        )
        question_list = "\n".join(f"- {s['text']}" for s in pool[:150])
        response = client.chat.completions.create(
            model=settings.litellm_model,
            messages=[
                {"role": "system", "content": KEYWORD_GROUPING_INSTRUCTIONS},
                {"role": "user", "content": f"Questions to group:\n\n{question_list}\n\nGroup them now."},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        groups = _parse_keyword_groups(content)
        if not groups:
            logger.warning("Keyword grouping produced no parsable groups")
            return {"groups": 0, "error": "The model returned no parsable keyword groups."}
        with _pool_lock:
            os.makedirs(os.path.dirname(STARTER_KEYWORDS_FILE), exist_ok=True)
            with open(STARTER_KEYWORDS_FILE, "w") as fh:
                json.dump(
                    {
                        "keywords": groups,
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "model": settings.litellm_model,
                        "pool_size": len(pool),
                    },
                    fh,
                    indent=2,
                )
        logger.info("Keyword grouping: wrote %d keyword group(s) to %s", len(groups), STARTER_KEYWORDS_FILE)
        return {"groups": len(groups)}
    except Exception as exc:
        logger.warning("Keyword grouping failed: %s", exc)
        return {"groups": 0, "error": str(exc)}


# ---- Public endpoints (any logged-in user) -----------------------------------

@router.get("/api/chat/conversation-starters")
def conversation_starters(n: int = 6, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """A random selection from the starter pool (mixing basic + interesting)."""
    n = max(1, min(n, 12))
    pool = _load_starter_pool() or list(FALLBACK_STARTERS)

    basic = [s for s in pool if s.get("kind") == "basic"]
    interesting = [s for s in pool if s.get("kind") != "basic"]
    random.shuffle(basic)
    random.shuffle(interesting)
    n_basic = min(len(basic), max(1, n // 3)) if basic else 0
    picked = basic[:n_basic] + interesting[: n - n_basic]
    if len(picked) < n:
        remaining = basic[n_basic:] + interesting[n - n_basic:]
        picked += remaining[: n - len(picked)]
    random.shuffle(picked)

    # Tag each starter with up to 3 keyword themes it belongs to (from the
    # grouping pass), matched on exact question text.
    keyword_index: dict[str, list[str]] = {}
    for group in _load_starter_keywords():
        kw = group["keyword"]
        for q in group.get("questions", []):
            keyword_index.setdefault(q.strip().lower(), []).append(kw)

    return {
        "starters": [
            {
                "text": s["text"],
                "kind": s.get("kind", "interesting"),
                "keywords": keyword_index.get(s["text"].strip().lower(), [])[:3],
            }
            for s in picked
        ],
        "pool_size": len(pool),
    }


@router.get("/api/chat/starter-keywords")
def starter_keywords(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """The keyword groups derived from the conversation-starter pool."""
    groups = _load_starter_keywords()
    return {
        "keywords": [
            {"keyword": g["keyword"], "count": len(g["questions"]), "questions": g["questions"][:6]}
            for g in groups
        ]
    }


# ---- Admin management endpoints (the /ai/starters manager page) --------------

@router.get("/api/chat/starters/manage")
def manage_starters(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Full pool + keyword grouping, for the admin manager page."""
    _require_admin(user)
    keywords_data = _load_starter_keywords_file()
    return {
        "chat_enabled": settings.chat_enabled,
        "model": settings.litellm_model,
        "starters": _load_starter_pool(),
        "keywords": _load_starter_keywords(),
        "keywords_meta": {
            "generated_at": keywords_data.get("generated_at"),
            "model": keywords_data.get("model"),
            "pool_size": keywords_data.get("pool_size"),
        },
    }


@router.post("/api/chat/starters/generate")
def admin_generate_starters(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Generate new starters, optionally steered by a direction/theme prompt.

    Runs synchronously (this is an admin page; the call can take a minute on a
    local model) and refreshes the keyword grouping afterwards.
    """
    _require_admin(user)
    direction = body.get("direction") if isinstance(body, dict) else None
    result = generate_conversation_starters(direction if isinstance(direction, str) else None)
    result["pool_size"] = len(_load_starter_pool())
    return result


@router.post("/api/chat/starters/regroup")
def admin_regroup_starters(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Re-run the keyword grouping over the current pool."""
    _require_admin(user)
    return group_starters_by_keyword()


@router.post("/api/chat/ping")
def admin_chat_ping(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Bare connectivity test for the configured model - NO catalog context, no
    prompts: exactly the client the app uses, a one-word request, three steps:
    list the proxy's models, one plain completion, one streamed completion.
    Every step reports its reply or its full error, so a broken endpoint, a
    wrong model name, or a streaming-only failure is pinpointed directly."""
    _require_admin(user)
    import time

    key = settings.litellm_api_key or ""
    result: dict[str, Any] = {
        "settings": {
            "chat_enabled": settings.chat_enabled,
            "base_url": settings.litellm_base_url,
            "model": settings.litellm_model,
            "api_key": (key[:4] + "..." + key[-4:]) if len(key) > 8 else ("(set)" if key else "(not set - 'sk-no-key' is sent)"),
        },
    }
    if not settings.chat_enabled:
        result["error"] = "LITELLM_BASE_URL is empty in the running process (was the container recreated after editing .env?)"
        return result
    client = _get_openai_client()
    prompt = [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}]

    t0 = time.time()
    try:
        names = [m.id for m in client.models.list().data]
        result["models"] = {"ok": True, "count": len(names), "names": names[:50],
                            "configured_model_listed": settings.litellm_model in names,
                            "ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        result["models"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800], "ms": int((time.time() - t0) * 1000)}

    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=settings.litellm_model, messages=prompt, temperature=0, max_tokens=5)
        result["completion"] = {"ok": True, "reply": (resp.choices[0].message.content or ""),
                                "finish_reason": resp.choices[0].finish_reason, "ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        result["completion"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800], "ms": int((time.time() - t0) * 1000)}

    t0 = time.time()
    try:
        chunks = []
        for chunk in client.chat.completions.create(model=settings.litellm_model, messages=prompt, temperature=0,
                                                    max_tokens=5, stream=True):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
        result["stream"] = {"ok": True, "reply": "".join(chunks), "chunks": len(chunks), "ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        result["stream"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800], "ms": int((time.time() - t0) * 1000)}

    result["ok"] = bool(result.get("completion", {}).get("ok")) and bool(result.get("stream", {}).get("ok"))
    return result


@router.post("/api/chat/starters/context-diagnostics")
def admin_context_diagnostics(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Context diagnostics for the admin page.

    Always returns catalog size estimates (thin catalog vs concept index vs full
    detail) plus LiteLLM's reported model limits. With {"probe_window": true} it
    additionally probes the ACTUAL accepted context size empirically by sending
    increasingly large filler prompts until the server rejects one — this can
    take minutes on a large local model.
    """
    _require_admin(user)
    from src.chat_retrieval import catalog_size_estimates
    from src.cohort_cache import get_cohorts_from_cache

    all_cohorts = get_cohorts_from_cache("")
    result: dict[str, Any] = {"sizes": catalog_size_estimates(all_cohorts)}
    result["sizes"]["current_catalog_context_tokens"] = round(len(build_context([], None)) / 4)

    # What LiteLLM claims about the model (configured metadata, not a measurement).
    result["model_info"] = None
    if settings.chat_enabled:
        try:
            import requests

            base = settings.litellm_base_url.rstrip("/")
            info_base = base[:-3] if base.endswith("/v1") else base
            resp = requests.get(
                f"{info_base}/model/info",
                headers={"Authorization": f"Bearer {settings.litellm_api_key or 'sk-no-key'}"},
                timeout=15,
            )
            if resp.ok:
                for entry in resp.json().get("data", []):
                    if entry.get("model_name") == settings.litellm_model:
                        info = entry.get("model_info", {}) or {}
                        result["model_info"] = {
                            "max_input_tokens": info.get("max_input_tokens"),
                            "max_tokens": info.get("max_tokens"),
                            "max_output_tokens": info.get("max_output_tokens"),
                        }
                        break
            else:
                result["model_info_error"] = f"/model/info returned {resp.status_code}"
        except Exception as exc:
            result["model_info_error"] = str(exc)

    # Empirical probe: the deployed server's real limit is whatever it accepts.
    if body.get("probe_window") and settings.chat_enabled:
        probe_results = []
        try:
            client = _get_openai_client()
            filler_word = " token"  # ~1 token per repetition
            for target in (4_000, 16_000, 64_000, 128_000, 256_000, 512_000, 1_000_000):
                prompt = "Reply with the single word OK." + filler_word * target
                try:
                    client.chat.completions.create(
                        model=settings.litellm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2,
                        temperature=0,
                    )
                    probe_results.append({"approx_tokens": target, "ok": True})
                except Exception as exc:
                    probe_results.append({"approx_tokens": target, "ok": False, "error": str(exc)[:300]})
                    break
        except HTTPException as exc:
            probe_results.append({"approx_tokens": 0, "ok": False, "error": str(exc.detail)})
        result["window_probe"] = probe_results
    return result


@router.post("/api/chat/starters/add")
def admin_add_starter(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Manually add a single conversation starter to the pool."""
    _require_admin(user)
    text = _clean(body.get("text")) if isinstance(body, dict) else ""
    if not text:
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string.")
    kind = body.get("kind") if isinstance(body, dict) else None
    kind = kind if kind in ("basic", "interesting") else "interesting"
    added = _append_to_starter_pool([{"text": text[:200], "kind": kind}])
    return {"added": added, "pool_size": len(_load_starter_pool())}


@router.post("/api/chat/starters/delete")
def admin_delete_starters(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Delete starters from the pool by exact text match."""
    _require_admin(user)
    texts = body.get("texts") if isinstance(body, dict) else None
    if not isinstance(texts, list) or not texts:
        raise HTTPException(status_code=400, detail="'texts' must be a non-empty list.")
    targets = {str(t).strip().lower() for t in texts}
    with _pool_lock:
        pool = _load_starter_pool()
        kept = [s for s in pool if s["text"].strip().lower() not in targets]
        deleted = len(pool) - len(kept)
        if deleted:
            _write_starter_pool(kept)
    return {"deleted": deleted, "remaining": len(kept)}


@router.post("/api/chat/stream")
def chat_stream(body: dict[str, Any], user: Any = Depends(get_current_user)) -> StreamingResponse:
    """Stream a chat completion as plain-text chunks for a live typing effect."""
    client = _get_openai_client()
    full_messages, model, temperature, info = _assemble_payload(body)

    def _generate() -> Any:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None
                if delta:
                    yield delta
        except Exception as exc:
            logger.warning("Chat stream failed: %s", exc)
            yield f"\n\n[Error contacting the model: {exc}]"

    # What went into the context, for the chat's progress display (read before
    # the first token arrives, which can take a while with a large context).
    return StreamingResponse(_generate(), media_type="text/plain; charset=utf-8",
                             headers={"X-Chat-Context": json.dumps(info)})


@router.get("/api/chat/mapping-status")
def chat_mapping_status(cohort_ids: str = "", user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Cache status for each pair of the given cohorts (comma-separated ids).

    Drives the chat UI: cached pairs are shown as available to the assistant;
    uncached pairs get a 'generate the mapping' button.
    """
    ids = [c.strip() for c in cohort_ids.split(",") if c.strip()]
    return {"pairs": mapping_pair_status(ids)}


# ---- Conversation history ----------------------------------------------------
#
# Conversations are persisted in a SQLite store (src/ai_history.py). The client
# upserts the full transcript after each completed turn (keyed by a
# client-generated conversation_id), so the dual summary/detailed answer
# variants and abandoned conversations are both captured without the streaming
# endpoint having to accumulate racing streams. Each user sees their own
# history; admins can view everyone's via scope=all.


def _user_email(user: Any) -> str:
    return (user.get("email") or "").strip().lower() if isinstance(user, dict) else ""


def _is_admin(user: Any) -> bool:
    return _user_email(user) in settings.admins_list


@router.post("/api/chat/history")
def save_conversation(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Upsert a conversation's transcript + metadata. Called by the client after
    each completed turn."""
    conv_id = _clean(body.get("conversation_id"))
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    try:
        ai_history.upsert_conversation(
            conv_id=conv_id,
            user_id=_user_email(user),
            arrival_path=_clean(body.get("arrival_path")) or "chat",
            model=_clean(body.get("model")) or settings.litellm_model,
            entry_context=body.get("entry_context") or {},
            summary_clicked=bool(body.get("summary_clicked")),
            messages=messages,
            started_at=_clean(body.get("started_at")) or None,
        )
    except ai_history.AccessError:
        raise HTTPException(status_code=403, detail="This conversation belongs to another user")
    except Exception as exc:  # storage must never break the chat experience
        logger.warning("Failed to save conversation %s: %s", conv_id, exc)
        raise HTTPException(status_code=500, detail="Could not save conversation")
    return {"ok": True, "id": conv_id}


@router.get("/api/chat/history")
def list_history(
    scope: str = "own",
    path: Optional[str] = None,
    search: Optional[str] = None,
    min_messages: Optional[int] = None,
    max_messages: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """List conversations, newest activity first. scope=all is admin-only."""
    return ai_history.list_conversations(
        viewer_id=_user_email(user),
        is_admin=_is_admin(user),
        scope=scope,
        path=path,
        search=search,
        min_messages=min_messages,
        max_messages=max_messages,
        limit=max(1, min(int(limit), 200)),
        offset=max(0, int(offset)),
    )


@router.get("/api/chat/history/summary")
def history_summary(scope: str = "own", user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Aggregate usage metrics for the dashboard. scope=all is admin-only."""
    return ai_history.usage_summary(
        viewer_id=_user_email(user), is_admin=_is_admin(user), scope=scope
    )


@router.get("/api/chat/history/{conv_id}")
def get_history(conv_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Full transcript for one conversation. Owner or admin only."""
    try:
        conv = ai_history.get_conversation(
            conv_id, viewer_id=_user_email(user), is_admin=_is_admin(user)
        )
    except ai_history.AccessError:
        raise HTTPException(status_code=403, detail="This conversation belongs to another user")
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv
