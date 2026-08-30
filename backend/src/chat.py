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

# Keep prompts bounded so a broad selection cannot blow up the context window.
MAX_CONTEXT_COHORTS = 25
MAX_VARS_PER_COHORT = 40
MAX_CATALOG_COHORTS = 200
# Variable names sampled per cohort in the CATALOG (no cohorts selected) view,
# so catalog-wide questions about variables can be answered with real names.
CATALOG_VARS_SAMPLE = 15
# Caps for client-supplied overrides (Glass Box & friends).
MAX_SYSTEM_PROMPT_CHARS = 8_000
MAX_CONTEXT_CHARS = 120_000

# How the platform works — shared so the assistant can guide users through the
# actual workflow, not just describe data.
PLATFORM_OVERVIEW = (
    "About the iCARE4CVD Cohort Explorer platform:\n"
    "- The Explorer's main (explore) page lets analysts discover cardiovascular studies/cohorts "
    "of interest and the variables each has uploaded (its metadata).\n"
    "- The explore page has a proper SEARCH BOX. It offers exactly two settings: WHERE to search "
    "(cohorts metadata, variables information, or all) and the MODE (OR search, AND search, or "
    "exact phrase). It searches variable names, labels, concept names and codes. There are NO "
    "other filters — no domain filter, no data-type filter, no visit filter — so never tell the "
    "user to apply one. When users need to find variables, point them to this search box — NEVER "
    "suggest browser tricks like Ctrl+F.\n"
    "- To actually analyse data, an analyst creates an analysis DCR (Data Clean Room): a secure "
    "computing enclave. The data owners (cohort admins) upload their real data into the DCR, and "
    "the analyst writes a script that computes over that data WITHOUT ever seeing the raw records — "
    "only permitted outputs leave the enclave.\n"
    "- Cross-cohort variable mapping is done from the dedicated MAPPING PAGE: the analyst picks a "
    "source cohort and target cohort(s) and the platform generates a mapping file of likely "
    "variable correspondences (suggested equivalences, not guarantees). This is the ONLY way to "
    "map variables across cohorts; there is no per-variable mapping control anywhere else.\n"
    "When relevant, explain how the user could act via these features (e.g. create a DCR to run an "
    "analysis, or generate a mapping from the mapping page to align variables across cohorts), but "
    "never claim to have run an analysis or seen raw data yourself."
)

SYSTEM_PROMPT = (
    "You are the iCARE4CVD Cohort Explorer assistant. You help researchers "
    "understand and compare cardiovascular research cohorts and their variables. "
    f"\n\n{PLATFORM_OVERVIEW}\n\n"
    "Answer using ONLY the cohort context provided in this conversation. "
    "INTERPRETING THE QUESTION: first decide what is being asked - a specific variable, a list "
    "of cohorts, an inventory of what is tracked (e.g. 'medications of X patients' = ALL "
    "medication variables, not one drug class), or data about a subgroup ('patients with X'). "
    "If the question supports different readings, SAY SO - the users are analysts and value "
    "having the ambiguity in their own question pinned down, even a subtle one. When one "
    "reading contains the other (an inventory contains a single flag), answer the broader and "
    "point out the narrower inside it. When the readings genuinely diverge, present each "
    "interpretation, sketch in a line what the search results say under each, and END BY "
    "ASKING which one is meant - never silently pick the narrower or more familiar reading, "
    "and never narrow a condition to one drug class just because it is clinically typical. "
    "Open with 'Interpreting this as ...' whenever you chose a reading. "
    "SUBGROUPS: the catalog holds variable METADATA, no patient rows. A question about "
    "'patients with X' can only be answered as 'these cohorts record X and also record Y - "
    "combining them happens inside a Data Clean Room', never as filtered patient results. "
    "COHORT NAMES: if the user's message contains a word matching a cohort's name or the first "
    "part of one (e.g. 'biostat' for BIOSTAT-CHF, 'aachen' for Aachen-HF, 'time' for TIME-CHF, "
    "'check' for CHECK-HF), ASSUME they are referring to that cohort: answer about the cohort, in "
    "the context of the conversation so far. If the match was only partial, end with ONE short "
    "question confirming the reading (e.g. 'I read \"biostat\" as the BIOSTAT-CHF cohort — did "
    "you mean something else?'). Never instead interpret such a word as a general topic (e.g. "
    "'biostat' is not biostatistics here). "
    "SEARCH RESULTS: when the conversation includes CATALOG SEARCH RESULTS (produced by the "
    "platform's built-in search tool, shown to the user in a search panel), any question about "
    "which cohorts or variables meet certain criteria MUST be answered from those results: "
    "name ALL the matching cohorts with their counts, list at most 10-15 variables per cohort, "
    "always state the full match counts so it is clear there are more, and use the "
    "equivalent-by-standard-code links to point out cross-cohort correspondences. "
    "IMPORTANT: focus your search, comparisons and suggestions on cohorts that "
    "have variables (metadata) uploaded to the Explorer — these are the only "
    "cohorts whose data can actually be explored here. Cohorts with 0 uploaded "
    "variables should not be suggested as places to find data; mention them only "
    "if the user asks about them directly. If the context does not contain the "
    "answer, say so plainly and suggest cohorts WITH uploaded variables the user "
    "could select or ask about instead. Be concise, use short paragraphs and "
    "bullet points, reference cohorts and variables by name, and never invent "
    "variables, values, or statistics that are not present in the context."
)

# Every question is asked twice — once per style — and the chat bubble lets the
# user toggle between the two answers.
STYLE_INSTRUCTIONS = {
    "summary": (
        "Answer style: SHORT SUMMARY. Give only the essential answer, in at most 4 "
        "sentences or up to 5 short bullet points. No preamble, no closing offer. "
        "COMPLETENESS STILL APPLIES: when the question asks which cohorts have "
        "something, being brief means saying MORE about fewer cohorts — never "
        "listing fewer cohorts. Detail the most relevant two or three, then close "
        "with one line naming EVERY remaining match, e.g. 'Also with beta-blocker "
        "variables: TIM-HF, GISSI-HF, GISSI-Prevenzione, CHECK-HF, ... — see the "
        "search results for the variables in each.' A question that pins down an "
        "ambiguous request ('did you mean A or B?') is part of the essential "
        "answer - brevity never drops it."
    ),
    "detailed": (
        "Answer style: DETAILED. Give a thorough, well-structured answer with "
        "specifics — cohort names, variable names, caveats — where the context "
        "supports them."
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


def _summarize_variable(var: Any) -> str:
    """One compact line describing a single variable."""
    name = _clean(getattr(var, "var_name", "")) or "?"
    label = _clean(getattr(var, "var_label", ""))
    vtype = _clean(getattr(var, "var_type", ""))
    units = _clean(getattr(var, "units", ""))
    bits = []
    if label and label.lower() != name.lower():
        bits.append(label)
    meta = []
    if vtype:
        meta.append(vtype)
    if units:
        meta.append(units)
    if meta:
        bits.append(f"[{', '.join(meta)}]")
    suffix = f" — {' '.join(bits)}" if bits else ""
    return f"{name}{suffix}"


def _summarize_cohort(cohort: Any, include_variables: bool = True) -> str:
    """Multi-line summary of one cohort with an optional sample of its variables."""
    lines = [f"### Cohort: {cohort.cohort_id}"]
    fields = [
        ("Institution", getattr(cohort, "institution", "")),
        ("Study type", getattr(cohort, "study_type", "")),
        ("Study design", getattr(cohort, "study_design", "")),
        ("Participants", getattr(cohort, "study_participants", "")),
        ("Population", getattr(cohort, "study_population", "")),
        ("Objective", getattr(cohort, "study_objective", "")),
        ("Primary outcome", getattr(cohort, "primary_outcome_spec", "")),
        ("Morbidity", getattr(cohort, "morbidity", "")),
        ("Location", getattr(cohort, "population_location", "")),
    ]
    for label, value in fields:
        cleaned = _clean(value)
        if cleaned:
            lines.append(f"- {label}: {cleaned}")

    variables = getattr(cohort, "variables", {}) or {}
    lines.append(f"- Variable count: {len(variables)}")
    if include_variables and variables:
        sample = list(variables.values())[:MAX_VARS_PER_COHORT]
        lines.append("- Variables (sample):")
        for var in sample:
            lines.append(f"    - {_summarize_variable(var)}")
        if len(variables) > MAX_VARS_PER_COHORT:
            lines.append(f"    - …and {len(variables) - MAX_VARS_PER_COHORT} more variables")
    return "\n".join(lines)


def build_context(cohort_ids: list[str], focus: Optional[str] = None) -> str:
    """Assemble the cohort context string injected as a system message.

    - With cohort_ids: deep per-cohort summaries (metadata + variable sample).
    - Without: a compact catalog of all cohorts (id + variable counts) so the
      model can still help the user pick relevant cohorts.
    """
    from src.cohort_cache import get_cohorts_from_cache

    all_cohorts = get_cohorts_from_cache("")
    parts: list[str] = []

    selected = [cid for cid in cohort_ids if cid in all_cohorts][:MAX_CONTEXT_COHORTS]
    if selected:
        parts.append(
            f"The user is focusing on {len(selected)} cohort(s): {', '.join(selected)}. "
            "Base your answer MAINLY on these cohorts and their variables; mention other "
            "cohorts only when directly relevant to the question."
        )
        for cid in selected:
            parts.append(_summarize_cohort(all_cohorts[cid], include_variables=True))
    else:
        with_vars = [c for c in all_cohorts.values() if getattr(c, "variables", None)]
        parts.append(
            f"No specific cohort is selected. Catalog of {len(all_cohorts)} cohorts "
            f"({len(with_vars)} with uploaded variables):"
        )
        catalog = []
        for cohort in list(all_cohorts.values())[:MAX_CATALOG_COHORTS]:
            variables = getattr(cohort, "variables", {}) or {}
            stype = _clean(getattr(cohort, "study_type", ""))
            descr = f" ({stype})" if stype else ""
            catalog.append(f"- {cohort.cohort_id}{descr}: {len(variables)} variables")
            # A sample of actual variable names so catalog-wide questions
            # ("which cohorts measure X?") can be answered without selecting
            # cohorts first. Deep dives still require selecting cohorts.
            if variables:
                names = [
                    _clean(getattr(v, "var_name", ""))[:40]
                    for v in list(variables.values())[:CATALOG_VARS_SAMPLE]
                ]
                names = [n for n in names if n]
                if names:
                    suffix = ", …" if len(variables) > CATALOG_VARS_SAMPLE else ""
                    catalog.append(f"    variables include: {', '.join(names)}{suffix}")
        parts.append("\n".join(catalog))

    if _clean(focus):
        parts.append(f"The user is particularly interested in: {_clean(focus)}")

    return "\n\n".join(parts)


# ---- Cross-cohort mapping files in chat context ------------------------------
#
# When the user has 2+ cohorts selected, the assistant should ground
# cross-cohort variable questions in the mapping files generated from the
# mapping page (CohortVarLinker), when such files exist in the cache. The
# transcript must make it unmistakable that a cached file was used and which
# one. When no mapping exists for a pair, the UI offers a "generate" button
# (see /api/chat/mapping-status) and the model is told the mapping is missing.

# Columns injected into context, in order, when present in the CSV. The full
# files carry ~19 columns; these are the ones useful for reasoning about
# variable equivalence without blowing up the context.
_MAPPING_CONTEXT_COLS = [
    "source", "slabel", "target", "tlabel", "category", "mapping type",
    "source_unit", "target_unit", "harmonization_status",
]
MAX_MAPPING_ROWS_PER_PAIR = 120
MAX_MAPPING_PAIRS = 6

MAPPING_USAGE_INSTRUCTIONS = (
    "CROSS-COHORT MAPPING FILES: cached mapping file(s) generated by the platform's mapping "
    "pipeline are included in the context below, each labelled with its exact filename.\n"
    "- Whenever your answer draws on a mapping file, make it VERY clear — state prominently "
    "(at the start of the relevant part of your answer) that you are using a cached mapping "
    "file and give its filename.\n"
    "- IMPORTANT: these files are OVER-GENERATED — they intentionally include many candidate "
    "correspondences, and not all are good. Exercise judgment: prefer rows whose labels, "
    "categories and units genuinely align (e.g. 'exact match' over loose semantic matches), "
    "and say so when a suggested mapping looks questionable.\n"
    "- These are suggested equivalences, not guarantees; recommend the user verify on the "
    "mapping page."
)


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


def _mapping_file_block(source: str, target: str, path: str) -> str:
    """Render one cached mapping CSV as a compact, capped context block."""
    import csv as _csv

    filename = os.path.basename(path)
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:
        logger.warning("Could not read mapping file %s: %s", filename, exc)
        return ""

    cols = [c for c in _MAPPING_CONTEXT_COLS if rows and c in rows[0]]
    if not cols:
        cols = list(rows[0].keys())[:6] if rows else []

    lines = [
        f"CACHED MAPPING FILE: {filename} (maps variables of '{source}' to '{target}'; "
        f"{len(rows)} candidate mappings, over-generated — apply judgment)."
    ]
    lines.append(" | ".join(cols))
    for row in rows[:MAX_MAPPING_ROWS_PER_PAIR]:
        lines.append(" | ".join(_clean(row.get(c, "")) for c in cols))
    if len(rows) > MAX_MAPPING_ROWS_PER_PAIR:
        lines.append(f"…and {len(rows) - MAX_MAPPING_ROWS_PER_PAIR} more rows not shown.")
    return "\n".join(lines)


def build_mapping_context(cohort_ids: list[str]) -> tuple[str, list[str]]:
    """(context block for cached pairs, human-readable list of uncached pairs)."""
    cached_blocks: list[str] = []
    uncached: list[str] = []
    try:
        output_dir = _linker_output_dir()
    except Exception:
        return "", []
    for pair in mapping_pair_status(cohort_ids)[:MAX_MAPPING_PAIRS]:
        if pair["cached"]:
            path = os.path.join(output_dir, pair["filename"])
            block = _mapping_file_block(pair["source"], pair["target"], path)
            if block:
                cached_blocks.append(block)
        else:
            uncached.append(f"{pair['source']} ↔ {pair['target']}")
    return "\n\n".join(cached_blocks), uncached


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


def _assemble_payload(body: dict[str, Any]) -> tuple[list[dict[str, str]], str, float]:
    """Turn the request body into (messages, model, temperature).

    Optional overrides (used by the Glass Box / Atlas layouts, which construct
    their payloads client-side for full transparency):
      - system_prompt: replaces the default SYSTEM_PROMPT.
      - context: replaces the server-built cohort context entirely, so what the
        user previews in the UI is exactly what the model receives.
    """
    messages = _normalize_messages(body.get("messages"))
    cohort_ids = body.get("cohort_ids") or []
    if not isinstance(cohort_ids, list):
        cohort_ids = []
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

    context_override = body.get("context")
    if isinstance(context_override, str) and context_override.strip():
        context = context_override.strip()[:MAX_CONTEXT_CHARS]
    else:
        context = build_context([str(c) for c in cohort_ids], focus)
        try:
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            note = cohort_name_note(last_user_msg) if last_user_msg else ""
            if note:
                context = f"{context}\n\n{note}"
        except Exception as exc:
            logger.warning("Cohort-name note failed: %s", exc)
        # Search results from the planning round (see /api/chat/plan-search):
        # the client runs the plan once per user turn and passes the structured
        # results into both style variants, so the model and the user's search
        # panel see exactly the same thing.
        search_results = body.get("search_results")
        runs_in, concepts_in, intersection_in = None, None, None
        if isinstance(search_results, dict):
            runs_in = search_results.get("runs")
            concepts_in = search_results.get("concepts")
            intersection_in = search_results.get("intersection")
        elif isinstance(search_results, list):
            runs_in = search_results
        if isinstance(runs_in, list) and runs_in:
            try:
                from src.chat_retrieval import format_search_context

                section = format_search_context(runs_in, concepts=concepts_in, intersection=intersection_in)
                if section:
                    context = f"{context}\n\n{section}"
            except Exception as exc:
                logger.warning("Search-results formatting failed: %s", exc)
        else:
            # Fallback single-round retrieval: mirror the cohorts-page search
            # (OR mode) over the user's question and inject matching variable
            # details, excluding terms that are too broad. See chat_retrieval.py.
            try:
                from src.chat_retrieval import retrieve_for_question
                from src.cohort_cache import get_cohorts_from_cache

                last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                if last_user:
                    section = retrieve_for_question(
                        last_user,
                        get_cohorts_from_cache(""),
                        restrict_to=[str(c) for c in cohort_ids],
                    )
                    if section:
                        context = f"{context}\n\n{section}"
            except Exception as exc:
                logger.warning("Question-based variable retrieval failed: %s", exc)

        # Cross-cohort mapping files: inject cached mappings for the selected
        # pairs, and tell the model plainly which pairs have none yet.
        if len(cohort_ids) >= 2:
            try:
                mapping_block, uncached_pairs = build_mapping_context([str(c) for c in cohort_ids])
                if mapping_block:
                    context = f"{context}\n\n{MAPPING_USAGE_INSTRUCTIONS}\n\n{mapping_block}"
                if uncached_pairs:
                    context = (
                        f"{context}\n\nNO CACHED MAPPING exists yet for: {'; '.join(uncached_pairs)}. "
                        "If the user asks how variables correspond across these cohorts, say the "
                        "mapping has not been generated yet and that they can generate it with the "
                        "button shown in this chat (or from the mapping page) — do NOT guess "
                        "variable correspondences yourself."
                    )
            except Exception as exc:
                logger.warning("Mapping context injection failed: %s", exc)

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Cohort context:\n\n{context}"},
    ]
    style = body.get("style")
    if isinstance(style, str) and style in STYLE_INSTRUCTIONS:
        full_messages.append({"role": "system", "content": STYLE_INSTRUCTIONS[style]})
    full_messages.extend(messages)
    return full_messages, model, temperature


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
    full_messages, model, temperature = _assemble_payload(body)
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
    "answer can show each interpretation with real results and ask the user which they meant. "
    "Treat a follow-up as a NEW question unless it clearly refines the previous one; do not "
    "just re-run the previous turn's terms.\n"
    "COHORT NAMES: the catalog's cohort names are listed below. A word matching a cohort name or "
    "its first part (biostat -> BIOSTAT-CHF, aachen -> Aachen-HF, time -> TIME-CHF) refers to THE "
    "COHORT, never to a topic — NEVER turn it into search terms (no 'biostatistics'). For a "
    "follow-up like 'what about <cohort>?', re-propose the search terms of the criteria already "
    "under discussion in the conversation, so the results cover that cohort too.\n"
    "If the question needs no catalog search (small talk, platform how-to, questions about "
    "already-shown results), return an empty list.\n"
    'Return STRICT JSON only: {"concepts": [{"name": "<short label>", "terms": ["term", ...]}, ...]} '
    '(empty list when no search is needed).'
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
        history = _normalize_messages(body.get("history"))[-6:] if body.get("history") else []
    except HTTPException:
        history = []
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history)
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
    except Exception as exc:
        logger.warning("Search planning failed: %s", exc)
        concepts, terms = [], []
    # Backstop: an identification-shaped question always searches, with terms
    # from the question itself if the planner offered none.
    if not terms and _IDENTIFICATION_RE.search(question):
        from src.chat_retrieval import extract_query_terms

        terms = extract_query_terms(question)[:6]
        concepts = [{"name": "", "terms": terms}] if terms else []
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
            "model": settings.litellm_model}


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
    "IMPORTANT: only reference cohorts that have variables (metadata) uploaded to the "
    "Explorer — i.e. those listed with 1 or more variables in the context. Never build a "
    "question around a cohort that has 0 uploaded variables, as there is no data to explore "
    "for it.\n\n"
    "Return STRICT JSON, no markdown fences and no commentary, exactly of the form:\n"
    '{"interesting": ["...", "..."], "basic": ["...", "..."]}\n\n'
    "- \"interesting\": 8 specific, research-oriented questions. Reference actual cohorts "
    "(WITH uploaded variables), domains or variables from the context where possible; favour "
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
            for target in (4_000, 8_000, 16_000, 32_000, 64_000, 100_000, 128_000):
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
    full_messages, model, temperature = _assemble_payload(body)

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

    return StreamingResponse(_generate(), media_type="text/plain; charset=utf-8")


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
