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
    "- The Explorer lets analysts discover cardiovascular studies/cohorts of interest and the "
    "variables each has uploaded (its metadata).\n"
    "- To actually analyse data, an analyst creates an analysis DCR (Data Clean Room): a secure "
    "computing enclave. The data owners (cohort admins) upload their real data into the DCR, and "
    "the analyst writes a script that computes over that data WITHOUT ever seeing the raw records — "
    "only permitted outputs leave the enclave.\n"
    "- A mapping functionality helps analysts identify which variables in one cohort LIKELY "
    "correspond to variables in another cohort (these are suggested equivalences, not guarantees).\n"
    "When relevant, explain how the user could act via these features (e.g. create a DCR to run an "
    "analysis, or use mapping to align variables across cohorts), but never claim to have run an "
    "analysis or seen raw data yourself."
)

SYSTEM_PROMPT = (
    "You are the iCARE4CVD Cohort Explorer assistant. You help researchers "
    "understand and compare cardiovascular research cohorts and their variables. "
    f"\n\n{PLATFORM_OVERVIEW}\n\n"
    "Answer using ONLY the cohort context provided in this conversation. "
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
        "sentences or up to 5 short bullet points. No preamble, no closing offer."
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
        # Single-round retrieval: mirror the cohorts-page search (OR mode) over
        # the user's question and inject matching variable details, excluding
        # terms that are too broad. See chat_retrieval.py.
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
