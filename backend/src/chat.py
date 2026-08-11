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
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.config import settings

router = APIRouter()

logger = logging.getLogger(__name__)

# Keep prompts bounded so a broad selection cannot blow up the context window.
MAX_CONTEXT_COHORTS = 25
MAX_VARS_PER_COHORT = 40
MAX_CATALOG_COHORTS = 200
# Caps for client-supplied overrides (Glass Box & friends).
MAX_SYSTEM_PROMPT_CHARS = 8_000
MAX_CONTEXT_CHARS = 120_000

SYSTEM_PROMPT = (
    "You are the iCARE4CVD Cohort Explorer assistant. You help researchers "
    "understand and compare cardiovascular research cohorts and their variables. "
    "Answer using ONLY the cohort context provided in this conversation. If the "
    "context does not contain the answer, say so plainly and suggest what the user "
    "could select or ask instead. Be concise, use short paragraphs and bullet "
    "points, reference cohorts and variables by name, and never invent variables, "
    "values, or statistics that are not present in the context."
)


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
        parts.append(f"The user is focusing on {len(selected)} cohort(s):")
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
            n_vars = len(getattr(cohort, "variables", {}) or {})
            stype = _clean(getattr(cohort, "study_type", ""))
            descr = f" ({stype})" if stype else ""
            catalog.append(f"- {cohort.cohort_id}{descr}: {n_vars} variables")
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

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Cohort context:\n\n{context}"},
        *messages,
    ]
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
