"""
study_context.py — Extract Studies design protocol Metadata to use as LLM validation as context
"""

from __future__ import annotations
from typing import Dict, Optional
import logging

from .query_builder import SPARQLQueryBuilder
from .utils import execute_query

logger = logging.getLogger(__name__)


def fetch_study_context(study_id: str) -> Optional[Dict[str, str]]:
    """Run the study-context SPARQL query and flatten the single result row."""
    query = SPARQLQueryBuilder.build_study_context_query(study_id.strip().lower())
    try:
        bindings = execute_query(query).get("results", {}).get("bindings", [])
    except Exception as e:
        logger.error(f"Study context query failed for '{study_id}': {e}")
        return None

    if not bindings:
        logger.warning(f"No study metadata found for '{study_id}'")
        return None

    return {k: v["value"] for k, v in bindings[0].items() if v.get("value")}


def format_study_context_block(src_study: str, tgt_study: str) -> str:
    """Build the compact context block prepended to the LLM user prompt.

    Returns an empty string when neither study has metadata — the pipeline
    behaves identically to the no-context path in that case.
    """
    src = fetch_study_context(src_study)
    tgt = fetch_study_context(tgt_study)

    if not src and not tgt:
        return ""

    lines = ["# STUDY CONTEXT"]
    for label, study_id, meta in [("Source", src_study, src), ("Target", tgt_study, tgt)]:
        if not meta:
            lines.append(f"{label}: {study_id} (no metadata)")
            continue

        parts = [f"{label}: {meta.get('study_name', study_id)}"]
        design = meta.get("study_design") or meta.get("study_type")
        inclusion_criteria =  meta.get("inclusion_criteria")
        # if design:
        #     parts.append(f"design={design}")
        n = meta.get("n_participants")
        if n:
            parts.append(f"total patients={n}")
        morb = meta.get("morbidities")
        if morb:
            parts.append(f"population={morb}")

        if inclusion_criteria:
            parts.append(f"inclusion criteria={inclusion_criteria}")
        # age = meta.get("age_distribution")
        # if age:
        #     parts.append(f"age={age}")
        lines.append(", ".join(parts))

    # ── Shared morbidity → degenerate-variable warning ─────────────
    def _morb_set(meta):
        if not meta:
            return set()
        return {m.strip().lower()
                for m in meta.get("morbidities", "").split(";")
                if m.strip()}

    # shared = _morb_set(src) & _morb_set(tgt)
    # if shared:
    #     lines.append(
    #         f"Both cohorts share study-level condition(s): {', '.join(sorted(shared))}. "
    #         "A variable encoding any of these shared conditions will be "
    #         "constant across both studies (zero variance) — classify such "
    #         "pairs as IMPOSSIBLE regardless of structural similarity."
    #     )

    return "\n".join(lines)
