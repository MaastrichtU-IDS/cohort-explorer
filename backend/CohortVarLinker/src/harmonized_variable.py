"""
Deterministic harmonized_variable (snake_case) for pairs that never reach the LLM.

When structural evidence is already decisive (IDENTICAL / COMPATIBLE), the
pipeline skips the LLM (`should_consult_llm` is false). This module derives a
single analysis-friendly name from shared OMOP metadata and the ontology graph
when available, mirroring the LLM rubric: prefer the coarsest shared concept.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .data_model import MatchLevel, MappingRelation, VariableNode
from .verdict import StructuralEvidence

_SLUG_MAX = 80


def _slugify(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:_SLUG_MAX] if s else ""


def _concept_name(graph: Any, concept_id: Optional[int]) -> str:
    if graph is None or concept_id is None:
        return ""
    try:
        cid = int(concept_id)
    except (TypeError, ValueError):
        return ""
    for attr in ("concept_name", "name"):
        name = graph.get_node_attr(cid, attr)
        if name and str(name).strip():
            return str(name).strip()
    return ""


def _name_from_pair_same_label(src: VariableNode, tgt: VariableNode) -> str:
    a = (src.main_label or "").strip()
    b = (tgt.main_label or "").strip()
    if a and a.lower() == b.lower():
        return _slugify(a)
    return ""


def _broader_display_name(graph: Any, src_id: Optional[int], tgt_id: Optional[int]) -> str:
    """Pick the broader (or shared) OMOP concept's display name using explain_path."""
    if graph is None or src_id is None or tgt_id is None:
        return ""
    try:
        sid, tid = int(src_id), int(tgt_id)
    except (TypeError, ValueError):
        return ""
    if sid == tid:
        return _concept_name(graph, sid)

    info = graph.explain_path(sid, tid, max_depth=6)
    pt = (info.get("path_type") or "").strip()
    path = info.get("path") or []

    if pt in ("exact_match",):
        return _concept_name(graph, sid)

    if pt in ("equivalence", "loinc_axis"):
        # Same clinical meaning class; either endpoint is fine for naming.
        return _concept_name(graph, sid) or _concept_name(graph, tid)

    if pt == "ancestor":
        # tgt is an ancestor of src → broader concept is tgt.
        return _concept_name(graph, tid) or _concept_name(graph, sid)

    if pt == "descendant":
        # tgt is a descendant of src → broader concept is src.
        return _concept_name(graph, sid) or _concept_name(graph, tid)

    if pt in ("sibling", "graph_traversal") and path:
        # Shared abstraction: use middle of shortest explanation path.
        mid = path[len(path) // 2]
        cid = mid[0] if isinstance(mid, (list, tuple)) and mid else None
        if cid is not None:
            return _concept_name(graph, cid) or ""

    return ""


def suggest_harmonized_variable_without_llm(
    src: VariableNode,
    tgt: VariableNode,
    structural: StructuralEvidence,
    *,
    graph: Any = None,
    verdict_level: MatchLevel,
) -> str:
    """Return snake_case harmonized_variable for non–LLM rows, or \"\".

    Intended for pairs where `should_consult_llm` is false (structural
    IDENTICAL / COMPATIBLE / NOT_APPLICABLE). For NOT_APPLICABLE verdicts,
    returns \"\" so we do not invent a pooled variable name.

    Resolution order:
    1. Same OMOP id → concept name from graph, else slugified shared label.
    2. Graph path between main ids → broader / shared concept name (see rubric).
    3. Identical normalized main labels (no graph) → slugify that label.
    4. Empty string if nothing grounded.
    """
    if verdict_level == MatchLevel.NOT_APPLICABLE:
        return ""

    relation = (structural.extra.get("mapping_relation") or "").strip().lower()

    sid, tid = src.main_id, tgt.main_id

    # Same concept id — strongest signal for symbolic / tight matches.
    if sid is not None and tid is not None and int(sid) == int(tid):
        raw = _concept_name(graph, sid)
        if raw:
            return _slugify(raw)
        same_lbl = _name_from_pair_same_label(src, tgt)
        if same_lbl:
            return same_lbl
        return _slugify(str(sid))

    # Ontology-backed broader/narrower or compatible different ids.
    if graph is not None and sid is not None and tid is not None:
        name = _broader_display_name(graph, sid, tid)
        if name:
            return _slugify(name)

    # Hierarchical SKOS edge without resolvable graph path: prefer coarse label heuristic.
    if MappingRelation.is_hierarchical(relation):
        s_l = (src.main_label or "").strip()
        t_l = (tgt.main_label or "").strip()
        if s_l and t_l:
            # Shorter label often corresponds to a broader clinical wording (weak fallback).
            coarse = s_l if len(s_l) <= len(t_l) else t_l
            return _slugify(coarse)

    # No graph path: identical preferred labels only.
    return _name_from_pair_same_label(src, tgt)
