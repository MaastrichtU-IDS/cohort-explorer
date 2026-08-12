"""
Deterministic harmonized_variable (snake_case) for pairs that never reach the LLM.

When structural evidence is already decisive (IDENTICAL / COMPATIBLE), the
pipeline skips the LLM (`should_consult_llm` is false). This module derives a
single analysis-friendly name from shared OMOP metadata and the ontology graph
when available, mirroring the LLM rubric: prefer the coarsest shared concept.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .data_model import MatchLevel, MappingRelation, StatisticalType, VariableNode
from .verdict import StructuralEvidence

_SLUG_MAX = 80

# The prompt caps the LLM's harmonized_variable at five words. This module must
# use the same budget, or the two generators produce names that cannot be
# compared: `weight_kg` from one and
# `alanine_aminotransferase_enzymatic_activity_volume_in_blood` from the other.
_MAX_WORDS = 5


def _slugify(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:_SLUG_MAX] if s else ""


def _bound_words(name: str, max_chars: int = _SLUG_MAX) -> str:
    """Bound a stem by dropping trailing words, never cutting mid-word.

    Depends on the stem alone. An earlier version reserved room for the axis
    suffix, which made the same concept truncate to different lengths depending
    on whether a suffix applied — `angiotensin_ii_receptor_blockers_arbs` in one
    row and `angiotensin_ii_receptor` in the next, for one concept id.
    """
    parts = [p for p in str(name or "").split("_") if p]
    while parts and len("_".join(parts)) > max_chars:
        parts.pop()
    return "_".join(parts)


def _unit_slug(unit: str) -> str:
    """ucum:mg/dL -> mg_per_dl. Only UCUM-tagged values count as units.

    The dictionaries also park date formats in this field ("DD.MM.YYYY"), which
    are a rendering rather than a unit and must not end up in the name.
    """
    u = str(unit or "").strip()
    if not u.lower().startswith("ucum:"):
        return ""
    return _slugify(re.sub(r"^ucum:", "", u, flags=re.I).replace("/", " per "))


def _is_date(v: VariableNode) -> bool:
    return str(getattr(v, "data_type", "") or "").strip().lower() in {"datetime", "date"}


def _axis_suffix(src: VariableNode, tgt: VariableNode) -> str:
    """Axis marker for the pooled variable — the coarsest side wins.

    Precedence follows the direction a harmonized column actually collapses in:
    a binary side forces a presence flag, a multi-class side forces categories,
    a date stays a date, and a unit applies only when both sides are quantities.
    Differing units yield no suffix — nothing here can say which is canonical,
    and asserting the wrong one is worse than omitting it.
    """
    types = {src.statistical_type, tgt.statistical_type}
    if StatisticalType.BINARY in types:
        return "yes_no"
    if StatisticalType.MULTI_CLASS in types:
        return "category"
    if _is_date(src) or _is_date(tgt):
        return "date"
    if StatisticalType.CONTINUOUS in types:
        src_unit, tgt_unit = _unit_slug(src.unit), _unit_slug(tgt.unit)
        if src_unit and tgt_unit and src_unit != tgt_unit:
            return ""
        return src_unit or tgt_unit
    return ""


def _with_axis(name: str, suffix: str) -> str:
    """Bound the concept stem, then append the axis marker.

    No five-word budget: that existed so this name could be compared with the
    LLM's, and the deterministic name is now the authoritative one. Truncating a
    vocabulary label only invites collisions between distinct concepts.
    """
    stem = _bound_words(name)
    if not stem:
        return ""
    if not suffix or stem == suffix or stem.endswith(f"_{suffix}"):
        return stem
    return f"{stem}_{suffix}"


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


def _base_name(
    src: VariableNode,
    tgt: VariableNode,
    structural: StructuralEvidence,
    *,
    graph: Any = None,
) -> str:
    """Concept stem for the harmonized variable, before the axis suffix.

    Resolution order:
    1. Same OMOP id → concept name from graph, else slugified shared label.
    2. Graph path between main ids → broader / shared concept name (see rubric).
    3. Identical normalized main labels (no graph) → slugify that label.
    4. Empty string if nothing grounded.
    """
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

    The name is the concept stem plus the information axis, under the same
    five-word budget the prompt gives the LLM, so both generators produce
    comparable names.
    """
    if verdict_level == MatchLevel.NOT_APPLICABLE:
        return ""

    return _with_axis(_base_name(src, tgt, structural, graph=graph),
                      _axis_suffix(src, tgt))


# ─────────────────────────────────────────────────────────────────────────────
# Subtype / parent pairs
# ─────────────────────────────────────────────────────────────────────────────

# Only a straight line of subsumption counts. `sibling` and `graph_traversal`
# are deliberately excluded: _broader_display_name answers those by taking the
# middle of the shortest path, which is a reasonable guess for a *stem* but far
# too weak to overrule a name the LLM already produced.
_SUBSUMPTION_PATHS = ("ancestor", "descendant")


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def subsumption_stem(src: VariableNode, tgt: VariableNode, *, graph: Any) -> str:
    """The parent concept's name when one side's concept subsumes the other.

    `""` unless the graph puts the two concepts in a direct ancestor/descendant
    line, so an unrelated or merely sibling pair is left alone.

    This is the rubric's "prefer the coarsest shared concept" applied to the one
    case that actually poses the question — two DIFFERENT concept ids, one a
    subtype of the other. `acute myocardial infarction of anterior wall`
    (434376) against `myocardial infarction` (4329847) must pool as the parent:
    naming it after the subtype claims every row is an anterior-wall infarct,
    which the broad side cannot support.
    """
    if graph is None:
        return ""
    sid, tid = _int_or_none(src.main_id), _int_or_none(tgt.main_id)
    if sid is None or tid is None or sid == tid:
        return ""

    path_type = (graph.explain_path(sid, tid, max_depth=6).get("path_type") or "").strip()
    if path_type not in _SUBSUMPTION_PATHS:
        return ""
    # Mirrors _broader_display_name: 'ancestor' means tgt is above src.
    broader = tid if path_type == "ancestor" else sid
    return _slugify(_concept_name(graph, broader))


# Structural words carry no clinical content, so they must not be what marks a
# name as claiming the subtype: "of" appears in half the vocabulary's labels.
_NAME_STOPWORDS = frozenset({
    "of", "the", "and", "or", "in", "at", "by", "with", "to", "a", "an",
    "for", "on", "from", "yes", "no", "history", "present", "status",
})


def _tokens(text: str) -> frozenset:
    return frozenset(t for t in _slugify(text).split("_") if t) - _NAME_STOPWORDS


def _distinguishing_tokens(narrow_label: str, broad_label: str) -> frozenset:
    """The words that make the subtype narrower than its parent.

    `acute myocardial infarction of anterior wall` minus `myocardial
    infarction` leaves {acute, anterior, wall} — the localisation the parent
    does not record. Computed per pair, so the same subtype yields a different
    set against a different parent: measured against `acute myocardial
    infarction` the surviving tokens are just {anterior, wall}, and a name
    reading `acute_myocardial_infarction` is then correctly left alone.
    """
    return _tokens(narrow_label) - _tokens(broad_label)


def _category_tokens(node: Any) -> frozenset:
    """Words appearing in a variable's own category values."""
    out: set = set()
    for attr in ("category_labels", "original_categories"):
        value = getattr(node, attr, None) or []
        if isinstance(value, str):
            value = re.split(r"[|;]+", value)
        for item in value:
            out |= _tokens(str(item))
    return frozenset(out)


def suggest_harmonized_variable_for_subsumption(
    src: VariableNode,
    tgt: VariableNode,
    current_name: str = "",
    *,
    graph: Any = None,
    verdict_level: Optional[MatchLevel] = None,
) -> str:
    """Parent-concept name for a subtype/parent pair whose name claims the subtype.

    The pooled variable must be one BOTH cohorts can emit, which is a question
    about derivability rather than ontology depth. Two shapes, opposite answers:

    * The parent is a bare presence flag. `ant_mi` (acute MI of anterior wall,
      yes/no) against a cohort recording any-MI history: `ant_mi = no` does not
      mean no infarct, and `MI = yes` does not mean anterior, so neither side
      derives the other. Pool at the parent — naming it for the anterior wall
      asserts a localisation the other cohort never recorded.

    * The parent ENCODES the subtype among its categories. TIME-CHF's `rales`
      (upper zone / middle zone / lower zone / no) against a cohort recording
      only `lower_rales` (yes/no): the multi-class side recodes down to the
      subtype cleanly, the binary side cannot reconstruct the zones. Pool at
      the SUBTYPE, and leave such a name alone — which is what the category
      check below is for.

    `""` in every other case too, including a subtype/parent pair already named
    something reasonable. That restraint is the point. An earlier version
    rewrote every pair the ontology could order and destroyed names the model
    had got right: `loop_diuretic_percent_target_dose` became
    `sulfonamides_plain_mg`, losing both the information axis and the unit, and
    `kidney_disease_history_yes_no` lost its temporal qualifier. The ontology
    knows which concept is broader; it does not know what the pooled column is
    *about*, and must not overrule a name that is not making the over-claim
    being repaired.

    `verdict_level` is optional so a post-hoc pass over finished output, which
    has a status string rather than a MatchLevel, can call it too.
    """
    if verdict_level is not None and verdict_level == MatchLevel.NOT_APPLICABLE:
        return ""
    if not _clean_cell(current_name):
        return ""
    if graph is None:
        return ""

    sid, tid = _int_or_none(src.main_id), _int_or_none(tgt.main_id)
    if sid is None or tid is None or sid == tid:
        return ""
    path_type = (graph.explain_path(sid, tid, max_depth=6).get("path_type") or "").strip()
    if path_type not in _SUBSUMPTION_PATHS:
        return ""

    broad_id, narrow_id = (tid, sid) if path_type == "ancestor" else (sid, tid)
    broad_label = _concept_name(graph, broad_id)
    narrow_label = _concept_name(graph, narrow_id)
    if not broad_label or not narrow_label:
        return ""

    distinguishing = _distinguishing_tokens(narrow_label, broad_label)
    if not distinguishing or not (_tokens(current_name) & distinguishing):
        return ""

    broad_side = tgt if broad_id == tid else src

    # Only a BINARY parent can be shown to be incapable of recording the
    # subtype: a presence flag has two values and neither is the subtype, so a
    # subtype name really does claim more than the column holds. Every other
    # shape might encode it and is left alone — a multi-class parent whose
    # categories are the subtypes (rales: upper/middle/lower zone), and equally
    # a free-text parent, which the dictionary cannot rule out: `descano` is a
    # qualitative ECG description carrying no categories at all, and nothing
    # here can say whether a pathological Q wave is written into it.
    if broad_side.statistical_type != StatisticalType.BINARY:
        return ""

    # Belt and braces: a binary parent whose two values somehow name the
    # subtype is still encoding it.
    if _category_tokens(broad_side) & distinguishing:
        return ""

    replacement = _with_axis(_slugify(broad_label), _axis_suffix(src, tgt))
    return replacement if replacement != _clean_cell(current_name) else ""


# ─────────────────────────────────────────────────────────────────────────────
# Reading the name back out of an output row
# ─────────────────────────────────────────────────────────────────────────────

_MISSING = ("", "nan", "none")


def _clean_cell(value: Any) -> str:
    """A CSV cell as text, with pandas' stand-ins for empty read as empty."""
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in _MISSING else text


def slugify_name(text: Any) -> str:
    """Public spelling of the snake_case rule every harmonized name follows."""
    return _slugify(_clean_cell(text))


def harmonized_name_from_row(row: Any) -> str:
    """The one harmonized name for an output row, wherever it was written.

    Three places have held this name, and every reader must agree on which
    wins, or the same row joins under two different keys:

    1. the ``harmonized_variable`` column — authoritative, written by run.py;
    2. the ``Mapping Description`` JSON blob — where it lived before the column
       existed, so every output already on disk carries it only here;
    3. ``derived_variable_name`` — last resort, and only in outputs written
       before run.py started dropping that column. It named the *formula*
       (``BSA_DuBois`` vs ``BSA_Mosteller``), not the concept, so it stands in
       only when neither namer produced anything, and is slugified on the way
       out so the column keeps one casing convention.
    """
    value = _clean_cell(row.get("harmonized_variable"))
    if value:
        return value

    raw = row.get("Mapping Description")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            value = _clean_cell(payload.get("harmonized_variable"))
            if value:
                return value

    return slugify_name(row.get("derived_variable_name"))
