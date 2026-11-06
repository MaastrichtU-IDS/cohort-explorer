"""
Each policy is a pure function which takes mapping mode, structural evidence, llm evidence and timepoint information to draw  Verdict.
"""

from __future__ import annotations
from typing import Optional

from .verdict import StructuralEvidence, LLMEvidence, Verdict, TimepointInfo, LLM_OK, LLM_SKIPPED
from .data_model import MatchLevel, TransformationType, MappingType, MappingRelation, ContextMatchType

def decide(mode: str,
           structural: StructuralEvidence,
           llm_verdit: Optional[LLMEvidence],
           llm_use:bool,
           timepoint: TimepointInfo) -> Verdict:
    
    """Top-level dispatch. The only place mode-conditional code lives."""
    if mode == MappingType.NE.value:
        return _ne_decide(structural, llm_verdit, timepoint,llm_use)
    if mode in (MappingType.OEH.value, MappingType.OEC.value):
        return _symbolic_neural_with_llm_decide(structural, llm_verdit, timepoint, llm_use)
    return _ontology_only_decide(structural, timepoint)


# Penalise broader vs specific pair matching

def _extra_info_from_llm(extra: dict, llm: Optional[LLMEvidence]) -> dict:
    out = dict(extra)

    if llm is not None:
        if llm.transform_direction:
            out["transform_direction"] = llm.transform_direction

        if llm.transform and not out.get("transformation_rule"):
            # The concrete "how" of the transformation. run.py lifts this into
            # the transformation_rule column, leaving transformation_type to
            # hold the category alone. A rule the structural layer computed
            # (a unit factor) is deterministic and is kept over the LLM's text.
            out["transformation_rule"] = llm.transform

        hv = (getattr(llm, "harmonized_variable", None) or "").strip()
        if hv:
            out["harmonized_variable"] = hv

    return out

    
def _demote_hierarchical(s: StructuralEvidence) -> tuple[MatchLevel, TransformationType, str]:

    relation = s.extra.get("mapping_relation", "")
    if MappingRelation.is_hierarchical(relation) and s.level in (
        MatchLevel.IDENTICAL, MatchLevel.COMPATIBLE
    ):
        transformation = (TransformationType.MANUAL_REVIEW
                          if s.transformation == TransformationType.NONE
                          else s.transformation)
        # phrase = MappingRelation.humanize(relation)
        reason = f"{s.reason}. Concepts are related but one is more generic, the other more specific; manual review recommended"
        return MatchLevel.PARTIAL, transformation, reason
    return s.level, s.transformation, s.reason


# A PARTIAL that no LLM ever saw still has a knowable direction: the candidate
# generator already stated which side is finer. A broadMatch means the target is
# the broader concept (torsemide -> "sulfonamides, plain"); a narrowMatch means
# the target is the narrower one (cerebrovascular disease -> cerebrovascular
# accident). Those map onto the ALIGNMENT DIRECTION vocabulary the LLM uses, so
# structural and LLM verdicts stay comparable downstream.
_HIERARCHICAL_DIRECTION = {
    MappingRelation.SymbolicBroadMatch.value:  "source to target",   # source finer
    MappingRelation.SymbolicNarrowMatch.value: "target to source",   # target finer
}


def _structural_direction(s: StructuralEvidence,
                          level: MatchLevel,
                          transformation: TransformationType) -> str:
    """transform_direction for a verdict decided without the LLM."""
    if level != MatchLevel.PARTIAL:
        return ""
    if transformation == TransformationType.DERIVATION or _is_derivation(s):
        # Both sides feed a third, derived variable.
        return "both for derivation"
    relation = (s.extra.get("mapping_relation") or "").strip().lower()
    return _HIERARCHICAL_DIRECTION.get(relation, "")


def _extra_with_direction(s: StructuralEvidence,
                          level: MatchLevel,
                          transformation: TransformationType) -> dict:
    """s.extra plus a transform_direction, when one can be derived structurally."""
    out = dict(s.extra)
    direction = _structural_direction(s, level, transformation)
    if direction and not out.get("transform_direction"):
        out["transform_direction"] = direction
    return out

# NE: neural matching only. The LLM is the primary semantic judge;

def _ne_decide(s: StructuralEvidence,
               llm: Optional[LLMEvidence],
               tp: TimepointInfo,
               llm_use:bool) -> Verdict:

    if llm is None or llm.status != LLM_OK:
        return _decide_without_llm_verdict(s.level, s.transformation, s.reason, s, llm, tp, llm_use)

    return _decide_with_llm(s, llm, tp)



# OEH/OEC: ontology gives the structural answer. LLM is consulted only
# when the ontology evidence is ambiguous, and acts as a tiebreaker on
# additional context.


def _symbolic_neural_with_llm_decide(s: StructuralEvidence,
                               llm: Optional[LLMEvidence],
                               tp: TimepointInfo,
               llm_use:bool) -> Verdict:
    if llm is None or llm.status != LLM_OK:
        level, transformation, reason = _demote_hierarchical(s)
        return _decide_without_llm_verdict(level, transformation, reason, s, llm, tp, llm_use)

    return _decide_with_llm(s, llm, tp)


def _decide_without_llm_verdict(level: MatchLevel,
                                transformation: TransformationType,
                                reason: str,
                                s: StructuralEvidence,
                                llm: Optional[LLMEvidence],
                                tp: TimepointInfo,
                                llm_use: bool) -> Verdict:
    """Verdict when no usable LLM verdict exists; records why in llm_status.

    Only a candidate the pipeline deliberately did not ask about (source-claim
    early exit) is dropped. When the LLM failed, or a verdict is missing that
    should exist, the structural verdict stands, flagged: losing the semantic
    check is not evidence that the pair does not match.
    """
    extra = _extra_with_direction(s, level, transformation)

    if not llm_use:
        extra["llm_status"] = "not_configured"
        return _build_verdict(level, transformation, reason, tp, extra)

    if llm is None and not should_consult_llm(s):
        extra["llm_status"] = "not_needed"           # structural answer was decisive
        return _build_verdict(level, transformation, reason, tp, extra)

    if llm is not None and llm.status == LLM_SKIPPED:
        extra["llm_status"] = LLM_SKIPPED
        if s.level == MatchLevel.PARTIAL and transformation != TransformationType.DERIVATION:
            level = MatchLevel.NOT_APPLICABLE
            reason = f"{reason} | {llm.reason}".strip(" |")
        return _build_verdict(level, transformation, reason, tp,
                              {**_extra_with_direction(s, level, transformation), "llm_status": LLM_SKIPPED})

    # The LLM was needed and gave no usable verdict.
    status = llm.status if llm is not None else "missing"
    why = (llm.reason if llm is not None and llm.reason else "no LLM verdict recorded for this pair")
    extra["llm_status"] = status
    return _build_verdict(
        level, transformation,
        f"{reason} | Semantic check not performed ({why}); structural verdict kept, review recommended.".strip(" |"),
        tp, extra,
    )


# The LLM decides meaning; the structural layer bounds what the representation
# allows. Shared by NE and OEH/OEC: once the LLM has spoken, the mode no longer
# changes how its verdict and the structural evidence combine.

_LLM_LEVEL = {
    "COMPLETE":   MatchLevel.IDENTICAL,
    "COMPATIBLE": MatchLevel.COMPATIBLE,
    "PARTIAL":    MatchLevel.PARTIAL,
    "IMPOSSIBLE": MatchLevel.NOT_APPLICABLE,
}


def _worse(a: MatchLevel, b: MatchLevel) -> MatchLevel:
    """The weaker of two levels (higher int = weaker match)."""
    return a if int(a) >= int(b) else b


def _binding_ceiling(s: StructuralEvidence) -> Optional[MatchLevel]:
    """Best level the structural evidence allows, or None if it deferred.

    A structural COMPATIBLE is binding: the handler found a recoding or a
    conversion, so values cannot merge as-is whatever the LLM says.
    """
    ceiling = s.ceiling
    if s.level == MatchLevel.COMPATIBLE:
        ceiling = MatchLevel.COMPATIBLE if ceiling is None else _worse(ceiling, MatchLevel.COMPATIBLE)
    return ceiling


def _transformation_for(level: MatchLevel, s: StructuralEvidence) -> TransformationType:
    """Transformation consistent with the final level."""
    if level == MatchLevel.IDENTICAL:
        return TransformationType.NONE
    if level == MatchLevel.COMPATIBLE:
        return _compatible_transformation(s)
    return TransformationType.MANUAL_REVIEW


def _decide_with_llm(s: StructuralEvidence, llm: LLMEvidence, tp: TimepointInfo) -> Verdict:
    extra = _extra_info_from_llm(s.extra, llm)
    extra["llm_status"] = LLM_OK

    if llm.verdict == "IMPOSSIBLE":
        return _build_verdict(MatchLevel.NOT_APPLICABLE, TransformationType.MANUAL_REVIEW,
                              f"LLM rejected pair: {llm.reason}".strip(), tp, extra)

    if llm.verdict == "PARTIAL":
        return _build_verdict(MatchLevel.PARTIAL, TransformationType.MANUAL_REVIEW,
                              f"LLM partial: {llm.reason}".strip(), tp, extra)

    level = _LLM_LEVEL[llm.verdict]
    reason = s.reason or "Structural check"
    if llm.reason:
        reason = f"{reason} | LLM {'confirmed' if llm.verdict == 'COMPLETE' else 'compatible'}: {llm.reason}"

    # The structural layer rejected the pair; the LLM accepts it. Neither is
    # trusted over the other (the structural NA is often an annotation error,
    # e.g. a day count typed as free text), so a human decides.
    if s.level == MatchLevel.NOT_APPLICABLE:
        extra["structural_llm_disagreement"] = True
        return _build_verdict(MatchLevel.PARTIAL, TransformationType.MANUAL_REVIEW,
                              f"{reason} | Structural and LLM evidence disagree; manual review required.",
                              tp, extra)

    ceiling = _binding_ceiling(s)
    if ceiling is not None and int(ceiling) > int(level):
        level = ceiling
        if s.extra.get("unit_range_check") == "contradicted":
            cause = "unit/value-range conflict"
        else:
            cause = s.extra.get("unit_relation") or "structural evidence"
        reason = f"{reason} | Capped at {level.to_str()} by {str(cause).replace('_', ' ')}."

    return _build_verdict(level, _transformation_for(level, s), reason, tp, extra)


def _ontology_only_decide(s: StructuralEvidence, tp: TimepointInfo) -> Verdict:
    """OO mode: no LLM, structural verdict is final by definition."""
    # level, transformation, reason = _demote_hierarchical(s)
    # return _build_verdict(level, transformation, reason, tp, s.extra)
    level, transformation, reason = _demote_hierarchical(s)

    ctx_type = s.extra.get("context_match_type")

    if (
        ctx_type == ContextMatchType.PENDING.value
    ):
        level = MatchLevel.NOT_APPLICABLE
        transformation = TransformationType.MANUAL_REVIEW
        reason = (
            f"{reason} | Ontology-only mode could not verify full concept-context equivalence; "
            "manual review required."
        ).strip()

    return _build_verdict(level, transformation, reason, tp,
                          _extra_with_direction(s, level, transformation))



# Helpers


def _compatible_transformation(s: StructuralEvidence) -> TransformationType:
    """Pick the right transformation for an LLM-judged compatible pair.

    Prefer the handler's transformation when it already implies a concrete
    operation (unit conversion, value normalization, aggregation).

    When it does not, two differing units are themselves the concrete operation:
    the handler returns a non-concrete transformation whenever it cannot verify
    unit equivalence ("units: (ucum:g/l); cannot verify equivalence"), and the
    LLM then confirms the pair is compatible precisely because the conversion is
    deterministic. Falling through to value normalization in that case labelled
    253 continuous conversions -- g/L to g/dL, mmol/L to mg/dL, L/L to %, umol/L
    to mg/dL -- as categorical recoding, which is not merely a wrong name: a
    consumer reading transformation_type to decide how to merge would look for
    category mappings on a variable that has no categories.

    Only differing units qualify. Equal units mean the compatibility came from
    something else, and value normalization remains the honest general label.
    """
    concrete = (
        TransformationType.UNIT_CONVERSION,
        TransformationType.VALUE_NORMALIZATION,
        TransformationType.AGGREGATION_OR_EXPANSION,
        TransformationType.UNIT_ALIGNMENT,
        TransformationType.BINARY_EXTRACTION,
        TransformationType.DERIVATION,
    )
    if s.transformation in concrete:
        return s.transformation

    # When the unit check ran, it is authoritative: equivalent spellings
    # (ng/mL vs ug/L) are not a conversion, a fixed factor is.
    relation = s.extra.get("unit_relation")
    if relation in ("scalable", "molar_mass_dependent"):
        return TransformationType.UNIT_CONVERSION
    if relation in ("equivalent", "incommensurable"):
        return TransformationType.VALUE_NORMALIZATION

    def _unit(key: str) -> str:
        value = s.extra.get(key)
        text = "" if value is None else str(value).strip()
        return "" if text.lower() in ("nan", "none", "null") else text

    src_unit, tgt_unit = _unit("source_unit"), _unit("target_unit")
    if src_unit and tgt_unit and src_unit.lower() != tgt_unit.lower():
        return TransformationType.UNIT_CONVERSION

    return TransformationType.VALUE_NORMALIZATION


def _build_verdict(level: MatchLevel,
                   transformation: TransformationType,
                   reason: str,
                   tp: TimepointInfo,
                   extra: dict) -> Verdict:
    return Verdict(
        level=level,
        transformation=transformation,
        description=reason,
        timepoint=tp,
        extra=dict(extra),  # defensive copy: caller can't mutate verdict
    )



# LLM-call gating — decides whether to spend an LLM call on a candidate.

def _is_derivation(s: StructuralEvidence) -> bool:
    transformation = (
        s.transformation.value
        if isinstance(s.transformation, TransformationType)
        else str(s.transformation)
    )

    return (
        transformation == TransformationType.DERIVATION.value
        or bool(s.extra.get("is_derived_pair"))
        or "derived variable" in str(s.extra.get("transformation_rule", "")).lower()
    )

def should_consult_llm(s: StructuralEvidence) -> bool:
    """Symmetric pre-filter: skip LLM whenever the structural answer is
    decisive in either direction.
    """

    if _is_derivation(s):
        return False
    if s.needs_review:
        return True
    # Decisive answers — no LLM needed.
    if s.level in (MatchLevel.IDENTICAL,
                    MatchLevel.COMPATIBLE,
                    MatchLevel.NOT_APPLICABLE):
        return False
    # PARTIAL is genuinely ambiguous — let the LLM weigh in.
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Source-claim early-exit support
#
# When a source variable is matched to a target with verdict COMPLETE or
# COMPATIBLE, we can skip the LLM on the same source's remaining ambiguous
# candidates ("source claim"). This requires a defensible per-source
# ordering so the *best* candidate gets the LLM first; otherwise a mediocre
# COMPATIBLE could claim the source before the LLM ever sees an IDENTICAL
# one further down the list.
# Ordering is built entirely from structural-layer signals that already
# exist on the candidate row.
# ─────────────────────────────────────────────────────────────────────────────


# Lower rank = considered first. Hierarchical relations are pushed last
# because _demote_hierarchical will demote them in policy anyway.
_RELATION_RANK = {
    MappingRelation.SymbolicExactMatch.value:   0,
    MappingRelation.SymbolicCloseMatch.value:   1,
    MappingRelation.NeuralMatch.value:          2,
    MappingRelation.SymbolicNarrowMatch.value:  3,
    MappingRelation.SymbolicBroadMatch.value:   4,
}

# Lower rank = considered first.
_CTX_RANK = {
    ContextMatchType.EXACT.value:           0,
    ContextMatchType.SUBSUMED.value:        1,
    ContextMatchType.COMPATIBLE.value:      2,
    ContextMatchType.PARTIAL.value:         3,
    ContextMatchType.PENDING.value:         4,
    ContextMatchType.NOT_APPLICABLE.value:  5,
}


def llm_priority_key(structural: StructuralEvidence) -> tuple:
    """Per-candidate ordering for source-claim early-exit.

    Returns a tuple suitable for ``sorted(key=...)``. Smaller values come
    first, i.e. the candidate most likely to be the source's best match
    gets the LLM call first.

    Ordering rationale (all already-computed structural signals):
      1. ``needs_review=False`` before ``needs_review=True``. A handler
         that produced a confident PARTIAL is more likely to flip to
         COMPATIBLE/COMPLETE under the LLM than one that punted.
      2. SymbolicExactMatch → SymbolicCloseMatch → NeuralMatch →
         hierarchical. This is the candidate generator's own ranking.
      3. EXACT → SUBSUMED → COMPATIBLE → PARTIAL → PENDING context.
      4. sim_score as a final tiebreaker only (descending).
    """
    extra = structural.extra or {}
    relation = (extra.get("mapping_relation") or "").strip().lower()
    # context_match_type is an IntEnum value (int), not a string — leave as-is.
    ctx_type = extra.get("context_match_type")
    try:
        sim_score = float(extra.get("sim_score") or 0.0)
    except (TypeError, ValueError):
        sim_score = 0.0

    return (
        1 if structural.needs_review else 0,
        _RELATION_RANK.get(relation, 99),
        _CTX_RANK.get(ctx_type, 99),
        -sim_score,
    )


# Verdicts that claim the source and shut down further LLM calls for it.
# PARTIAL is deliberately excluded — a PARTIAL doesn't harmonize the
# source, so a later candidate might still produce a real match.
CLAIMING_VERDICTS = frozenset({"COMPLETE", "COMPATIBLE"})
