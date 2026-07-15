"""
Each policy is a pure function which takes mapping mode, structural evidence, llm evidence and timepoint information to draw  Verdict.
"""

from __future__ import annotations
from typing import Optional

from .verdict import StructuralEvidence, LLMEvidence, Verdict, TimepointInfo
from .data_model import MatchLevel, TransformationType, MappingType, MappingRelation, ContextMatchType
import json

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





def _extra_info_from_llm(extra: dict, llm: Optional[LLMEvidence]) -> dict:
    out = dict(extra)

    if llm is not None:
        if llm.transform_direction:
            out["transform_direction"] = llm.transform_direction

        if llm.transform:
            out["llm_transform"] = llm.transform

        out["llm_verdict"] = llm.verdict
        out["llm_confidence"] = llm.confidence

        uncertainty = {
            "logprob_usable": getattr(llm, "logprob_usable", False),
            "distribution_type": getattr(llm, "logprob_distribution_type", ""),
            "observability": getattr(llm, "logprob_observability", 0.0),
            "logprob_confidence": getattr(llm, "logprob_confidence", 0.0),
            "top_label": getattr(llm, "logprob_top_label", ""),
            "top_prob": getattr(llm, "logprob_top_prob", 0.0),
            "runner_up": getattr(llm, "logprob_runner_up", ""),
            "margin": getattr(llm, "logprob_margin", 0.0),
            "raw_margin": getattr(llm, "logprob_raw_margin", 0.0),
            "confidence_source": getattr(llm, "confidence_source", ""),
            "dist": getattr(llm, "logprob_dist", {}),
        }

        out["llm_uncertainty"] = json.dumps(uncertainty, ensure_ascii=False)

        hv = (getattr(llm, "harmonized_variable", None) or "").strip()
        if hv:
            out["harmonized_variable"] = hv

    return out

    
def _demote_hierarchical(s: StructuralEvidence) -> tuple[MatchLevel, TransformationType, str]:
    # print(s)
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

# NE: neural matching only. The LLM is the primary semantic judge;

def _ne_decide(s: StructuralEvidence,
               llm: Optional[LLMEvidence],
               tp: TimepointInfo,
               llm_use:bool) -> Verdict:

    # No LLM consulted → structural verdict is final.
    if llm is None:
        level = MatchLevel.NOT_APPLICABLE if llm_use and s.level == MatchLevel.PARTIAL  else s.level
        return _build_verdict(level, s.transformation, s.reason, tp, s.extra)

    if llm.verdict == "IMPOSSIBLE":
        return _build_verdict(
            MatchLevel.NOT_APPLICABLE,
            TransformationType.MANUAL_REVIEW,
            f"LLM rejected pair: {llm.reason}".strip(),
            tp, _extra_info_from_llm(s.extra, llm),
        )

    if llm.verdict == "COMPLETE":
        # Preserve a structural COMPATIBLE verdict; otherwise allow IDENTICAL cap.
        if s.level == MatchLevel.COMPATIBLE:
            capped = MatchLevel.COMPATIBLE
        else:
            # Lower int = better match; min picks the better of IDENTICAL vs structural.
            capped = min(MatchLevel.IDENTICAL, s.level, key=int)
        transformation = llm.transform or TransformationType.NONE   # COMPLETE ⇒ no transform
        reason = s.reason or "Handler verdict"
        if llm.reason:
            reason = f"{reason} | LLM confirmed: {llm.reason}"
        return _build_verdict(capped, transformation, reason, tp, _extra_info_from_llm(s.extra, llm))
       

    if llm.verdict == "COMPATIBLE":
        # The class that was unreachable before this fix.
        return _build_verdict(
            MatchLevel.COMPATIBLE,
            _compatible_transformation(s),
            f"LLM compatible: {llm.reason}".strip(),
            tp, _extra_info_from_llm(s.extra, llm),
        )

    if llm.verdict == "PARTIAL":
        return _build_verdict(
            MatchLevel.PARTIAL,
            TransformationType.MANUAL_REVIEW,
            f"LLM partial: {llm.reason}".strip(),
            tp,  _extra_info_from_llm(s.extra, llm),
        )

    # Unknown verdict (shouldn't happen — LLMEvidence.__post_init__
    # normalizes invalid verdicts to IMPOSSIBLE — but be safe).
    return _build_verdict(s.level, s.transformation, s.reason, tp, s.extra)



# OEH/OEC: ontology gives the structural answer. LLM is consulted only
# when the ontology evidence is ambiguous, and acts as a tiebreaker on
# additional context.


def _symbolic_neural_with_llm_decide(s: StructuralEvidence,
                               llm: Optional[LLMEvidence],
                               tp: TimepointInfo,
               llm_use:bool) -> Verdict:
    if llm is None:
        level, transformation, reason = _demote_hierarchical(s)
        level = MatchLevel.NOT_APPLICABLE if (llm_use and s.level == MatchLevel.PARTIAL and transformation != TransformationType.DERIVATION)  else level
        return _build_verdict(level, transformation, reason, tp, s.extra)
    
    if llm.verdict == "IMPOSSIBLE":
        return _build_verdict(
            MatchLevel.NOT_APPLICABLE,
            TransformationType.MANUAL_REVIEW,
            f"LLM rejected pair: {llm.reason}".strip(),
            tp, _extra_info_from_llm(s.extra, llm),
        )

    if llm.verdict == "COMPLETE":
        if s.level == MatchLevel.COMPATIBLE:
            capped = MatchLevel.COMPATIBLE
        else:
            capped = min(MatchLevel.IDENTICAL, s.level, key=int)
        transformation = llm.transform or TransformationType.NONE   # COMPLETE ⇒ no transform
        reason = s.reason or "Ontology match"
        if llm.reason:
            reason = f"{reason} | LLM confirmed: {llm.reason}"
        # Ontology + LLM both agree → structural verdict survives.
        return _build_verdict(capped, transformation, reason, tp, _extra_info_from_llm(s.extra, llm))

    if llm.verdict == "COMPATIBLE":
        if s.transformation == TransformationType.NONE:
            capped = min(MatchLevel.IDENTICAL, s.level, key=int)
        else:
            capped = MatchLevel.COMPATIBLE
        transformation = llm.transform or _compatible_transformation(s)
        reason = s.reason or "Ontology match"
        if llm.reason:
            reason = f"{reason} | LLM confirmed: {llm.reason}"
        return _build_verdict(
            capped,
            transformation,
            reason,
            tp, _extra_info_from_llm(s.extra, llm),
        )
    if llm.verdict == "PARTIAL":
        return _build_verdict(
            MatchLevel.PARTIAL,
            TransformationType.MANUAL_REVIEW,
            f"LLM partial: {llm.reason}".strip(),
            tp,  _extra_info_from_llm(s.extra, llm),
        )

    return _build_verdict(s.level, s.transformation, s.reason, tp, s.extra)


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

    return _build_verdict(level, transformation, reason, tp, s.extra)



# Helpers


def _compatible_transformation(s: StructuralEvidence) -> TransformationType:
    """Pick the right transformation for an LLM-judged compatible pair.

    Prefer the handler's transformation when it already implies a concrete
    operation (unit conversion, value normalization, aggregation).
    Otherwise default to value normalization, which is the most general
    'compatible-after-transform' label.
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

CLAIMING_VERDICTS = frozenset({"COMPLETE", "COMPATIBLE"})
