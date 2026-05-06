"""
policy.py — Per-mode decision functions.

Each policy is a pure function: (mode, structural, llm, timepoint) -> Verdict.
The level is computed exactly once.

The cap collapsed two distinct LLM verdicts (COMPATIBLE, PARTIAL) into a
single output level (PARTIAL), making MatchLevel.COMPATIBLE structurally
unreachable in NE+LLM mode. This module replaces those three steps with
one explicit decision per mode.
"""

from __future__ import annotations
from typing import Optional

from .verdict import StructuralEvidence, LLMEvidence, Verdict, TimepointInfo
from .data_model import MatchLevel, TransformationType, MappingType, MappingRelation, ContextMatchType


def decide(mode: str,
           structural: StructuralEvidence,
           llm: Optional[LLMEvidence],
           timepoint: TimepointInfo) -> Verdict:
    
    """Top-level dispatch. The only place mode-conditional code lives."""
    if mode == MappingType.NE.value:
        return _ne_decide(structural, llm, timepoint)
    if mode in (MappingType.OEH.value, MappingType.OEC.value):
        return _symbolic_neural_with_llm_decide(structural, llm, timepoint)
    return _ontology_only_decide(structural, timepoint)


# Penalise broader vs specific pair matching

def _extra_info_from_llm(extra: dict, llm: Optional[LLMEvidence]) -> dict:
    out = dict(extra)

    if llm is not None:
        if llm.transform_direction:
            out["transformation_direction"] = llm.transform_direction
        if llm.transform:
            out["llm_transform"] = llm.transform
        out["llm_verdict"] = llm.verdict
        out["llm_confidence"] = llm.confidence
        print(out)
    return out
def _demote_hierarchical(s: StructuralEvidence) -> tuple[MatchLevel, TransformationType, str]:
    print(s)
    relation = s.extra.get("mapping_relation", "")
    if MappingRelation.is_hierarchical(relation) and s.level in (
        MatchLevel.IDENTICAL, MatchLevel.COMPATIBLE
    ):
        transformation = (TransformationType.MANUAL_REVIEW
                          if s.transformation == TransformationType.NONE
                          else s.transformation)
        # phrase = MappingRelation.humanize(relation)
        reason = f"{s.reason} | Concepts are related but one is more generic, the other more specific; manual review recommended"
        return MatchLevel.PARTIAL, transformation, reason
    return s.level, s.transformation, s.reason

# NE: neural matching only. The LLM is the primary semantic judge;

def _ne_decide(s: StructuralEvidence,
               llm: Optional[LLMEvidence],
               tp: TimepointInfo) -> Verdict:

    # No LLM consulted → structural verdict is final.
    if llm is None:
        return _build_verdict(s.level, s.transformation, s.reason, tp, s.extra)

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
                               tp: TimepointInfo) -> Verdict:
    if llm is None:
        level, transformation, reason = _demote_hierarchical(s)
        
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


def should_consult_llm(s: StructuralEvidence) -> bool:
    """Symmetric pre-filter: skip LLM whenever the structural answer is
    decisive in either direction.
    """
    if s.needs_review:
        return True
    # Decisive answers — no LLM needed.
    if s.level in (MatchLevel.IDENTICAL,
                    MatchLevel.COMPATIBLE,
                    MatchLevel.NOT_APPLICABLE):
        return False
    # PARTIAL is genuinely ambiguous — let the LLM weigh in.
    return True
