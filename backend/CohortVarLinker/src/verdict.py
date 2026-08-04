"""
verdict.py — Immutable contracts for the matching pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from .data_model import MatchLevel, TransformationType




@dataclass(frozen=True)
class StructuralEvidence:
    """Output of the type-specific handler dispatch.

    Represents what the structural rules (units, statistical types, category
    sets, ranges) can determine about a candidate pair, before any LLM call.
    """
    level: MatchLevel
    transformation: TransformationType
    reason: str
    needs_review: bool = False
    """True when the handler could not produce a confident answer
    (e.g. asymmetric units, type mismatch with manual-review transform).
    Used by should_consult_llm() to decide whether the LLM is worth calling.
    """
    extra: Dict[str, Any] = field(default_factory=dict)
    """Handler-specific context: source_unit, target_unit, source_range,
    target_range, categories, etc. Survives into the final Verdict.extra."""


@dataclass(frozen=True)
class LLMEvidence:
    """The LLM's verdict on a candidate pair.

    Carries the raw verdict string ('COMPLETE' | 'COMPATIBLE' | 'PARTIAL'
    | 'IMPOSSIBLE') rather than a ContextMatchType — keeping the LLM's
    semantic protocol explicit and out of the rest of the pipeline.

    """
    verdict: str
    confidence: float
    reason: str = ""
    transform: str = ""
    transform_direction: str = ""

    VALID_VERDICTS = ("COMPLETE", "COMPATIBLE", "PARTIAL", "IMPOSSIBLE")

    def __post_init__(self):
        if self.verdict not in self.VALID_VERDICTS:
            object.__setattr__(self, "verdict", "IMPOSSIBLE")


@dataclass(frozen=True)
class TimepointInfo:
    """Timepoint annotation for a candidate. Informational only —
    does not change level. The old pipeline mixed timepoint annotation
    into the same mutable details dict; here it's a separate record.
    """
    aligned: bool
    source_visit: str = ""
    target_visit: str = ""
    status: str = ""
    """'aligned' | 'mismatch' | 'undetermined'. Splits what `aligned=False`
    used to conflate: a genuine repeated-measures mismatch (baseline vs
    month 6) versus a side that names no protocol timepoint at all (Aachen-HF
    records everything against a generic 'visit date'). `aligned` stays False
    for both so existing log consumers keep working."""
    undetermined_side: Optional[str] = None
    candidate_timepoints: Tuple[str, ...] = ()
    """When one side is undetermined, the exhaustive set of counterpart
    timepoints this pair could resolve to — this row is one of them."""
    resolved_timepoint: str = ""
    """The determinate side's canonical period: which candidate this row
    stands for. Empty unless exactly one side is undetermined."""
    note: str = ""

    @property
    def is_undetermined(self) -> bool:
        return self.status == "undetermined"


# Keys that already exist as their own CSV column or inside the LLMEvidence
# column. Measured identical on every row where both were present, so carrying
# them in the details blob only doubled the file and created two places for the
# same fact to drift apart. harmonized_variable is deliberately NOT here —
# run.py reads it back out of details to decide whether to synthesise one.
_REDUNDANT_DETAIL_KEYS = frozenset({
    "mapping_relation",                     # own column
    "source_type", "target_type",           # own columns
    "source_unit", "target_unit",           # own columns
    "transform_direction",                  # in LLMEvidence
})


@dataclass(frozen=True)
class Verdict:
    """The single, final, immutable decision for a candidate pair.

    Produced once by policy.decide(). Nothing downstream rewrites it.
    """
    level: MatchLevel
    # mapping_relation: str # neural match, symbolic:exactmatch, symbolic:broadmatch, symbolic:narrowmatch, Symbolic:closeMatch
    transformation: TransformationType
    description: str
    timepoint: TimepointInfo
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_tuple(self) -> tuple:
        """Adapter for run.py's existing output schema:
        returns (details_dict, harmonization_status_str).

        Keeps evaluate_cross_study.py's input format unchanged, so
        no downstream consumer needs to know about Verdict at all.
        
        """
        details = {k: v for k, v in self.extra.items()
                   if k not in _REDUNDANT_DETAIL_KEYS}
        details["description"] = self._compose_description()
        details["transformation"] = (
            self.transformation.value
            if isinstance(self.transformation, TransformationType)
            else (self.transformation or TransformationType.NONE.value)
        )
        details["timepoint_transformation"] = (
                TransformationType.TIMEPOINT_ALIGNMENT.value
                if (not self.timepoint.aligned and self.level != MatchLevel.NOT_APPLICABLE)
                else TransformationType.NONE.value
            )
        details["timepoint_aligned"] = "yes" if self.timepoint.aligned else "no"
        if self.timepoint.status:
            details["timepoint_status"] = self.timepoint.status
        # No separate "requires verification" flag: timepoint_status ==
        # 'undetermined' already carries it, and undetermined_timepoint_side
        # says which side. A third field would only be able to disagree.
        if not self.timepoint.aligned and self.level != MatchLevel.NOT_APPLICABLE:
            details["source_timepoint"] = self.timepoint.source_visit
            details["target_timepoint"] = self.timepoint.target_visit
            if self.timepoint.is_undetermined:
                details["undetermined_timepoint_side"] = self.timepoint.undetermined_side or ""
                # Exhaustive: the caller can enumerate every resolution this
                # row stands for instead of assuming baseline.
                details["candidate_timepoints"] = list(self.timepoint.candidate_timepoints)
                if self.timepoint.resolved_timepoint:
                    details["resolved_timepoint"] = self.timepoint.resolved_timepoint
        return details, self.level.to_str()

    def _compose_description(self) -> str:
        desc = self.description.strip()
        if (not self.timepoint.aligned
                and self.level != MatchLevel.NOT_APPLICABLE):
            note = self.timepoint.note.strip() or (
                f"Timepoints differ "
                f"({self.timepoint.source_visit} vs {self.timepoint.target_visit}).")
            desc = f"{desc} {note}".strip() if desc else note
        return desc
