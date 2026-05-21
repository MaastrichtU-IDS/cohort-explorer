"""
verdict.py — Immutable contracts for the matching pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
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
        details = dict(self.extra)
        details["description"] = self._compose_description()
        details["transformation"] = (
            self.transformation.value
            if isinstance(self.transformation, TransformationType)
            else (self.transformation or TransformationType.NONE.value)
        )
        details["timepoint_aligned"] = "yes" if self.timepoint.aligned else "no"
        if not self.timepoint.aligned and self.level != MatchLevel.NOT_APPLICABLE:
            details["source_timepoint"] = self.timepoint.source_visit
            details["target_timepoint"] = self.timepoint.target_visit
        return details, self.level.to_str()

    def _compose_description(self) -> str:
        desc = self.description.strip()
        if (not self.timepoint.aligned
                and self.level != MatchLevel.NOT_APPLICABLE):
            note = (f"Timepoints differ "
                    f"({self.timepoint.source_visit} vs {self.timepoint.target_visit}).")
            desc = f"{desc} {note}".strip() if desc else note
        return desc
