"""
verdict.py — Immutable contracts for the matching pipeline.

These dataclasses define the data flow between pipeline stages:

  StructuralEvidence  ← produced by handlers in constraints.py
  LLMEvidence         ← produced by llm_call.py (when LLM is consulted)
  Verdict             ← produced once, by policy.decide(), never mutated

"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from .data_model import MatchLevel, TransformationType, ContextMatchType




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
    ceiling: Optional[MatchLevel] = None
    """Best level the representation allows, set when a handler found a fact
    the LLM cannot override (units that need a factor, units of different
    dimensions). None means the handler deferred: the LLM verdict stands."""

# Outcome of consulting the LLM, kept apart from its verdict: a failure to get
# an answer must never read as the LLM rejecting the pair.
LLM_OK = "ok"                     # the LLM answered with a valid verdict
LLM_NOT_REACHED = "not_reached"   # API errors / timeouts on every attempt
LLM_UNREADABLE = "unreadable"     # answered, but no usable verdict in the reply
LLM_SKIPPED = "skipped"           # deliberately not asked (source-claim early exit)
LLM_FAILED = (LLM_NOT_REACHED, LLM_UNREADABLE)


@dataclass(frozen=True)
class LLMEvidence:
    verdict: str
    confidence: float
    reason: str = ""
    transform: str = ""
    transform_direction: str = ""
    harmonized_variable: str = ""

    VALID_VERDICTS = ("COMPLETE", "COMPATIBLE", "PARTIAL", "IMPOSSIBLE")

    # Logprob-derived uncertainty evidence
    logprob_usable: bool = False
    logprob_distribution_type: str = ""   # complete_four_class | observed_alternatives | unusable
    logprob_observability: float = 0.0    # observed class codes / 4

    logprob_dist: dict = field(default_factory=dict)
    logprob_confidence: float = 0.0
    logprob_top_label: str = ""
    logprob_top_prob: float = 0.0
    logprob_runner_up: str = ""
    logprob_margin: float = 0.0
    logprob_raw_margin: float = 0.0

    confidence_source: str = ""

    status: str = LLM_OK
    """LLM_OK, LLM_NOT_REACHED, LLM_UNREADABLE or LLM_SKIPPED. Only an LLM_OK
    evidence carries a verdict; the others carry the reason in `reason`."""

    def __post_init__(self):
        # An answer without a known verdict is unreadable, not a rejection.
        if self.status == LLM_OK and self.verdict not in self.VALID_VERDICTS:
            object.__setattr__(self, "status", LLM_UNREADABLE)
            object.__setattr__(self, "reason", f"unknown verdict {self.verdict!r}. {self.reason}".strip())
        if self.status != LLM_OK:
            object.__setattr__(self, "verdict", "")


_CTX_TYPE_TO_VERDICT = {
    ContextMatchType.EXACT.value:           "COMPLETE",
    ContextMatchType.COMPATIBLE.value:      "COMPATIBLE",
    ContextMatchType.SUBSUMED.value:        "COMPATIBLE",
    ContextMatchType.PARTIAL.value:         "PARTIAL",
    ContextMatchType.NOT_APPLICABLE.value:  "IMPOSSIBLE",
}


def _as_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def llm_evidence_from_tuple(parsed_tuple) -> LLMEvidence:
    """LLMEvidence from llm_call's (matched, ctx_type, conf, reason_json) tuple.

    `conf` is already logprob-based confidence if llm_call.py applied
    _apply_logprob_confidence(). A tuple with matched=None (or a pending
    context type) is a failure to get a verdict, recorded as such.
    """
    matched, ctx_type, conf, reason_json = parsed_tuple

    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
        if not isinstance(d, dict):
            d = {}
    except (json.JSONDecodeError, TypeError):
        d = {"reason": str(reason_json)}

    if matched is None or ctx_type == ContextMatchType.PENDING.value:
        reason = str(d.get("reason") or reason_json or "no answer from the LLM")
        # Failures written before llm_status existed: retries exhausted in
        # assess() or an exception in the call were API problems.
        not_reached = (str(d.get("status", "")).upper() == "NOT_REACHED"
                       or reason.startswith("pending_after")
                       or "eval_pair_exception" in reason)
        return LLMEvidence(verdict="", confidence=0.0, reason=reason,
                           status=LLM_NOT_REACHED if not_reached else LLM_UNREADABLE)

    if matched is False or ctx_type == ContextMatchType.NOT_APPLICABLE.value:
        verdict = "IMPOSSIBLE"
    else:
        verdict = _CTX_TYPE_TO_VERDICT.get(ctx_type, "")   # unknown -> unreadable

    return LLMEvidence(
        verdict=verdict,
        confidence=_as_float(d.get("llm_self_reported_confidence"), _as_float(conf)),
        reason=d.get("reason", "") or "",
        transform=d.get("transform", "") or "",
        transform_direction=d.get("alignment_direction", "") or "",
        harmonized_variable=(d.get("harmonized_variable") or "").strip(),
        confidence_source=d.get("confidence_source") or "",
    )


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


_REDUNDANT_DETAIL_KEYS = frozenset({
    "mapping_relation",                     # own column
    "source_type", "target_type",           # own columns
    "source_unit", "target_unit",           # own columns
})
# transform_direction is deliberately NOT stripped. It used to be, on the
# grounds that it already travels in LLMEvidence — but a verdict decided
# without the LLM has no LLMEvidence, and policy now derives a direction for
# those from the mapping relation. Keeping the key here makes the direction
# readable in one place for every row, LLM-judged or not.


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
