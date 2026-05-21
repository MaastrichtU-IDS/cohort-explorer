"""
constraints.py — Pure Structural Constraint Pipeline

Architecture:
  Context scoring (concept equivalence, embedding similarity) is computed
  upstream in resolve_matches(). By the time pairs reach this pipeline,
  context_match_type and sim_score are already populated.

  This pipeline performs STRUCTURAL validation only:
    1. ContextGate     — pre-gate using pre-computed context scores
    2. StatisticalLogic — type-specific structural analysis (units, categories, ranges)
    3. SubsumedCap      — downgrade to PARTIAL if context was subsumed/close_match

  Decision flow:
    context mismatch  → NOT_APPLICABLE (stop, no structural analysis)
    context exact     → continue to structural analysis, no cap
    context subsumed  → continue, cap to PARTIAL after structural analysis
    no context data   → continue (neither side has context)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Dict, Any, Tuple, List, Optional, Set, Protocol, runtime_checkable
from unicodedata import category

import numpy as np
from .config import settings
from sklearn.metrics.pairwise import cosine_similarity
from .fuzz_match import FuzzyMatcher
from .data_model import (
    VariableNode,
    StatisticalType,
    TransformationType,
    MatchLevel,
    ContextMatchType,
    DATA_TYPE,
    MappingType,
    MappingRelation
)
from .utils import is_absolute_vs_percent_dose
# from .verdict import StructuralEvidence
_GENERIC_BOOL = frozenset({"0", "1"})
_BOOL_POS = frozenset({"1", "yes", "true", "y", "positive", "present", "on"})
_BOOL_NEG = frozenset({"0", "no", "false", "n", "negative", "absent", "off"})

def _canonical_cats(cats) -> frozenset:
    out = set()
    for c in cats:
        lc = str(c).lower().strip()
        out.add("1" if lc in _BOOL_POS else ("0" if lc in _BOOL_NEG else lc))
    return frozenset(out)


# 1. Protocols & Data Classes


@runtime_checkable
class MatcherProtocol(Protocol):
    """Interface for categorical subsumption checking."""
    def check_categorical_subsumption(
        self,
        category_concept: Tuple[str, str],
        main_concept: Tuple[str, str],
        target_study: str,
    ) -> Tuple[bool, Optional[str]]: ...

# @dataclass
# class TimepointResult:
#     """Configuration for timepoint-gated results."""
#     current_level: MatchLevel
#     reason_exact: str
#     reason_undetermined: str
#     reason_na: str
#     extra_details: Optional[Dict] = None

@dataclass
class CandidateContext:
    """Mutable state for handler chain. Access variable data via src/tgt directly.

    NOTE: Used internally by handlers via mutate-then-read. Compute_structural()
    converts the final state to an immutable StructuralEvidence at the
    boundary so the rest of the pipeline never sees this mutability.
    """
    src: VariableNode
    tgt: VariableNode
    current_level: MatchLevel = MatchLevel.PARTIAL
    details: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    matcher: Optional[MatcherProtocol] = None
    mapping_mode: MappingType = MappingType.OEH.value
    mapping_relation:MappingRelation = MappingRelation.UnMatched.value
    # ── Only properties that add logic beyond VariableNode ────
    @property
    def is_exact_timepoint(self) -> bool:
        return self.src.visit.lower().strip() == self.tgt.visit.lower().strip()

    @property
    def context_match_type(self) -> Optional[str]:
        return self.src.context_match_type or self.tgt.context_match_type

    @property
    def sim_score(self) -> Optional[float]:
        return self.src.sim_score

    @property
    def src_original_categories(self) -> List[str]:
        return [c.lower() for c in self.src.original_categories]

    @property
    def is_derived_variable(self) -> bool:
        return self.src.is_derived or self.tgt.is_derived

    @property
    def is_same_category(self) -> bool:
        return self.src.category == self.tgt.category


    @property
    def timepoint_details(self) -> Dict[str, str]:
        return {"source_timepoint": self.src.visit, "target_timepoint": self.tgt.visit}

    # ── Result Setters ───────────────────────────────────────
    def set_result(self, level: MatchLevel, reason: str = "",
                   extra_details: Optional[Dict] = None, should_stop: bool = False):
        self.current_level = level
        existing = self.details.get("description", "").strip()
        self.details["description"] = f"{existing} {reason}".strip() if existing else reason
        if extra_details:
            self.details.update(extra_details)
        self.should_stop = should_stop


class CategoryMapper:
    """Semantic matching between categorical value labels."""

    _label_embedding_cache: Dict[str, np.ndarray] = {}
    _label_omop_cache: Dict[str, int] = {}              # label.lower() -> omop_id
    _alignment_cache: Dict[Tuple[int, int], bool] = {}  # (sid, tid) -> reachable?

    @staticmethod
    def _fuzzy_match_labels(ctx: CandidateContext, src_labels: List[str],
                            tgt_labels: List[str], threshold: float = settings.ADAPTIVE_THRESHOLD) -> Dict[str, str]:
        """Four-tier label matching:
        Tier 0: canonical boolean equivalence  (yes/no/true/false/0/1)
        Tier 1: exact lowercase string
        Tier 2: OMOP graph reachability        (OO, OEH modes only)
        Tier 3: embedding cosine + context pass (NE, OEH modes only)
        """
        if not src_labels or not tgt_labels:
            return {}

        src_norm = {s.lower(): s for s in src_labels}   # lowercase → original label
        tgt_norm = {t.lower(): t for t in tgt_labels}
        matches: Dict[str, str] = {}

        def _unmatched():
            used_tgt = set(matches.values())
            return (
                [s for s in src_norm if s not in matches],
                [t for t in tgt_norm if t not in used_tgt],
            )

        # ── Tier 0: canonical boolean equivalence ────────────────────────────
        # Build canonical key → first label mapping (setdefault avoids overwrite)
        canon_src: Dict[frozenset, str] = {}
        for s in src_norm:
            canon_src.setdefault(_canonical_cats([s]), s)
        canon_tgt: Dict[frozenset, str] = {}
        for t in tgt_norm:
            canon_tgt.setdefault(_canonical_cats([t]), t)
        for cs, s_lbl in canon_src.items():
            if cs in canon_tgt:
                matches[s_lbl] = canon_tgt[cs]
        if len(matches) == len(src_norm):
            return matches

        # ── Tier 1: exact lowercase string ───────────────────────────────────
        u_src, u_tgt = _unmatched()
        tgt_set = set(u_tgt)
        for s in u_src:
            if s in tgt_set:
                matches[s] = s
        if len(matches) == len(src_norm):
            return matches

        # ── Tier 2: OMOP graph reachability (OO + OEH only) ──────────────────
        if ctx.mapping_mode in (MappingType.OO.value, MappingType.OEH.value):
            u_src, u_tgt = _unmatched()
            for s_lbl in u_src:
                sid = CategoryMapper._label_omop_cache.get(s_lbl)
                if not sid:
                    continue
                for t_lbl in u_tgt:
                    if t_lbl in matches.values():
                        continue
                    tid = CategoryMapper._label_omop_cache.get(t_lbl)
                    if not tid:
                        continue
                    if sid == tid:
                        matches[s_lbl] = t_lbl
                        break
                    key = (min(sid, tid), max(sid, tid))
                    if CategoryMapper._alignment_cache.get(key):
                        matches[s_lbl] = t_lbl
                        break
            if len(matches) == len(src_norm):
                return matches

        # ── Tier 3: embedding cosine (NE + OEH only) ─────────────────────────
        if ctx.mapping_mode not in (MappingType.NE.value, MappingType.OEH.value):
            return matches

        u_src, u_tgt = _unmatched()
        cache = CategoryMapper._label_embedding_cache

        src_embs = [(lbl, cache[lbl]) for lbl in u_src if lbl in cache]
        tgt_embs = [(lbl, cache[lbl]) for lbl in u_tgt if lbl in cache]
        if not src_embs or not tgt_embs:
            return matches

        src_lbls, S = zip(*src_embs)
        tgt_lbls, T = zip(*tgt_embs)
        sim = cosine_similarity(np.array(S), np.array(T))   # (n_src, n_tgt)

        needs_context: List[Tuple[str, str]] = []
        for i, s_lbl in enumerate(src_lbls):
            j = int(sim[i].argmax())
            t_lbl, score = tgt_lbls[j], float(sim[i, j])
            if score < threshold:
                needs_context.append((s_lbl, tgt_lbls[j]))
                continue
            if FuzzyMatcher._is_negation_pair(s_lbl, t_lbl)[0]:
                continue
            s_tok, t_tok = FuzzyMatcher.tokenize(s_lbl), FuzzyMatcher.tokenize(t_lbl)
            if (s_tok.issubset(t_tok) or t_tok.issubset(s_tok)) and abs(len(s_tok) - len(t_tok)) >= 2:
                needs_context.append((s_lbl, t_lbl))
                continue
            matches[s_lbl] = t_lbl

        # Context-enhanced second pass for ambiguous pairs
        sv, tv = ctx.src.main_label, ctx.tgt.main_label
        for s_lbl, t_lbl in needs_context:
            sk, tk = f"{sv}::{s_lbl}", f"{tv}::{t_lbl}"
            if sk in cache and tk in cache and FuzzyMatcher._has_token_overlap(s_lbl, t_lbl):
                score = float(cosine_similarity([cache[sk]], [cache[tk]])[0, 0])
                if score >= threshold:
                    matches[s_lbl] = t_lbl

        return matches
    @staticmethod
    def build_label_mapping(ctx: CandidateContext,
                            src_categories_labels: List[str], src_original_categories: List[str],
                            tgt_categories_labels: List[str], tgt_original_categories: List[str],
                            use_fuzzy: bool = True) -> Dict[str, Any]:
        """Build categorical value mapping between source and target."""

        # GENERIC_TERMS = {"yes", "no", "true", "false", "y", "n", "0", "1", "unknown", "other", "n/a", "none", "not applicable"}

       
        def c2l(values, labels):
            return {val: lab for val, lab in zip(values, labels) if val and lab}

        src_c2l = c2l(src_original_categories, src_categories_labels)
        tgt_c2l = c2l(tgt_original_categories, tgt_categories_labels)

        fuzzy_map = {}
        if use_fuzzy and src_categories_labels and tgt_categories_labels:
            fuzzy_map = CategoryMapper._fuzzy_match_labels(
                ctx, src_categories_labels, tgt_categories_labels, threshold=settings.ADAPTIVE_THRESHOLD
            )



        def l2codes_fuzzy(c2l_map, fuzzy_map=None):
            lab2codes, pretty = {}, {}
            for cat, lab in c2l_map.items():
                key = lab.lower()
                if fuzzy_map:
                    key = fuzzy_map.get(key) or fuzzy_map.get(next(iter(_canonical_cats([key])), key), key)
                lab2codes.setdefault(key, []).append(cat)
                pretty.setdefault(key, lab)
            for k in lab2codes:
                lab2codes[k].sort()
            return lab2codes, pretty

        src_lab, src_pretty = l2codes_fuzzy(src_c2l, fuzzy_map)
        tgt_lab, tgt_pretty = l2codes_fuzzy(tgt_c2l)

        overlap_keys = sorted(set(src_lab) & set(tgt_lab))
        items, code_map = [], {}
        for k in overlap_keys:
            label = src_pretty.get(k, tgt_pretty.get(k, k))
            s_codes, t_codes = src_lab[k], tgt_lab[k]
            items.append(f"{label}: {', '.join(s_codes)}<->{', '.join(t_codes)}")
            code_map[s_codes[0]] = t_codes[0]

        unmapped_src = sorted(src_pretty[k] for k in src_lab.keys() - set(overlap_keys))
        unmapped_tgt = sorted(tgt_pretty[k] for k in tgt_lab.keys() - set(overlap_keys))

        identical = (set(src_lab) == set(tgt_lab)) and all(
            len(src_lab[k]) == len(tgt_lab[k]) for k in overlap_keys
        )

        # if one side has yes no and other has other values and if symbolic relationship is 
        return {
            "mapping_str": "; ".join(items) if items else None,
            "code_map": code_map,
            "overlap_labels": [src_pretty[k] for k in overlap_keys],
            "unmapped_source_labels": "; ".join(unmapped_src),
            "unmapped_target_labels": "; ".join(unmapped_tgt),
            "has_overlap": bool(overlap_keys),
            # "is_purely_generic_overlap": is_purely_generic_overlap,
            "identical": identical,
            "overlap_count": len(overlap_keys),
            "src_unique_labels": len(src_lab),
            "tgt_unique_labels": len(tgt_lab),
            "fuzzy_matches": len(fuzzy_map) if use_fuzzy else 0,
        }





# 4. Statistical Handlers


class Constraint(ABC):
    @abstractmethod
    def apply(self, ctx: CandidateContext):
        pass


class ContinuousHandler:
    """Unit compatibility for continuous variables."""



    @staticmethod
    def apply(ctx: CandidateContext):
        src_unit = str(ctx.src.unit).lower() if ctx.src.unit else None
        tgt_unit = str(ctx.tgt.unit).lower() if ctx.tgt.unit else None
        unit_info = {"source_unit": src_unit, "target_unit": tgt_unit}
        dosage_unit = is_absolute_vs_percent_dose(src_unit, tgt_unit)
    
        units_same   = (src_unit == tgt_unit) if (src_unit and tgt_unit) else False
        units_differ = (src_unit != tgt_unit) if (src_unit and tgt_unit) else False
        no_unit      = (not src_unit and not tgt_unit)
        

        # Case 1: Neither side has a unit — concept is unitless on both sides.

        if no_unit:
            if ctx.context_match_type == ContextMatchType.PENDING.value and ctx.matcher.llm_matcher is not None:
                # LLM-mode rows that haven't been resolved yet — defer.
                ctx.set_result(
                    level=MatchLevel.PARTIAL,
                    reason="Unit dimension not applicable; concept equivalence pending semantic check.",
                    extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW},
                )
            else:
                ctx.set_result(
                    level=MatchLevel.IDENTICAL,
                    reason="Unit dimension not applicable.",
                    extra_details={**unit_info, "transformation": TransformationType.NONE}
                )

            return

        # Case 2: Same units.
        if units_same and ctx.context_match_type:
            if ctx.context_match_type == ContextMatchType.PENDING.value:
                # LLM-mode rows that haven't been resolved yet — defer.
                ctx.set_result(
                    level=MatchLevel.PARTIAL,
                    reason="Same units; concept equivalence pending semantic check.",
                    extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW},
                )
                return
            ctx.set_result(
                level=MatchLevel.IDENTICAL,
                reason="Same units.",
                extra_details={**unit_info, "transformation": TransformationType.NONE}
            )

            return
        
        # Case 3a: Both sides have units but they differ → unit conversion.
        if units_differ and ctx.matcher.llm_matcher is None:
            if dosage_unit and ctx.mapping_relation not in [MappingRelation.SymbolicBroadMatch.value, MappingRelation.SymbolicNarrowMatch.value]:
                   # Case 4: Asymmetric — one side has a unit,             q the other doesn't. Genuinely incomplete.
                ctx.set_result(
                        level=MatchLevel.PARTIAL,
                        reason=f"units: ({src_unit or tgt_unit}) conversion requires external knowledge.",
                        extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW}
                    )
            else:
                  ctx.set_result(
                    level=MatchLevel.COMPATIBLE,
                    reason=f"Units differ ({src_unit} vs {tgt_unit}).",
                    extra_details={**unit_info, "transformation": TransformationType.UNIT_CONVERSION}
                )
              
            return
        # Case 3b: Derived variable → recomputation required (not the same as unit scaling).
        if ctx.is_derived_variable:
            ctx.set_result(
                level=MatchLevel.COMPATIBLE,
                reason="Derived variable; recomputation required.",
                extra_details={**unit_info, "transformation": TransformationType.DERIVATION}
            )
          
            return

        # Case 4: Asymmetric — one side has a unit,             q the other doesn't. Genuinely incomplete.
        ctx.set_result(
                level=MatchLevel.PARTIAL,
                reason=f"units: ({src_unit or tgt_unit}); cannot verify equivalence.",
                extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW}
            )
      
class CategoricalHandler:
    """Same-type categorical comparison (binary-binary or multi-multi)."""
    
    @staticmethod
    def apply(ctx: CandidateContext, s_cats: Set[str], t_cats: Set[str],
              cat_label_map: Optional[Dict]):
        s_raw_norm = _canonical_cats(ctx.src.original_categories)
        t_raw_norm = _canonical_cats(ctx.tgt.original_categories)
        _range_match = RangeHelper.exact_match(ctx)
        s_lbl_norm = _canonical_cats(s_cats)
        t_lbl_norm = _canonical_cats(t_cats)
        has_overlap = cat_label_map.get("has_overlap", False) if cat_label_map else False

        # trusted_raw_match = (s_raw == t_raw) and (has_overlap or context_is_exact)
        # Case 1: Equality
       
        if s_raw_norm == t_raw_norm or s_lbl_norm == t_lbl_norm or _range_match:
            if s_raw_norm == t_raw_norm:
                is_generic = (s_raw_norm == _GENERIC_BOOL)
                ctx_is_exact = (ctx.context_match_type == ContextMatchType.EXACT.value)
                if is_generic and not ctx_is_exact:
                    level = MatchLevel.PARTIAL
                    reason = "Same raw 0/1 codes; semantic equivalence unverified."
                    extra = {"categories": s_lbl_norm,
                     "transformation": TransformationType.MANUAL_REVIEW}
                else:
                    level = MatchLevel.IDENTICAL
                    reason = "Same raw codes." 
                    extra = {"categories": s_lbl_norm, "transformation": TransformationType.NONE}
            else:
                level = MatchLevel.COMPATIBLE
                reason = "Same categories (different format)."
                extra = {"categories": s_lbl_norm, "transformation": TransformationType.VALUE_NORMALIZATION} 
            if level == MatchLevel.COMPATIBLE and cat_label_map:
                extra.update(cat_label_map)
            
            ctx.set_result(
                level=level,
                reason=reason,
                extra_details=extra
            )
            return

        # Case 2: Strict subset (canonical)
        if s_lbl_norm < t_lbl_norm or t_lbl_norm < s_lbl_norm:
            subset, superset = (s_lbl_norm, t_lbl_norm) if s_lbl_norm < t_lbl_norm else (t_lbl_norm, s_lbl_norm)
            extra = {"transformation": TransformationType.AGGREGATION_OR_EXPANSION,
                    "subset_categories": subset, "superset_categories": superset}
            if cat_label_map:
                extra.update(cat_label_map)
            ctx.set_result(
                level=MatchLevel.PARTIAL,
                reason="One variable is categorical subset of other.",
                extra_details=extra
            )
           
            return

        # Case 3: Semantic overlap from label mapping
        if has_overlap:
            extra = {"transformation": TransformationType.MANUAL_REVIEW}
            extra.update(cat_label_map)
            ctx.set_result(
                level=MatchLevel.PARTIAL,
                reason= "Some overlapping categories.",
                extra_details=extra
            )
           
            return
        

        # Case 4: No overlap
        # case 4.1: Detect if exactly ONE side is generic boolean (0/1/yes/no)

        is_s_generic = s_lbl_norm.issubset(_GENERIC_BOOL) and bool(s_lbl_norm)
        is_t_generic = t_lbl_norm.issubset(_GENERIC_BOOL) and bool(t_lbl_norm)

        if is_s_generic != is_t_generic:  # XOR: One is generic, the other is not
            generic_side = "source" if is_s_generic else "target"
            specific_side = "target" if is_s_generic else "source"
            specific_cats = t_lbl_norm if is_s_generic else s_lbl_norm
            
            extra = {
                "transformation": TransformationType.MANUAL_REVIEW,
                "generic_variable_side": generic_side,
                "specific_variable_side": specific_side,
                "specific_categories": specific_cats,
                "mapping_hint": "binary_extraction_candidate"
            }
            if cat_label_map:
                extra.update(cat_label_map)

            ctx.set_result(
                level=MatchLevel.PARTIAL,
                reason=f"Potential dichotomization: generic boolean ({generic_side}) vs specific categories ({specific_side}).",
                extra_details=extra
            )
            return

        # Or no overlap
        ctx.set_result(MatchLevel.NOT_APPLICABLE, 
                    reason="No overlap in categorical codes.",
                     extra_details=
                        {"transformation": TransformationType.MANUAL_REVIEW,
                        "source_categories": s_lbl_norm, "target_categories": t_lbl_norm
                        },
                     )


class ContinuousCategoricalHandler:
    """Continuous vs Categorical handler."""

    @staticmethod
    def apply(ctx: CandidateContext):
        cont = StatisticalType.CONTINUOUS.value
        categorical = StatisticalType.MULTI_CLASS.value

        if ctx.src.statistical_type == cont and ctx.tgt.statistical_type == categorical:
            continuous_side, categorical_side = "source", "target"
        elif ctx.src.statistical_type == categorical and ctx.tgt.statistical_type == cont:
            continuous_side, categorical_side = "target", "source"
        else:
            return
        ctx.set_result(
            level=MatchLevel.PARTIAL,
            reason=(
                f"Categorical ({categorical_side}) extractable from "
                f"continuous ({continuous_side}) via threshold"
            ),
            extra_details={
                "transformation": TransformationType.AGGREGATION_OR_EXPANSION,
                "extraction_direction": "continuous_to_binary",
                "continuous_variable_side": continuous_side,
                "categorical_variable_side": categorical_side,
                "source_type": ctx.src.statistical_type, "target_type": ctx.tgt.statistical_type,
            }
        )
       
class ContinuousBinaryHandler:
    """Continuous vs Binary: binary can be extracted from continuous via threshold."""

    @staticmethod
    def apply(ctx: CandidateContext):
        cont   = StatisticalType.CONTINUOUS.value
        binary = StatisticalType.BINARY.value
        dt_val = DATA_TYPE.DATETIME.value

        if ctx.src.statistical_type == cont and ctx.tgt.statistical_type == binary:
            cont_side, bin_side = "source", "target"
        elif ctx.src.statistical_type == binary and ctx.tgt.statistical_type == cont:
            cont_side, bin_side = "target", "source"
        else:
            return

        common = {
            "continuous_variable_side": cont_side,
            "binary_variable_side":     bin_side,
            "source_type":              ctx.src.statistical_type,
            "target_type":              ctx.tgt.statistical_type,
        }

        if dt_val in {ctx.src.data_type, ctx.tgt.data_type}:
            ctx.set_result(
                level=MatchLevel.PARTIAL,
                reason=f"Binary ({bin_side}) extractable from continuous "
                              f"({cont_side}); presence of date implies True.",
                extra_details={
                    **common,
                    "transformation": TransformationType.BINARY_EXTRACTION,
                    "extraction_direction": "continuous_to_binary",
                }
             )
           
            return
            
        ctx.set_result(
            level=MatchLevel.PARTIAL,
            reason=f"Continuous ({cont_side}) vs binary ({bin_side}) requires a "
                        f"domain-specific clinical threshold; not derivable from metadata.",
            extra_details={
                **common,
                "transformation": TransformationType.MANUAL_REVIEW,
            }
             )
       
   
class MixedCategoricalHandler:
    """Binary vs Multi-class comparison."""

    @staticmethod
    def apply(ctx: CandidateContext, s_cats: Set[str],
              t_cats: Set[str], cat_label_map: Optional[Dict]):
        if ctx.src.statistical_type == StatisticalType.BINARY.value and ctx.tgt.statistical_type == StatisticalType.MULTI_CLASS.value:
            binary_cats, multi_cats = s_cats, t_cats
            binary_code_id, binary_label = ctx.src.main_id, ctx.src.main_label or ctx.src.description
            multi_code_ids, multi_labels = ctx.tgt.category_ids, ctx.tgt.category_labels or ctx.tgt.original_categories
            binary_side, study = 'source', ctx.tgt.study
        elif ctx.src.statistical_type == StatisticalType.MULTI_CLASS.value and ctx.tgt.statistical_type == StatisticalType.BINARY.value:
            binary_cats, multi_cats = t_cats, s_cats
            binary_code_id, binary_label = ctx.tgt.main_id, ctx.tgt.main_label or ctx.tgt.description
            multi_code_ids, multi_labels = ctx.src.category_ids, ctx.src.category_labels or ctx.src.original_categories
            binary_side, study = 'target', ctx.src.study
        else:
            return
        binary_raw_norm = _canonical_cats(
            ctx.src.original_categories if binary_side == 'source' else ctx.tgt.original_categories
        )
        multi_raw_norm = _canonical_cats(
            ctx.tgt.original_categories if binary_side == 'source' else ctx.src.original_categories
        )
        _BOOL_ALL = _BOOL_POS | _BOOL_NEG
        is_dir_subset = (
            binary_raw_norm.issubset(multi_raw_norm)
            and bool(binary_raw_norm - _BOOL_ALL)   # binary has at least one non-generic code
            and bool(multi_raw_norm - _BOOL_ALL)    # multi also has at least one non-generic code
        )
        has_overlap = cat_label_map.get("has_overlap", False) if cat_label_map else False
        # overlap_count = cat_label_map.get("overlap_count", 0) if cat_label_map else 0
        # unmapped_src = cat_label_map.get("unmapped_source_labels", "") if cat_label_map else ""

        base = {
            "transformation": TransformationType.AGGREGATION_OR_EXPANSION,
            "binary_categories": binary_cats, "multi_categories": multi_cats,
        }
        if cat_label_map:
            base["code_map"] = cat_label_map.get("code_map")
            base["overlap_labels"] = cat_label_map.get("overlap_labels")
            base["mapping_str"] = cat_label_map.get("mapping_str")

        # CASE 1: Subsumption
        if MixedCategoricalHandler._try_subsumption(
            ctx, binary_code_id, binary_label, multi_code_ids, multi_labels, binary_side, study
        ):
            return

        # CASE 2: Direct subset or has overlap
        if is_dir_subset or has_overlap:

            base["transformation"] = TransformationType.BINARY_EXTRACTION

            ctx.set_result(
            MatchLevel.PARTIAL,
            reason="Binary variable is subset of Multi-class variable.",
            extra_details=base
            )
           
            return

      

        # CASE 6: Binary aggregation pattern
        if MixedCategoricalHandler._try_binary_aggregation(ctx, binary_cats, multi_cats, cat_label_map):
            return

        # No valid mapping found
        ctx.set_result(
            MatchLevel.NOT_APPLICABLE,
            reason="No valid mapping between Binary and Multi-class categories found.",
            extra_details={
                "transformation": TransformationType.MANUAL_REVIEW,
                "binary_categories": binary_cats, "multi_categories": multi_cats,
            },
        )

    

    @staticmethod
    def _try_subsumption(ctx, binary_id, binary_label, multi_ids, multi_labels, binary_side, study) -> bool:
        if not multi_labels:
            return False
        binary_lower = binary_label.lower().strip()
        for cat_label in multi_labels:
            if binary_lower == cat_label.lower().strip():
                ctx.set_result(
                    level=MatchLevel.PARTIAL,
                    reason=f"Binary extracts '{cat_label}' (direct label match).",
                    extra_details={
                        "transformation": TransformationType.BINARY_EXTRACTION,
                        "binary_variable_side": binary_side,
                        "matched_category_label": cat_label,
                        "mapping_relation": "direct_label_match"},
                    )
             
                return True

        if not (ctx.matcher and multi_labels):
            return False
        ids = multi_ids if isinstance(multi_ids, list) and len(multi_ids) == len(multi_labels) else [''] * len(multi_labels)
        for cat_id, cat_label in zip(ids, multi_labels):
            is_match, method = ctx.matcher.check_categorical_subsumption(
                category_concept=(str(cat_id or ''), cat_label),
                main_concept=(str(binary_id or ''), binary_label),
                target_study=study,
            )
            if is_match:
                ctx.set_result(
                    level=MatchLevel.PARTIAL,
                    reason=f"Binary extracts '{cat_label}' ({method}).",
                    extra_details={
                        "transformation": TransformationType.BINARY_EXTRACTION,
                        "binary_variable_side": binary_side,
                        "matched_category_label": cat_label,
                        "matched_category_omop": cat_id,
                        "mapping_relation": method,
                    }
                )   
              
                return True
        return False

    @staticmethod
    def _try_binary_aggregation(ctx, binary_cats, multi_cats, cat_label_map) -> bool:
        """Detect binary aggregation pattern (one matches, other aggregates rest)."""
        if len(binary_cats) != 2:
            return False

        binary_list = list(binary_cats)
        is_neg_pair, neg_cat, pos_cat = FuzzyMatcher._is_negation_pair(binary_list[0], binary_list[1])

        if cat_label_map and cat_label_map.get('overlap_labels'):
            overlap_lower = {l.lower() for l in cat_label_map.get('overlap_labels', [])}
            overlap = {b for b in binary_cats if b.lower() in overlap_lower}
        else:
            overlap = {b for b in binary_cats if b.lower() in {m.lower() for m in multi_cats}}

        if len(overlap) != 1:
            return False

        matched_cat = list(overlap)[0]
        aggregate_cat = list(binary_cats - overlap)[0]
        aggregated_multi = {m for m in multi_cats if m.lower() != matched_cat.lower()}

        if len(aggregated_multi) < 2 and not is_neg_pair:
            return False

        category_mapping = {matched_cat: matched_cat}
        for agg in aggregated_multi:
            category_mapping[agg] = aggregate_cat

        extra = {
            "transformation": TransformationType.BINARY_EXTRACTION,
            "matched_category": matched_cat,
            "aggregate_category": aggregate_cat,
            "aggregated_values": sorted(aggregated_multi),
            "category_mapping": category_mapping,
            "mapping_direction": "multi_to_binary",
            "mapping_summary": f"{matched_cat} → {matched_cat}; {{{', '.join(sorted(aggregated_multi))}}} → {aggregate_cat}",
        }
        if is_neg_pair:
            extra.update(detection_method="negation_pair",
                         negative_category=neg_cat, positive_category=pos_cat)

      
        ctx.set_result(
            level=MatchLevel.PARTIAL,
            reason=f"'{aggregate_cat}' aggregates {len(aggregated_multi)} categories.",
            extra_details=extra
        )
        return True



# 5. Range Helper


class RangeHelper:
    """Numeric range comparison utilities."""

    @staticmethod
    def get_minmax(ctx: CandidateContext):
        s_min, s_max = ctx.src.statistics.min_val, ctx.src.statistics.max_val
        t_min, t_max = ctx.tgt.statistics.min_val, ctx.tgt.statistics.max_val

        for side, (mn, mx), cats in [
            ('src', (s_min, s_max), ctx.src.original_categories),
            ('tgt', (t_min, t_max), ctx.tgt.original_categories),
        ]:
            if mn is None or mx is None:
                nums = []
                for c in cats:
                    try:
                        nums.append(float(c))
                    except (ValueError, TypeError):
                        pass
                if nums:
                    if side == 'src':
                        s_min, s_max = min(nums), max(nums)
                    else:
                        t_min, t_max = min(nums), max(nums)

        return s_min, s_max, t_min, t_max

    @staticmethod
    def exact_match(ctx: CandidateContext) -> bool:
        s_min, s_max, t_min, t_max = RangeHelper.get_minmax(ctx)
        if any(v is None for v in (s_min, s_max, t_min, t_max)):
            return False
        if s_min > s_max:
            s_min, s_max = s_max, s_min
        if t_min > t_max:
            t_min, t_max = t_max, t_min
        return s_min == t_min and s_max == t_max



# 6. StatisticalLogicConstraint (dispatcher)


class StatisticalLogicConstraint(Constraint):
    """Routes to the appropriate type-specific handler."""

    def apply(self, ctx: CandidateContext):
        s_type = ctx.src.statistical_type
        t_type = ctx.tgt.statistical_type
        cat_types = {StatisticalType.BINARY.value, StatisticalType.MULTI_CLASS.value}
        cont = StatisticalType.CONTINUOUS.value
        binary = StatisticalType.BINARY.value
        valid_types = {st.value for st in StatisticalType}
        qual = StatisticalType.QUALITATIVE.value
        
        if s_type not in valid_types or t_type not in valid_types:
            if ctx.is_derived_variable:
                ctx.set_result(MatchLevel.COMPATIBLE, 
                reason="Variables are compatible",
                extra_details={"transformation": TransformationType.DERIVATION})
            else:
                ctx.set_result(MatchLevel.NOT_APPLICABLE,
                    reason= f"Invalid or missing statistical types: {s_type} vs {t_type}.",
                    extra_details={"transformation": TransformationType.NONE}
                )
            return

        # Same type
        if s_type == t_type:
            if s_type == StatisticalType.CONTINUOUS.value:
                ContinuousHandler.apply(ctx)
            elif s_type in cat_types:
                cat_map = self._build_cat_map(ctx)
                CategoricalHandler.apply(ctx, set(ctx.src.category_labels), set(ctx.tgt.category_labels), cat_map)
            else:
                ctx.set_result(
                    level=MatchLevel.IDENTICAL,
                    reason="Variables are compatible.",
                    extra_details={
                   "transformation": TransformationType.NONE
                    }
                )
              
            return
        # Type mismatch: continuous vs qualitative
        if {s_type, t_type} == {cont, qual}:
            ctx.set_result(MatchLevel.NOT_APPLICABLE, 
                reason= "Continuous vs qualitative (free-text); no structural harmonization possible.",
                extra_details={
                   "transformation": TransformationType.NONE
                })
            return
          
            
        # Qualitative vs Categorical
        if qual in {s_type, t_type} and (s_type in cat_types or t_type in cat_types):
            ctx.set_result(
                MatchLevel.PARTIAL,
                reason=f"Statistical type mismatch:{s_type} vs {t_type}.",
                extra_details={"transformation": TransformationType.MANUAL_REVIEW},
            )
            return
       
        # Continuous vs Categorical
        if cont in {s_type, t_type} and (cat_types & {s_type, t_type}):
            if {s_type, t_type} == {cont, binary}:
                ContinuousBinaryHandler.apply(ctx)
                return
            if RangeHelper.exact_match(ctx):
                s_min, s_max, t_min, t_max = RangeHelper.get_minmax(ctx)
                ctx.set_result(
                    level=MatchLevel.PARTIAL,
                    reason=f"continuous vs categorical with identical range [{s_min}-{s_max}].",
                    extra_details={
                        "transformation": TransformationType.VALUE_NORMALIZATION,
                        "source_range": f"[{s_min}, {s_max}]",
                        "target_range": f"[{t_min}, {t_max}]",
                    }
                )
                
            else:
                ctx.set_result(
                level=MatchLevel.NOT_APPLICABLE,
                reason="Continuous vs Multi-class mismatch; no valid numeric or structural overlap.",
                extra_details={"transformation": TransformationType.MANUAL_REVIEW}
                )
                return
           

    
        if s_type in cat_types or t_type in cat_types:
            cat_map = self._build_cat_map(ctx)
            MixedCategoricalHandler.apply(
                ctx, set(ctx.src.category_labels), set(ctx.tgt.category_labels), cat_map
            )
            return

        ctx.set_result(
            MatchLevel.NOT_APPLICABLE,
            reason=f"Statistical type mismatch: {s_type} vs {t_type}.",
            extra_details={"transformation": TransformationType.NONE},
        )

    def _build_cat_map(self, ctx: CandidateContext) -> Optional[Dict]:
        s_cats = set(ctx.src.category_labels)
        t_cats = set(ctx.tgt.category_labels)
        if s_cats and t_cats:
            return CategoryMapper.build_label_mapping(
                ctx=ctx,
                src_categories_labels=ctx.src.category_labels,
                src_original_categories=ctx.src.original_categories,
                tgt_categories_labels=ctx.tgt.category_labels,
                tgt_original_categories=ctx.tgt.original_categories,
            )
        return None



# 7. compute_structural — entry point into the new pipeline

def compute_structural(src: VariableNode,
                        tgt: VariableNode,
                        mapping_mode: MappingType = MappingType.NE.value,
                        matcher: Optional[MatcherProtocol] = None):

    """Run the structural rules against a candidate pair and return an
    immutable StructuralEvidence record.
    """
    from .verdict import StructuralEvidence
    ctx = CandidateContext(src=src, tgt=tgt, matcher=matcher,
                            mapping_mode=mapping_mode)

    StatisticalLogicConstraint().apply(ctx)
    reason = ctx.details.get("description", "").strip()
    # Convert mutable ctx to frozen StructuralEvidence.
    level = ctx.current_level
    # if level == MatchLevel.IDENTICAL:
    #     # Assuming ContextMatchType.PARTIAL.value maps to your '4' and SUBSUMED is '2'/'3'
    #     if str(ctx.context_match_type) not in {ContextMatchType.EXACT.value, ContextMatchType.PENDING.value}:
    #         level = MatchLevel.PARTIAL
    #         reason += f" (Capped to Partial due to context match type {ctx.context_match_type})."
    
    transform_raw = ctx.details.get("transformation", TransformationType.NONE)
    transformation = (transform_raw if isinstance(transform_raw, TransformationType)
                       else TransformationType(transform_raw)
                       if transform_raw in {t.value for t in TransformationType}
                       else TransformationType.NONE)
    # needs_review: handler couldn't reach a confident structural answer.
    needs_review = (transformation == TransformationType.MANUAL_REVIEW) or (ctx.context_match_type == ContextMatchType.PENDING.value)

    # Strip transformation/description from extras (they're top-level fields).
    extra = {k: v for k, v in ctx.details.items()
             if k not in ("transformation", "description")}
    extra={**extra, "mapping_relation": (src.mapping_relation or tgt.mapping_relation or ""),
        "context_match_type": ctx.context_match_type}

    return StructuralEvidence(
        level=level,
        transformation=transformation,
        reason=reason,
        needs_review=needs_review,
        extra=extra,
    )

def make_timepoint_info(src: VariableNode, tgt: VariableNode):
    """Build TimepointInfo from a candidate pair.

    Pulled out so run.py doesn't need to know about VariableNode internals.
    """
    from .verdict import TimepointInfo
    s_visit = src.visit or ""
    t_visit = tgt.visit or ""
    aligned = s_visit.lower().strip() == t_visit.lower().strip()
    return TimepointInfo(
        aligned=aligned,
        source_visit=s_visit,
        target_visit=t_visit,
    )
        

       
        
