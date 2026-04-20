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
)


# =====================================================================
# 1. Protocols & Data Classes
# =====================================================================

@runtime_checkable
class MatcherProtocol(Protocol):
    """Interface for categorical subsumption checking."""
    def check_categorical_subsumption(
        self,
        category_concept: Tuple[str, str],
        main_concept: Tuple[str, str],
        target_study: str,
    ) -> Tuple[bool, Optional[str]]: ...


@dataclass
class TimepointResult:
    """Configuration for timepoint-gated results."""
    current_level: MatchLevel
    reason_exact: str
    reason_undetermined: str
    reason_na: str
    extra_details: Optional[Dict] = None

@dataclass
class CandidateContext:
    """Mutable state for constraint chain. Access variable data via src/tgt directly."""
    src: VariableNode
    tgt: VariableNode
    current_level: MatchLevel = MatchLevel.IDENTICAL
    details: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    matcher: Optional[MatcherProtocol] = None
    context_is_subsumed: bool = False
    mapping_mode: MappingType = MappingType.OEH.value
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

    def set_result_with_timepoint(self, config: TimepointResult, should_stop: bool = False):
        if self.is_exact_timepoint:
            self.set_result(config.current_level, config.reason_exact,
                            {**(config.extra_details or {})}, should_stop)
        else:
            extra = {**(config.extra_details or {})}
            if 'transformation' not in extra:
                extra['transformation'] = TransformationType.MANUAL_REVIEW
            self.set_result(config.current_level, config.reason_undetermined, extra, should_stop)
# =====================================================================
# 2. CategoryMapper
# =====================================================================

class CategoryMapper:
    """Semantic matching between categorical value labels."""

    _label_embedding_cache: Dict[str, np.ndarray] = {}
    _label_omop_cache: Dict[str, int] = {}              # label.lower() -> omop_id
    _alignment_cache: Dict[Tuple[int, int], bool] = {}  # (sid, tid) -> reachable?

    @staticmethod
  
    def _fuzzy_match_labels(ctx: CandidateContext, src_labels: List[str],
                            tgt_labels: List[str], threshold: float = 0.8) -> Dict[str, str]:
        """Three-tier match: exact string (all modes) → OMOP graph (OO, OEH) →
        embedding cosine (NE, OEH, via pre-populated cache)."""
        if not src_labels or not tgt_labels:
            return {}

        src_norm = {s.lower(): s for s in src_labels}
        tgt_norm = {t.lower(): t for t in tgt_labels}

        # ── Tier 1: exact lowercase string ───────────────────────────────
        matches = {s: t for s in src_norm for t in tgt_norm if s == t}
        if len(matches) == len(src_norm):
            return matches

        # ── Tier 2: OMOP graph reachability (OO + OEH) ───────────────────
        use_ontology = ctx.mapping_mode in (MappingType.OO.value, MappingType.OEH.value)
        if use_ontology is not None:
            unmatched_src = [s for s in src_norm if s not in matches]
            unmatched_tgt = [t for t in tgt_norm if t not in matches.values()]
            for s_lbl in unmatched_src:
                sid = CategoryMapper._label_omop_cache.get(s_lbl)
                if not sid:
                    continue
                for t_lbl in unmatched_tgt:
                    if t_lbl in matches.values():      # another src already claimed it
                        continue
                    tid = CategoryMapper._label_omop_cache.get(t_lbl)
                    if not tid or sid == tid:
                        # same OMOP ID → tier 1 already would have caught it via label;
                        # different labels mapping to same ID also count as aligned
                        if sid and tid and sid == tid:
                            matches[s_lbl] = t_lbl
                            break
                        continue
                    key = (sid, tid) if sid <= tid else (tid, sid)
                    hit = CategoryMapper._alignment_cache.get(key)
                    # if hit is None:
                    #     hit = bool(ctx.graph.source_to_targets_paths(sid, {tid}, max_depth=1))
                    #     CategoryMapper._alignment_cache[key] = hit
                    if hit:
                        matches[s_lbl] = t_lbl
                        break
            if len(matches) == len(src_norm):
                return matches

        # ── Tier 3: embedding cosine (cache empty in OO → no-op) ─────────
        unmatched_src = [s for s in src_norm if s not in matches]
        unmatched_tgt = [t for t in tgt_norm if t not in matches.values()]
        src_embs, src_with_emb, tgt_embs, tgt_with_emb = [], [], [], []
        for lbl in unmatched_src:
            if lbl in CategoryMapper._label_embedding_cache:
                src_embs.append(CategoryMapper._label_embedding_cache[lbl]); src_with_emb.append(lbl)
        for lbl in unmatched_tgt:
            if lbl in CategoryMapper._label_embedding_cache:
                tgt_embs.append(CategoryMapper._label_embedding_cache[lbl]); tgt_with_emb.append(lbl)

        if not src_embs or not tgt_embs:
            return matches

        sim = cosine_similarity(np.array(src_embs), np.array(tgt_embs))
        needs_context = []
        for i, s_lbl in enumerate(src_with_emb):
            j = sim[i].argmax()
            t_lbl, score = tgt_with_emb[j], sim[i, j]
            if score >= threshold:
                if FuzzyMatcher._is_negation_pair(s_lbl, t_lbl)[0]:
                    continue
                s_tok, t_tok = FuzzyMatcher.tokenize(s_lbl), FuzzyMatcher.tokenize(t_lbl)
                if (s_tok.issubset(t_tok) or t_tok.issubset(s_tok)) and abs(len(s_tok) - len(t_tok)) >= 2:
                    needs_context.append((s_lbl, t_lbl)); continue
                matches[s_lbl] = t_lbl
            else:
                needs_context.append((s_lbl, t_lbl))

        # Context-enhanced second pass
        if needs_context:
            sv, tv = ctx.src.main_label, ctx.tgt.main_label
            for s_lbl, t_lbl in needs_context:
                sk, tk = f"{sv}::{s_lbl}", f"{tv}::{t_lbl}"
                if (sk in CategoryMapper._label_embedding_cache
                    and tk in CategoryMapper._label_embedding_cache
                    and FuzzyMatcher._has_token_overlap(s_lbl, t_lbl)):
                    score = cosine_similarity(
                        [CategoryMapper._label_embedding_cache[sk]],
                        [CategoryMapper._label_embedding_cache[tk]])[0][0]
                    if score >= threshold:
                        matches[s_lbl] = t_lbl

        return matches
    #   @staticmethod
    # def _fuzzy_match_labels(ctx: CandidateContext, src_labels: List[str],
    #                         tgt_labels: List[str], threshold: float = 0.8) -> Dict[str, str]:
    #     """Find semantic matches using pre-cached embeddings."""
    #     if not src_labels or not tgt_labels:
    #         return {}

    #     src_norm = {s.lower(): s for s in src_labels}
    #     tgt_norm = {t.lower(): t for t in tgt_labels}

    #     quick_matches = {}
    #     for s_key, s_orig in src_norm.items():
    #         for t_key, t_orig in tgt_norm.items():
    #             if s_orig == t_orig:
    #                 quick_matches[s_key] = t_key

    #     if len(quick_matches) == len(src_labels):
    #         return quick_matches

    #     unique_src = list(set(s.lower() for s in src_labels))
    #     unique_tgt = list(set(t.lower() for t in tgt_labels))

    #     src_embs, src_with_emb = [], []
    #     tgt_embs, tgt_with_emb = [], []

    #     for lbl in unique_src:
    #         if lbl in CategoryMapper._label_embedding_cache:
    #             src_embs.append(CategoryMapper._label_embedding_cache[lbl])
    #             src_with_emb.append(lbl)

    #     for lbl in unique_tgt:
    #         if lbl in CategoryMapper._label_embedding_cache:
    #             tgt_embs.append(CategoryMapper._label_embedding_cache[lbl])
    #             tgt_with_emb.append(lbl)

    #     if not src_embs or not tgt_embs:
    #         return quick_matches

    #     sim_matrix = cosine_similarity(np.array(src_embs), np.array(tgt_embs))

    #     matches = {}
    #     needs_context = []
    #     for i, src_lbl in enumerate(src_with_emb):
    #         if src_lbl in quick_matches:
    #             matches[src_lbl] = quick_matches[src_lbl]
    #             continue

    #         best_idx = sim_matrix[i].argmax()
    #         best_score = sim_matrix[i, best_idx]
    #         tgt_lbl = tgt_with_emb[best_idx]

    #         if best_score >= threshold:
    #             if FuzzyMatcher._is_negation_pair(src_lbl, tgt_lbl)[0]:
    #                 continue
    #             s_tok = FuzzyMatcher.tokenize(src_lbl)
    #             t_tok = FuzzyMatcher.tokenize(tgt_lbl)
    #             if (s_tok.issubset(t_tok) or t_tok.issubset(s_tok)) and abs(len(s_tok) - len(t_tok)) >= 2:
    #                 needs_context.append((src_lbl, tgt_lbl, best_score))
    #                 continue
    #             matches[src_lbl] = tgt_lbl
    #         else:
    #             needs_context.append((src_lbl, tgt_lbl, best_score))

    #     # Context-enhanced matching for ambiguous pairs
    #     if needs_context:
    #         src_var = ctx.src.main_label
    #         tgt_var = ctx.tgt.main_label
    #         for src_lbl, tgt_lbl, _ in needs_context:
    #             src_key = f"{src_var}::{src_lbl}"
    #             tgt_key = f"{tgt_var}::{tgt_lbl}"
    #             if src_key not in CategoryMapper._label_embedding_cache:
    #                 continue
    #             if not FuzzyMatcher._has_token_overlap(src_lbl, tgt_lbl):
    #                 continue
    #             if tgt_key in CategoryMapper._label_embedding_cache:
    #                 score = cosine_similarity(
    #                     [CategoryMapper._label_embedding_cache[src_key]],
    #                     [CategoryMapper._label_embedding_cache[tgt_key]]
    #                 )[0][0]
    #                 if score >= threshold:
    #                     matches[src_lbl] = tgt_lbl

    #     return {**quick_matches, **matches}

    @staticmethod
    def build_label_mapping(ctx: CandidateContext,
                            src_categories_labels: List[str], src_original_categories: List[str],
                            tgt_categories_labels: List[str], tgt_original_categories: List[str],
                            use_fuzzy: bool = True) -> Dict[str, Any]:
        """Build categorical value mapping between source and target."""
        def c2l(values, labels):
            return {v: l for v, l in zip(values, labels) if v and l}

        src_c2l = c2l(src_original_categories, src_categories_labels)
        tgt_c2l = c2l(tgt_original_categories, tgt_categories_labels)

        fuzzy_map = {}
        if use_fuzzy and src_categories_labels and tgt_categories_labels:
            fuzzy_map = CategoryMapper._fuzzy_match_labels(
                ctx, src_categories_labels, tgt_categories_labels, threshold=0.8
            )

        def l2codes_fuzzy(c2l_map, fuzzy_map=None):
            lab2codes, pretty = {}, {}
            for c, l in c2l_map.items():
                key = l.lower()
                if fuzzy_map and key in fuzzy_map:
                    key = fuzzy_map[key]
                lab2codes.setdefault(key, []).append(c)
                pretty.setdefault(key, l)
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

        return {
            "mapping_str": "; ".join(items) if items else None,
            "code_map": code_map,
            "overlap_labels": [src_pretty[k] for k in overlap_keys],
            "unmapped_source_labels": "; ".join(unmapped_src),
            "unmapped_target_labels": "; ".join(unmapped_tgt),
            "has_overlap": bool(overlap_keys),
            "identical": identical,
            "overlap_count": len(overlap_keys),
            "src_unique_labels": len(src_lab),
            "tgt_unique_labels": len(tgt_lab),
            "fuzzy_matches": len(fuzzy_map) if use_fuzzy else 0,
        }


# =====================================================================
# 3. ContextGate — reads pre-computed context scores
# =====================================================================

class ContextGate:
    """Pre-gate using context scores computed upstream in resolve_matches.

    Decision table:
      context_match_type | action
      ───────────────────┼─────────────────────────────
      'exact'            | pass through, no cap
      'close_match'      | pass through, flag for PARTIAL cap
      'subsumed'         | pass through, flag for PARTIAL cap
      'mismatch'         | NOT_APPLICABLE, stop
      None + low score   | NOT_APPLICABLE, stop
      None + high score  | pass through
      no context         | pass through
      one-sided context  | NOT_APPLICABLE, stop
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

  
    # def apply(self, ctx: CandidateContext):

    #     def _has_context(labels):
    #      return bool(labels) and any(l.strip().lower() != 'nan' for l in labels)
    #     src_ctx = ctx.src.context_labels
    #     tgt_ctx = ctx.tgt.context_labels

    #     # Neither has context → pass through
    #     if not src_ctx and not tgt_ctx:
    #         return

    #     # One has context, other doesn't → mismatch
    #     if _has_context(src_ctx) != _has_context(tgt_ctx):
    #         print(f"One variable has context, the other doesn't. {src_ctx} {tgt_ctx}")
    #         ctx.set_result(
    #             MatchLevel.NOT_APPLICABLE,
    #             reason="One variable has context, the other doesn't.",
    #             extra_details={"transformation": TransformationType.MANUAL_REVIEW},
    #             should_stop=True,
    #         )
    #         return

    #     # Both have context — check pre-computed match type
    #     ctx_type = ctx.context_match_type

    #     if ctx_type  == ContextMatchType.EXACT.value:
    #         return

    #     if ctx_type in [ContextMatchType.SUBSUMED.value, ContextMatchType.COMPATIBLE.value, ContextMatchType.PARTIAL.value]:
    #         ctx.context_is_subsumed = True
    #         return

    #     if ctx_type == ContextMatchType.NOT_APPLICABLE.value:
    #         ctx.set_result(
    #             MatchLevel.NOT_APPLICABLE,
    #             reason="Context mismatch, variables measure different things.",
    #             extra_details={"transformation": TransformationType.MANUAL_REVIEW},
    #             should_stop=True,
    #         )
    #         return

    #     # ctx_type is None → use sim_score threshold
    #     if ctx.sim_score is not None:
    #         score = round(ctx.sim_score, 2)
    #         if score < self.threshold:
    #             ctx.set_result(
    #                 MatchLevel.NOT_APPLICABLE,
    #                 reason=f"Context similarity {score:.2f} below threshold {self.threshold}.",
    #                 extra_details={"transformation": TransformationType.MANUAL_REVIEW},
    #                 should_stop=True,
    #             )
    #             return

    def apply(self, ctx: CandidateContext):
        def _has_context(labels):
            return bool(labels) and any(l.strip().lower() != 'nan' for l in labels)



     
        # ── Trust pre-computed context match type FIRST ──────────
        ctx_type = ctx.context_match_type

        if isinstance(ctx_type, str) and ctx_type == "pending":
            ctx_type = None
        if ctx_type is not None:
            if ctx_type == ContextMatchType.EXACT.value:
                return
            if ctx_type in [ContextMatchType.SUBSUMED.value, ContextMatchType.COMPATIBLE.value, ContextMatchType.PARTIAL.value] and ctx.sim_score is not None and ctx.sim_score < 0.85:
                ctx.context_is_subsumed = True
                return
            if ctx_type == ContextMatchType.NOT_APPLICABLE.value:
                ctx.set_result(
                    MatchLevel.NOT_APPLICABLE,
                    reason="Context mismatch, variables measure different things.",
                    extra_details={"transformation": TransformationType.MANUAL_REVIEW},
                    should_stop=True,
                )
                return

        # ── No pre-computed type → fall back to raw context_labels ──
        src_ctx = ctx.src.context_labels
        tgt_ctx = ctx.tgt.context_labels

        if not src_ctx and not tgt_ctx:
            return
        
        if _has_context(src_ctx) != _has_context(tgt_ctx):
            ctx.set_result(
                MatchLevel.NOT_APPLICABLE,
                reason="One variable has context, the other doesn't.",
                extra_details={"transformation": TransformationType.MANUAL_REVIEW},
                should_stop=True,
            )
            return

        # ctx_type is None and both have context → use sim_score threshold
        if ctx.sim_score is not None:
            score = round(ctx.sim_score, 2)
            if score < self.threshold:
                ctx.set_result(
                    MatchLevel.NOT_APPLICABLE,
                    reason=f"Context similarity {score:.2f} below threshold {self.threshold}.",
                    extra_details={"transformation": TransformationType.MANUAL_REVIEW},
                    should_stop=True,
                )
                return


# =====================================================================
# 4. Statistical Handlers
# =====================================================================

class Constraint(ABC):
    @abstractmethod
    def apply(self, ctx: CandidateContext):
        pass


class ContinuousHandler:
    """Unit compatibility for continuous variables."""

    @staticmethod
    def apply(ctx: CandidateContext):
        # src_unit = str(ctx.src.unit).lower() if ctx.src.unit else None
        # tgt_unit = str(ctx.tgt.unit).lower() if ctx.tgt.unit else None
        # units_same = (src_unit == tgt_unit ) if (src_unit and tgt_unit) else False
        # 
        # unit_info = {"source_unit": src_unit, "target_unit": tgt_unit}

        src_unit = str(ctx.src.unit).lower() if ctx.src.unit else None
        tgt_unit = str(ctx.tgt.unit).lower() if ctx.tgt.unit else None
        unit_info = {"source_unit": src_unit, "target_unit": tgt_unit}

        # Trust LLM: if context was judged EXACT, treat units as matched
        comparable_flag = (ctx.context_match_type in {ContextMatchType.EXACT.value}
                    and ctx.mapping_mode == MappingType.NE.value)   # we use in hybrid or neural matching only (rely on LLM as unit is passed to LLM along with variable description)
        units_same = (src_unit == tgt_unit) if (src_unit and tgt_unit) else False
        units_differ = (src_unit != tgt_unit) and not comparable_flag if (src_unit and tgt_unit) else False

        if (units_same or comparable_flag):
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.IDENTICAL,
                reason_exact="Same units, same timepoint.",
                reason_undetermined="Same units but undetermined timepoint.",
                reason_na="Same units but different timepoint.",
                extra_details={**unit_info, "transformation": TransformationType.VALUE_NORMALIZATION,
                            **ctx.timepoint_details},
            ))
        elif units_differ or ctx.is_derived_variable:
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.COMPATIBLE,
                reason_exact=f"Units differ ({src_unit} vs {tgt_unit}), same timepoint.",
                reason_undetermined=f"Units differ ({src_unit} vs {tgt_unit}), undetermined timepoint.",
                reason_na="Different units and different timepoint.",
                extra_details={**unit_info, "transformation": TransformationType.UNIT_CONVERSION},
            ))
        else:
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact="Incomplete unit information.",
                reason_undetermined="Incomplete unit information.",
                reason_na="Incomplete unit information.",
                extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW},
            ))
        # if (units_same or exact_match_flag)  and not ctx.is_derived_variable:
        
        #     ctx.set_result_with_timepoint(TimepointResult(
        #         current_level=MatchLevel.IDENTICAL,
        #         reason_exact="Same units, same timepoint.",
        #         reason_undetermined="Same units but undetermined timepoint.",
        #         reason_na="Same units but different timepoint.",
        #         extra_details={**unit_info, "transformation": TransformationType.VALUE_NORMALIZATION,
        #                        **ctx.timepoint_details},
        #     ))
        # elif units_differ or ctx.is_derived_variable:
        #     print(f"units_differ: {units_differ} and derived variable: {ctx.is_derived_variable}")
        #     ctx.set_result_with_timepoint(TimepointResult(
        #         current_level=MatchLevel.COMPATIBLE,
        #         reason_exact=f"Units differ ({src_unit} vs {tgt_unit}), same timepoint.",
        #         reason_undetermined=f"Units differ ({src_unit} vs {tgt_unit}), undetermined timepoint.",
        #         reason_na="Different units and different timepoint.",
        #         extra_details={**unit_info, "transformation": TransformationType.UNIT_CONVERSION},
        #     ))
        # else:
        #     ctx.set_result_with_timepoint(TimepointResult(
        #         current_level=MatchLevel.PARTIAL,
        #         reason_exact="Incomplete unit information.",
        #         reason_undetermined="Incomplete unit information.",
        #         reason_na="Incomplete unit information.",
        #         extra_details={**unit_info, "transformation": TransformationType.MANUAL_REVIEW},
        #     ))


class CategoricalHandler:
    """Same-type categorical comparison (binary-binary or multi-multi)."""

    @staticmethod
    def apply(ctx: CandidateContext, s_cats: Set[str], t_cats: Set[str],
              cat_label_map: Optional[Dict]):
        s_raw = set(ctx.src.original_categories)
        t_raw = set(ctx.tgt.original_categories)
        _range_match = RangeHelper.exact_match(ctx)

        # Case 1: Equality
        if (s_cats == t_cats) or (s_raw == t_raw) or _range_match:
            if s_raw == t_raw:
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.IDENTICAL,
                    reason_exact="Same codes, same raw values, same timepoint.",
                    reason_undetermined="Same codes, same raw values, undetermined timepoint.",
                    reason_na="Same values but different timepoint.",
                    extra_details={"categories": s_cats, "transformation": TransformationType.VALUE_NORMALIZATION},
                ), should_stop=True)
            else:
                extra = {"transformation": TransformationType.VALUE_NORMALIZATION}
                if cat_label_map:
                    extra.update(cat_label_map)
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.COMPATIBLE,
                    reason_exact=" Same codes (diff format), same timepoint.",
                    reason_undetermined="Same codes (diff format), undetermined timepoint.",
                    reason_na="Format differs and timepoint differs.",
                    extra_details=extra,
                ), should_stop=True)
            return

        # Case 2: Strict subset
        if (s_cats < t_cats) or (t_cats < s_cats):
            subset = s_cats if s_cats < t_cats else t_cats
            superset = t_cats if s_cats < t_cats else s_cats
            extra = {
                "transformation": TransformationType.AGGREGATION_OR_EXPANSION,
                "subset_categories": subset, "superset_categories": superset,
            }
            if cat_label_map:
                extra.update(cat_label_map)
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact="Strict subset, same timepoint.",
                reason_undetermined="Strict subset, undetermined timepoint.",
                reason_na="Subset valid but timepoint differs.",
                extra_details=extra,
            ), should_stop=True)
            return

        # Case 3: Partial overlap
        if cat_label_map and cat_label_map.get("has_overlap"):
            extra = {"transformation": TransformationType.AGGREGATION_OR_EXPANSION}
            extra.update(cat_label_map)
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact="Some overlapping categories, same timepoint.",
                reason_undetermined="Some overlapping categories, undetermined timepoint.",
                reason_na="Overlap valid but timepoint differs.",
                extra_details=extra,
            ), should_stop=True)
            return

        # Case 4: No overlap
        if ctx.sim_score is not None and ctx.sim_score == 1.0:
            ctx.set_result(
                MatchLevel.PARTIAL,
                reason="No code overlap but high label similarity.",
                extra_details={
                    "transformation": TransformationType.AGGREGATION_OR_EXPANSION,
                    "source_categories": s_cats, "target_categories": t_cats,
                }, should_stop=True,
            )
        else:
            ctx.set_result(
                MatchLevel.NOT_APPLICABLE,
                reason="No overlap in categorical codes.",
                extra_details={
                    "transformation": TransformationType.MANUAL_REVIEW,
                    "source_categories": s_cats, "target_categories": t_cats,
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

        ctx.set_result_with_timepoint(TimepointResult(
            current_level=MatchLevel.PARTIAL,
            reason_exact=(
                f"Categorical ({categorical_side}) extractable from "
                f"continuous ({continuous_side}) via threshold, same timepoint."
            ),
            reason_undetermined="Binary extractable from continuous, undetermined timepoint.",
            reason_na="Categorical expansion or aggregation after manual review and timepoint alignment.",
            extra_details={
                "transformation": TransformationType.AGGREGATION_OR_EXPANSION,
                "extraction_direction": "continuous_to_binary",
                "continuous_variable_side": continuous_side,
                "categorical_variable_side": categorical_side,
                "source_type": ctx.src.statistical_type, "target_type": ctx.tgt.statistical_type,
            },
        ))


class ContinuousBinaryHandler:
    """Continuous vs Binary: binary can be extracted from continuous via threshold."""

    @staticmethod
    def apply(ctx: CandidateContext):
        cont = StatisticalType.CONTINUOUS.value
        binary = StatisticalType.BINARY.value

        if ctx.src.statistical_type == cont and ctx.tgt.statistical_type == binary:
            continuous_side, binary_side = "source", "target"
        elif ctx.src.statistical_type == binary and ctx.tgt.statistical_type == cont:
            continuous_side, binary_side = "target", "source"
        else:
            return

        ctx.set_result_with_timepoint(TimepointResult(
            current_level=MatchLevel.PARTIAL,
            reason_exact=(
                f"Binary ({binary_side}) extractable from "
                f"continuous ({continuous_side}) via threshold, same timepoint."
            ),
            reason_undetermined="Binary extractable from continuous, undetermined timepoint.",
            reason_na="Binary extraction valid but timepoint differs.",
            extra_details={
                "transformation": TransformationType.BINARY_EXTRACTION,
                "extraction_direction": "continuous_to_binary",
                "continuous_variable_side": continuous_side,
                "binary_variable_side": binary_side,
                "source_type": ctx.src.statistical_type, "target_type": ctx.tgt.statistical_type,
            },
        ))


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
            is_dir_subset = s_cats.issubset(t_cats)
        elif ctx.src.statistical_type == StatisticalType.MULTI_CLASS.value and ctx.tgt.statistical_type == StatisticalType.BINARY.value:
            binary_cats, multi_cats = t_cats, s_cats
            binary_code_id, binary_label = ctx.tgt.main_id, ctx.tgt.main_label or ctx.tgt.description
            multi_code_ids, multi_labels = ctx.src.category_ids, ctx.src.category_labels or ctx.src.original_categories
            binary_side, study = 'target', ctx.src.study
            is_dir_subset = t_cats.issubset(s_cats)
        else:
            return

        has_overlap = cat_label_map.get("has_overlap", False) if cat_label_map else False
        overlap_count = cat_label_map.get("overlap_count", 0) if cat_label_map else 0
        unmapped_src = cat_label_map.get("unmapped_source_labels", "") if cat_label_map else ""

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
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact="Binary subset of Multi-class, same timepoint.",
                reason_undetermined="Binary subset of Multi-class, undetermined timepoint.",
                reason_na="Subset valid but timepoint differs.",
                extra_details=base,
            ))
            return

        # CASE 3: Full semantic overlap
        if has_overlap and not unmapped_src:
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact=f"All {overlap_count} binary labels match multi-class, same timepoint.",
                reason_undetermined=f"All {overlap_count} binary labels match, undetermined timepoint.",
                reason_na="Semantic match valid but timepoint differs.",
                extra_details=base,
            ), should_stop=True)
            return

        # CASE 4: Partial semantic overlap
        if has_overlap and unmapped_src:
            base["transformation"] = TransformationType.AGGREGATION_OR_EXPANSION
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact=f"{overlap_count} labels matched, unmapped: [{unmapped_src}].",
                reason_undetermined=f"{overlap_count} labels matched, unmapped: [{unmapped_src}], undetermined.",
                reason_na="Partial overlap but timepoint differs.",
                extra_details={
                    **base,
                    "unmapped_source_labels": unmapped_src,
                    "unmapped_target_labels": cat_label_map.get("unmapped_target_labels", "") if cat_label_map else "",
                },
            ), should_stop=True)
            return

        # CASE 5: Collapsible categories
        if cat_label_map and overlap_count > 0:
            ctx.set_result_with_timepoint(TimepointResult(
                current_level=MatchLevel.PARTIAL,
                reason_exact="Multi-class contains sub-types of the Binary state.",
                reason_undetermined="Potential collapsible categories, undetermined timepoint.",
                reason_na="Different timepoints for collapsible categories.",
                extra_details={
                    "transformation": TransformationType.BINARY_EXTRACTION,
                    "mapped_categories": cat_label_map.get("overlap_labels"),
                    "unmapped_target_labels": cat_label_map.get("unmapped_target_labels"),
                },
            ))
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
    def _try_subsumption(ctx, binary_ids,
                         binary_label, multi_ids, multi_labels, 
                         binary_side, study) -> bool:
        """Check subsumption via direct label match or OMOP graph."""
        binary_lower = binary_label.lower().strip()

        for cat_label in multi_labels:
            if binary_lower == cat_label.lower().strip():
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.PARTIAL,
                    reason_exact=f"Binary extracts '{cat_label}' (direct label match).",
                    reason_undetermined=f"Binary extracts '{cat_label}', undetermined timepoint.",
                    reason_na="Extraction valid but timepoint differs.",
                    extra_details={
                        "transformation": TransformationType.BINARY_EXTRACTION,
                        "binary_variable_side": binary_side,
                        "matched_category_label": cat_label,
                        "mapping_relation": "direct_label_match",
                    },
                ), should_stop=True)
                return True

        if ctx.matcher and multi_labels:
            for cat_id, cat_label in zip(
                multi_ids if len(multi_ids) == len(multi_labels) else [''] * len(multi_labels),
                multi_labels
            ):
            # for cat_omop, cat_label in zip(multi_omops, multi_labels):
                is_match, method = ctx.matcher.check_categorical_subsumption(
                    category_concept=(str(cat_id or ''), cat_label),
                    main_concept=(str(binary_ids or ''), binary_label),
                    target_study=study,
                )
                if is_match:
                    ctx.set_result_with_timepoint(TimepointResult(
                        current_level=MatchLevel.PARTIAL,
                        reason_exact=f"Binary extracts '{cat_label}' ({method}).",
                        reason_undetermined=f"Binary extracts '{cat_label}', undetermined.",
                        reason_na="Extraction valid but timepoint differs.",
                        extra_details={
                            "transformation": TransformationType.BINARY_EXTRACTION,
                            "binary_variable_side": binary_side,
                            "matched_category_label": cat_label,
                            "matched_category_omop": cat_id,
                            "mapping_relation": method,
                        },
                    ), should_stop=True)
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

        ctx.set_result_with_timepoint(TimepointResult(
            current_level=MatchLevel.PARTIAL,
            reason_exact=f"'{aggregate_cat}' aggregates {len(aggregated_multi)} categories.",
            reason_undetermined="Binary aggregates multi-class, undetermined timepoint.",
            reason_na="Aggregation or expansion after manual review and timepoint alignment.",
            extra_details=extra,
        ))
        return True


# =====================================================================
# 5. Range Helper
# =====================================================================

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


# =====================================================================
# 6. StatisticalLogicConstraint (dispatcher)
# =====================================================================

class StatisticalLogicConstraint(Constraint):
    """Routes to the appropriate type-specific handler."""

    def apply(self, ctx: CandidateContext):
        s_type = ctx.src.statistical_type
        t_type = ctx.tgt.statistical_type
        cat_types = {StatisticalType.BINARY.value, StatisticalType.MULTI_CLASS.value}
        cont = StatisticalType.CONTINUOUS.value
        binary = StatisticalType.BINARY.value
        valid_types = {st.value for st in StatisticalType}

        # Invalid or missing types
        if s_type not in valid_types or t_type not in valid_types:
            if ctx.is_derived_variable:
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.COMPATIBLE,
                    reason_exact="Variables are compatible with same timepoint.",
                    reason_undetermined="Variables are compatible with undetermined timepoint.",
                    reason_na="Different timepoints.",
                    extra_details={"transformation": TransformationType.DERIVATION},
                ))
            else:
                ctx.set_result(MatchLevel.NOT_APPLICABLE, extra_details={
                    "reason": f"Invalid or missing statistical types: {s_type} vs {t_type}."
                })
            return

        # Same type
        if s_type == t_type:
            if s_type == StatisticalType.CONTINUOUS.value:
                ContinuousHandler.apply(ctx)
            elif s_type in cat_types:
                cat_map = self._build_cat_map(ctx)
                CategoricalHandler.apply(ctx, set(ctx.src.category_labels), set(ctx.tgt.category_labels), cat_map)
            else:
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.IDENTICAL,
                    reason_exact="Variables are compatible with same timepoint.",
                    reason_undetermined="Variables are compatible with undetermined timepoint.",
                    reason_na="Different timepoints.",
                ))
            return

        # Type mismatch
        cont = StatisticalType.CONTINUOUS.value
        qual = StatisticalType.QUALITATIVE.value

        # Type mismatch: continuous vs qualitative
        if {s_type, t_type} == {cont, qual}:
            src_dt = ctx.src.data_type
            tgt_dt = ctx.tgt.data_type
            datetime_val = DATA_TYPE.DATETIME.value
            non_temporal = {DATA_TYPE.INTEGER.value, DATA_TYPE.STRING.value, DATA_TYPE.FLOAT.value}

            if src_dt and tgt_dt and datetime_val in {src_dt, tgt_dt} and {src_dt, tgt_dt} & non_temporal:
                ctx.set_result(MatchLevel.NOT_APPLICABLE, extra_details={
                    "reason": f"Data type mismatch: {src_dt} vs {tgt_dt} (temporal vs non-temporal)."
                })
            else:
                ctx.set_result(
                    MatchLevel.PARTIAL,
                    reason=f"Statistical type mismatch: {s_type} vs {t_type}.",
                    extra_details={"transformation": TransformationType.MANUAL_REVIEW},
                )
            return
        # Qualitative vs Categorical
        if qual in {s_type, t_type} and (s_type in cat_types or t_type in cat_types):
            ctx.set_result(
                MatchLevel.PARTIAL,
                reason=f"Statistical type mismatch:{s_type} vs {t_type}.",
                extra_details={"transformation": TransformationType.MANUAL_REVIEW},
            )
            return
        if s_type in cat_types and t_type in cat_types:
            cat_map = self._build_cat_map(ctx)
            CategoricalHandler.apply(ctx, set(ctx.src.category_labels), set(ctx.tgt.category_labels), cat_map)
            return

        if {s_type, t_type} == {cont, binary}:
            ContinuousBinaryHandler.apply(ctx)
            return
        # Continuous vs Categorical
        is_cont_vs_cat = (
            (s_type == cont and t_type in cat_types) or
            (t_type == cont and s_type in cat_types)
        )
        if is_cont_vs_cat:
            if RangeHelper.exact_match(ctx):
                s_min, s_max, t_min, t_max = RangeHelper.get_minmax(ctx)
                ctx.set_result_with_timepoint(TimepointResult(
                    current_level=MatchLevel.PARTIAL,
                    reason_exact=f"continuous vs categorical with identical range [{s_min}-{s_max}].",
                    reason_undetermined="continuous vs categorical with identical range, undetermined timepoint.",
                    reason_na="Different timepoints for range-matched variables.",
                    extra_details={
                        "transformation": TransformationType.VALUE_NORMALIZATION,
                        "source_range": f"[{s_min}, {s_max}]",
                        "target_range": f"[{t_min}, {t_max}]",
                    },
                ))
            else:
                ContinuousCategoricalHandler.apply(ctx)
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
            extra_details={"transformation": TransformationType.MANUAL_REVIEW},
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


# =====================================================================
# 7. ConstraintSolver — Orchestrator
# =====================================================================

class ConstraintSolver:
    """Pipeline: ContextGate → StatisticalLogic → SubsumedCap.

    Context scores are pre-computed upstream (in resolve_matches).
    ContextGate reads them as a strict gate. If mismatch, structural
    analysis is never performed. If subsumed, result is capped
    to PARTIAL after structural analysis.
    """

    def __init__(self, matcher: Optional[MatcherProtocol] = None):
        self.neurosymbolic_matcher = matcher
        self.context_gate = ContextGate(threshold=0.75)
        self.statistical_logic = StatisticalLogicConstraint()

    @staticmethod
    def check_visit_string(visit_str_1: str, visit_str_2: str) -> str:
        s_low = visit_str_1.lower()
        t_low = visit_str_2.lower()
        for hint in settings.DATE_HINTS:
            if hint in s_low and hint in t_low:
                return visit_str_1
            elif hint in s_low:
                return visit_str_2
            elif hint in t_low:
                return visit_str_1
        return visit_str_1

    def solve(self, src: VariableNode, tgt: VariableNode, mapping_mode: MappingType = MappingType.OEH.value) -> Tuple[Dict, str]:
        ctx = CandidateContext(src=src, tgt=tgt, matcher=self.neurosymbolic_matcher, mapping_mode=mapping_mode)

        def _finalize(ctx: CandidateContext):
            t = ctx.details.get("transformation")
            ctx.details["transformation"] = t.value if isinstance(t, TransformationType) else (t or TransformationType.NONE.value)
            return ctx.details, ctx.current_level.to_str()

        self.context_gate.apply(ctx)
        if ctx.should_stop:
            return _finalize(ctx)

        self.statistical_logic.apply(ctx)

        if ctx.context_is_subsumed:
            # existing = ctx.details.get("transformation")
            # if not existing or existing == TransformationType.NONE:
            ctx.details["transformation"] = TransformationType.MANUAL_REVIEW
            ctx.details["description"] = f"Need further review and analysis before harmonization. {TransformationType.MANUAL_REVIEW.value}"
            ctx.current_level = MatchLevel.PARTIAL
        return _finalize(ctx)
        

       
        
