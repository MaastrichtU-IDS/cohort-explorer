"""Neuro-symbolic matching with typed VariableNode inputs.
"""

import json
import os
from collections import defaultdict
import pandas as pd
from typing import Any, Dict, List
from .llm_call import LLMConceptMatcher
from .config import settings
from .data_model import (
    MappingType, VariableNode, StatisticalType, 
    ContextMatchType, MappingRelation
)
from .vector_db import search_category_by_id, search_in_db, _embed_cache,  _cache_key
from .fuzz_match import FuzzyMatcher
from .derived_variables import compute_derived_variables
from .graph_similarity import (
    score_context,
    parse_post_cordinating_concepts_labels
)
from .utils import setup_logger, clean_label_remove_temporal_context

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

logger = setup_logger('llm_matcher.log')

def _node_to_profile_cols(node: VariableNode, side: str) -> Dict[str, Any]:
    """Flatten VariableNode profile fields into prefixed DataFrame columns.

    Produces the exact column names expected by VariableNode.from_source_row()
    and VariableNode.from_target_row(), so the constraint solver gets typed inputs.
    """
    st = node.statistical_type
    stats = node.statistics
    if isinstance(stats, tuple):
        stats = stats[0] if stats else None
    return {
        f"{side}_type": st.value if isinstance(st, StatisticalType) else (st or ""),
        f"{side}_unit": node.unit,
        f"{side}_categories_labels": "||".join(node.category_labels) if node.category_labels else "",
        f"{side}_categories_omop_ids": "||".join(str(x) for x in node.category_ids) if node.category_ids else "",
        f"{side}_original_categories": "||".join(node.original_categories) if node.original_categories else "",
        f"{side}_composite_code_labels": "||".join(node.context_labels) if node.context_labels else "",
        f"{side}_composite_code_omop_ids": "||".join(str(x) for x in node.context_ids) if node.context_ids else "",
        f"{side}_composite_code_codes": "||".join(str(x) for x in node.context_codes) if node.context_codes else "",
        f"{side}_min_val": stats.min_val  if stats else None ,
        f"{side}_max_val": stats.max_val  if stats else None ,
        f"{side}_data_type": node.data_type or "",
    }

        
def _build_match_dict(
    src: VariableNode, tgt: VariableNode,
    relation: str, ctx_type: str, sim_score: float,
) -> Dict[str, Any]:
    """Build a complete match dict including profile columns.

    The resulting dict can be directly used in pd.DataFrame() and contains
    every column the constraint solver expects.
    """
    return {
        "source": src.name,
        "target": tgt.name,
        "source_label": src.description,
        "target_label": tgt.description,
        "somop_id": src.main_id,
        "tomop_id": tgt.main_id,
        "scode": src.main_code,
        "tcode": tgt.main_code,
        "slabel": src.main_label,
        "tlabel": tgt.main_label,
        "source_visit": src.visit,
        "target_visit": tgt.visit,
        "category": src.category or tgt.category,
        "mapping_relation": relation,
        "context_match_type": ctx_type,
        "sim_score": sim_score,
        
        **_node_to_profile_cols(src, "source"),
        **_node_to_profile_cols(tgt, "target"),
    }


# When one (source, target) pair is reached by several routes, the strongest
# relation is kept: symbolic evidence first, embedding similarity last.
_RELATION_RANK = {r.value: i for i, r in enumerate((
    MappingRelation.SymbolicExactMatch, MappingRelation.SymbolicCloseMatch,
    MappingRelation.SymbolicNarrowMatch, MappingRelation.SymbolicBroadMatch,
    MappingRelation.NeuralMatch))}


def _relation_rank(relation: str) -> int:
    return _RELATION_RANK.get((relation or "").strip().lower(), len(_RELATION_RANK))


# =====================================================================
# NeuroSymbolicMatcher
# =====================================================================

class NeuroSymbolicMatcher:
    def __init__(self, vector_db: Any, embed_model: Any, graph: Any = None,
                 collection_name: str = "concepts",
                 mapping_mode: str = MappingType.OEH.value, llm_model: List[str] = None, similarity_threshold: float = 0.8):
        self.vector_db = vector_db
        self.embed_model = embed_model
        self.graph = graph
        self.collection_name = collection_name
        self.mapping_mode = mapping_mode
        self.llm_matcher = None
        self.similarity_threshold = similarity_threshold
        # self.floor_threshold = floor_threshold
        #self.similarity_threshold = 0.5 if mapping_mode == MappingType.NE.value else self.similarity_threshold
        if llm_model is not None:
            self.llm_matcher = LLMConceptMatcher(
                            models=[llm_model],
                            temperature=0, mode=mapping_mode)
        # self.unit_converter = (UnitConverter.from_csv(unit_csv)
        #                        if unit_csv else UnitConverter())
        # (f"similarity threshold: {self.similarity_threshold} for model: {embed_model.model_name}")


    # =================================================================
    # Vector query text for Qdrant search
    # =================================================================

    def _vector_query(self, node: VariableNode) -> str:
        """Build query text for vector DB search from a VariableNode."""
        query = clean_label_remove_temporal_context(node.description or node.name)
        if self.mapping_mode == MappingType.OEH.value:
            # query = node.description
            if node.main_label not in query:
                query += f", {node.main_label}"
            for label in node.context_labels:
                if label not in query:
                    query += f", {label}"
        elif self.mapping_mode == MappingType.OEC.value:
            query = node.main_label
            for label in node.context_labels:
                if label not in query:
                    query += f", {label}"
      
        return query

    def _label_vector_query(self, node: VariableNode) -> str:
        """Cleaned study label only: used by NE and OEH label channel."""
        return clean_label_remove_temporal_context(
            node.description or node.name or ""
        ).lower().strip()

    def _concept_vector_query(self, node: VariableNode) -> str:
        """Main concept + context concepts: used by OEC and OEH concept channel."""
        parts = []
        if node.main_label:
            parts.append(str(node.main_label).lower().strip())
        for label in node.context_labels:
            if label:
                parts.append(str(label).lower().strip())
        parts = FuzzyMatcher._deduplicate_parts(parts=parts, threshold=0.8)
        return " ".join(parts).strip() or self._label_vector_query(node)

        

    def _compute_pair_context_graph_only(self, src_node, tgt_node):
        if not self.graph or self.mapping_mode == MappingType.NE.value:
            return ContextMatchType.PENDING.value, 0.0
        src_codes = list(src_node.context_ids)
        if src_node.main_id and src_node.main_id not in src_codes:
            src_codes.insert(0, src_node.main_id)
        tgt_codes = list(tgt_node.context_ids)
        if tgt_node.main_id and tgt_node.main_id not in tgt_codes:
            tgt_codes.insert(0, tgt_node.main_id)
        ct, score = score_context(
            src_codes=src_codes, tgt_codes=tgt_codes,
            src_values=list(src_node.category_ids),
            tgt_values=list(tgt_node.category_ids),
            src_labels=list(src_node.context_labels),
            tgt_labels=list(tgt_node.context_labels),
            src_val_labels=list(src_node.category_labels),
            tgt_val_labels=list(tgt_node.category_labels),
            graph=self.graph, embed_model=None,
            mapping_mode=self.mapping_mode,
        )
        # exact/subsumed → graph confirmed, trust it, else pass to LLM for final verdict
        return (ct, score) if ct in (ContextMatchType.EXACT.value, ContextMatchType.SUBSUMED.value, ContextMatchType.COMPATIBLE.value) else (ContextMatchType.PENDING.value, 0.0)

    def _build_evidence_block(self, src_node, tgt_node):
        if not self.graph or self.mapping_mode == MappingType.NE.value:
            return ""
        def _concept_context(concept_id):
            """Return concept's immediate parents and vocabulary — one line."""
            if not concept_id or not self.graph:
                return ""
            name = self.graph.get_node_attr(concept_id, "concept_name")
            vocab = self.graph.get_node_attr(concept_id, "vocabulary")
            parents = self.graph.get_parents(concept_id)
            parent_names = [self.graph.get_node_attr(p, "concept_name") for p in parents[:3]]
            parent_names = [n for n in parent_names if n]
            parts = []
            if name:
                parts.append(name)
            if vocab:
                parts.append(f"[{vocab}]")
            if parent_names:
                parts.append(f"is_a: {', '.join(parent_names)}")
            return " ".join(parts) if parts else ""
        src_codes = ([src_node.main_id] if src_node.main_id else []) + list(src_node.context_ids)
        tgt_codes = ([tgt_node.main_id] if tgt_node.main_id else []) + list(tgt_node.context_ids)
        lines = []

        shared = set(src_codes) & set(tgt_codes)
        if shared:
            names = [self.graph.get_node_attr(c, "concept_name") for c in shared if c]
            lines.append(f"shared_concepts: {', '.join(n for n in names if n)}")

        if src_node.main_id and tgt_node.main_id and src_node.main_id != tgt_node.main_id:
            result = self.graph.explain_path(src_node.main_id, tgt_node.main_id, max_depth=4)
            if result["path_type"] != "no_path":
                lines.append(f"ontology_relation: {result['explanation']}")
            else:
                # No connection — give each concept's position separately
                src_ctx = _concept_context(src_node.main_id)
                tgt_ctx = _concept_context(tgt_node.main_id)
                if src_ctx:
                    lines.append(f"source_hierarchy: {src_ctx}")
                if tgt_ctx:
                    lines.append(f"target_hierarchy: {tgt_ctx}")

        if (set(src_codes) & set(tgt_node.category_ids) or
            set(tgt_codes) & set(src_node.category_ids)):
            lines.append("note: concept appears in other's value set")
        evidence = " | ".join(lines) if lines else ""
        # print(f"evidence: {evidence}")
        return evidence



    # =================================================================
    # Main Matching — Pipeline modularized
    # =================================================================
    def generate_candidates(self, src_collection, tgt_collection, target_study):
        """Phase 1: Discover candidates via Graph and Vector DB."""
        final_matches = []
        is_ne = self.mapping_mode == MappingType.NE.value

        # Trace one source concept end-to-end through candidate generation.
        # Set TRACE_SOURCE_CONCEPT to an omop id (e.g. 4060077) to print which
        # route proposed each target and what relation it carried.
        _trace_cid = os.getenv("TRACE_SOURCE_CONCEPT")
        _trace_cid = int(_trace_cid) if _trace_cid and _trace_cid.strip().isdigit() else None

        if is_ne:
            src_grouped, tgt_map, tgt_by_desc, tgt_name_to_desc = {}, {}, {}, {}
            for v in src_collection.variables:
                key = clean_label_remove_temporal_context(v.description or v.name)
                src_grouped.setdefault(key, []).append(v)
            for v in tgt_collection.variables:
                nk = (v.name or "").lower()
                dk = clean_label_remove_temporal_context(v.description or v.name)
                tgt_map.setdefault(nk, []).append(v)
                tgt_name_to_desc[nk] = dk
                tgt_by_desc.setdefault(dk, []).append(v)
            unique_tgt_ids = set(tgt_map.keys())
  
        else:
            src_grouped = src_collection._by_omop_id or {}
            tgt_map = tgt_collection._by_omop_id or {}
            unique_tgt_ids = tgt_collection.omop_ids
        # print(f"tgt_name_to_desc.values()[0] = {tgt_name_to_desc.values()[0]}")
        # print(f"tgt_by_desc.values()[0] = {tgt_by_desc.values()[0]}")
        use_graph = bool(self.graph) and not is_ne
        use_vector = (self.vector_db is not None and self.embed_model is not None
                      and self.mapping_mode != MappingType.OO.value)

        if use_vector:
            # One query per distinct variable text (description + own concepts),
            # not one per main OMOP id.
            all_query_texts = list({
                self._vector_query(node)
                for nodes in src_grouped.values() for node in nodes
            })
            uncached = [t for t in all_query_texts
                        if _cache_key(self.embed_model.model_name, t) not in _embed_cache]
            if uncached:
                vectors = self.embed_model.embed_batch(uncached, is_query=True)
                for text, vec in zip(uncached, vectors):
                    _embed_cache[_cache_key(self.embed_model.model_name, text)] = vec.tolist()
                # print(f" ✅ Pre-embedded {len(uncached)} query texts (1 batch)")

        # One entry per (source variable, target variable). Each pair keeps its
        # own relation; a pair reached by several routes keeps the strongest.
        pairs = {}

        def add_pair(src_node, tgt_node, relation, score, trace):
            if not FuzzyMatcher.visits_compatible(src_node.visit, tgt_node.visit):
                return
            key = (id(src_node), id(tgt_node))
            held = pairs.get(key)
            if trace:
                logger.info(f"[TRACE {trace}]     {src_node.name} -> {tgt_node.name} "
                            f"(tgt.main_id={tgt_node.main_id}) rel={relation}"
                            + ("" if held is None else f" :: pair already held '{held[2]}'"))
            if held is None:
                pairs[key] = [src_node, tgt_node, relation, score]
            elif _relation_rank(relation) < _relation_rank(held[2]):
                held[2], held[3] = relation, score

        def targets_of(tid, names=None):
            if is_ne:
                tk = (tid or "").lower()
                tdesc = tgt_name_to_desc.get(tk)
                return tgt_by_desc.get(tdesc, []) if tdesc else tgt_map.get(tk, [])
            nodes = tgt_map.get(tid, [])
            if names:
                # Only the target variables of the retrieved point. An index
                # built from an older dictionary may name none of the current
                # variables; the hit then falls back to the whole main id.
                wanted = {n.strip().lower() for n in names}
                own = [n for n in nodes if (n.name or "").strip().lower() in wanted]
                if own:
                    return own
            return nodes

        for sid, s_group in src_grouped.items():
            graph_candidates = set()

            _trace = sid if (_trace_cid is not None and sid == _trace_cid) else None
            if _trace:
                logger.info(f"[TRACE {sid}] source group: {[n.name for n in s_group]}")

            if use_graph:
                if sid in unique_tgt_ids:
                    graph_candidates.add((sid, MappingRelation.SymbolicExactMatch.value))
                    if _trace:
                        logger.info(f"[TRACE {sid}]   self-id route -> tid={sid} exactmatch"
                                    f" | tgt_map[{sid}] = {[n.name for n in tgt_map.get(sid, [])]}")
                others = unique_tgt_ids - {sid}
                if others:
                    reachable = self.graph.source_to_targets_paths(
                        sid, others, max_depth=settings.DEFAULT_GRAPH_DEPTH)
                    if reachable:
                        graph_candidates.update((tid, rel) for tid, rel in reachable)
                    if _trace:
                        logger.info(f"[TRACE {sid}]   graph route  -> {len(reachable or [])} target(s): "
                                    f"{sorted(reachable or [])[:12]}")

            # Self-id and graph routes work on concept ids, so they pair every
            # variable of the group.
            for tid, relation in sorted(graph_candidates, key=lambda c: (str(c[0]), str(c[1]))):
                for tgt_node in targets_of(tid):
                    for src_node in s_group:
                        add_pair(src_node, tgt_node, relation, None, _trace)

            if use_vector:
                # Each variable searches with its own description and concepts;
                # variables with identical query text share one search.
                graph_tids = {tid for tid, _ in graph_candidates}
                by_query = {}
                for node in s_group:
                    by_query.setdefault(self._vector_query(node), []).append(node)
                for query, members in by_query.items():
                    raw_matches = search_in_db(
                        vectordb=self.vector_db,
                        embedding_model=self.embed_model,
                        query_text=query,
                        target_study=[target_study],
                        limit=settings.LIMIT,
                        min_threshold=self.similarity_threshold,
                        collection_name=self.collection_name,
                        mapping_mode=self.mapping_mode,
                        with_names=not is_ne,
                    )
                    if _trace:
                        logger.info(f"[TRACE {sid}]   vector route {[n.name for n in members]} -> "
                                    f"{len(raw_matches)} hit(s): {raw_matches[:12]}")
                    seen_descs = set()
                    for item in raw_matches:
                        item = item if isinstance(item, tuple) else (item, 0.0)
                        tid, score = item[0], item[1]
                        names = item[2] if len(item) > 2 else None
                        if tid in graph_tids:
                            continue
                        if is_ne:
                            tdesc = tgt_name_to_desc.get((tid or "").lower())
                            if not tdesc or tdesc in seen_descs:
                                continue
                            seen_descs.add(tdesc)
                        for tgt_node in targets_of(tid, names):
                            for src_node in members:
                                add_pair(src_node, tgt_node, MappingRelation.NeuralMatch.value, score, _trace)

        for src_node, tgt_node, relation, score in pairs.values():
            rel_str = relation.strip().lower() if relation else "unknown"
            final_matches.append(_build_match_dict(
                src_node, tgt_node, rel_str, ContextMatchType.PENDING.value, score))
        return final_matches

    def resolve_pending_with_llm(self,
                                      ambiguous_indices,
                                      structurals,
                                      src_study,
                                      tgt_study,
                                      tuple_to_evidence):
        """LLM consultation for the three-phase pipeline.

        Returns:
            Dict[int, LLMEvidence] — keyed by the input indices. An index
            absent from the result means no LLM evidence is available;
            policy.decide() will fall back to structural-only.
        """
        if not self.llm_matcher or self.mapping_mode == MappingType.OO.value:
            return {}

        # ── Step 1: collapse rows by concept_key so identical concepts
  
        concept_matches = {}
        idx_to_ckey = {}
        for idx in ambiguous_indices:
            src_node, tgt_node, _ = structurals[idx]
            ckey = self._concept_key(src_node, tgt_node)
            if ckey not in concept_matches:
                concept_matches[ckey] = {
                    "src_rep": src_node,
                    "tgt_rep": tgt_node,
                    "indices": [],
                }
            concept_matches[ckey]["indices"].append(idx)
            idx_to_ckey[idx] = ckey

        if not concept_matches:
            return {}

        # ── Step 2: build LLM groups (mirrors _llm_resolve_concepts shape).
        is_ne = self.mapping_mode == MappingType.NE.value
        src_groups = defaultdict(list)
        for ckey in concept_matches.keys():
            src_sig = (ckey[0], ckey[1])
            src_groups[src_sig].append(ckey)

        groups, flat_keys = [], []
        for src_sig, keys in src_groups.items():
            entry0 = concept_matches[keys[0]]
            src = entry0["src_rep"]
            targets = []
            for ckey in keys:
                entry = concept_matches[ckey]
                tgt = entry["tgt_rep"]
                tgt_entry = {
                    "tgt_cats": " | ".join(ckey[3]),
                    "tgt_unit": tgt.unit or "",
                    "desc": self._desc_for_mode(tgt),
                }
                if not is_ne:
                    tgt_entry["tgt_concepts"] = ckey[2]
                evidence = self._build_evidence_block(src, tgt)
                if evidence:
                    tgt_entry["evidence"] = evidence
                targets.append(tgt_entry)
                flat_keys.append(ckey)

            group = {
                "src_cats": " | ".join(src_sig[1]),
                "src_unit": src.unit or "",
                "src_desc": self._desc_for_mode(src),
                "targets": targets,
            }
            if not is_ne:
                group["src_concepts"] = src_sig[0]
            groups.append(group)

        # ── Step 3: call the LLM once across all groups.
        case_ids = [f"P{i}" for i in range(len(flat_keys))]
        if len(flat_keys) > 1:
            logger.info(f"  🤖 LLM Validating : {len(flat_keys)} concept pairs")
        else:
            logger.debug(f"  🤖 LLM Validating : {len(flat_keys)} concept pair")
        grouped_results, stats = self.llm_matcher.assess(groups, case_ids=case_ids)
        if stats.get("not_reached") or stats.get("unreadable"):
            logger.warning(
                f"  🤖 LLM {stats.get('model')}: {stats.get('total_targets')} pairs, "
                f"{stats.get('not_reached', 0)} not reached, {stats.get('unreadable', 0)} unreadable "
                f"({stats.get('total_llm_calls')} calls); their structural verdicts are kept and flagged"
            )

        # ── Step 4: convert verdicts to LLMEvidence

        ckey_to_evidence = {}
        flat_pos = 0
        for grp_verdicts in grouped_results:
            for verdict_tuple in grp_verdicts:
                ckey = flat_keys[flat_pos]
                # A missing result is a failure to get a verdict, not a rejection.
                ckey_to_evidence[ckey] = tuple_to_evidence(
                    verdict_tuple if verdict_tuple else
                    (None, ContextMatchType.PENDING.value, 0.0,
                     json.dumps({"status": "NOT_REACHED", "reason": "no result returned for this pair"}))
                )
                flat_pos += 1

        # ── Step 5: fan out concept-level verdicts to every row that
        #           mapped to that concept. Runs ONCE, after all groups
        #           have been processed.
        results = {}
        for idx, ckey in idx_to_ckey.items():
            ev = ckey_to_evidence.get(ckey)
            if ev is not None:
                results[idx] = ev
        
        return results

    

    def compute_derived_variables(self, df: pd.DataFrame, src_collection, tgt_collection) -> pd.DataFrame:
        """Phase 3: Append derived variables to the mapped DataFrame (see derived_variables.py)."""
        if self.mapping_mode == MappingType.NE.value:
            return df
        return compute_derived_variables(
            df, src_collection, tgt_collection, self.graph, settings.DERIVED_VARIABLES_LIST)

    def _concept_key(self, src_node, tgt_node):
        """Build a concept-level key that collapses visit variants."""
        use_ontology = self.mapping_mode != MappingType.NE.value
        if use_ontology:
            s_parts = [l.strip() for l in src_node.context_labels if l.strip()] or [src_node.main_label]
            t_parts = [l.strip() for l in tgt_node.context_labels if l.strip()] or [tgt_node.main_label]
        else:
            s_parts = [clean_label_remove_temporal_context(src_node.description) or src_node.name]
            t_parts = [clean_label_remove_temporal_context(tgt_node.description) or tgt_node.name]
        if src_node.unit: s_parts.append(src_node.unit)
        if tgt_node.unit: t_parts.append(tgt_node.unit)

        # s_cats = tuple(sorted(l.strip() for l in
        #     (src_node.category_labels if use_ontology else src_node.category_labels) if l.strip()))
        # t_cats = tuple(sorted(l.strip() for l in
        #     (tgt_node.category_labels if use_ontology else tgt_node.category_labels) if l.strip()))

        # s_cats = tuple(sorted(l.strip() for l in src_node.category_labels if l.strip()))
        # t_cats = tuple(sorted(l.strip() for l in tgt_node.category_labels if l.strip()))

        s_cat_src = src_node.category_pairs or src_node.category_labels
        t_cat_src = tgt_node.category_pairs or tgt_node.category_labels
        s_cats = tuple(sorted(l.strip() for l in s_cat_src if l.strip()))
        t_cats = tuple(sorted(l.strip() for l in t_cat_src if l.strip()))

        return (" | ".join(s_parts), s_cats, " | ".join(t_parts), t_cats)

    def _llm_resolve_concepts(self, pending_keys, concept_matches):
        is_ne = self.mapping_mode == MappingType.NE.value

        src_groups = defaultdict(list)
        for key in pending_keys:
            src_sig = (key[0], key[1])
            src_groups[src_sig].append(key)

        groups, flat_keys = [], []
        for src_sig, keys in src_groups.items():
            entry0 = concept_matches[keys[0]]
            src = entry0["src_rep"]
            targets = []
            for key in keys:
                entry = concept_matches[key]
                tgt = entry["tgt_rep"]
                tgt_entry = {
                    "tgt_cats": " | ".join(key[3]),
                    "tgt_unit": tgt.unit or "",
                    "desc": self._desc_for_mode(tgt),
                }
                if not is_ne:
                    tgt_entry["tgt_concepts"] = key[2]
                evidence = self._build_evidence_block(src, tgt)
                if evidence:
                    tgt_entry["evidence"] = evidence
                targets.append(tgt_entry)
                flat_keys.append(key)

            group = {
                "src_cats": " | ".join(src_sig[1]),
                "src_unit": src.unit or "",
                "src_desc": self._desc_for_mode(src),
                "targets": targets,
            }
            if not is_ne:
                group["src_concepts"] = src_sig[0]
            groups.append(group)

        case_ids = [f"P{i}" for i in range(len(flat_keys))]
        logger.info(f"  🤖 LLM: {len(flat_keys)} concept pairs")

        grouped_results, stats = self.llm_matcher.assess(groups, case_ids=case_ids)
        
        results = {}
        flat_pos = 0
        for grp_verdicts in grouped_results:
            for verdict in grp_verdicts:
                key = flat_keys[flat_pos]
                if verdict and verdict[0] is True:
                    results[key] = (verdict[1], verdict[2], verdict[3])
                else:  # False OR None/empty → explicitly reject
                    results[key] = (ContextMatchType.NOT_APPLICABLE.value, 0.0)
                flat_pos += 1

        return results

    def _desc_for_mode(self, node):
        if self.mapping_mode == MappingType.OEC.value:
            parts = [l.strip() for l in node.context_labels if l.strip()]
            return ", ".join(parts) if parts else node.main_label
        return clean_label_remove_temporal_context(node.description) or node.name

  

    def check_categorical_subsumption(self, category_concept, main_concept, target_study):
        cid, clabel = category_concept
        mid, mlabel = main_concept

        if clabel.lower().strip() == mlabel.lower().strip():
            return True, "Direct Label Match"

        if self.mapping_mode == MappingType.NE.value:
            if self.embed_model:
                c_key = clean_label_remove_temporal_context(clabel.lower().strip())
                m_key = clean_label_remove_temporal_context(mlabel.lower().strip())
                c_vec = _embed_cache.get(_cache_key(self.embed_model.model_name, c_key)) or self.embed_model.embed_text(c_key, is_query=True)
                m_vec = _embed_cache.get(_cache_key(self.embed_model.model_name, m_key)) or self.embed_model.embed_text(m_key, is_query=True)
                if float(cos_sim([c_vec], [m_vec])[0][0]) >= self.similarity_threshold:
                    return True, MappingRelation.NeuralMatch.value
            return False, None

        cid_valid = str(cid).strip().lstrip('-').isdigit()   # guard both paths
        mid_valid = str(mid).strip().lstrip('-').isdigit()
        # if not cid_valid or not mid_valid:
        #     print(f"category label {cid} and main var label {mid}")
        if self.graph and cid_valid and mid_valid:
            if self.graph.source_to_targets_paths(int(cid), {int(mid)}, max_depth=1):
                return True, MappingRelation.SymbolicCloseMatch.value

        # Only enter vector fallback when both IDs are real — prevents empty-string false positives
        if self.embed_model and cid_valid and mid_valid:
            t_raw_matches = search_category_by_id(
                vectordb=self.vector_db, category_id=mid,
                embedding_model=self.embed_model,
                query_text=clabel.lower().strip(),
                target_study=[target_study],
                collection_name=self.collection_name,
                limit=settings.LIMIT,
            )
            if t_raw_matches:
                for match in t_raw_matches:
                    if str(match.get("parent_variable_concept", '')).strip() == str(cid).strip():
                        return True, MappingRelation.NeuralMatch.value

        return False, None
    def _precompute_catvalues_similarity(self, df: pd.DataFrame, model_object: Any) -> pd.DataFrame:
        from .constraints import CategoryMapper
        if df.empty:
            return df

        label_cache = CategoryMapper._label_omop_cache
        emb_cache   = CategoryMapper._label_embedding_cache
        align_cache = CategoryMapper._alignment_cache

        all_labels, ctx_pairs = set(), set()
        src_ids_per_row, tgt_ids_per_row = [], []

        cols = ['slabel', 'tlabel',
                'source_categories_labels', 'source_categories_omop_ids',
                'target_categories_labels', 'target_categories_omop_ids']
        sub = df.reindex(columns=cols).fillna('')

        for slabel, tlabel, s_lbls, s_oids, t_lbls, t_oids in sub.itertuples(index=False, name=None):
            s_row, t_row = [], []
            for lbls_raw, oids_raw, concept, dst in (
                (s_lbls, s_oids, slabel, s_row),
                (t_lbls, t_oids, tlabel, t_row),
            ):
                if not lbls_raw or not str(lbls_raw).strip():
                    continue
                labels = [l.strip().lower() for l in parse_post_cordinating_concepts_labels(lbls_raw) if l.strip()]
                omops  = str(oids_raw).split('||') if oids_raw else []
                n_o = len(omops)
                for i, lbl in enumerate(labels):
                    all_labels.add(lbl)
                    if concept:
                        ctx_pairs.add((concept, lbl))
                    if i < n_o:
                        o = omops[i].strip()
                        if o.isdigit():
                            oid = int(o)
                            label_cache[lbl] = oid
                            dst.append(oid)
            src_ids_per_row.append(s_row)
            tgt_ids_per_row.append(t_row)

        # print(f"🔤 Ontology cache: {len(label_cache)} label→OMOP mappings")

        if self.graph is not None:
            pairs_to_check = {
                (sid, tid) if sid <= tid else (tid, sid)
                for s_ids, t_ids in zip(src_ids_per_row, tgt_ids_per_row)
                for sid in s_ids for tid in t_ids if sid != tid
            } - align_cache.keys()
            if pairs_to_check:
                # print(f"🧭 Resolving {len(pairs_to_check)} unique OMOP-pair alignments via graph...")
                resolve = self.graph.source_to_targets_paths
                aligned = 0
                for sid, tid in pairs_to_check:
                    hit = bool(resolve(sid, {tid}, max_depth=1))
                    align_cache[(sid, tid)] = hit
                    aligned += hit
                # print(f"✅ Alignment cache: {aligned} aligned / {len(pairs_to_check)} checked")

        if self.mapping_mode == MappingType.OO.value or not all_labels:
            return df

        # Encode ONLY what isn't already cached — biggest speedup across repeated calls
        new_lbls = [l for l in all_labels if l not in emb_cache]
        new_ctx  = [(c, l) for c, l in ctx_pairs if f"{c}::{l}" not in emb_cache]
        # cached_n = (len(all_labels) - len(new_lbls)) + (len(ctx_pairs) - len(new_ctx))

        if new_lbls or new_ctx:
            texts = new_lbls + [f"{c} {l}" for c, l in new_ctx]
            keys  = new_lbls + [f"{c}::{l}" for c, l in new_ctx]
            # print(f"🔤 Pre-encoding {len(texts)} NEW embeddings "
                # f"({len(new_lbls)} labels + {len(new_ctx)} contextualized; {cached_n} reused from cache)...")
            emb_cache.update(zip(keys, model_object.embed_batch(texts, show_progress=False)))
        # else:
        #     print(f"🔤 All {cached_n} embeddings reused from cache — no encoding needed.")

        # print(f"✅ Cache size: {len(emb_cache)} embeddings")
        return df