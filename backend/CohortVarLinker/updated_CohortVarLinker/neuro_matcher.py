"""Neuro-symbolic matching with typed VariableNode inputs.
"""

from collections import defaultdict
import pandas as pd
from typing import Any, Dict, List, Set,Tuple
from .llm_call import LLMConceptMatcher
from .config import settings
from .data_model import (
    MappingType, VariableNode, StatisticalType, 
    ContextMatchType,TransformationType
)
from .vector_db import search_category_by_id, search_in_db, _embed_cache,  _cache_key
from .fuzz_match import FuzzyMatcher
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
    return {
        f"{side}_type": st.value if isinstance(st, StatisticalType) else (st or ""),
        f"{side}_unit": node.unit,
        f"{side}_categories_labels": "||".join(node.category_labels) if node.category_labels else "",
        f"{side}_categories_omop_ids": "||".join(str(x) for x in node.category_ids) if node.category_ids else "",
        f"{side}_original_categories": "||".join(node.original_categories) if node.original_categories else "",
        f"{side}_composite_code_labels": "||".join(node.context_labels) if node.context_labels else "",
        f"{side}_composite_code_omop_ids": "||".join(str(x) for x in node.context_ids) if node.context_ids else "",
        f"{side}_composite_code_codes": "||".join(str(x) for x in node.context_codes) if node.context_codes else "",
        f"{side}_min_val": node.statistics.min_val,
        f"{side}_max_val": node.statistics.max_val,
        f"{side}_data_type": node.data_type,
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


# =====================================================================
# NeuroSymbolicMatcher
# =====================================================================

class NeuroSymbolicMatcher:
    def __init__(self, vector_db: Any, embed_model: Any, graph: Any = None,
                 collection_name: str = "concepts",
                 mapping_mode: str = MappingType.OEH.value, llm_models: List[str] = None, similarity_threshold: float = 0.8):
        self.vector_db = vector_db
        self.embed_model = embed_model
        self.graph = graph
        self.collection_name = collection_name
        self.mapping_mode = mapping_mode
        self.llm_matcher = None
        self.similarity_threshold = similarity_threshold
        # self.floor_threshold = floor_threshold
        #self.similarity_threshold = 0.5 if mapping_mode == MappingType.NE.value else self.similarity_threshold
        if llm_models is not None:
            self.llm_matcher = LLMConceptMatcher(
                            models=llm_models,
                            temperature=0, mode=mapping_mode)
        print(f"similarity threshold: {self.similarity_threshold} for model: {embed_model.model_name}")

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
        # print(f"query: {query}")
        # elif self.mapping_mode == MappingType.NE.value:
        #     if node.unit and str(node.unit).strip():
        #         query += f", unit:{node.unit.strip()}"
        #     elif node.original_categories and len(node.original_categories) > 0:
        #         query += f", values:{'|'.join(node.original_categories)}"
        return query



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
    # Derived Variables (dict-based, converted at call site)
    # =================================================================

    def _extend_with_derived_variables(
        self,
        single_source: dict,
        standard_derived_variable: tuple,
        parameters_omop_ids: list,
        variable_name: str,
        category: str,
        stats_type: str = "continuous_variable",
        unit_label: str = "",
    ) -> List[Dict]:
        """Check if a derived variable can be synthesized. Returns list of match dicts."""
        data_context = single_source.copy()

        def _find_omop_id_rows(data_list: list, omop_code: int, code_key: str = "omop_id") -> list:
            found = []
            for row in data_list:
                try:
                    curr_val = int(row.get(code_key, 0))
                except (ValueError, TypeError):
                    continue
                if curr_val == int(omop_code):
                    found.append(row)
            return found

        def _get_parameter_visits(data: dict, parameters_codes: list, side: str = "source") -> Tuple[bool, Set[str]]:
            code_key_mapped = "somop_id" if side == "source" else "tomop_id"
            visit_key_mapped = "source_visit" if side == "source" else "target_visit"
            visits = set()
            all_found = True
            for code in parameters_codes:
                unmapped_rows = _find_omop_id_rows(data[side], code, code_key="omop_id")
                mapped_rows = _find_omop_id_rows(data['mapped'], code, code_key=code_key_mapped)
                if unmapped_rows:
                    for row in unmapped_rows:
                        v = str(row.get('visit', '')).strip().lower()
                        if v:
                            visits.add(v)
                elif mapped_rows:
                    for row in mapped_rows:
                        v = str(row.get(visit_key_mapped, '')).strip().lower()
                        if v:
                            visits.add(v)
                else:
                    all_found = False
                    break
            return all_found, visits

        def _find_aligned_visit_pairs(source_visits: Set[str], target_visits: Set[str]) -> Dict[Tuple[str, str], bool]:
            aligned_pairs = {}
            for src_vis in source_visits:
                for tgt_vis in target_visits:
                    if FuzzyMatcher.check_visit_string(src_vis, tgt_vis):
                        aligned_pairs[(src_vis, tgt_vis)] = True
            return aligned_pairs

        def _get_varname_for_visit(derived_rows: list, visit: str, default_name: str, key: str) -> str:
            for row in derived_rows:
                row_visit = str(row.get('visit', '')).strip().lower()
                if row_visit == visit:
                    return row.get(key, default_name)
            return default_name

        target_omop_id = int(standard_derived_variable[2])
        source_derived_rows = _find_omop_id_rows(data_context["source"], target_omop_id, code_key="omop_id")
        target_derived_rows = _find_omop_id_rows(data_context["target"], target_omop_id, code_key="omop_id")

        source_can_calc, source_param_visits = _get_parameter_visits(data_context, parameters_omop_ids, side="source")
        target_can_calc, target_param_visits = _get_parameter_visits(data_context, parameters_omop_ids, side="target")

        for row in source_derived_rows:
            v = str(row.get('visit', '')).strip().lower()
            if v:
                source_param_visits.add(v)
        for row in target_derived_rows:
            v = str(row.get('visit', '')).strip().lower()
            if v:
                target_param_visits.add(v)

        source_valid = (len(source_derived_rows) > 0) or source_can_calc
        target_valid = (len(target_derived_rows) > 0) or target_can_calc
        if not (source_valid and target_valid):
            return []

        aligned_visit_pairs = _find_aligned_visit_pairs(source_param_visits, target_param_visits)
        if not aligned_visit_pairs:
            return []

        final_stats_type = stats_type
        all_rows = source_derived_rows + target_derived_rows
        if all_rows and all_rows[0].get('stats_type'):
            final_stats_type = all_rows[0].get('stats_type')

        src_unit, tgt_unit = "", ""
        if source_derived_rows and source_derived_rows[0].get('unit_label'):
            src_unit = source_derived_rows[0].get('unit_label')
        elif target_derived_rows and target_derived_rows[0].get('unit_label'):
            tgt_unit = target_derived_rows[0].get('unit_label')

        final_results = []
        for (src_visit, tgt_visit) in aligned_visit_pairs.keys():
            if not src_visit or not tgt_visit:
                continue
            source_varname = _get_varname_for_visit(
                source_derived_rows, src_visit, f"derived_{variable_name}", "source"
            )
            target_varname = _get_varname_for_visit(
                target_derived_rows, tgt_visit, f"derived_{variable_name}", "target"
            )
            final_results.append({
                "source": source_varname, "target": target_varname,
                "somop_id": target_omop_id, "tomop_id": target_omop_id,
                "scode": standard_derived_variable[0], "slabel": standard_derived_variable[1],
                "tcode": standard_derived_variable[0], "tlabel": standard_derived_variable[1],
                "source_visit": src_visit, "target_visit": tgt_visit,
                "category": category,
                "source_type": final_stats_type, "target_type": final_stats_type,
                "source_unit": src_unit if src_unit else unit_label,
                "target_unit": tgt_unit if tgt_unit else unit_label,
                "source_composite_code_labels": standard_derived_variable[1],
                "source_composite_code_omop_ids": f"{target_omop_id}",
                "target_composite_code_labels": standard_derived_variable[1],
                "target_composite_code_omop_ids": f"{target_omop_id}",
                "mapping_relation": "Symbolic:closeMatch",
                "context_match_type": ContextMatchType.EXACT.value,
                "sim_score": 1.0,
                "transformation_rule": {
                    "description": f"Derived variable {variable_name} using parameter columns {parameters_omop_ids}.",
                }
            })
        return final_results


    # =================================================================
    # Main Matching — Pipeline modularized
    # =================================================================
    def generate_candidates(self, src_collection, tgt_collection, target_study):
        """Phase 1: Discover candidates via Graph and Vector DB."""
        final_matches = []
        is_ne = self.mapping_mode == MappingType.NE.value

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
            # for v in tgt_collection.variables:
            #     tgt_map.setdefault(v.name, []).append(v)
            # tgt_name_to_desc = {v.name: clean_label_remove_temporal_context(
            #     v.description or v.name) for v in tgt_collection.variables}
            
            # for v in tgt_collection.variables:
            #     key = clean_label_remove_temporal_context(v.description or v.name)
            #     tgt_by_desc.setdefault(key, []).append(v)
            # unique_tgt_ids = set(tgt_map.keys())
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
            all_query_texts = list({
                self._vector_query(nodes[0])
                for nodes in src_grouped.values() if nodes
            })
            uncached = [t for t in all_query_texts
                        if _cache_key(self.embed_model.model_name, t) not in _embed_cache]
            if uncached:
                vectors = self.embed_model.embed_batch(uncached, is_query=True)
                for text, vec in zip(uncached, vectors):
                    _embed_cache[_cache_key(self.embed_model.model_name, text)] = vec.tolist()
                print(f" ✅ Pre-embedded {len(uncached)} query texts (1 batch)")

        concept_matches = {}
        total_groups = len(src_grouped)

        for idx, (sid, s_group) in enumerate(src_grouped.items()):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"resolve_matches: {idx + 1}/{total_groups} source groups...")

            rep = s_group[0]
            matched_candidates = set()

            if use_graph:
                if sid in unique_tgt_ids:
                    matched_candidates.add((sid, "Symbolic:exactMatch", None))
                others = unique_tgt_ids - {sid}
                if others:
                    reachable = self.graph.source_to_targets_paths(
                        sid, others, max_depth=settings.DEFAULT_GRAPH_DEPTH)
                    if reachable:
                        matched_candidates.update((tid, rel, None) for tid, rel in reachable) 

            if use_vector:
                graph_tids = {tid for tid, _, score in matched_candidates}
                query = self._vector_query(rep)
                # print(f"query ={query}")
                raw_matches = search_in_db(
                    vectordb=self.vector_db,
                    embedding_model=self.embed_model,
                    query_text=query,
                    target_study=[target_study],
                    limit=settings.LIMIT,
                    min_threshold=self.similarity_threshold,
                    collection_name=self.collection_name,
                    mapping_mode=self.mapping_mode,
                )
                if is_ne:
                    seen_descs = set()
                    for item in raw_matches:
                        tid, score = item if isinstance(item, tuple) else (item, 0.0)
                        tk = (tid or "").lower()
                        tdesc = tgt_name_to_desc.get(tk)
                        if tdesc and tdesc not in seen_descs and tid not in graph_tids:
                            seen_descs.add(tdesc)
                            matched_candidates.add((tid, "neural match", score))
                else:
                    for item in raw_matches:
                        tid, score = item if isinstance(item, tuple) else (item, 0.0)
                        if tid not in graph_tids:
                            matched_candidates.add((tid, "neural match", score))  # score preserved

            for tid, relation, score in matched_candidates:
                if is_ne:
                    tdesc = tgt_name_to_desc.get(tid)
                    tgt_nodes = tgt_by_desc.get(tdesc, []) if tdesc else tgt_map.get(tid, [])
                else:
                    tgt_nodes = tgt_map.get(tid, [])
                
                # seen_pairs = set()
                for tgt_node in tgt_nodes:
                    for src_node in s_group:
                        if not FuzzyMatcher.check_visit_string(src_node.visit, tgt_node.visit):
                            continue
                        ckey = self._concept_key(src_node, tgt_node)
                        if ckey not in concept_matches:
                            concept_matches[ckey] = {
                                "relation": relation,
                                "src_rep": src_node,
                                "tgt_rep": tgt_node,
                                "group_pairs": [],
                                # "visit":(src_node.visit, tgt_node.visit)
                            }
                        concept_matches[ckey]["group_pairs"].append((src_node, tgt_node, score))
        
        print(f"total concept_matches unique pairs: {len(concept_matches)}")
        for ckey, entry in concept_matches.items():
            rel_str = entry["relation"].strip().lower() if entry["relation"] else "unknown"
            for src_node, tgt_node, score in entry["group_pairs"]:
                final_matches.append(_build_match_dict(
                    src_node, tgt_node, rel_str, ContextMatchType.PENDING.value, score))
        return final_matches

    def resolve_pending_with_llm(self, df: pd.DataFrame, src_study: str, tgt_study: str) -> pd.DataFrame:
        """Phase 2: Extract unresolved rows, apply structural pre-filter, and run LLM."""
        if not self.llm_matcher or self.mapping_mode == MappingType.OO.value:
            return df
            
        from .constraints import StatisticalLogicConstraint, CandidateContext
        from .data_model import ContextMatchType, MatchLevel
        
        stat_logic = StatisticalLogicConstraint()
        
        # After compute_context_scores, unresolved rows will be NOT_APPLICABLE or pending/nan.
        needs_llm = df['context_match_type'].isna() | \
                    (df['context_match_type'] == ContextMatchType.PENDING.value) | \
                    (df['context_match_type'] == ContextMatchType.NOT_APPLICABLE.value)
        
        
        pending_indices = df[needs_llm].index
        concept_matches = {}
        pending_keys = set()
        
        for idx in pending_indices:
            row = df.loc[idx]
            src_node = VariableNode.from_source_row(row, study=src_study)
            tgt_node = VariableNode.from_target_row(row, study=tgt_study)
            
            # 🛡️ Structural Pre-filter (Blocks hallucinated pairings from reaching LLM)
            c_ctx = CandidateContext(src=src_node, tgt=tgt_node, matcher=self, mapping_mode=self.mapping_mode)
            stat_logic.apply(c_ctx)
            transform = c_ctx.details.get("transformation")
            needs_review = (transform ==TransformationType.MANUAL_REVIEW and c_ctx.current_level == MatchLevel.PARTIAL)
            if c_ctx.current_level == MatchLevel.NOT_APPLICABLE and not needs_review:
                df.at[idx, 'context_match_type'] = ContextMatchType.NOT_APPLICABLE.value
                continue
                
            ckey = self._concept_key(src_node, tgt_node)
            if ckey not in concept_matches:
                concept_matches[ckey] = {"src_rep": src_node, "tgt_rep": tgt_node, "indices": []}
            concept_matches[ckey]["indices"].append(idx)
            pending_keys.add(ckey)
            
        if pending_keys:
            print(f" 🤖 LLM resolving {len(pending_keys)} pending concept pairs (Structurally Pre-Filtered)...")
            llm_results = self._llm_resolve_concepts(list(pending_keys), concept_matches)
            
            for ckey, (ctx_type, score) in llm_results.items():
                for idx in concept_matches[ckey]["indices"]:
                    df.at[idx, 'context_match_type'] = ctx_type
                    df.at[idx, 'sim_score'] = score
                    
        return df

    def compute_derived_variables(self, df: pd.DataFrame, src_collection, tgt_collection) -> pd.DataFrame:
        """Phase 3: Append derived variables to the mapped DataFrame."""
        if self.mapping_mode == MappingType.NE.value:
            return df
            
        single_source_context = {
            "source": [n.to_element_dict("source") for n in src_collection.variables],
            "target": [n.to_element_dict("target") for n in tgt_collection.variables],
            "mapped": df.to_dict('records'),
        }
        
        new_matches = []
        for derived_def in settings.DERIVED_VARIABLES_LIST:
            derived_match = self._extend_with_derived_variables(
                single_source=single_source_context,
                standard_derived_variable=(
                    derived_def["code"], derived_def["label"], derived_def["omop_id"]),
                parameters_omop_ids=derived_def["required_omops"],
                variable_name=derived_def["name"],
                category=derived_def["category"],
                unit_label=derived_def.get("unit", ""),
            )
            if derived_match:
                new_matches.extend(derived_match)
                
        if new_matches:
            new_df = pd.DataFrame(new_matches)
            df = pd.concat([df, new_df], ignore_index=True)
            
        return df
    # def resolve_matches(self, src_collection, tgt_collection, target_study):
    #     final_matches = []
    #     is_ne = self.mapping_mode == MappingType.NE.value

    #     if is_ne:
    #         src_grouped = {}
    #         for v in src_collection.variables:
    #             key = clean_label_remove_temporal_context(v.description or v.name)
    #             src_grouped.setdefault(key, []).append(v)
    #         tgt_map = {}
    #         for v in tgt_collection.variables:
    #             tgt_map.setdefault(v.name, []).append(v)
    #         tgt_name_to_desc = {v.name: clean_label_remove_temporal_context(
    #             v.description or v.name) for v in tgt_collection.variables}
    #         tgt_by_desc = {}
    #         for v in tgt_collection.variables:
    #             key = clean_label_remove_temporal_context(v.description or v.name)
    #             tgt_by_desc.setdefault(key, []).append(v)
    #         unique_tgt_ids = set(tgt_map.keys())
    #     else:
    #         src_grouped = src_collection._by_omop_id or {}
    #         tgt_map = tgt_collection._by_omop_id or {}
    #         unique_tgt_ids = tgt_collection.omop_ids

    #     use_graph = bool(self.graph) and not is_ne # for symbolic match in stage 1
    #     use_vector = (self.vector_db is not None and self.embed_model is not None
    #                 and self.mapping_mode != MappingType.OO.value) # for semantic similatity match  in stage 1
    #     use_llm = (self.llm_matcher is not None # for further reasoning of uncertain pairs in stage 2
    #             and self.mapping_mode != MappingType.OO.value)

    #     # if use_vector:
    #     #     all_query_texts = list({
    #     #         self._vector_query(nodes[0])
    #     #         for nodes in src_grouped.values() if nodes
    #     #     })
    #     #     uncached = [t for t in all_query_texts if t not in _embed_cache]
    #     #     if uncached:
    #     #         vectors = self.embed_model.embed_batch(uncached, is_query=True)
    #     #         for text, vec in zip(uncached, vectors):
    #     #             _embed_cache[text] = vec.tolist()
    #     #         print(f" ✅ Pre-embedded {len(uncached)} query texts (1 batch)")

    #     # Line 334-338: use model-prefixed cache keys
    #     if use_vector:
    #         all_query_texts = list({
    #             self._vector_query(nodes[0])
    #             for nodes in src_grouped.values() if nodes
    #         })
    #         uncached = [t for t in all_query_texts
    #                     if _cache_key(self.embed_model.model_name, t) not in _embed_cache]
    #         if uncached:
    #             vectors = self.embed_model.embed_batch(uncached, is_query=True)
    #             for text, vec in zip[tuple[str, Any]](uncached, vectors):
    #                 _embed_cache[_cache_key(self.embed_model.model_name, text)] = vec.tolist()
    #             print(f" ✅ Pre-embedded {len(uncached)} query texts (1 batch)")

    #     # ── Stage 1: Discover similar target candidates via graph/similarity ──
    #     # concept_key → { "relation": str, "src_rep": node, "tgt_rep": node,
    #     #                  "visit_pairs": [(src_node, tgt_node), ...] }
    #     concept_matches = {}
    #     total_groups = len(src_grouped)

    #     for idx, (sid, s_group) in enumerate(src_grouped.items()):
    #         if (idx + 1) % 50 == 0 or idx == 0:
    #             print(f"resolve_matches: {idx + 1}/{total_groups} source groups...")

    #         rep = s_group[0]
    #         matched_candidates = set()

    #         if use_graph:
    #             if sid in unique_tgt_ids:
    #                 # check if both source variable and target variable has equal number of exact concepts 
    #                 matched_candidates.add((sid, "Symbolic:exactMatch", None))
    #             others = unique_tgt_ids - {sid}
    #             if others:
    #                 reachable = self.graph.source_to_targets_paths(
    #                     sid, others, max_depth=settings.DEFAULT_GRAPH_DEPTH)
    #                 if reachable:
    #                     matched_candidates.update(reachable)

    #         if use_vector:
    #             graph_tids = {tid for tid, _ in matched_candidates}
    #             query = self._vector_query(rep)
    #             raw_matches = search_in_db(
    #                 vectordb=self.vector_db,
    #                 embedding_model=self.embed_model,
    #                 query_text=query,
    #                 target_study=[target_study],
    #                 limit=settings.LIMIT,
    #                 # min_threshold=self.similarity_threshold,
    #                 # start_threshold=self.similarity_threshold,
    #                 collection_name=self.collection_name,
    #                 mapping_mode=self.mapping_mode,
    #             )
    #             if is_ne:
    #                 seen_descs = set()
    #                 for tid in raw_matches:
    #                     tid, score = tid if is_ne else (tid, 0.0)

    #                     tdesc = tgt_name_to_desc.get(tid)
    #                     if tdesc and tdesc not in seen_descs and tid not in graph_tids:
    #                         seen_descs.add(tdesc)
    #                         matched_candidates.add((tid, "Neural Match", score))
    #             else:
    #                 for tid in raw_matches:
    #                     if tid not in graph_tids:
    #                         matched_candidates.add((tid, "Neural Match", None))

    #         # ── Collect visit-aligned pairs, keyed by concept signature ──
    #         for tid, relation in matched_candidates:
    #             if is_ne:
    #                 tdesc = tgt_name_to_desc.get(tid)
    #                 tgt_nodes = tgt_by_desc.get(tdesc, []) if tdesc else tgt_map.get(tid, [])
    #             else:
    #                 tgt_nodes = tgt_map.get(tid, [])

    #             for tgt_node in tgt_nodes:
    #                 for src_node in s_group:
    #                     if not FuzzyMatcher.check_visit_string(src_node.visit, tgt_node.visit):
    #                         continue
    #                     ckey = self._concept_key(src_node, tgt_node)
    #                     if ckey not in concept_matches:
    #                         concept_matches[ckey] = {
    #                             "relation": relation,
    #                             "src_rep": src_node,
    #                             "tgt_rep": tgt_node,
    #                             "visit_pairs": [],
    #                         }
    #                     concept_matches[ckey]["visit_pairs"].append((src_node, tgt_node))

    #     n_concepts = len(concept_matches)
    #     n_visit_pairs = sum(len(v["visit_pairs"]) for v in concept_matches.values())
    #     print(f"📊 {n_concepts} unique concept pairs ({n_visit_pairs} visit-expanded)")

    #     # ── Phase 2: Graph context on CONCEPT representatives only ──
    #     concept_keys = list(concept_matches.keys())
    #     ctx_results = {}
    #     for ckey in concept_keys:
    #         entry = concept_matches[ckey]
    #         ctx_results[ckey] = self._compute_pair_context_graph_only(
    #             entry["src_rep"], entry["tgt_rep"])

    #     # ── Phase 3: LLM for pending concept pairs ──
    #     if use_llm:
    #         pending_keys = [k for k in concept_keys if ctx_results[k][0] == "pending"]
    #         if pending_keys:
    #             print(f" 🤖 LLM resolving {len(pending_keys)}/{n_concepts} pending concept pairs...")
    #             llm_results = self._llm_resolve_concepts(
    #                 pending_keys, concept_matches)
    #             ctx_results.update(llm_results)

    #     # ── Phase 4: Expand to all visit variants ──
    #     for ckey, entry in concept_matches.items():
    #         ctx_type, sim_score = ctx_results.get(ckey, ("pending", 0.0))
    #         rel_str = entry["relation"].strip().lower() if entry["relation"] else "unknown"
    #         for src_node, tgt_node in entry["visit_pairs"]:
    #             final_matches.append(_build_match_dict(
    #                 src_node, tgt_node, rel_str, ctx_type, sim_score))

    #     # ── Derived variables ──
    #     if not is_ne:
    #         single_source_context = {
    #             "source": [n.to_element_dict("source") for n in src_collection.variables],
    #             "target": [n.to_element_dict("target") for n in tgt_collection.variables],
    #             "mapped": final_matches,
    #         }
    #         for derived_def in settings.DERIVED_VARIABLES_LIST:
    #             derived_match = self._extend_with_derived_variables(
    #                 single_source=single_source_context,
    #                 standard_derived_variable=(
    #                     derived_def["code"], derived_def["label"], derived_def["omop_id"]),
    #                 parameters_omop_ids=derived_def["required_omops"],
    #                 variable_name=derived_def["name"],
    #                 category=derived_def["category"],
    #                 unit_label=derived_def.get("unit", ""),
    #             )
    #             if derived_match:
    #                 print(f"derived_match: {derived_match}")
    #                 final_matches.extend(derived_match)

    #     return final_matches

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

        s_cats = tuple(sorted(l.strip() for l in
            (src_node.category_labels if use_ontology else src_node.original_categories) if l.strip()))
        t_cats = tuple(sorted(l.strip() for l in
            (tgt_node.category_labels if use_ontology else tgt_node.original_categories) if l.strip()))

        return (" | ".join(s_parts), s_cats, " | ".join(t_parts), t_cats)

    def _llm_resolve_concepts_batch(self, pending_keys, concept_matches):
      
        pending_keys = sorted(list(pending_keys))
        MAX_TARGETS_PER_BATCH = 3
        is_ne = self.mapping_mode == MappingType.NE.value

        src_groups = defaultdict(list)
        for key in pending_keys:
            src_sig = (key[0], key[1])
            src_groups[src_sig].append(key)
       
        groups, flat_keys = [], []
       
        for src_sig, keys in sorted(src_groups.items()):

            entry0 = concept_matches[keys[0]]
            src = entry0["src_rep"]
            keys.sort()
            for chunk_start in range(0, len(keys), MAX_TARGETS_PER_BATCH):
                chunk = keys[chunk_start:chunk_start + MAX_TARGETS_PER_BATCH]
                targets = []
                for key in chunk:
                    tgt = concept_matches[key]["tgt_rep"]
                    tgt_entry = {
                        "tgt_cats": " | ".join(key[3]),
                        "tgt_unit": tgt.unit or "",
                        "desc": self._desc_for_mode(tgt),
                    }
                    if not is_ne:
                        tgt_entry["tgt_concepts"] = key[2]
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
        logger.info(f"  🤖 LLM: {len(flat_keys)} concept pairs in {len(groups)} groups (max {MAX_TARGETS_PER_BATCH}/group)")

        grouped_results, _ = self.llm_matcher.assess_batch(groups, case_ids=case_ids)

        results = {}
        flat_pos = 0
        for grp_verdicts in grouped_results:
            for verdict in grp_verdicts:
                key = flat_keys[flat_pos]
                if verdict and verdict[0] is True:
                    results[key] = (verdict[1], verdict[2])
                else:
                    results[key] = (ContextMatchType.NOT_APPLICABLE.value, 0.0)
                flat_pos += 1

        return results
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
                    results[key] = (verdict[1], verdict[2])
                else:  # False OR None/empty → explicitly reject
                    results[key] = (ContextMatchType.NOT_APPLICABLE.value, 0.0)
                flat_pos += 1

        return results

    def _desc_for_mode(self, node):
        if self.mapping_mode == MappingType.OEC.value:
            parts = [l.strip() for l in node.context_labels if l.strip()]
            return ", ".join(parts) if parts else node.main_label
        return clean_label_remove_temporal_context(node.description) or node.name

    
    # def check_categorical_subsumption(self, category_concept, main_concept, target_study):
    #     cid, clabel = category_concept
    #     mid, mlabel = main_concept

    #     # Direct string match — always valid
    #     if clabel.lower().strip() == mlabel.lower().strip():
    #         return True, "Direct Label Match"

    #     if self.mapping_mode == MappingType.NE.value:
    #         # Pure embedding similarity between category label and main concept label
    #         if self.embed_model:
    #             # from numpy import dot
    #             # from numpy.linalg import norm
    #             c_key = clean_label_remove_temporal_context(clabel.lower().strip())
    #             m_key = clean_label_remove_temporal_context(mlabel.lower().strip())
                
    #             c_vec = _embed_cache.get(_cache_key(self.embed_model.model_name, c_key)) or self.embed_model.embed_text(c_key, is_query=True)
    #             m_vec = _embed_cache.get(_cache_key(self.embed_model.model_name, m_key)) or self.embed_model.embed_text(m_key, is_query=True)    

    #             sim = float(cos_sim([c_vec], [m_vec])[0][0])
    #             if sim >= self.similarity_threshold:
    #                 return True, "Neural:closeMatch"
    #         return False, None

    #     # OMOP paths for ontology modes
    #     if self.graph:
    #         if cid and mid and str(cid).strip() and str(mid).strip():
    #             if self.graph.source_to_targets_paths(int(cid), {int(mid)}, max_depth=1):
    #                 return True, "Symbolic:closeMatch"
    #             # print(f"cid={cid}, mid={mid}")

    #     if self.embed_model:
    #         t_raw_matches = search_category_by_id(
    #             vectordb=self.vector_db, category_id=mid,
    #             embedding_model=self.embed_model,
    #             query_text=clabel.lower().strip(),
    #             target_study=[target_study],
    #             collection_name=self.collection_name,
    #             limit=settings.LIMIT,
    #         )
    #         if t_raw_matches:
    #             for match in t_raw_matches:
    #                 if str(match.get("parent_variable_concept", '')).lower().strip() == cid.lower().strip():
    #                     return True, "Neural:closeMatch"

    #     return False, None

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
                    return True, "Neural:closeMatch"
            return False, None

        cid_valid = str(cid).strip().lstrip('-').isdigit()   # guard both paths
        mid_valid = str(mid).strip().lstrip('-').isdigit()
        # if not cid_valid or not mid_valid:
        #     print(f"category label {cid} and main var label {mid}")
        if self.graph and cid_valid and mid_valid:
            if self.graph.source_to_targets_paths(int(cid), {int(mid)}, max_depth=1):
                return True, "Symbolic:closeMatch"

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
                        return True, "Neural:closeMatch"

        return False, None
    def _precompute_catvalues_similarity(self, df: pd.DataFrame, model_object: Any) -> pd.DataFrame:
        from .constraints import CategoryMapper
        if df.empty:
            return df

        all_labels, contextualized_pairs = set(), set()
        src_ids_per_row, tgt_ids_per_row = [], []

        for _, row in df.iterrows():
            row_ids = {'source_categories_omop_ids': [], 'target_categories_omop_ids': []}
            for lbl_col, omop_col, concept in [
                ('source_categories_labels', 'source_categories_omop_ids', row.get('slabel', '')),
                ('target_categories_labels', 'target_categories_omop_ids', row.get('tlabel', '')),
            ]:
                labels_raw = row.get(lbl_col)
                omops_raw  = row.get(omop_col)
                if not pd.notna(labels_raw) or not str(labels_raw).strip():
                    continue
                labels = [l.strip().lower() for l in parse_post_cordinating_concepts_labels(labels_raw) if l.strip()]
                omops  = [o.strip() for o in str(omops_raw).split('||')] if pd.notna(omops_raw) else []

                for i, lbl in enumerate(labels):
                    all_labels.add(lbl)
                    if concept:
                        contextualized_pairs.add((concept, lbl))
                    if i < len(omops) and omops[i].isdigit():
                        oid = int(omops[i])
                        CategoryMapper._label_omop_cache[lbl] = oid
                        row_ids[omop_col].append(oid)

            src_ids_per_row.append(row_ids['source_categories_omop_ids'])
            tgt_ids_per_row.append(row_ids['target_categories_omop_ids'])

        print(f"🔤 Ontology cache: {len(CategoryMapper._label_omop_cache)} label→OMOP mappings")

        # ── Pass 2: populate alignment cache  — runs in OO and OEH ──
        if self.graph is not None:
            pairs_to_check = set()
            for s_ids, t_ids in zip(src_ids_per_row, tgt_ids_per_row):
                for sid in s_ids:
                    for tid in t_ids:
                        if sid == tid:
                            continue
                        pairs_to_check.add((sid, tid) if sid <= tid else (tid, sid))

            if pairs_to_check:
                print(f"🧭 Resolving {len(pairs_to_check)} unique OMOP-pair alignments via graph...")
                aligned = 0
                for sid, tid in pairs_to_check:
                    hit = bool(self.graph.source_to_targets_paths(sid, {tid}, max_depth=1))
                    CategoryMapper._alignment_cache[(sid, tid)] = hit
                    aligned += int(hit)
                print(f"✅ Alignment cache: {aligned} aligned / {len(pairs_to_check)} checked")

        # ── Embedding precompute — OO skips, OEH/NE continue ──
        if self.mapping_mode == MappingType.OO.value or not all_labels:
            return df

        texts, keys = [], []
        for lbl in all_labels:
            texts.append(lbl); keys.append(lbl)
        for concept, lbl in contextualized_pairs:
            texts.append(f"{concept} {lbl}"); keys.append(f"{concept}::{lbl}")

        print(f"🔤 Pre-encoding {len(texts)} categorical embeddings "
            f"({len(all_labels)} labels + {len(contextualized_pairs)} contextualized)...")
        embeddings = model_object.embed_batch(texts, show_progress=False)
        for k, e in zip(keys, embeddings):
            CategoryMapper._label_embedding_cache[k] = e
        print(f"✅ Encoded {len(all_labels)} label-only + {len(contextualized_pairs)} contextualized embeddings")
        return df
    # def _precompute_catvalues_similarity(self, df: pd.DataFrame, model_object: Any) -> pd.DataFrame:
    #     from .constraints import CategoryMapper
    #     if self.mapping_mode == MappingType.OO.value:
    #         return df

    #     all_labels = set()
    #     contextualized_pairs = set()

    #     for _, row in df.iterrows():
    #         s_concept = row.get('slabel', '')
    #         t_concept = row.get('tlabel', '')

    #         for cat_col, fallback_col, concept in [
    #             ('source_categories_labels', 'source_original_categories', s_concept),
    #             ('target_categories_labels', 'target_original_categories', t_concept),
    #         ]:
    #             col = cat_col
    #             if not pd.notna(row.get(col)) or not str(row.get(col, '')).strip():
    #                 col = fallback_col
    #             if pd.notna(row.get(col)) and str(row.get(col, '')).strip():
    #                 labels = [l.strip().lower() for l in parse_post_cordinating_concepts_labels(row[col]) if l.strip()]
    #                 for label in labels:
    #                     all_labels.add(label)
    #                     if concept:
    #                         contextualized_pairs.add((concept, label))

    #     if not all_labels:
    #         return df
        
    #     texts_to_encode = []
    #     cache_keys = []

    #     for label in all_labels:
    #         texts_to_encode.append(label)
    #         cache_keys.append(label)

    #     for concept, label in contextualized_pairs:
    #         texts_to_encode.append(f"{concept} {label}")
    #         cache_keys.append(f"{concept}::{label}")

    #     print(f"🔤 Pre-encoding {len(texts_to_encode)} categorical embeddings "
    #         f"({len(all_labels)} labels + {len(contextualized_pairs)} contextualized)...")

    #     embeddings = model_object.embed_batch(texts_to_encode, show_progress=False)

    #     for cache_key, emb in zip(cache_keys, embeddings):
    #         CategoryMapper._label_embedding_cache[cache_key] = emb

    #     print(f"✅ Encoded {len(all_labels)} label-only + {len(contextualized_pairs)} contextualized embeddings")
    #     return df
   