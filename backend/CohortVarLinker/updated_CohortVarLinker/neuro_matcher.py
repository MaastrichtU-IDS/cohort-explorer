"""Neuro-symbolic matching with typed VariableNode inputs.
"""

from collections import defaultdict
import pandas as pd
from typing import Any, Dict, List, Set,Tuple
from .llm_call import LLMConceptMatcher
from .config import settings
from .data_model import (
    MappingType, VariableNode, StatisticalType, 
    ContextMatchType, MappingRelation
)
from .vector_db import search_category_by_id, search_in_db, _embed_cache,  _cache_key
from .fuzz_match import FuzzyMatcher
from .graph_similarity import (
    score_context,
    parse_post_cordinating_concepts_labels
)
from .utils import setup_logger, clean_label_remove_temporal_context

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# logger = setup_logger('llm_matcher.log')

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
                "mapping_relation": MappingRelation.SymbolicCloseMatch.value,
                "context_match_type": ContextMatchType.EXACT.value,
                "sim_score": 1.0,
                "transformation_rule": f"Derived variable {variable_name} using parameter columns {parameters_omop_ids}.",
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
                    matched_candidates.add((sid, MappingRelation.SymbolicExactMatch.value, None))
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
                            matched_candidates.add((tid, MappingRelation.NeuralMatch.value, score))
                else:
                    for item in raw_matches:
                        tid, score = item if isinstance(item, tuple) else (item, 0.0)
                        if tid not in graph_tids:
                            matched_candidates.add((tid, MappingRelation.NeuralMatch.value, score))  # score preserved

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
        # logger.info(f"  🤖 LLM Validating : {len(flat_keys)} concept pairs")
        grouped_results, _stats = self.llm_matcher.assess(groups, case_ids=case_ids)

        # ── Step 4: convert verdicts to LLMEvidence

        ckey_to_evidence = {}
        flat_pos = 0
        for grp_verdicts in grouped_results:
            for verdict_tuple in grp_verdicts:
                ckey = flat_keys[flat_pos]
                # tuple_to_evidence handles missing/empty tuples → IMPOSSIBLE.
                ckey_to_evidence[ckey] = tuple_to_evidence(
                    verdict_tuple if verdict_tuple else
                    (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "")
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

        s_cats = tuple(sorted(l.strip() for l in src_node.category_labels if l.strip()))
        t_cats = tuple(sorted(l.strip() for l in tgt_node.category_labels if l.strip()))

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
        # logger.info(f"  🤖 LLM: {len(flat_keys)} concept pairs")

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
        cached_n = (len(all_labels) - len(new_lbls)) + (len(ctx_pairs) - len(new_ctx))

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