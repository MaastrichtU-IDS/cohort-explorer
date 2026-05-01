
import json
import pandas as pd
from typing import Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import settings
from .query_builder import SPARQLQueryBuilder
from .constraints import ConstraintSolver
from .data_model import (
    MappingType, VariableNode, VariableCollection,
    Statistics, StatisticalType, _safe_float
)
from .neuro_matcher import NeuroSymbolicMatcher
from .variable_profile import VariableProfile

from .graph_similarity import compute_context_scores
from .utils import clean_label_remove_temporal_context, execute_query, parse_post_cordinating_concepts_labels
# from .llm_call import LocalLLMConceptMatcher


class StudyMapper:

    def __init__(self, vector_db: Any, embedding_model: Any, omop_graph: Any = None,
                 vector_collection: str = "studies",
                 mapping_mode: str = MappingType.OEH.value, llm_models: List[str] = None):
        self.collection_name = vector_collection
        self.llm_models = llm_models
        self.embed_model = embedding_model
        self.similarity_threshold = settings.ADAPTIVE_THRESHOLD
        # self.similarity_threshold =

        # 0.6 if self.embed_model.model_name  in  else 0.8 # not using coder here
        self.matcher = NeuroSymbolicMatcher(
            vector_db,
            embed_model=embedding_model,
            graph=omop_graph,
            collection_name=vector_collection,
            mapping_mode=mapping_mode,
            llm_models=llm_models,
            similarity_threshold=self.similarity_threshold

        )
        self.graph = omop_graph
        self.solver = ConstraintSolver(self.matcher)
        # self.llm_models = llm_models
   
 
    # Step 1a: SPARQL → typed VariableCollections
 

    # def _build_collections_from_vectordb(self, src_study: str, tgt_study: str) -> Tuple[VariableCollection, VariableCollection]:
    #     """Build VariableCollections from vector DB payloads — no SPARQL, no OMOP."""
    #     from qdrant_client import models

    #     def _fetch_study_vars(study: str) -> List[VariableNode]:
    #         results = self.matcher.vector_db.scroll(
    #             collection_name=self.matcher.collection_name,
    #             scroll_filter=models.Filter(must=[
    #                 models.FieldCondition(key='study_name', match=models.MatchValue(value=study)),
    #                 models.FieldCondition(key='is_category', match=models.MatchValue(value=0)),
    #             ]),
    #             limit=10000, with_payload=True, with_vectors=False,
    #         )[0]
    #         nodes = []
    #         for point in results:
    #             p = point.payload
    #             description = clean_label_remove_temporal_context(p.get('variable_label', ''))
    #             original_categories = parse_post_cordinating_concepts_labels(p.get('original_categories'))
    #             category_labels = parse_post_cordinating_concepts_labels(p.get('original_categories_labels'))
    #             unit = p.get('unit', '') or ''
    #             min_val = _safe_float(p.get('min', None))
    #             max_val = _safe_float(p.get('max', None))
    #             # stat_type = p.get('statistical_type', None)
    #             # if not stat_type:
    #             #     n_cats = len(original_categories) or len(category_labels)
    #             #     if n_cats == 2: stat_type = StatisticalType.BINARY.value
    #             #     elif n_cats > 2: stat_type = StatisticalType.MULTI_CLASS.value
    #             #     elif unit or (min_val is not None and max_val is not None): stat_type = StatisticalType.CONTINUOUS.value
    #             #     else: stat_type = StatisticalType.QUALITATIVE.value

    #             if p.get('is_category', 0) != 0:
    #                 continue

    #             # Expand grouped visits into individual nodes
    #             visits_raw = p.get('visit', '')
    #             visits = [v.strip() for v in visits_raw.split('|') if v.strip()] if visits_raw else ['']
    #             names_raw = p.get('variable_names', p.get('variable_name', ''))
    #             names = [n.strip() for n in names_raw.split('|') if n.strip()] if names_raw else ['']

    #             for vi, visit in enumerate(visits):
    #                 name = names[vi] if vi < len(names) else names[0]
    #                 node = VariableNode(
    #                     name=name, description=description, main_label=description,
    #                     study=study, role="source" if study == src_study else "target",
    #                     unit=unit, visit=visit, category=p.get('domain', ''),
    #                     statistics=Statistics(min_val=min_val, max_val=max_val),
    #                     original_categories=original_categories, category_labels=category_labels,
    #                 )
    #                 # node.statistical_type = StatisticalType.from_string(stat_type)
    #                 nodes.append(node)
    #         print(f"fetched {len(nodes)} variables from {study}")
    #         return nodes
       
    #     src_nodes = _fetch_study_vars(src_study)
    #     tgt_nodes = _fetch_study_vars(tgt_study)
    #     return VariableCollection(study=src_study, variables=src_nodes), VariableCollection(study=tgt_study, variables=tgt_nodes)

    def _fetch_unmapped_variables(self, study: str, graph_repo: str, role: str = None,use_filter:bool=False) -> List[VariableNode]:
        """Raw unmapped variables — self-sufficient (no enrichment needed)."""
        query = SPARQLQueryBuilder.build_unmapped_variables_query(study, graph_repo, use_filter)
        bindings = execute_query(query).get("results", {}).get("bindings", [])
        nodes = []
        for b in bindings:
            label = b.get("var_label", {}).get("value", "")
            unit = b.get("unit_label", {}).get("value", "") or None
            stat = b.get("stat_label", {}).get("value", "")
            node = VariableNode(
                name=b.get("var", {}).get("value", ""),
                description=label, main_label=label, study=study, role=role,
                visit=b.get("visit_label", {}).get("value", "baseline"),
                category=b.get("domain_val", {}).get("value", ""),
                unit=unit, main_id=None, main_code=None,
            )
            if stat:
                node.statistical_type = StatisticalType.from_string(stat)
            nodes.append(node)
        # print(f"first node format = {nodes[0]}")
        return nodes
    def _parse_sparql_results(
        self, query: str, src_study: str, tgt_study: str,
    ) -> Tuple[VariableCollection, VariableCollection]:
        """Parse SPARQL alignment query into typed VariableCollections.

        Each SPARQL binding produces VariableNodes with:
            name, main_id, main_label, main_code, category, visit, var_label (extra)

        Profile fields (statistical_type, unit, context_ids, category_ids, etc.)
        are NOT set here — they come from _enrich_with_profiles.
        """
        bindings = execute_query(query).get("results", {}).get("bindings", [])
        src_nodes: List[VariableNode] = []
        tgt_nodes: List[VariableNode] = []

        for result in bindings:
            omop = int(result["omop_id"]["value"])
            code_label = result.get("code_label", {}).get("value", "")
            code_value = result.get("code_value", {}).get("value", "")
            src_cat = result["source_domain"]["value"].strip().lower()
            tgt_cat = result["target_domain"]["value"].strip().lower()

            def parse_raw(raw: str) -> List[Tuple[str, str, str]]:
                out = []
                if not raw:
                    return out
                for entry in raw.split("||"):
                    parts = (entry.split(";;") + ["N/A"] * 3)[:3]
                    out.append((parts[0], parts[1], parts[2]))
                return out

            raw_src = result.get("source_definition", {}).get("value", "")
            raw_tgt = result.get("target_definition", {}).get("value", "")

            for name, label, visit in parse_raw(raw_src):
                node = VariableNode(
                    name=name, description =label, study=src_study, role="source",
                    main_id=omop, main_label=code_label, main_code=code_value,
                    category=src_cat, visit=visit,
                )
                # node.var_label = label  # extra field (Config extra="allow")
                src_nodes.append(node)

            for name, label, visit in parse_raw(raw_tgt):
                node = VariableNode(
                    name=name, description =label, study=tgt_study, role="target",
                    main_id=omop, main_label=code_label, main_code=code_value,
                    category=tgt_cat, visit=visit,
                )
                # node.var_label = label
                tgt_nodes.append(node)

        src_col = VariableCollection(study=src_study, variables=src_nodes)
        tgt_col = VariableCollection(study=tgt_study, variables=tgt_nodes)
        return src_col, tgt_col

 
    # Step 1b: Enrich VariableNodes with profile data — O(n) per study
 
    
    @staticmethod
    def _enrich_with_profiles(collection: VariableCollection, graph_repo: str, mapping_mode:str=MappingType.OEH.value):
        """Fetch profiles from KG and merge INTO VariableNode attributes in-place.

        After this call, every node has: statistical_type, unit, context_labels,
        context_ids, category_labels, category_ids, original_categories, statistics.

        Note: Pydantic v2 validators don't fire on attribute assignment,
        so we parse pipe-separated strings into typed lists before setting.
        """
        var_names = [v.name for v in collection.variables]
        if not var_names:
            return

        profiles_df = VariableProfile.fetch_profiles(var_names, collection.study, graph_repo)
        if profiles_df.empty:
            return

        def _split_labels(val) -> list:
            if not val:
                return []
            return [x.strip() for x in str(val).split("||") if x.strip()]

        def _split_ids(val) -> list:
            if not val:
                return []
            result = []
            for x in str(val).replace("||", ";").split(";"):
                try:
                    result.append(int(x.strip()))
                except (ValueError, TypeError):
                    continue
            return result

        # Build lookup: identifier → profile row
        prof_map = {row["identifier"]: row for _, row in profiles_df.iterrows()}

        for node in collection.variables:
            p = prof_map.get(node.name)
            if p is None:
                continue
            # Parse into typed values BEFORE assignment (validators don't fire post-init)
            node.statistical_type = StatisticalType.from_string(p.get("stat_label")) if mapping_mode != MappingType.NE.value else node.statistical_type
            node.unit = p.get("unit_label", "") if mapping_mode != MappingType.NE.value else node.unit
            node.context_labels = _split_labels(p.get("composite_code_labels")) if mapping_mode != MappingType.NE.value else []
            node.context_ids = _split_ids(p.get("composite_code_omop_ids"))   if mapping_mode != MappingType.NE.value else []
            node.category_labels = _split_labels(p.get("categories_labels"))
            node.category_ids = _split_ids(p.get("categories_omop_ids"))  if mapping_mode != MappingType.NE.value else []
            node.original_categories = _split_labels(p.get("original_categories"))
            node.statistics = Statistics(
                min_val=_safe_float(p.get("min_val")),
                max_val=_safe_float(p.get("max_val")),
            )

   
 
    # Run Pipeline
 

    def run_pipeline(
        self, src_study: str, tgt_study: str,
        mapping_mode: str = MappingType.OEH.value
    ) -> pd.DataFrame:
        

        src_col = VariableCollection(study=src_study, variables=[])
        tgt_col = VariableCollection(study=tgt_study, variables=[])

        # ── Step 1a: Parse SPARQL → typed VariableCollections ─────
        if mapping_mode == MappingType.NE.value:
            src_col = VariableCollection(study=src_study,
                variables=self._fetch_unmapped_variables(src_study, settings.GRAPH_REPO, role="source"))
            tgt_col = VariableCollection(study=tgt_study,
                variables=self._fetch_unmapped_variables(tgt_study, settings.GRAPH_REPO, role="target"))

        else:
            query = SPARQLQueryBuilder.build_alignment_query(src_study, tgt_study, settings.GRAPH_REPO)
            src_col, tgt_col = self._parse_sparql_results(query, src_study, tgt_study)
        
        # ── Step 1b: Fetch unmapped variables (OEH/NE) ─────
        if mapping_mode == MappingType.OEH.value:
            for col, study in [(src_col, src_study), (tgt_col, tgt_study)]:
                unmapped = self._fetch_unmapped_variables(study, settings.GRAPH_REPO, use_filter=True)
                if unmapped:
                    col.variables.extend(unmapped)
                    col._by_omop_id = None
                    col._by_name = None
                    col._build_indexes()
                    print(f"  📎 Added {len(unmapped)} unmapped variables from {study}")

        self._enrich_with_profiles(src_col, settings.GRAPH_REPO, mapping_mode)
        self._enrich_with_profiles(tgt_col, settings.GRAPH_REPO, mapping_mode)
        print(f"📊 Parsed {len(src_col)} source, {len(tgt_col)} target variables")

        # ── Step 2: Concept matching + context scoring ────────────
        # ns_matches = self.matcher.resolve_matches(
        #     src_collection=src_col, tgt_collection=tgt_col, target_study=tgt_study
        # )

        # # ── Build DataFrame (profile columns already in dicts) ────
        # df = pd.DataFrame(ns_matches)
        # if df.empty:
        #     return pd.DataFrame(columns=["source", "target", "harmonization_status"])
        # df = df.drop_duplicates(subset=["source", "target"]).dropna(subset=["source", "target"])

        # # ── Step 3: Pre-embed categorical labels ──────────────────
        # # if mapping_mode != MappingType.OO.value:
        # df = self.matcher._precompute_catvalues_similarity(df, model_object=self.embed_model)
        # if not self.llm_models:
        #     df = compute_context_scores(df, graph=self.graph, embed_model=self.embed_model, mapping_mode=mapping_mode, threshold=self.similarity_threshold)

        # ── Step 2: Discover Candidates ────────────
        ns_matches = self.matcher.generate_candidates(src_collection=src_col, tgt_collection=tgt_col, target_study=tgt_study)

        df = pd.DataFrame(ns_matches)
        if df.empty:
            return pd.DataFrame(columns=["source", "target", "harmonization_status"])
        df = df.drop_duplicates(subset=["source", "target"])

        # ── Step 3: Embeddings & Context Scoring ──────────────────
        df = self.matcher._precompute_catvalues_similarity(df, model_object=self.embed_model)
        llm_use = True if self.llm_models else False
        df = compute_context_scores(df, graph=self.graph, embed_model=self.embed_model, mapping_mode=mapping_mode, threshold=self.similarity_threshold, llm=llm_use)
        # ── Step 4: Symbolic Rules & LLM ──────────────────
        if self.llm_models:
            df = self.matcher.resolve_pending_with_llm(df, src_study, tgt_study)

        # ── Step 5: Expand Derived Variables ──────────────────
        df = self.matcher.compute_derived_variables(df, src_col, tgt_col)

        # ── Step 6: Constraint solving ────────────────────────────
        total_rows = len(df)
        # print(f"🔧 Solving constraints for {total_rows} rows (parallel)...")

        def solve_row(idx_row):
            idx, row = idx_row
            src_node = VariableNode.from_source_row(row, study=src_study)
            tgt_node = VariableNode.from_target_row(row, study=tgt_study)
            details, status = self.solver.solve(src_node, tgt_node, mapping_mode=mapping_mode)
            return idx, (str(details), status)

        results = [None] * total_rows
        done_count = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(solve_row, (i, row)): i
                for i, (_, row) in enumerate(df.iterrows())
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                done_count += 1
                if done_count % 500 == 0:
                    print(f"  solve row: {done_count}/{total_rows}...")

        print(f"  solve row: {total_rows}/{total_rows} done")
        df[["transformation_rule", "harmonization_status"]] = pd.DataFrame(results, index=df.index)

        for col in df.columns:
            if col in {"source", "target"}:
                continue
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                df[col] = df[col].apply(json.dumps)
            elif df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(str)

        df.dropna(subset=["source", "target"], inplace=True)
        return df.drop_duplicates(keep="first")
