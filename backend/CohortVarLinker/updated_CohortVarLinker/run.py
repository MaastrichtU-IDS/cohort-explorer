
import json
import pandas as pd
from typing import Any, List, Tuple
from .config import settings
from .query_builder import SPARQLQueryBuilder
from .constraints import compute_structural, make_timepoint_info
from .policy import decide, should_consult_llm
from .verdict import LLMEvidence
from .data_model import (
    MappingType, VariableNode, VariableCollection,
    Statistics, StatisticalType, ContextMatchType, _safe_float
)
from .utils import setup_logger
from .neuro_matcher import NeuroSymbolicMatcher
from .variable_profile import VariableProfile

from .graph_similarity import compute_context_scores
from .utils import execute_query

# from concurrent.futures import ThreadPoolExecutor
# logger = setup_logger('storelog.log')

_CTX_TYPE_TO_VERDICT = {
    ContextMatchType.EXACT.value:           "COMPLETE",
    ContextMatchType.COMPATIBLE.value:      "COMPATIBLE",
    ContextMatchType.SUBSUMED.value:        "COMPATIBLE",
    ContextMatchType.PARTIAL.value:         "PARTIAL",
    ContextMatchType.NOT_APPLICABLE.value:  "IMPOSSIBLE",
}

def _llm_tuple_to_evidence(parsed_tuple) -> LLMEvidence:
    """Adapter: llm_call.py returns (matched_or_none, ctx_type, conf, reason_json).
    Convert to the policy-facing LLMEvidence record."""
    matched, ctx_type, conf, reason_json = parsed_tuple
    if matched is False or ctx_type == ContextMatchType.NOT_APPLICABLE.value:
        verdict = "IMPOSSIBLE"
    else:
        verdict = _CTX_TYPE_TO_VERDICT.get(ctx_type, "IMPOSSIBLE")

    reason = transform = transform_dir = ""
    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
        reason = d.get("reason", "") or ""
        transform = d.get("transform", "") or ""
        transform_dir = d.get("transform_direction", "") or ""
    except (json.JSONDecodeError, TypeError):
        pass

    return LLMEvidence(
        verdict=verdict,
        confidence=float(conf) if conf is not None else 0.0,
        reason=reason,
        transform=transform,
        transform_direction=transform_dir,
    )

class StudyMapper:

    def __init__(self, vector_db: Any, embedding_model: Any, omop_graph: Any = None,
                 vector_collection: str = "studies",
                 mapping_mode: str = MappingType.OEH.value, llm_model:str = None):
        self.collection_name = vector_collection
        self.llm_model = llm_model
        self.embed_model = embedding_model
        self.similarity_threshold =  self._select_mode_specific_threshold(mapping_mode)
        # self.similarity_threshold =

        # 0.6 if self.embed_model.model_name  in  else 0.8 # not using coder here
        self.matcher = NeuroSymbolicMatcher(
            vector_db,
            embed_model=embedding_model,
            graph=omop_graph,
            collection_name=vector_collection,
            mapping_mode=mapping_mode,
            llm_model=llm_model,
            similarity_threshold=self.similarity_threshold

        )
        self.graph = omop_graph

   
    
    def _select_mode_specific_threshold(self, mapping_mode):
        if mapping_mode == MappingType.NE.value:
            return 0.45  # lower a bit due to low label quality
        return settings.ADAPTIVE_THRESHOLD
    
    # Step 1a: SPARQL → typed VariableCollections
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
        # prof_map = {row["identifier"]: row for _, row in profiles_df.iterrows()}
        prof_map = profiles_df.set_index("identifier").to_dict("index")
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
                    # logger.info(f"  📎 Added {len(unmapped)} unmapped variables from {study}")

        self._enrich_with_profiles(src_col, settings.GRAPH_REPO, mapping_mode)
        self._enrich_with_profiles(tgt_col, settings.GRAPH_REPO, mapping_mode)
        # logger.info(f"📊 Parsed {len(src_col)} source, {len(tgt_col)} target variables")

        # ── Step 2: Discover Candidates ────────────
        ns_matches = self.matcher.generate_candidates(src_collection=src_col, tgt_collection=tgt_col, target_study=tgt_study)

        df = pd.DataFrame(ns_matches)
        if df.empty:
            return pd.DataFrame(columns=["source", "target", "harmonization_status"])
        df = df.drop_duplicates(subset=["source", "target"])

        # ── Step 3: Embeddings & Context Scoring ──────────────────
        df = self.matcher._precompute_catvalues_similarity(df, model_object=self.embed_model)
        llm_use = True if self.llm_model else False
        df = compute_context_scores(df, graph=self.graph, embed_model=self.embed_model, mapping_mode=mapping_mode, threshold=self.similarity_threshold, llm=llm_use)

        # ── Step 4: Expand Derived Variables ──────────────────
        df = self.matcher.compute_derived_variables(df, src_col, tgt_col)
        if not df.empty:
            str_cols = [
                "source_label", "target_label", "slabel", "tlabel",
                "source_unit", "target_unit", "source_visit", "target_visit",
                "source_composite_code_labels", "target_composite_code_labels",
                "source_categories_labels", "target_categories_labels",
                "source_categories_omop_ids", "target_categories_omop_ids",
                "source_original_categories", "target_original_categories",
                "source_data_type", "target_data_type",
                "category", "mapping_relation",
            ]
            df.loc[:, [c for c in str_cols if c in df.columns]] = (
                df.loc[:, [c for c in str_cols if c in df.columns]].fillna("")
            )
            recs = df.to_dict("records")

            # Phase A: structural evidence
            # logger.info(f"🧱 Phase A: structural evidence for {len(recs)} candidates...")
            structurals = {}
            for idx, row in enumerate(recs):
     
                s = VariableNode.from_source_row(row, study=src_study)
                t = VariableNode.from_target_row(row, study=tgt_study)
                ev = compute_structural(s, t, mapping_mode=mapping_mode, matcher=self.matcher)
                structurals[idx] = (s, t, ev)

            # Phase B: LLM only for ambiguous cases (symmetric pre-filter)
            llm_evidence = {}
            if self.llm_model:
                ambiguous = [idx for idx, (_, _, ev) in structurals.items()
                              if should_consult_llm(ev)]
                n_skipped = len(recs) - len(ambiguous)
                logger.info(f"🤖 Phase B: LLM consulted on {len(ambiguous)}/{len(recs)} "
                      f"candidates ({n_skipped} skipped by structural confidence)")
                if ambiguous:
                    llm_evidence = self.matcher.resolve_pending_with_llm(
                        ambiguous, structurals, src_study, tgt_study,
                        tuple_to_evidence=_llm_tuple_to_evidence,
                    )

            # Phase C: one policy.decide() per row, immutable verdict, single write
            # logger.info(f"⚖️  Phase C: policy decision for {len(structurals)} candidates...")
            descriptions, transformations, statuses = [], [], []
            for idx in range(len(recs)):
                s, t, struct_ev = structurals[idx]
                tp = make_timepoint_info(s, t)
                verdict = decide(mapping_mode, struct_ev, llm_evidence.get(idx), tp)
                
                details, status = verdict.to_legacy_tuple()
                transformations.append(details.pop("transformation", "") or "")
                descriptions.append(json.dumps(details, default=str, ensure_ascii=False))
                statuses.append(status)
                # if llm_use: 
                #     logger.info(f"final verdict for source {s} and target {t} is {verdict}")

            df["transformation_type"]   = transformations
            df["Mapping Description"]   = descriptions
            df["harmonization_status"]  = statuses
        
        df.dropna(subset=["source", "target"], inplace=True)
        df.drop(columns="context_match_type", inplace=True, errors="ignore")
        return df
       