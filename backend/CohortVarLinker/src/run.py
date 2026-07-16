
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Any, Dict, List, Tuple
from .config import settings
from .query_builder import SPARQLQueryBuilder
from .constraints import compute_structural, make_timepoint_info
from .policy import decide, should_consult_llm, llm_priority_key, CLAIMING_VERDICTS
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

from .harmonized_variable import suggest_harmonized_variable_without_llm
logger = setup_logger('storelog.log')
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
    harmonized = ""
    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
        reason = d.get("reason", "") or ""
        transform = d.get("transform", "") or ""
        transform_dir = d.get("alignment_direction", "") or ""
        harmonized = (d.get("harmonized_variable") or "").strip()
    except (json.JSONDecodeError, TypeError):
        pass

    return LLMEvidence(
        verdict=verdict,
        confidence=float(conf) if conf is not None else 0.0,
        reason=reason,
        transform=transform,
        transform_direction=transform_dir,
        harmonized_variable=harmonized,
    )

class StudyMapper:

    def __init__(self, vector_db: Any, embedding_model: Any, omop_graph: Any = None,
                 vector_collection: str = "studies",
                 mapping_mode: str = MappingType.OEH.value, llm_model:str = None,
                 enable_source_claim_early_exit: bool = True,
                 phase_a_workers: int | None = None):
        self.collection_name = vector_collection
        self.llm_model = llm_model
        self.embed_model = embedding_model
        self.similarity_threshold =  self._select_mode_specific_threshold(mapping_mode)

        # When True, once a source variable is matched COMPLETE or COMPATIBLE
        # against any target, the remaining ambiguous candidates for that
        # source are skipped (no LLM call). This is an ablation knob —
        # default False so paper numbers stay reproducible against the
        # no-claim baseline.
        self.enable_source_claim_early_exit = enable_source_claim_early_exit
        default_workers = int(os.getenv("PHASE_A_WORKERS", "6"))
        self.phase_a_workers = max(1, min(8, phase_a_workers or default_workers))

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
    
    # def fetch_studies_context(src_study_id:str, tgt_study_id:str):
        
     
    #     src_query = SPARQLQueryBuilder.build_study_context_query(src_study_id )
    #     tgt_query = SPARQLQueryBuilder.build_study_context_query(tgt_study_id)

    #     bindings = execute_query(src_query).get("results", {}).get("bindings", [])
    #     bindings = execute_query(tgt_query).get("results", {}).get("bindings", [])

    # Step 1a: SPARQL → typed VariableCollections
    def _fetch_unmapped_variables(self, study: str, role: str = None,use_filter:bool=False) -> List[VariableNode]:
        """Raw unmapped variables — self-sufficient (no enrichment needed)."""
        query = SPARQLQueryBuilder.build_unmapped_variables_query(study, use_filter)

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
            src_cat = result.get("source_domain", {}).get("value", "").strip().lower()
            tgt_cat = result.get("target_domain", {}).get("value", "").strip().lower()

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
    def _enrich_with_profiles(collection: VariableCollection, mapping_mode:str=MappingType.OEH.value):
        """Fetch profiles from KG and merge INTO VariableNode attributes in-place.
        """
        var_names = [v.name for v in collection.variables]
        dup_counts = {name: count for name, count in Counter(var_names).items() if count > 1}
        if dup_counts:
            logger.debug( 
                "Variables with multiple OMOP alignments in %s (expected for composite-coded vars): %s",
                collection.study,
                            dup_counts,
            )
        if not var_names:
            return

        profiles_df = VariableProfile.fetch_profiles(var_names, collection.study)
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

       
        prof_map = (
                profiles_df.drop_duplicates(subset="identifier", keep="last")
                        .set_index("identifier")
                        .to_dict("index")
            )
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

    # ──────────────────────────────────────────────────────────────────
    # Source-claim early-exit (ablation): once a source has been matched
    # COMPLETE or COMPATIBLE, skip the LLM on its remaining ambiguous
    # candidates. Gated by self.enable_source_claim_early_exit.
    # ──────────────────────────────────────────────────────────────────
    def _resolve_llm_with_source_claim(
        self, ambiguous_indices, structurals, src_study, tgt_study,
    ):
        """Per-source LLM resolution with COMPLETE/COMPATIBLE early-exit.

        For each source variable:
          1. Order its ambiguous candidates by ``llm_priority_key`` so the
             structural layer's best guess is judged first.
          2. Call the LLM on candidates one at a time (the inner
             ``LLMMatcher.assess()`` still uses its own ThreadPool for the
             actual network calls — we don't add a second pool here).
          3. Stop as soon as a verdict in ``CLAIMING_VERDICTS`` comes back.
             Remaining candidates for that source are skipped — policy
             then falls back to structural-only for them.

        Returns
        -------
        (llm_evidence, n_skipped) :
            llm_evidence : Dict[int, LLMEvidence]
            n_skipped    : count of candidates not sent to the LLM due
                           to source claim.

        Notes
        -----
        * Sources are processed serially. Claim semantics require serial
          calls within a source; parallelizing across sources would nest a
          thread pool inside ``assess()``'s own pool, multiplying the
          effective worker count and risking rate limits.
        * Source identity is (name, visit); two visits of the same
          variable are independent sources.
        * Target-side claiming is NOT applied — a target can legitimately
          receive multiple source mappings.
        """
        from collections import defaultdict

        # Group ambiguous candidates by source identity, ordered.
        by_source = defaultdict(list)
        for idx in ambiguous_indices:
            s, _t, ev = structurals[idx]
            src_key = (s.name, s.visit)
            by_source[src_key].append((idx, llm_priority_key(ev)))

        for src_key in by_source:
            by_source[src_key].sort(key=lambda pair: pair[1])

        llm_evidence = {}
        total_skipped = 0

        for src_key, ordered in by_source.items():
            claimed = False
            for idx, _prio in ordered:
                if claimed:
                    total_skipped += 1
                    continue
                # Reuse the existing concept-key-aware path. Single-element
                # lists preserve caching and concept dedup semantics.
                ev_dict = self.matcher.resolve_pending_with_llm(
                    [idx], structurals, src_study, tgt_study,
                    tuple_to_evidence=_llm_tuple_to_evidence,
                )
                if idx in ev_dict:
                    llm_evidence[idx] = ev_dict[idx]
                    if ev_dict[idx].verdict in CLAIMING_VERDICTS:
                        claimed = True

        return llm_evidence, total_skipped

    def _run_phase_a_structural(
        self,
        recs: List[Dict[str, Any]],
        src_col: VariableCollection,
        tgt_col: VariableCollection,
        src_study: str,
        tgt_study: str,
        mapping_mode: str,
    ) -> Dict[int, Tuple[VariableNode, VariableNode, Any]]:
        """Phase A: structural evidence for all candidates (optional parallel)."""
        n = len(recs)
        workers = min(self.phase_a_workers, n)
        matcher = self.matcher
        structurals: Dict[int, Tuple[VariableNode, VariableNode, Any]] = {}

     
        def _one(item):
            idx, row = item
            try:
                s = VariableNode.for_match_pair(src_col, row, side="source", study=src_study)
                t = VariableNode.for_match_pair(tgt_col, row, side="target", study=tgt_study)
                ev = compute_structural(s, t, mapping_mode=mapping_mode, matcher=matcher)
                return idx, (s, t, ev)
            except Exception as e:
                logger.error(f"Phase A worker failed on row {idx}: {e}")
                raise  # or return a fallback
        if workers <= 1:
            for idx, row in enumerate(recs):
                _, triple = _one((idx, row))
                structurals[idx] = triple
                # if (idx + 1) % 500 == 0 or (idx + 1) == n:
                #     logger.info(f"Phase A: {idx + 1}/{n}")
            return structurals

        # logger.info(f"Phase A: using {workers} worker threads")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, (i, r)) for i, r in enumerate(recs)]
            for fut in as_completed(futures):
                idx, triple = fut.result()
                structurals[idx] = triple
                done += 1
                # if done % 500 == 0 or done == n:
                #     logger.info(f"Phase A: {done}/{n}")
        return structurals

    # Run Pipeline
    def run_pipeline(
        self, src_study: str, tgt_study: str,
        mapping_mode: str = MappingType.OEH.value
    ) -> pd.DataFrame:
        

        src_col = VariableCollection(study=src_study, variables=[])
        tgt_col = VariableCollection(study=tgt_study, variables=[])

        # get studies design protocols
        if self.llm_model:
            from .study_context import format_study_context_block
            study_context = format_study_context_block(src_study, tgt_study)
            if study_context:
                logger.info(f"📋 Study context fetched ({len(study_context)} chars):\n{study_context}")
                # Store on the LLM matcher so assess() can prepend it to prompts.
                self.matcher.llm_matcher.set_study_context(study_context)
        # ── Step 1a: Parse SPARQL → typed VariableCollections ─────
        if mapping_mode == MappingType.NE.value:
            src_col = VariableCollection(study=src_study,
                variables=self._fetch_unmapped_variables(src_study, role="source"))
            tgt_col = VariableCollection(study=tgt_study,
                variables=self._fetch_unmapped_variables(tgt_study, role="target"))

        else:
            query = SPARQLQueryBuilder.build_alignment_query(src_study, tgt_study)
            src_col, tgt_col = self._parse_sparql_results(query, src_study, tgt_study)
        
        # ── Step 1b: Fetch unmapped variables (OEH/NE) ─────
        if mapping_mode == MappingType.OEH.value:
            for col, study in [(src_col, src_study), (tgt_col, tgt_study)]:
                unmapped = self._fetch_unmapped_variables(study,  use_filter=True)
                if unmapped:
                    col.variables.extend(unmapped)
                    col._by_omop_id = None
                    col._by_name = None
                    col._build_indexes()
                    # logger.info(f"  📎 Added {len(unmapped)} unmapped variables from {study}")

        self._enrich_with_profiles(src_col,  mapping_mode)
        self._enrich_with_profiles(tgt_col,  mapping_mode)
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

            # Phase A: structural evidence (collection lookup + parallel workers)
            # logger.info(
            #     f"🧱 Phase A: structural evidence for {len(recs)} candidates "
            #     f"(workers={min(self.phase_a_workers, len(recs))})..."
            # )
            structurals = self._run_phase_a_structural(
                recs, src_col, tgt_col, src_study, tgt_study, mapping_mode,
            )

            # Phase B: LLM only for ambiguous cases (symmetric pre-filter)
            llm_evidence = {}
            if self.llm_model:
                ambiguous = [idx for idx, (_, _, ev) in structurals.items()
                              if should_consult_llm(ev)]
                # n_skipped_structural = len(recs) - len(ambiguous)

                if self.enable_source_claim_early_exit and ambiguous:
                    llm_evidence, n_skipped_claim = self._resolve_llm_with_source_claim(
                        ambiguous, structurals, src_study, tgt_study,
                    )
                    # logger.info(
                    #     f"🤖 Phase B: LLM consulted on {len(llm_evidence)}/{len(recs)} "
                    #     f"candidates "
                    #     f"({n_skipped_structural} skipped by structural confidence, "
                    #     f"{n_skipped_claim} skipped by source-claim early-exit)"
                    # )
                else:
                    # logger.info(
                    #     f"🤖 Phase B: LLM consulted on {len(ambiguous)}/{len(recs)} "
                    #     f"candidates ({n_skipped_structural} skipped by structural confidence)"
                    # )
                    if ambiguous:
                        llm_evidence = self.matcher.resolve_pending_with_llm(
                            ambiguous, structurals, src_study, tgt_study,
                            tuple_to_evidence=_llm_tuple_to_evidence,
                        )

            # Phase C: one policy.decide() per row, immutable verdict, single write
            # logger.info(f"⚖️  Phase C: policy decision for {len(structurals)} candidates...")
            descriptions, transformations, statuses = [], [], []
            # llm_use = True if self.llm_model else False
            for idx in range(len(recs)):
                s, t, struct_ev = structurals[idx]
                tp = make_timepoint_info(s, t)

                verdict = decide(mapping_mode, struct_ev, llm_evidence.get(idx), llm_use, tp)
                
                details, status = verdict.to_legacy_tuple()
                if not (details.get("harmonized_variable") or "").strip():
                    hv = suggest_harmonized_variable_without_llm(
                        s,
                        t,
                        struct_ev,
                        graph=self.graph,
                        verdict_level=verdict.level,
                    )
                    if hv:
                        details["harmonized_variable"] = hv
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
       