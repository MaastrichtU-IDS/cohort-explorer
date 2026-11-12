import logging
import re
import threading

from SPARQLWrapper import JSON, SPARQLWrapper

from .config import settings
from .utils import sparql_prefixes, term_iri

# Study graphs are published under the cohort id as written in the cohorts
# spreadsheet ("https://w3id.org/CMEO/graph/Aachen-HF"), while the mapping
# pipeline works with lowercased study names ("aachen-hf"). IRIs are case
# sensitive, so the graph IRI is looked up among the graphs that actually exist
# instead of being assembled from the lowercased name.
_graph_names_lock = threading.Lock()
_graph_names: dict[str, str] = {}  # lowercased graph name -> graph IRI


def _separators_collapsed(name: str) -> str:
    return re.sub(r"[\s_-]+", "", str(name).strip().lower())


def _load_graph_names() -> dict[str, str]:
    query = f"""
        SELECT DISTINCT ?g WHERE {{
            GRAPH ?g {{ ?s ?p ?o }}
            FILTER(STRSTARTS(STR(?g), "{settings.GRAPH_REPO}/"))
        }}
    """
    endpoint = SPARQLWrapper(settings.query_endpoint)
    endpoint.setReturnFormat(JSON)
    endpoint.setQuery(query)
    names: dict[str, str] = {}
    for b in endpoint.query().convert()["results"]["bindings"]:
        iri = b["g"]["value"]
        names[iri[len(settings.GRAPH_REPO) + 1:].lower()] = iri
    return names


class _Terms:
    """T.minimum_value -> <ontology IRI>, from the same table the KG generator uses,
    so the queries and the graph cannot drift apart."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return f"<{term_iri(name)}>"


T = _Terms()

class SPARQLQueryBuilder:
    """Responsible solely for constructing valid SPARQL queries."""
    
    PREFIXES = sparql_prefixes()
    METADATA_GRAPH = "https://w3id.org/CMEO/graph/studies_metadata"
    GRAPHS_URI = settings.GRAPH_REPO

    @classmethod
    def graph_iri(cls, study: str) -> str:
        """IRI of the study's named graph, whatever the case of `study`.

        Exact (case-insensitive) name first, then a match ignoring spaces,
        underscores and hyphens when it is unique. The graph list is cached and
        re-read once on a miss, so a study uploaded after start-up is found.
        Falls back to the name as given when no graph matches.
        """
        key = str(study).strip().lower()
        global _graph_names
        for attempt in range(2):
            with _graph_names_lock:
                if attempt == 1 or not _graph_names:
                    try:
                        _graph_names = _load_graph_names()
                    except Exception as e:
                        logging.warning("Could not list study graphs in the triplestore: %s", e)
                names = dict(_graph_names)
            if key in names:
                return names[key]
            collapsed = _separators_collapsed(key)
            loose = [iri for name, iri in names.items() if _separators_collapsed(name) == collapsed]
            if len(loose) == 1:
                return loose[0]
        logging.warning("No graph in the triplestore for study '%s'; queries will match nothing", study)
        return f"{cls.GRAPHS_URI}/{study}"

    @staticmethod
    def literal(value: str) -> str:
        """A SPARQL string literal for value. Escapes \\ first, then ", so a
        name like height"cm or dose\\day still matches the stored identifier
        instead of breaking the query; the server returns the original value."""
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'



    @classmethod
    def _alignment_side(cls, study: str, side: str) -> str:
        """One study's coded variables, grouped by main OMOP concept.

        Every value reaching an aggregate is bound: Oxigraph >= 0.4.7 treats an
        unbound value inside GROUP_CONCAT as an error and blanks the whole
        group, which is what emptied every concept shared by both studies.
        """
        return f"""
                SELECT ?omop_id
                    (SAMPLE(?lbl) AS ?{side}_lbl)
                    (SAMPLE(?val) AS ?{side}_code)
                    (GROUP_CONCAT(DISTINCT ?de_str; SEPARATOR="||") AS ?{side}_de_uris)
                    (GROUP_CONCAT(DISTINCT ?combined; SEPARATOR="||") AS ?{side}_definition)
                    (GROUP_CONCAT(DISTINCT ?domain; SEPARATOR="||") AS ?{side}_domain)
                WHERE {{
                    GRAPH <{cls.graph_iri(study)}> {{
                        ?de a {T.data_element} ;
                            rdfs:label ?var_label ;
                            dc:identifier ?var ;
                            skos:closeMatch ?codeSet .
                        ?codeSet rdf:_1 ?codeNode .
                        ?codeNode rdfs:label ?lbl ;
                                  {T.has_value} ?val ;
                                  {T.denotes}/{T.has_value} ?omop_id .
                        OPTIONAL {{ ?de {T.has_attribute}/skos:closeMatch/rdfs:label ?vis_label . }}
                        OPTIONAL {{ ?de {T.has_annotation}/{T.has_value} ?domain_val . }}
                        BIND(CONCAT(?var, ";;", ?var_label, ";;", COALESCE(?vis_label, "N/A")) AS ?combined)
                        BIND(COALESCE(?domain_val, "") AS ?domain)
                        BIND(STR(?de) AS ?de_str)
                    }}
                }}
                GROUP BY ?omop_id"""

    @classmethod
    def build_alignment_query(cls, source: str, target: str) -> str:
        # Each study is grouped on its own, then the two are merged per concept.
        # A concept yields at most one row per side, so the outer GROUP_CONCAT
        # with an empty separator just picks that side's value (or ""). The
        # concept label and code prefer the source study's, so they no longer
        # depend on which row SAMPLE happens to pick.
        query = f"""
            {cls.PREFIXES}
            SELECT ?omop_id
            (IF(GROUP_CONCAT(COALESCE(?src_lbl, ""); SEPARATOR="") != "",
                GROUP_CONCAT(COALESCE(?src_lbl, ""); SEPARATOR=""),
                GROUP_CONCAT(COALESCE(?tgt_lbl, ""); SEPARATOR="")) AS ?code_label)
            (IF(GROUP_CONCAT(COALESCE(?src_code, ""); SEPARATOR="") != "",
                GROUP_CONCAT(COALESCE(?src_code, ""); SEPARATOR=""),
                GROUP_CONCAT(COALESCE(?tgt_code, ""); SEPARATOR="")) AS ?code_value)
            (GROUP_CONCAT(COALESCE(?src_de_uris, ""); SEPARATOR="") AS ?source_de_uris)
            (GROUP_CONCAT(COALESCE(?tgt_de_uris, ""); SEPARATOR="") AS ?target_de_uris)
            (GROUP_CONCAT(COALESCE(?src_definition, ""); SEPARATOR="") AS ?source_definition)
            (GROUP_CONCAT(COALESCE(?tgt_definition, ""); SEPARATOR="") AS ?target_definition)
            (GROUP_CONCAT(COALESCE(?src_domain, ""); SEPARATOR="") AS ?source_domain)
            (GROUP_CONCAT(COALESCE(?tgt_domain, ""); SEPARATOR="") AS ?target_domain)
            WHERE {{
                {{ {cls._alignment_side(source, "src")}
                }}
                UNION
                {{ {cls._alignment_side(target, "tgt")}
                }}
            }}
            GROUP BY ?omop_id
            ORDER BY ?omop_id
        """
        # print("Generated SPARQL Query:")
        # print(query)
        return query


    


    @classmethod   
    def build_unmapped_variables_query(cls, study: str, use_filter:bool=False) -> str:
        """Variables with no skos:closeMatch — raw label/visit/unit/stat for NE mode."""
        if use_filter:
            return f"""
            {cls.PREFIXES}
            SELECT ?var ?var_label ?visit_label ?domain_val ?stat_label ?unit_label
            WHERE {{
                GRAPH <{cls.graph_iri(study)}> {{
                    ?de a {T.data_element} ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                    FILTER NOT EXISTS {{ ?de skos:closeMatch ?any }}
                    OPTIONAL {{ ?de {T.is_denoted_by}/{T.has_value} ?stat_label. }}
                    OPTIONAL {{ ?de {T.has_attribute} ?visit.
                                ?visit a {T.visit_type};
                                  {T.has_value} ?visit_label. }}
                    OPTIONAL {{ ?de {T.has_annotation}/{T.has_value} ?domain_val. }}
                    OPTIONAL {{ ?de {T.has_measurement_unit_label}/{T.has_value} ?unit_label. }}
                }}
            }}
            """
        else:
            return f"""
            {cls.PREFIXES}
            SELECT ?var ?var_label ?visit_label ?domain_val ?stat_label ?unit_label
            WHERE {{
                GRAPH <{cls.graph_iri(study)}> {{
                    ?de a {T.data_element} ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                  
                    OPTIONAL {{ ?de {T.is_denoted_by}/{T.has_value} ?stat_label. }}
                    OPTIONAL {{ ?de {T.has_attribute} ?visit.
                                  ?visit a {T.visit_type};
                                  {T.has_value} ?visit_label . }}
                    OPTIONAL {{ ?de {T.has_annotation}/{T.has_value} ?domain_val. }}
                    OPTIONAL {{ ?de {T.has_measurement_unit_label}/{T.has_value} ?unit_label. }}
                }}
            }}
            """
    @classmethod
    def build_statistic_query(cls, source: str, values_str: str) -> str:
        query =  f"""
        {cls.PREFIXES}
        SELECT 
        ?identifier 
        (SAMPLE(?_stat_label)    AS ?stat_label)
        (SAMPLE(?_unit_label)    AS ?unit_label)
        (SAMPLE(?_unit_omop_id)  AS ?unit_omop_id)
        (SAMPLE(?_data_type_val) AS ?data_type_val)
        (MAX(?_min_v)            AS ?min_val)
        (MAX(?_max_v)            AS ?max_val)
        # COALESCE: on Oxigraph >= 0.4.7 one unbound value (e.g. a category with
        # no OMOP code) blanks the whole GROUP_CONCAT for the variable. The ""
        # parts it leaves are skipped by the parser (_split_labels / _split_ids).
        (GROUP_CONCAT(DISTINCT COALESCE(STR(?cat_omop_id), ""); SEPARATOR="||") AS ?cat_omop_ids)
        (GROUP_CONCAT(DISTINCT COALESCE(?_catL, ""); SEPARATOR="||") AS ?all_cat_labels)
        (GROUP_CONCAT(DISTINCT COALESCE(?_catV, ""); SEPARATOR="||") AS ?all_original_cat_values)
        (GROUP_CONCAT(DISTINCT COALESCE(?_catPair, ""); SEPARATOR="||") AS ?all_cat_pairs)

        (SAMPLE(?_ordered_code_labels) AS ?code_label)
        (SAMPLE(?_ordered_code_values) AS ?code_value)
        (SAMPLE(?_ordered_omop_ids) AS ?omop_ids)

        WHERE {{
            GRAPH <{cls.graph_iri(source)}> {{
            VALUES ?identifier {{ {values_str} }}
            ?dataElement a {T.data_element} ;
                dc:identifier ?identifier .

            OPTIONAL {{
            ?dataElement {T.is_denoted_by}/{T.has_value} ?_stat_label .
        }}

            OPTIONAL {{
            ?dataElement {T.has_attribute} ?dt .
            ?dt a {T.data_type} ;
            {T.has_value} ?_data_type_val .
        }}

            BIND(URI(CONCAT(STR(?dataElement), "/statistic")) AS ?statUri)

            OPTIONAL {{
            ?statUri {T.has_part} ?minPart .
            ?minPart a {T.minimum_value} ;
            {T.has_value} ?_min_v .
        }}

            OPTIONAL {{
            ?statUri {T.has_part} ?maxPart .
            ?maxPart a {T.maximum_value} ;
            {T.has_value} ?_max_v .
        }}

        #unit
        OPTIONAL {{
            ?dataElement {T.has_measurement_unit_label} ?unit_node .

            OPTIONAL {{
                ?unit_node skos:closeMatch/{T.has_value} ?standard_unit .
                }}

            OPTIONAL {{
                ?unit_node {T.has_value} ?raw_unit .
            }}

            # The unit's OMOP concept. UCUM strings are spelled many ways for one
            # unit across these dictionaries, so the id is what says two units are
            # the same; the label is kept for display and for the conversion table.
            OPTIONAL {{
                ?unit_node skos:closeMatch/{T.denotes}/{T.has_value} ?_unit_omop_id .
            }}

            BIND(COALESCE(?standard_unit, ?raw_unit) AS ?_unit_label)
        }}

            # Categories
            OPTIONAL {{
            ?cat_val a {T.categorical_value_specification} ;
                {T.specifies_value_of} ?dataElement ;
                {T.has_value} ?_catV .
            OPTIONAL {{
            ?cat_val skos:closeMatch ?cat_code .
            ?cat_code a skos:Concept ;
            rdfs:label ?_catL .
            OPTIONAL {{ ?cat_code {T.denotes}/{T.has_value} ?cat_omop_id . }}
        
        }}
        BIND(CONCAT(STR(?_catV), "=", COALESCE(STR(?_catL), STR(?_catV))) AS ?_catPair)

        }}

            # Codes subquery - OUTSIDE the GRAPH block but joined on ?dataElement
            OPTIONAL {{
            SELECT ?dataElement 
            (GROUP_CONCAT(?cL; SEPARATOR="||") AS ?_ordered_code_labels)
            (GROUP_CONCAT(?cV; SEPARATOR="||") AS ?_ordered_code_values)
            (GROUP_CONCAT(str(?omop_id); SEPARATOR="||") AS ?_ordered_omop_ids)
            WHERE {{
            GRAPH <{cls.graph_iri(source)}> {{
            SELECT ?dataElement ?cL ?cV ?omop_id ?seqNum WHERE {{
            ?dataElement skos:closeMatch ?codeSet .
            ?codeSet ?seqPred ?codeNode .
            ?codeNode a skos:Concept ;
            rdfs:label ?cL ;
            {T.has_value} ?cV;
            {T.denotes}/{T.has_value} ?omop_id.
            FILTER(STRSTARTS(STR(?seqPred), CONCAT(STR(rdf:), "_")))
            BIND(xsd:integer(STRAFTER(STR(?seqPred), CONCAT(STR(rdf:), "_"))) AS ?seqNum)
        }}
            ORDER BY ?dataElement ?seqNum
        }}
        }}
            GROUP BY ?dataElement
        }}
        }}
        }}
        GROUP BY ?identifier
        ORDER BY ?identifier

                
        """
        # print("Generated SPARQL Query:")
        # print(query)
        return query
        
    @staticmethod
    def _separator_insensitive(study_id: str) -> str:
        """Collapse spaces, underscores and hyphens so the same study matches
        regardless of how its name was punctuated in the source spreadsheet.
        Mirrors the REPLACE() applied to dc:identifier inside the query."""
        return re.sub(r"[\s_-]+", " ", str(study_id).strip().lower())

    @classmethod
    def build_study_context_query(cls,study_id: str) -> str:
        # Study ids reach us as cohort folder / graph names ("gissi-hf_outcomes"),
        # while dc:identifier carries the spreadsheet spelling of the same study
        # ("gissi-hf outcomes"). Collapse separators on both sides so either
        # spelling matches instead of rewriting one into the other.
        # Reads studies_metadata, which cohort-explorer's backend/src/upload.py still
        # builds with the old IRIs and without dc:identifier on the study design
        # execution; see the note on study_kg.generate_studies_kg.
        study_key = cls._separator_insensitive(study_id)
        query =  f"""
        {cls.PREFIXES}
                SELECT
                    ?study_name
                    (SAMPLE(?design_val)    AS ?study_design)
                    (SAMPLE(?type_label)    AS ?study_type)
                    (SAMPLE(?n_val)         AS ?n_participants)
                    (GROUP_CONCAT(DISTINCT ?morb_label; SEPARATOR="; ") AS ?morbidities)
                    (SAMPLE(?age_val)       AS ?age_distribution)
                    (SAMPLE(?loc_val)       AS ?location)
                    (SAMPLE(?obj_val)       AS ?objective)
                    (GROUP_CONCAT(DISTINCT ?inc_val; SEPARATOR="; ")    AS ?inclusion_criteria)
                WHERE {{
                    GRAPH <{cls.METADATA_GRAPH}> {{

                        ?sde a {T.study_design_execution} ;
                            dc:identifier ?study_name .
                        FILTER(REPLACE(LCASE(STR(?study_name)), "[ _-]+", " ") = "{study_key}")

                        # ── Study design ──
                        OPTIONAL {{ ?sde {T.concretizes} ?sdA . ?sdA {T.has_value} ?design_val . }}

                        # ── Study descriptor ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdB .
                            ?sdB {T.is_about} ?desc .
                            ?desc a {T.descriptor} ; rdfs:label ?type_label .
                        }}

                        # ── Number of participants ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdC .
                            ?sdC {T.has_part} ?protC .
                            ?protC a {T.protocol} ; {T.has_part} ?np .
                            ?np a {T.number_of_participants} ; {T.has_value} ?n_val .
                        }}

                        # ── Population morbidity ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdD .
                            ?sdD {T.has_part} ?protD .
                            ?protD {T.has_part} ?eligD .
                            ?eligD a {T.eligibility_criterion} ; {T.is_concretized_by} ?enrollD .
                            ?enrollD {T.has_output} ?popD .
                            ?popD {T.has_characteristic} ?morb .
                            ?morb a {T.morbidity} ; rdfs:label ?morb_label .
                        }}

                        # ── Age distribution ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdE .
                            ?sdE {T.has_part} ?protE .
                            ?protE {T.has_part} ?eligE .
                            ?eligE a {T.eligibility_criterion} ; {T.is_concretized_by} ?enrollE .
                            ?enrollE {T.has_output} ?popE .
                            ?popE {T.has_characteristic} ?age .
                            ?age a {T.age_distribution} ; {T.has_value} ?age_val .
                        }}

                        # ── Population location ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdF .
                            ?sdF {T.has_part} ?protF .
                            ?protF {T.has_part} ?eligF .
                            ?eligF a {T.eligibility_criterion} ; {T.is_concretized_by} ?enrollF .
                            ?enrollF {T.has_output} ?popF .
                            ?site a {T.site} ; {T.is_about} ?popF ; {T.has_value} ?loc_val .
                        }}

                        # ── Study objective ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdG .
                            ?sdG {T.has_part} ?protG .
                            ?protG a {T.protocol} ; {T.has_part} ?obj .
                            ?obj a {T.objective_specification} ; {T.has_value} ?obj_val .
                        }}

                        # ── Inclusion criteria ──
                        OPTIONAL {{
                            ?sde {T.concretizes} ?sdH .
                            ?sdH {T.has_part} ?protH .
                            ?protH {T.has_part} ?eligH .
                            ?eligH a {T.eligibility_criterion} ; {T.has_part} ?inc_crit .
                            ?inc_crit a {T.inclusion_criterion} ; {T.has_part} ?inc_item .
                            ?inc_item {T.has_value} ?inc_val .
                        }}
                    }}
                }}
                GROUP BY ?study_name
                """
        # print(query)
        return query
