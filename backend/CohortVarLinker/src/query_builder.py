from .config import settings


class SPARQLQueryBuilder:
    """Responsible solely for constructing valid SPARQL queries."""
    
    PREFIXES = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX sio:  <http://semanticscience.org/ontology/sio.owl/>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc:   <http://purl.org/dc/elements/1.1/>
        PREFIX iao:  <http://purl.obolibrary.org/obo/iao.owl/>
        PREFIX cmeo: <https://w3id.org/CMEO/>
        PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
        PREFIX stato: <http://purl.obolibrary.org/obo/stato.owl/>
        PREFIX ro:    <http://purl.obolibrary.org/obo/ro.owl/>
        PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#> 
        PREFIX bfo: <http://purl.obolibrary.org/obo/bfo.owl/>
    """
    METADATA_GRAPH = "https://w3id.org/CMEO/graph/studies_metadata"
    GRAPHS_URI = settings.GRAPH_REPO


    @classmethod
    def build_alignment_query(cls, source: str, target: str) -> str:
        # Optimized query structure
        query= f""" 
            {cls.PREFIXES}
            SELECT ?omop_id
            (SAMPLE(?lbl) AS ?code_label)
            (SAMPLE(?val) AS ?code_value)
            (GROUP_CONCAT(DISTINCT ?src_de_str; SEPARATOR="||") AS ?source_de_uris)
            (GROUP_CONCAT(DISTINCT ?tgt_de_str; SEPARATOR="||") AS ?target_de_uris)
            (GROUP_CONCAT(DISTINCT ?src_combined; SEPARATOR="||") AS ?source_definition)
            (GROUP_CONCAT(DISTINCT ?tgt_combined; SEPARATOR="||") AS ?target_definition)
            (GROUP_CONCAT(DISTINCT ?src_val; SEPARATOR="||") AS ?source_domain)
            (GROUP_CONCAT(DISTINCT ?tgt_val; SEPARATOR="||") AS ?target_domain)
            WHERE {{
                {{
                    SELECT ?omop_id  ?lbl ?val ?src_de_str  ?src_combined ?src_val 
                    WHERE {{
                        GRAPH <{cls.GRAPHS_URI}/{source}> {{
                             ?deA a cmeo:data_element ; 
                                rdfs:label ?src_var_label ;
                                dc:identifier ?src_var ;
                                  skos:closeMatch ?codeSetA .
                             ?codeSetA rdf:_1 ?codeNodeA .
                             ?codeNodeA rdfs:label ?lbl ;
                                        cmeo:has_value ?val ;
                                        iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deA sio:has_attribute/skos:closeMatch/rdfs:label ?src_vis_label . }}
                             OPTIONAL {{ ?deA sio:has_annotation/cmeo:has_value ?src_val . }}
                            BIND(IF(BOUND(?src_vis_label), 
                                CONCAT(?src_var, ";;", ?src_var_label, ";;", ?src_vis_label), 
                                CONCAT(?src_var, ";;", ?src_var_label, ";;N/A")
                            ) AS ?src_combined)

                            BIND(STR(?deA) AS ?src_de_str)
                        }}
                    }}
                }}
                UNION
                {{
                     SELECT ?omop_id ?lbl ?val ?tgt_de_str ?tgt_combined ?tgt_val 
                    WHERE {{
                        GRAPH <{cls.GRAPHS_URI}/{target}> {{
                             ?deB a cmeo:data_element ; 
                                rdfs:label ?tgt_var_label ;
                                dc:identifier ?tgt_var ;
                                  skos:closeMatch ?codeSetB .
                             ?codeSetB rdf:_1 ?codeNodeB .
                             ?codeNodeB rdfs:label ?lbl ;
                                        cmeo:has_value ?val ;
                                        iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deB sio:has_attribute/skos:closeMatch/rdfs:label ?tgt_vis_label . }}
                             OPTIONAL {{ ?deB sio:has_annotation/cmeo:has_value ?tgt_val . }}
                            BIND(IF(BOUND(?tgt_vis_label), 
                            CONCAT(?tgt_var, ";;", ?tgt_var_label, ";;", ?tgt_vis_label), 
                            CONCAT(?tgt_var, ";;", ?tgt_var_label, ";;N/A")
                        ) AS ?tgt_combined)

                        BIND(STR(?deB) AS ?tgt_de_str)
                        }}
                    }}
                }}
            }}
            GROUP BY ?omop_id
            ORDER BY ?omop_id
        """
        # print("Generated SPARQL Query:")
        # print(query)
        return query

    # query_builder.py — add new method

    @classmethod


   
    def build_unmapped_variables_query(cls, study: str, use_filter:bool=False) -> str:
        """Variables with no skos:closeMatch — raw label/visit/unit/stat for NE mode."""
        if use_filter:
            return f"""
            {cls.PREFIXES}
            SELECT ?var ?var_label ?visit_label ?domain_val ?stat_label ?unit_label
            WHERE {{
                GRAPH <{cls.GRAPHS_URI}/{study}> {{
                    ?de a cmeo:data_element ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                    FILTER NOT EXISTS {{ ?de skos:closeMatch ?any }}
                    OPTIONAL {{ ?de iao:is_denoted_by/cmeo:has_value ?stat_label. }}
                    OPTIONAL {{ ?de sio:has_attribute ?visit.
                                ?visit a cmeo:visit_type;
                                  cmeo:has_value ?visit_label. }}
                    OPTIONAL {{ ?de sio:has_annotation/cmeo:has_value ?domain_val. }}
                    OPTIONAL {{ ?de obi:has_measurement_unit_label/cmeo:has_value ?unit_label. }}
                }}
            }}
            """
        else:
            return f"""
            {cls.PREFIXES}
            SELECT ?var ?var_label ?visit_label ?domain_val ?stat_label ?unit_label
            WHERE {{
                GRAPH <{cls.GRAPHS_URI}/{study}> {{
                    ?de a cmeo:data_element ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                  
                    OPTIONAL {{ ?de iao:is_denoted_by/cmeo:has_value ?stat_label. }}
                    OPTIONAL {{ ?de sio:has_attribute ?visit.
                                  ?visit a cmeo:visit_type;
                                  cmeo:has_value ?visit_label . }}
                    OPTIONAL {{ ?de sio:has_annotation/cmeo:has_value ?domain_val. }}
                    OPTIONAL {{ ?de obi:has_measurement_unit_label/cmeo:has_value ?unit_label. }}
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
        (SAMPLE(?_data_type_val) AS ?data_type_val)
        (MAX(?_min_v)            AS ?min_val)
        (MAX(?_max_v)            AS ?max_val)
        (GROUP_CONCAT(DISTINCT STR(?cat_omop_id); SEPARATOR="||") AS ?cat_omop_ids)
        (GROUP_CONCAT(DISTINCT ?_catL; SEPARATOR="||") AS ?all_cat_labels)
        (GROUP_CONCAT(DISTINCT ?_catV; SEPARATOR="||") AS ?all_original_cat_values)
        (SAMPLE(?_ordered_code_labels) AS ?code_label)
        (SAMPLE(?_ordered_code_values) AS ?code_value)
        (SAMPLE(?_ordered_omop_ids) AS ?omop_ids)

        WHERE {{
            GRAPH <{cls.GRAPHS_URI}/{source}> {{
            VALUES ?identifier {{ {values_str} }}
            ?dataElement a cmeo:data_element ;
                dc:identifier ?identifier .

            OPTIONAL {{
            ?dataElement iao:is_denoted_by/cmeo:has_value ?_stat_label .
        }}

            OPTIONAL {{
            ?dataElement sio:has_attribute ?dt .
            ?dt a cmeo:data_type ;
            cmeo:has_value ?_data_type_val .
        }}

            BIND(URI(CONCAT(STR(?dataElement), "/statistic")) AS ?statUri)

            OPTIONAL {{
            ?statUri ro:has_part ?minPart .
            ?minPart a stato:minimum_value ;
            cmeo:has_value ?_min_v .
        }}

            OPTIONAL {{
            ?statUri ro:has_part ?maxPart .
            ?maxPart a stato:maximum_value ;
            cmeo:has_value ?_max_v .
        }}

        #unit 
        OPTIONAL {{
            ?dataElement obi:has_measurement_unit_label ?unit_node .

            OPTIONAL {{
                ?unit_node skos:closeMatch/cmeo:has_value ?standard_unit .
                }}

            OPTIONAL {{
                ?unit_node cmeo:has_value ?raw_unit .
            }}

            BIND(COALESCE(?standard_unit, ?raw_unit) AS ?_unit_label)
        }}
            # Categories
            OPTIONAL {{
            ?cat_val a obi:categorical_value_specification ;
                obi:specifies_value_of ?dataElement ;
                cmeo:has_value ?_catV .
            OPTIONAL {{
            ?cat_val skos:closeMatch ?cat_code .
            ?cat_code a skos:concept ;
            rdfs:label ?_catL .
            OPTIONAL {{ ?cat_code iao:denotes/cmeo:has_value ?cat_omop_id . }}
        
        }}
        }}

            # Codes subquery - OUTSIDE the GRAPH block but joined on ?dataElement
            OPTIONAL {{
            SELECT ?dataElement 
            (GROUP_CONCAT(?cL; SEPARATOR="||") AS ?_ordered_code_labels)
            (GROUP_CONCAT(?cV; SEPARATOR="||") AS ?_ordered_code_values)
            (GROUP_CONCAT(str(?omop_id); SEPARATOR="||") AS ?_ordered_omop_ids)
            WHERE {{
            GRAPH <{cls.GRAPHS_URI}/{source}> {{
            SELECT ?dataElement ?cL ?cV ?omop_id ?seqNum WHERE {{
            ?dataElement skos:closeMatch ?codeSet .
            ?codeSet ?seqPred ?codeNode .
            ?codeNode a skos:concept ;
            rdfs:label ?cL ;
            cmeo:has_value ?cV;
            iao:denotes/cmeo:has_value ?omop_id.
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
        
    @classmethod
    def build_study_context_query(cls,study_id: str) -> str:
        study_id =  study_id.replace("_", " ")
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
                
                        # ── Anchor: study design execution ──
                        ?sde  a  obi:study_design_execution ;
                            dc:identifier  ?study_name .
                        FILTER(LCASE(STR(?study_name)) = LCASE("{study_id}"))
                
                        # ── Study design type (e.g., "randomized_controlled_trial") ──
                        OPTIONAL {{
                            ?sde  ro:concretizes  ?sd .
                            ?sd   cmeo:has_value  ?design_val .
                        }}
                
                        # ── Study descriptor (e.g., "interventional") ──
                        OPTIONAL {{
                            ?sd   iao:is_about  ?desc .
                            ?desc a sio:descriptor ;
                                rdfs:label ?type_label .
                        }}
                
                        # ── Number of participants ──
                        OPTIONAL {{
                            ?sd   ro:has_part  ?protocol .
                            ?protocol  a  obi:protocol .
                            ?protocol  ro:has_part  ?np .
                            ?np   a  cmeo:number_of_participants ;
                                cmeo:has_value  ?n_val .
                        }}
                
                        # ── Population morbidity (key: tells LLM what conditions are guaranteed) ──
                        OPTIONAL {{
                            ?sd   ro:has_part  ?protocol2 .
                            ?protocol2 ro:has_part ?elig .
                            ?elig a obi:eligibility_criterion .
                            ?elig ro:is_concretized_by ?enroll .
                            ?enroll ro:has_output ?pop .
                            ?pop  ro:has_characteristic ?morb .
                            ?morb a obi:morbidity ;
                                rdfs:label ?morb_label .
                        }}
                
                        # ── Age distribution ──
                        OPTIONAL {{
                            ?pop  ro:has_characteristic ?age .
                            ?age  a obi:age_distribution ;
                                cmeo:has_value ?age_val .
                        }}
                
                        # ── Population location ──
                        OPTIONAL {{
                            ?site a bfo:site ;
                                iao:is_about ?pop ;
                                cmeo:has_value ?loc_val .
                        }}
                
                        # ── Study objective ──
                        OPTIONAL {{
                            ?protocol ro:has_part ?obj .
                            ?obj a obi:objective_specification ;
                                cmeo:has_value ?obj_val .
                        }}
                
                        # ── Inclusion criteria (concatenated) ──
                        OPTIONAL {{
                            ?elig ro:has_part ?inc_crit .
                            ?inc_crit a obi:inclusion_criterion .
                            ?inc_crit ro:has_part ?inc_item .
                            ?inc_item cmeo:has_value ?inc_val .
                        }}
                    }}
                }}
                GROUP BY ?study_name
                """
        # print(query)
        return query
