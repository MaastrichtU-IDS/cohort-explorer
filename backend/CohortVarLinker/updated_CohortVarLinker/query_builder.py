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
    """

    @classmethod
    def build_alignment_query(cls, source: str, target: str, graph_repo: str) -> str:
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
                        GRAPH <{graph_repo}/{source}> {{
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
                        GRAPH <{graph_repo}/{target}> {{
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


   
    def build_unmapped_variables_query(cls, study: str, graph_repo: str, use_filter:bool=False) -> str:
        """Variables with no skos:closeMatch — raw label/visit/unit/stat for NE mode."""
        if use_filter:
            return f"""
            {cls.PREFIXES}
            SELECT ?var ?var_label ?visit_label ?domain_val ?stat_label ?unit_label
            WHERE {{
                GRAPH <{graph_repo}/{study}> {{
                    ?de a cmeo:data_element ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                    FILTER NOT EXISTS {{ ?de skos:closeMatch ?any }}
                    OPTIONAL {{ ?de iao:is_denoted_by/cmeo:has_value ?stat_label. }}
                    OPTIONAL {{ ?de sio:has_attribute ?visit.
                                ?visit a cmeo:visit_type;
                                  rdfs:label ?visit_label. }}
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
                GRAPH <{graph_repo}/{study}> {{
                    ?de a cmeo:data_element ;
                        dc:identifier ?var ;
                        rdfs:label ?var_label .
                  
                    OPTIONAL {{ ?de iao:is_denoted_by/cmeo:has_value ?stat_label. }}
                    OPTIONAL {{ ?de sio:has_attribute ?visit.
                                  ?visit a cmeo:visit_type;
                                  rdfs:label ?visit_label . }}
                    OPTIONAL {{ ?de sio:has_annotation/cmeo:has_value ?domain_val. }}
                    OPTIONAL {{ ?de obi:has_measurement_unit_label/cmeo:has_value ?unit_label. }}
                }}
            }}
            """
    @classmethod
    def build_statistic_query(cls, source: str, values_str: str, graph_repo: str) -> str:
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
            GRAPH <{graph_repo}/{source}> {{
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

            OPTIONAL {{
            ?dataElement obi:has_measurement_unit_label/skos:closeMatch/cmeo:has_value ?_unit_label .
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
            GRAPH <{graph_repo}/{source}> {{
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
        
        
       