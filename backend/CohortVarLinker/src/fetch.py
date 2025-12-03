from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper

from collections import defaultdict
from .config import settings
from .utils import apply_rules
from .vector_db import search_in_db

import json

# @dataclass
# class Element:
#     role: str
#     name: str
#     visit: str
#     omop_id: int
#     code: str
#     code_label: str
#     category: str

BASELINE_TIME_HINTS = ["6 months prior to baseline", "prior to baseline visit"]
DATE_HINTS = ["visit date", "date of visit","date of event"]

# we may later seperate "6 months prior to baseline", "prior to baseline visit" as a match but not to baseline time
DERIVED_VARIABLES_LIST= [
    
     {
                    "name": "BMI-derived",
                    "omop_id": 3038553,           
                    "code": "loinc:39156-5",
                    "label": "Body mass index (BMI) [Ratio]",
                    "unit": "kg/m2",
                    "required_omops": [3016723, 3025315],
                    "category": "measurement",
                    "data_type": "continuous_variable"
                },
                {
                    "name": "eGFR_CG-derived",
                    "omop_id": 37169169,          
                    "code": "snomed:1556501000000100",
                    "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
                    "unit": "ml/min",
                    "required_omops": [3016723, 3022304, 46235213],
                    "category": "measurement",
                    "data_type": "continuous_variable"
                }
]

def check_visit_string(visit_str_src: str, visit_str_tgt:str) -> str:
    # if src or tgt visit string contains any of the time hints, return the value of the visit that is not in time hint
    # print(f"Checking visit strings: src='{visit_str_src}', tgt='{visit_str_tgt}'")
    for hint in DATE_HINTS:
        if hint in visit_str_src.lower():
            return visit_str_tgt
        if hint in visit_str_tgt.lower():
            return visit_str_src

    for hint in BASELINE_TIME_HINTS:
        if hint in visit_str_src.lower() or hint in visit_str_tgt.lower():
            return 'baseline time'
    return visit_str_src




def _build_alignment_query(
    source: str, target: str, graph_repo: str
) -> str:
    """Return the SPARQL query used to retrieve variables of both studies."""
    
    return f""" 
            
            PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
            PREFIX dc:    <http://purl.org/dc/elements/1.1/>
            PREFIX ro:    <http://purl.obolibrary.org/obo/ro.owl/>
            PREFIX obi:   <http://purl.obolibrary.org/obo/obi.owl/>
            PREFIX iao:   <http://purl.obolibrary.org/obo/iao.owl/>
            PREFIX bfo:   <http://purl.obolibrary.org/obo/bfo.owl/>
            PREFIX cmeo:  <https://w3id.org/CMEO/>

            SELECT
            ?omop_id ?code_label ?code_value ?val
            (GROUP_CONCAT(DISTINCT ?varNameA; SEPARATOR=", ") AS ?source)
            (GROUP_CONCAT(DISTINCT ?varNameB; SEPARATOR=", ") AS ?target)
            (GROUP_CONCAT(DISTINCT ?visitsA ; SEPARATOR=", ") AS ?source_visit)
            (GROUP_CONCAT(DISTINCT ?visitsB ; SEPARATOR=", ") AS ?target_visit)
            WHERE {{
            {{
                # ---------- TIME-CHF (source) ----------
                SELECT
                ?omop_id ?code_label ?code_value ?val
                (GROUP_CONCAT(DISTINCT ?var_nameA; SEPARATOR=", ") AS ?varNameA)
                (GROUP_CONCAT(DISTINCT ?pairA   ; SEPARATOR=", ") AS ?visitsA)
                ("{source}" AS ?source)
                WHERE {{
                GRAPH <{graph_repo}/{source}>  {{
                    # 1) Most selective: mapping from data element -> standardized code (rdf:_1) -> OMOP
                    ?stdProcessA a cmeo:data_standardization ;
                                obi:has_specified_output ?codeSetA ;
                                obi:has_specified_input  ?dataElementA .
                    ?codeSetA rdf:_1 ?codeNodeA .
                    ?codeNodeA a cmeo:code ;
                            cmeo:has_value ?code_value ;
                            rdfs:label     ?code_label ;
                            iao:denotes    ?omopClassA .
                    ?omopClassA a cmeo:omop_id ; cmeo:has_value ?omop_id .

                    # 2) Data element identity (single valued)
                    ?dataElementA a cmeo:data_element ; dc:identifier ?var_nameA .

                    # 3) Optional category value (kept single per DE; if multi, choose one deterministically)
                    OPTIONAL {{
                    ?catProcessA a cmeo:categorization_process ;
                                obi:has_specified_input  ?dataElementA ;
                                obi:has_specified_output ?catOutA .
                    ?catOutA cmeo:has_value ?val .
                    }}

                    # 4) Visits per data element — pre-aggregate to avoid fan-out
                    OPTIONAL {{
                    {{
                        SELECT ?dataElementA (GROUP_CONCAT(DISTINCT ?visitLblA; SEPARATOR="|") AS ?visitStrA)
                        WHERE {{
                        ?visitDatumA a cmeo:visit_measurement_datum ;
                                    iao:is_about ?dataElementA ;
                                    obi:is_specified_input_of ?vsProcA .
                        ?vsProcA obi:has_specified_output ?visitCodeA .
                        ?visitCodeA rdfs:label ?visitLblA .
                        }}
                        GROUP BY ?dataElementA
                    }}
                    }}

                    # 5) Make the (var || visits) pair once, not multiplicatively
                    BIND(COALESCE(?visitStrA, "") AS ?visA)
                    BIND(CONCAT(STR(?var_nameA), "||", ?visA) AS ?pairA)
                }}
                }}
                GROUP BY ?omop_id ?code_label ?code_value ?val
            }}
            UNION
            {{
                # ---------- GISSI-HF (target) ----------
                SELECT
                ?omop_id ?code_label ?code_value ?val
                (GROUP_CONCAT(DISTINCT ?var_nameB; SEPARATOR=", ") AS ?varNameB)
                (GROUP_CONCAT(DISTINCT ?pairB   ; SEPARATOR=", ") AS ?visitsB)
                ("{target}" AS ?target)
                WHERE   {{
                GRAPH <{graph_repo}/{target}> {{
                    ?stdProcessB a cmeo:data_standardization ;
                                obi:has_specified_output ?codeSetB ;
                                obi:has_specified_input  ?dataElementB .
                    ?codeSetB rdf:_1 ?codeNodeB .
                    ?codeNodeB a cmeo:code ;
                            cmeo:has_value ?code_value ;
                            rdfs:label     ?code_label ;
                            iao:denotes    ?omopClassB .
                    ?omopClassB a cmeo:omop_id ; cmeo:has_value ?omop_id .

                    ?dataElementB a cmeo:data_element ; dc:identifier ?var_nameB .

                    OPTIONAL {{
                    ?catProcessB a cmeo:categorization_process ;
                                obi:has_specified_input  ?dataElementB ;
                                obi:has_specified_output ?catOutB .
                    ?catOutB cmeo:has_value ?val .
                    }}

                    OPTIONAL {{
                    {{
                        SELECT ?dataElementB (GROUP_CONCAT(DISTINCT ?visitLblB; SEPARATOR="|") AS ?visitStrB)
                        WHERE {{
                        ?visitDatumB a cmeo:visit_measurement_datum ;
                                    iao:is_about ?dataElementB ;
                                    obi:is_specified_input_of ?vsProcB .
                        ?vsProcB obi:has_specified_output ?visitCodeB .
                        ?visitCodeB rdfs:label ?visitLblB .
                        }}
                        GROUP BY ?dataElementB
                    }}
                    }}

                    BIND(COALESCE(?visitStrB, "") AS ?visB)
                    BIND(CONCAT(STR(?var_nameB), "||", ?visB) AS ?pairB)
                }}
                }}
                GROUP BY ?omop_id ?code_label ?code_value ?val
            }}
            }}
            GROUP BY ?omop_id ?code_label ?code_value ?val
            ORDER BY ?omop_id

        """
    # return f"""
        
    #         PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    #         PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    #         PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
    #         PREFIX dc:    <http://purl.org/dc/elements/1.1/>
    #         PREFIX ro:    <http://purl.obolibrary.org/obo/ro.owl/>
    #         PREFIX obi:   <http://purl.obolibrary.org/obo/obi.owl/>
    #         PREFIX iao:   <http://purl.obolibrary.org/obo/iao.owl/>
    #         PREFIX sio:   <http://semanticscience.org/ontology/sio.owl/>
    #         PREFIX bfo:   <http://purl.obolibrary.org/obo/bfo.owl/>
    #         PREFIX ncbi:  <http://purl.bioontology.org/ontology/NCBITAXON/>
    #         PREFIX cmeo:  <https://w3id.org/CMEO/>
    #         SELECT
    #         ?omop_id ?code_label ?code_value ?val
    #         (GROUP_CONCAT(DISTINCT ?varNameA; SEPARATOR=", ") AS ?source)
    #         (GROUP_CONCAT(DISTINCT ?varNameB; SEPARATOR=", ") AS ?target)
    #         (GROUP_CONCAT(DISTINCT STR(?visitsA); SEPARATOR=", ") AS ?source_visit)
    #         (GROUP_CONCAT(DISTINCT STR(?visitsB); SEPARATOR=", ") AS ?target_visit)
            
    #         WHERE 
    #         {{
    #          {{
    #                     SELECT
    #                     ?omop_id ?code_label ?code_value ?val

    #                     (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountA)
    #                     (GROUP_CONCAT(DISTINCT STR(?var_nameA); SEPARATOR=", ") AS ?varNameA)
    #                     (GROUP_CONCAT(CONCAT(STR(?var_nameA), "||", STR(?visitcodelabelA)); SEPARATOR=", ") AS ?visitsA)
    #                     ("{source}" AS ?source)
    #                     WHERE {{
    #                     GRAPH <{graph_repo}/{source}> 
    #                     {{
    #                                 ?dataElementA rdf:type cmeo:data_element ;
    #                                                 dc:identifier ?var_nameA ;
    #                                                 obi:is_specified_input_of ?catProcessA, ?stdProcessA .
    #                                  OPTIONAL {{
    #                                 ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
    #                                             iao:is_about ?dataElementA ;
    #                                             obi:is_specified_input_of ?vs_stdProcessA .
                                    
                                    
    #                                 ?vs_stdProcessA obi:has_specified_output ?visit_code.
    #                                 ?visit_code rdfs:label ?visitcodelabelA.
    #                                 }}
    #                                 ?catProcessA rdf:type cmeo:categorization_process ;
    #                                             obi:has_specified_output ?cat_outputA .
    #                                 ?cat_outputA cmeo:has_value ?val .
    #                                 #FILTER(?val IN ("measurement", "drug_exposure"))

    #                                 ?stdProcessA rdf:type cmeo:data_standardization ;
    #                                             obi:has_specified_output ?codeA .
    #                                 ?codeA rdf:_1 ?primary_code_literal .
    #                                 ?primary_code_literal iao:denotes ?omop_id_uri ;
    #                                         cmeo:has_value ?code_value ;
    #                                         rdfs:label ?code_label .
    #                                 ?omop_id_uri rdf:type cmeo:omop_id ;
    #                                             cmeo:has_value ?omop_id .
    #                         }}
    #                     }}
    #                     GROUP BY ?omop_id ?code_label ?code_value ?val
    #             }}
    #         UNION
    #         {{
    #                 SELECT
    #                 ?omop_id ?code_label  ?code_value ?val
    #                 (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountB)
    #                 (GROUP_CONCAT(DISTINCT STR(?var_nameB); SEPARATOR=", ") AS ?varNameB)
    #                  (GROUP_CONCAT(CONCAT(STR(?var_nameB), "||", STR(?visitcodelabelB)); SEPARATOR=", ") AS ?visitsB)
    #                 ("{target}" AS ?target)
    #                     WHERE 
    #                     {{
    #                             GRAPH <{graph_repo}/{target}> 
    #                             {{
    #                                 ?dataElementB rdf:type cmeo:data_element ;
    #                                 dc:identifier ?var_nameB ;
    #                                 obi:is_specified_input_of ?catProcessB, ?stdProcessB.
                                    
    #                                 OPTIONAL {{
    #                                 ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
    #                                             iao:is_about ?dataElementB ;
    #                                             obi:is_specified_input_of ?vs_stdProcessAB .
                                    
    #                                 ?vs_stdProcessAB obi:has_specified_output ?visit_code.
    #                                 ?visit_code rdfs:label ?visitcodelabelB.
                                    
    #                                 }}
    #                                 ?catProcessB rdf:type cmeo:categorization_process ;
    #                                 obi:has_specified_output ?cat_outputB .
    #                                 ?cat_outputB cmeo:has_value ?val .
    #                                 #FILTER(?val IN ("measurement", "drug_exposure"))

    #                                 ?stdProcessB rdf:type cmeo:data_standardization ;
    #                                         obi:has_specified_output ?codeB .
    #                                 ?codeB rdf:_1 ?primary_code_literal .
    #                                 ?primary_code_literal iao:denotes ?omop_id_uri ;
    #                                 cmeo:has_value ?code_value;
    #                                 rdfs:label ?code_label.
    #                                 ?omop_id_uri rdf:type cmeo:omop_id ;
    #                                 cmeo:has_value ?omop_id.
    #                             }}
    #                     }}

    #                 GROUP BY ?omop_id  ?code_label ?code_value  ?val
    #             }}
    #         }}
    #         GROUP BY ?omop_id ?code_label ?code_value ?val
    #         #HAVING (COUNT(DISTINCT ?source) < 3)
    #         ORDER BY ?omop_id
    # """


def _execute_query(query: str) -> Iterable[Dict[str, Any]]:
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def _parse_bindings(bindings: Iterable[Dict[str, Any]]) -> tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Return source elements, target elements and exact matches."""
    source_elems, target_elems, matches = [], [], []
    for result in bindings:
        omop = int(result["omop_id"]["value"])
        code_label = result["code_label"]["value"]
        code_value = result["code_value"]["value"]
        category = result["val"]["value"].strip().lower()

        src_vars = result["source"]["value"].split(", ") if result["source"]["value"] else []
        tgt_vars = result["target"]["value"].split(", ") if result["target"]["value"] else []
        src_visits = [
            r.split("||")[-1].strip() for r in result["source_visit"]["value"].split(", ")
        ] if result["source_visit"]["value"] else ["baseline time"] * len(src_vars)
        tgt_visits = [
            r.split("||")[-1].strip() for r in result["target_visit"]["value"].split(", ")
        ] if result["target_visit"]["value"] else ["baseline time"] * len(tgt_vars)
        print(f"src vars: {src_vars}, src visits: {src_visits}")
        assert len(src_vars) == len(src_visits), (
            f"Visit column Length mismatch with variable labels: {len(src_vars)} != {len(src_visits)}"
        )
        print(f"tgt vars: {tgt_vars}, tgt visits: {tgt_visits}")
        assert len(tgt_vars) == len(tgt_visits), (
            f"Visit column Length mismatch with variable labels: {len(tgt_vars)} != {len(tgt_visits)}"
        )
        matches.extend(
            _exact_match_records(src_vars, tgt_vars, src_visits, tgt_visits,
                                omop, code_value, code_label, category)
        )
        source_elems.extend(
            _build_elements("source", src_vars, src_visits, omop, code_value, code_label, category)
        )
        target_elems.extend(
            _build_elements("target", tgt_vars, tgt_visits, omop, code_value, code_label, category)
        )
    return source_elems, target_elems, matches


# {
#             'omop_id': omop_id,
#             'code': code_id.strip(),
#             'code_label': code_label,
#             role: el,
#             'category': domain,
#             'visit': vis,
#         }

def _build_elements(
    role: str,
    variables: List[str],
    visits: List[str],
    omop_id: int,
    code: str,
    code_label: str,
    category: str,
) -> List[dict[str, Any]]:
    return [
        {
            'omop_id': omop_id,
            'code': code.strip(),
            'code_label': code_label,
            role: el,
            'category': category,
            'visit': vis,
        }
        for el, vis in zip(variables, visits)
    ]

def _exact_match_records(
    src_vars: List[str],
    tgt_vars: List[str],
    src_visits: List[str],
    tgt_visits: List[str],
    omop: int,
    code_value: str,
    code_label: str,
    category: str,
) -> List[Dict[str, Any]]:
    """Return records where source and target share OMOP id and visit."""
    res = []
    for s, sv in zip(src_vars, src_visits):
        for t, tv in zip(tgt_vars, tgt_visits):
            # sv = check_visit_string(sv)
            # tv = check_visit_string(tv)
            if check_visit_string(sv, tv) == check_visit_string(tv, sv):
                res.append(
                    {
                        "source": s,
                        "target": t,
                        "somop_id": omop,
                        "tomop_id": omop,
                        "scode": code_value.strip(),
                        "slabel": code_label,
                        "tcode": code_value.strip(),
                        "tlabel": code_label,
                        "category": category,
                        "mapping type": "code match",
                        "source_visit": sv,
                        "target_visit": tv,
                    }
                )
    return res


def extend_with_derived_variables(single_source: dict, 
                                  standard_derived_variable: tuple, 
                                  parameters_omop_ids: list, 
                                  variable_name: str, category: str) -> dict:
    """
    If source and target both can derive a standard variable (e.g. BMI, eGFR),
    create a new mapping row for the derived variable.
    `single_source` should have keys 'source', 'target', and 'mapped' (all lists of dicts).
    `standard_derived_variable`: tuple(code, label, omop_id)
    `parameters_omop_ids`: list of OMOP IDs needed to compute variable.
    `variable_name`: string, e.g., "bmi"
    """
    single_source = single_source.copy()  # Avoid in-place mutation

    def find_omop_id_rows(data: list, omop_code: str, code_key: str = "omop_id") -> list:
        found = []
        for row in data:
            code_value = row.get(code_key, "")
            if int(code_value) == int(omop_code):
                found.append(row)
        return found

    def can_produce_variable(data: dict, parameters_codes: list, side: str = "source") -> bool:
        code_key = "somop_id" if side == "source" else "tomop_id"
        has_parameter_un_mapped = all(
            len(find_omop_id_rows(data[side], code, code_key="omop_id")) > 0 for code in parameters_codes
        )
        has_parameters_mapped = all(
            len(find_omop_id_rows(data['mapped'], code, code_key=code_key)) > 0 for code in parameters_codes
        )
        return has_parameter_un_mapped or has_parameters_mapped

    source_derived_rows = find_omop_id_rows(single_source["source"], standard_derived_variable[2], code_key="omop_id")
    target_derived_rows = find_omop_id_rows(single_source["target"], standard_derived_variable[2], code_key="omop_id")

    # If neither side has the variable, and cannot produce it, do nothing
    if not source_derived_rows and not target_derived_rows:
        return {}
    source_can = can_produce_variable(single_source, parameters_omop_ids, side="source")
    target_can = can_produce_variable(single_source, parameters_omop_ids, side="target")
    if not (source_can and target_can):
        return {}

    if source_derived_rows:
        source_varname = source_derived_rows[0]["source"]
    else:
        source_varname = f"{variable_name}(derived)"
    if target_derived_rows:
        target_varname = target_derived_rows[0]["target"]
    else:
        target_varname = f"{variable_name} (derived)"

    mapping_type = "derived match" if ("derived" in source_varname.lower() or "derived" in target_varname.lower()) else "code match"
    return {
        "source": source_varname,
        "target": target_varname,
        "somop_id": standard_derived_variable[2],
        "tomop_id": standard_derived_variable[2],
        "scode": standard_derived_variable[0],
        "slabel": standard_derived_variable[1],
        "tcode": standard_derived_variable[0],
        "tlabel": standard_derived_variable[1],
        "mapping type": mapping_type,
        "source_visit": "baseline time",
        "target_visit": "baseline time",
        "category": category,
        "transformation_rule": {
            "description": f"Derived variable {variable_name} using variable columns  {parameters_omop_ids} from original dataset. Consider the timeline of the longitudinal data when using this variable.",
        }
    }

def _graph_vector_matches(
    src: List[dict[str, Any]],
    tgt: List[dict[str, Any]],
    graph: Any,
    vector_db: Any,
    embed_model: Any,
    target_study: str,
    collection_name: str,
) -> List[Dict[str, Any]]:
    """Match remaining variables using the OMOP graph or embedding search."""
    final: List[Dict[str, Any]] = []
    src_map: Dict[tuple, List[dict[str, Any]]] = defaultdict(list)
    tgt_map: Dict[tuple, List[dict[str, Any]]] = defaultdict(list)

    for el in src:
        src_map[(el["omop_id"], el["category"])].append(el)
    for el in tgt:
        tgt_map[(el["omop_id"], el["category"])].append(el)

    unique_targets: Dict[str, set[int]] = {}
    for el in tgt:
        unique_targets.setdefault(el["category"], set()).add(el["omop_id"])

    for (sid, category), s_elems in src_map.items():
        tgt_ids = unique_targets.get(category, set()) - {sid}
        if not tgt_ids:
            continue
        label = s_elems[0]["code_label"]
        reachable = None
        if category in {"drug_exposure", "drug_era"}:
            reachable = graph.bfs_bidirectional_reachable(sid, tgt_ids, max_depth=2, domain ='drug')
            
        elif category in {"condition_occurrence", "condition_era"}:
            reachable = graph.bfs_bidirectional_reachable(sid, tgt_ids, max_depth=2, domain ='condition')
        elif category in { "measurement", "procedure_occurrence", "observation", "device_exposure", "visit_occurrence", "specimen"}:
            reachable = graph.only_upward_or_downward(sid, tgt_ids, max_depth=1)

        if reachable:
            matched = set(reachable)
        else:
            # score = 0.65 if category in {"drug_exposure", "drug_era"} else 0.7
            score = 0.65
            matched = set(
                search_in_db(
                    vectordb=vector_db,
                    embedding_model=embed_model,
                    query_text=label,
                    target_study=target_study,
                    limit=100,
                    omop_domain=[category],
                    min_score=score,
                    collection_name=collection_name,
                )
            )

        for tid in matched:
            key = (tid, category)
            if key not in tgt_map:
                continue
            for se in s_elems:
                for te in tgt_map[key]:
                    tv = te['visit']
                    sv = se['visit']
                    if se["category"] != te["category"] or check_visit_string(sv, tv) != check_visit_string(tv, sv):
                        continue
                    final.append(
                        {
                            "source": se.get("source", ""),
                            "target": te.get("target", ""),
                            "source_visit": se.get("visit", ""),
                            "target_visit": te.get("visit", ""),
                            "somop_id": se.get("omop_id", ""),
                            "tomop_id": te.get("omop_id", ""),
                            "scode": se.get("code", ""),
                            "slabel": se.get("code_label", ""),
                            "tcode": te.get("code", ""),
                            "tlabel": te.get("code_label", ""),
                            "category": category,
                            "mapping type": "graph hierarchy match" if reachable else "semantic text match"
                        }
                    )
    return final



def fetch_variables_statistic_type(var_names_list:list[str], study_name:str) -> pd.DataFrame:

    data_dict = []
    # split var_names_list with 
    # make multiple lists by having 30 items in each list
    var_names_list_ = [var_names_list[i:i + 50] for i in range(0, len(var_names_list), 50)]
    # print(f"length of var_names_list: {len(var_names_list_)}")
    for var_list in var_names_list_:
        values_str = " ".join(f'"{v}"' for v in var_list)
        query = f"""
            PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dc:   <http://purl.org/dc/elements/1.1/>
            PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
            PREFIX cmeo: <https://w3id.org/CMEO/>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
            PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
            SELECT DISTINCT
                ?identifier
                ?stat_label
                ?unit_label
                ?data_type_val
                (GROUP_CONCAT(DISTINCT ?cate_code_label; separator="; ") AS ?all_cat_labels)
                (GROUP_CONCAT(DISTINCT ?original_cat_val_value; separator="; ") AS ?all_original_cat_values)

                WHERE {{
                GRAPH <https://w3id.org/CMEO/graph/{study_name}> {{
                
                    # Input: dc:identifier values
                    VALUES ?identifier {{ {values_str}
                    }}

                    ?dataElement dc:identifier ?identifier .

                    # Optional: Statistical description
                    OPTIONAL {{
                    ?dataElement iao:is_denoted_by ?stat .
                    ?stat cmeo:has_value ?stat_label.
                    ?data_type a cmeo:data_type;
                        iao:is_about ?dataElement;
                        cmeo:has_value ?data_type_val.
                    }}
                    # Optional: Measurement unit
                    OPTIONAL {{

                        ?dataElement obi:has_measurement_unit_label ?unit .
                        ?unit a obi:measurement_unit_label; obi:is_specified_input_of ?mu_standardization.
                        
                        ?mu_standardization obi:has_specified_output ?unit_code_node .
                        ?unit_code_node a cmeo:code; cmeo:has_value ?unit_label .

                }}
                # Optional: permissible values
                    OPTIONAL {{

                    ?cat_val a obi:categorical_value_specification;
                        obi:specifies_value_of ?dataElement;
                        obi:is_specified_input_of ?mu_standardization;
                        cmeo:has_value ?original_cat_val_value .
                        
                    ?mu_standardization obi:has_specified_output ?cat_codes .
                    ?cat_codes rdfs:label  ?cate_code_label
                    }}
                }}
                }}
                GROUP BY ?identifier ?stat_label ?unit_label ?data_type_val
                ORDER BY ?identifier

        """
        # print(query)
        sparql = SPARQLWrapper(settings.query_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        for result in results["results"]["bindings"]:
            identifier = result['identifier']['value']
            if identifier in var_names_list:
                data_dict.append({
                    'identifier': identifier,
                    'stat_label': result['stat_label']['value'] if 'stat_label' in result else None,
                    'unit_label': result['unit_label']['value'] if 'unit_label' in result else None,
                    'data_type': result['data_type_val']['value'] if 'data_type_val' in result else None,
                    "categories_labels": result['all_cat_labels']['value'] if 'all_cat_labels' in result else None,
                    'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result else None
                })
    data_dict = pd.DataFrame.from_dict(data_dict)
    # print(f"head of data dict: {data_dict.head()}")
    return data_dict




def fetch_variables_eda(var_names_list:list[str], study_name:str) -> pd.DataFrame:

    data_dict = []
    # split var_names_list with 
    # make multiple lists by having 30 items in each list
    var_names_list_ = [var_names_list[i:i + 50] for i in range(0, len(var_names_list), 50)]
    # print(f"length of var_names_list: {len(var_names_list_)}")
    for var_list in var_names_list_:
        values_str = " ".join(f'"{v}"' for v in var_list)
        query = f"""
            PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dc:   <http://purl.org/dc/elements/1.1/>
            PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
            PREFIX cmeo: <https://w3id.org/CMEO/>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
            PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
            PREFIX stato: <http://purl.obolibrary.org/obo/stato.owl/>
            SELECT DISTINCT
                ?identifier
                ?stat_label
                GROUP_CONCAT(DISTINCT ?statistic_part; separator=";") AS ?all_statistic_parts
                GROUP_CONCAT(DISTINCT ?stat_value; separator="; ") AS ?all_stat_values
                

                WHERE {{
                GRAPH <https://w3id.org/CMEO/graph/{study_name}> {{
                
                    # Input: dc:identifier values
                    VALUES ?identifier {{ {values_str}}}

                    ?dataElement dc:identifier ?identifier .

                    # Optional: Statistical description
                    OPTIONAL {{
                    ?dataElement iao:is_denoted_by ?stat .
                    ?stat cmeo:has_value ?stat_label.
                    
                    ?dataset iao:is_about  ?stat.
                       obi:is_specified_input_of ?eda_process.
                    ?eda_process a cmeo:exploratory_data_analysis ;
                        obi:has_specified_output ?eda_output.
                    
                    ?eda_output a stato:statistic.
                    
                    
                       OPTIONAL {{

                               ?eda_output  ro:has_part ?statistic_part.
                                 ?statistic_part cmeo:has_value ?stat_value.
                           
                           }}
                    
                    
                      
                    
                    }}
                   
                }}
                }}
                GROUP BY ?identifier ?stat_label
                ORDER BY ?identifier

        """
        print(query)
        sparql = SPARQLWrapper(settings.query_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        for result in results["results"]["bindings"]:
            identifier = result['identifier']['value']
            if identifier in var_names_list:
                data_dict.append({
                    'identifier': identifier,
                    'stat_label': result['stat_label']['value'] if 'stat_label' in result else None,
                    'unit_label': result['unit_label']['value'] if 'unit_label' in result else None,
                    'data_type': result['data_type_val']['value'] if 'data_type_val' in result else None,
                    "categories_labels": result['all_cat_labels']['value'] if 'all_cat_labels' in result else None,
                    'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result else None
                })
    data_dict = pd.DataFrame.from_dict(data_dict)
    # print(f"head of data dict: {data_dict.head()}")
    return data_dict




def _attach_statistics(
    df: pd.DataFrame, source_vars: List[str], target_vars: List[str], src_study: str, tgt_study: str
) -> pd.DataFrame:
    src_stats = fetch_variables_statistic_type(source_vars, src_study)
    tgt_stats = fetch_variables_statistic_type(target_vars, tgt_study)
    # src_eda = fetch_variables_eda(source_vars, src_study)
    # print(src_eda.head(3))

    if not src_stats.empty and "identifier" in src_stats.columns:
        df = df.merge(
            src_stats.rename(
                columns={
                    "identifier": "source",
                    "stat_label": "source_type",
                    "unit_label": "source_unit",
                    "data_type": "source_data_type",
                    "categories_labels": "source_categories_labels",
                    "original_categories": "source_original_categories"
                }
            ),
            on="source",
            how="left",
        )
    else:
        print(f"Warning: No source statistics found for {src_study}")

    df = df.merge(
        tgt_stats.rename(
            columns={
                "identifier": "target",
                "stat_label": "target_type",
                "unit_label": "target_unit",
                "data_type": "target_data_type",
                "categories_labels": "target_categories_labels",
                "original_categories": "target_original_categories"
            }
        ),
        on="target",
        how="left",
    )
    return df

# def _cross_category_matches(
#     source_elements: List[Dict[str, Any]],
#     target_elements: List[Dict[str, Any]],
#     target_study: str,
#     vector_db: Any,
#     embedding_model: Any,
#     collection_name: str,
# ) -> Iterable[Dict[str, Any]]:
#     """
#     Generate cross-category pairings that share the same ``omop_id`` and visit
#     (code match), then additionally semantic label matches across categories
#     using the embedding search.

#     Much cheaper than the original version:
#     - Exact matches: uses indexed lookups.
#     - Embedding stage: at most one search_in_db per (source_label, source_category).
#     """
#     # Categories where cross-category makes sense
#     CROSS_CATS = {
#         "measurement",
#         "observation",
#         "condition_occurrence",
#         "condition_era",
#         "observation_period",
#     }

#     # -----------------------------
#     # 1. Exact cross-category code matches (same omop_id & visit)
#     # -----------------------------
#     src_index: Dict[tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
#     tgt_index: Dict[tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)

#     # Index source by (omop_id, normalized_visit)
#     for s in source_elements:
#         visit_norm = s["visit"]
#         key = (s["omop_id"], visit_norm)
#         src_index[key].append(s)

#     # Index target by (omop_id, normalized_visit)
#     for t in target_elements:
#         visit_norm = t["visit"]
#         key = (t["omop_id"], visit_norm)
#         tgt_index[key].append(t)

#     # Yield code matches where both categories are in CROSS_CATS
#     for key, src_list in src_index.items():
#         if key not in tgt_index:
#             continue
#         tgt_list = tgt_index[key]
#         _, visit_norm = key

#         for s in src_list:
#             s_cat = s["category"].strip().lower()
#             if s_cat not in CROSS_CATS:
#                 continue

#             for t in tgt_list:
#                 t_cat = t["category"].strip().lower()
#                 if t_cat not in CROSS_CATS:
#                     continue

#                 # Same normalized visit already enforced by key, but recheck if you want:
#                 svisit = check_visit_string(s["visit"], t["visit"])
#                 tvisit = check_visit_string(t["visit"], s["visit"])
#                 if svisit != tvisit:
#                     continue

#                 yield {
#                     "source": s["source"],
#                     "target": t["target"],
#                     "somop_id": s["omop_id"],
#                     "tomop_id": t["omop_id"],
#                     "scode": s["code"],
#                     "slabel": s["code_label"],
#                     "tcode": t["code"],
#                     "tlabel": t["code_label"],
#                     "category": f"{s['category']}|{t['category']}",
#                     "mapping type": "code match",
#                     "source_visit": s["visit"],
#                     "target_visit": t["visit"],
#                 }

#     # -----------------------------
#     # 2. Semantic label matches across categories using embeddings
#     #    (no nested source×target search_in_db)
#     # -----------------------------

#     # Index targets by omop_id for quick lookup
#     targets_by_omop: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#     for t in target_elements:
#         targets_by_omop[t["omop_id"]].append(t)

#     # Cache embedding results per (source_label, source_category)
#     embed_cache: Dict[tuple[str, str], set[int]] = {}

#     for s in source_elements:
#         s_label = s["code_label"]
#         s_cat = s["category"].strip().lower()

#         # Only bother for relevant categories
#         if s_cat not in CROSS_CATS:
#             continue

#         cache_key = (s_label, s_cat)
#         if cache_key in embed_cache:
#             matched_omops = embed_cache[cache_key]
#         else:
#             # You had 0.65 as default score; keep that
#             score = 0.65
#            # score = 0.65 if s_category in {"drug_exposure", "drug_era"} else 0.85

#             # We allow matches to any of the CROSS_CATS in the target
#             matched_omops = set(
#                 search_in_db(
#                     vectordb=vector_db,
#                     embedding_model=embedding_model,
#                     query_text=s_label,
#                     target_study=target_study,
#                     limit=100,
#                     omop_domain=list(CROSS_CATS),
#                     min_score=score,
#                     collection_name=collection_name,
#                 )
#             )
#             embed_cache[cache_key] = matched_omops

#         if not matched_omops:
#             continue

#         for omop_id in matched_omops:
#             for t in targets_by_omop.get(omop_id, []):
#                 t_cat = t["category"].strip().lower()
#                 if t_cat not in CROSS_CATS:
#                     continue

#                 # Visit constraint
#                 svisit = check_visit_string(s["visit"], t["visit"])
#                 tvisit = check_visit_string(t["visit"], s["visit"])
#                 if svisit != tvisit:
#                     continue

#                 yield {
#                     "source": s["source"],
#                     "target": t["target"],
#                     "somop_id": s["omop_id"],
#                     "tomop_id": t["omop_id"],
#                     "scode": s["code"],
#                     "slabel": s["code_label"],
#                     "tcode": t["code"],
#                     "tlabel": t["code_label"],
#                     "category": f"{s['category']}|{t['category']}",
#                     "mapping type": "semantic label match",
#                     "source_visit": s["visit"],
#                     "target_visit": t["visit"],
#                 }


def _cross_category_matches(
    source_elements: List[Dict[str, Any]],
    target_elements: List[Dict[str, Any]],
    target_study: str,
    vector_db: Any,
    embedding_model: Any,
    collection_name: str,
) -> Iterable[Dict[str, Any]]:
    """Generate cross‑category pairings that share the same ``omop_id`` **and** visit.

    Parameters
    ----------
    source_elements / target_elements
        Flat lists produced in the first pass of ``build_mappings``.

    Returns
    -------
    List[Dict[str, Any]]
        New mapping dictionaries labelled ``cross‑category exact match``.
    """
    # final: List[Dict[str, Any]] = []
    CROSS_CATS = {
        "measurement",
        "observation",
        "condition_occurrence",
        "condition_era",
        "observation_period",
    }
    src_index: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for s in source_elements:
        s['visit_'] = check_visit_string(s['visit'], s['visit'])
        src_index[(s["omop_id"], s['visit'])].append(s)

    for t in target_elements:
        t['visit_'] = check_visit_string(t['visit'], s['visit'])    
        key = (t["omop_id"],  t['visit'])
        for s in src_index.get(key, []):
            print(s)
            if s['category'].strip().lower() in ["measurement", "observation", "condition_occurrence", "condition_era", "observation_period"] and t['category'].strip().lower() in ["measurement", "observation", "condition_occurrence", "condition_era","observation_period"]:
                # tvisit= check_visit_string(t['visit'], visit_constraint)
                # svisit = check_visit_string(s['visit'], visit_constraint)
                tvisit = check_visit_string(t['visit_'], s['visit_'])
                svisit = check_visit_string(s['visit_'], t['visit_'])
                # print(f"source visit: {svisit} and target visit: {tvisit}")
                # mapping_type = "code match"
                if svisit == tvisit:
                    yield {
                        "source": s["source"],
                        "target": t["target"],
                        "somop_id": s["omop_id"],
                        "tomop_id": t["omop_id"],
                        "scode": s["code"],
                        "slabel": s["code_label"],
                        "tcode": t["code"],
                        "tlabel": t["code_label"],
                        "category": f"{s['category']}|{t['category']}",
                        "mapping type": "code match",
                        "source_visit": s['visit'],
                        "target_visit":  t['visit'],
                    }
                    
    # ALSO CHECK SEMANTIC SIMILARITY ACROSS CATEGORIES IF NO EXACT MATCHES FOUND USING EMBEDDING SEARCH
    targets_by_omop: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for t in target_elements:
        targets_by_omop[t["omop_id"]].append(t)

    # Cache embedding results per (source_label, source_category)
    embed_cache: Dict[tuple[str, str], set[int]] = {}

    for s in source_elements:
        s_label = s["code_label"]
        s_cat = s["category"].strip().lower()

        # Only bother for relevant categories
        if s_cat not in CROSS_CATS:
            continue

        cache_key = (s_label, s_cat)
        if cache_key in embed_cache:
            matched_omops = embed_cache[cache_key]
        else:
            # You had 0.65 as default score; keep that
            score = 0.65
           # score = 0.65 if s_category in {"drug_exposure", "drug_era"} else 0.85

            # We allow matches to any of the CROSS_CATS in the target
            matched_omops = search_in_db(
                    vectordb=vector_db,
                    embedding_model=embedding_model,
                    query_text=s_label,
                    target_study=target_study,
                    limit=100,
                    omop_domain=list(CROSS_CATS),
                    min_score=score,
                    collection_name=collection_name,
                )
            
            embed_cache[cache_key] = matched_omops

        if not matched_omops:
            continue

        for omop_id in matched_omops:
            for t in targets_by_omop.get(omop_id, []):
                t_cat = t["category"].strip().lower()
                if t_cat not in CROSS_CATS:
                    continue

                # Visit constraint
                svisit = check_visit_string(s["visit"], t["visit"])
                tvisit = check_visit_string(t["visit"], s["visit"])
                if svisit != tvisit:
                    continue

                yield {
                    "source": s["source"],
                    "target": t["target"],
                    "somop_id": s["omop_id"],
                    "tomop_id": t["omop_id"],
                    "scode": s["code"],
                    "slabel": s["code_label"],
                    "tcode": t["code"],
                    "tlabel": t["code_label"],
                    "category": f"{s['category']}|{t['category']}",
                    "mapping type": "semantic label match",
                    "source_visit": s["visit"],
                    "target_visit": t["visit"],
                }

def map_source_target(
    source_study_name: str,
    target_study_name: str,
    vector_db: Any,
    embedding_model: Any,
    graph_db_repo: str = "https://w3id.org/CMEO/graph",
    collection_name: str = "studies_metadata",
    graph: Any = None,
) -> pd.DataFrame:
    """
    Align variables between two studies using OMOP graph relations
    first, then fall back to embedding similarity.
    """
    # source_related_studies = find_related_studies(source_study_name)
    # target_related_studies = find_related_studies(target_study_name)
    default_columns = [
            "source", "target", "somop_id", "tomop_id", "scode", "slabel", "tcode", "tlabel", "category", "source_visit", "target_visit", "source_type", "source_unit", "source_data_type", "source_categories_codes", "source_original_categories", "target_type", "target_unit", "target_data_type", "target_categories_codes", "target_original_categories", "mapping type", "transformation_rule", "harmonization_status"
        ]
    query = _build_alignment_query(source_study_name, target_study_name, graph_db_repo)
    bindings = _execute_query(query)

    source_elems, target_elems, matches = _parse_bindings(bindings)
    print(f"Source elements: {len(source_elems)}, Target elements: {len(target_elems)}, Matches: {len(matches)}")
    if not target_elems and not matches:
        print(f"No matches found for {source_study_name} and {target_study_name}.")
        columns = [
            f"{source_study_name}_variable",
            f"{target_study_name}_variable",
            "somop_id",
            "tomop_id",
            "scode",
            "slabel",
            "tcode",
            "tlabel",
            "category",
            "mapping type",
            "source_visit",
            "target_visit",
        ]
        return pd.DataFrame(columns=columns)

    
    
    
    # Build up your matching dict
    single_source = {
        "source": source_elems,
        "target": target_elems,
        "mapped": matches
    }
    for derived in DERIVED_VARIABLES_LIST:
        derived_row = extend_with_derived_variables(
            single_source=single_source,
            standard_derived_variable=(derived["code"], derived["label"], derived["omop_id"]),
            parameters_omop_ids=derived["required_omops"],
            variable_name=derived["name"],
            category=derived["category"],
        )
        if derived_row:
            matches.append(derived_row)
            
    cross_category_matches = list(_cross_category_matches(
        source_elements=source_elems,
        target_elements=target_elems,
        target_study=target_study_name,
        embedding_model=embedding_model,
        vector_db=vector_db,
        collection_name=collection_name,
    ))
    if cross_category_matches and len(cross_category_matches) > 0:
        matches.extend(cross_category_matches)
    
    matches.extend(_graph_vector_matches(
        source_elems,
        target_elems,
        graph,
        vector_db,
        embedding_model,
        target_study_name,
        collection_name,
    ))

    print(f"Total matches found: {len(matches)}")
    
    df = pd.DataFrame(matches).drop_duplicates(subset=["source", "target"])
    if df.empty:
        print(f"No matches found for {source_study_name} and {target_study_name}.")
        df = pd.DataFrame(columns=default_columns)
        return df
    df = _attach_statistics(
        df,
        df["source"].dropna().unique().tolist(),
        df["target"].dropna().unique().tolist(),
        source_study_name,
        target_study_name,
    )

    
    # move "mapping type" to the end
    if "mapping type" in df.columns:
        mapping_type = df.pop("mapping type")
        df["mapping type"] = mapping_type
    
    df[["transformation_rule", "harmonization_status"]]   = df.apply(
        lambda row: apply_rules(
            domain=row.get("category", "") if "category" in row and pd.notna(row.get("category")) else "",
            src_info={
                "var_name": row.get("source", ""),
                "omop_id": row.get("somop_id", ""),
                "stats_type": row.get("source_type", ""),
                "unit": row.get("source_unit", ""),
                "data_type": row.get("source_data_type", ""),
                "categories_labels": row.get("source_categories_labels", ""),
                "original_categories": row.get("source_original_categories", "")
            },
            tgt_info={
                "var_name": row.get("target", ""),
                "omop_id": row.get("tomop_id", ""),
                "stats_type": row.get("target_type", ""),
                "unit": row.get("target_unit", ""),
                "data_type": row.get("target_data_type", ""),
                "categories_labels": row.get("target_categories_labels", ""),
                "original_categories": row.get("target_original_categories", "")
            },
        ),
        axis=1,
        result_type="expand",
    )
    
    # normalize json columns
    df["transformation_rule"] = df["transformation_rule"].apply(
    lambda v: json.dumps(v) if isinstance(v, dict) else v
)
 

    
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].apply(json.dumps)
        elif df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(str)

    return df.drop_duplicates(keep="first")
