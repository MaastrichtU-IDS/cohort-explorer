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


DERIVED_VARIABLES_LIST= [
    
     {
                    "name": "BMI-derived",
                    "omop_id": 3038553,           
                    "code": "loinc:39156-5",
                    "label": "Body mass index (BMI) [Ratio]",
                    "unit": "kg/m2",
                    "required_omops": [3016723, 3025315],
                    "category": "measurement"
                },
                {
                    "name": "eGFR_CG-derived",
                    "omop_id": 37169169,          
                    "code": "snomed:1556501000000100",
                    "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
                    "unit": "ml/min",
                    "required_omops": [3016723, 3022304, 46235213],
                    "category": "measurement"
                }
]
def _build_alignment_query(
    source: str, target: str, graph_repo: str
) -> str:
    """Return the SPARQL query used to retrieve variables of both studies."""
    return f"""
        
            PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
            PREFIX dc:   <http://purl.org/dc/elements/1.1/>
            PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
            PREFIX cmeo: <https://w3id.org/CMEO/>
            PREFIX bfo:  <http://purl.obolibrary.org/obo/bfo.owl/>
            PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>

            SELECT
            ?omop_id ?code_label ?code_value ?val
            (GROUP_CONCAT(DISTINCT ?varNameA; SEPARATOR=", ") AS ?source)
            (GROUP_CONCAT(DISTINCT ?varNameB; SEPARATOR=", ") AS ?target)
            (GROUP_CONCAT(DISTINCT STR(?visitsA); SEPARATOR=", ") AS ?source_visit)
            (GROUP_CONCAT(DISTINCT STR(?visitsB); SEPARATOR=", ") AS ?target_visit)
            
            WHERE 
            {{
             {{
                        SELECT
                        ?omop_id ?code_label ?code_value ?val

                        (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountA)
                        (GROUP_CONCAT(DISTINCT STR(?var_nameA); SEPARATOR=", ") AS ?varNameA)
                        (GROUP_CONCAT(CONCAT(STR(?var_nameA), "||", STR(?visitcodelabelA)); SEPARATOR=", ") AS ?visitsA)
                        ("{source}" AS ?source)
                        WHERE {{
                        GRAPH <{graph_repo}/{source}> 
                        {{
                                    ?dataElementA rdf:type cmeo:data_element ;
                                                    dc:identifier ?var_nameA ;
                                                    obi:is_specified_input_of ?catProcessA, ?stdProcessA .
                                     OPTIONAL {{
                                    ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
                                                iao:is_about ?dataElementA ;
                                                obi:is_specified_input_of ?vs_stdProcessA .
                                    
                                    
                                    ?vs_stdProcessA obi:has_specified_output ?visit_code.
                                    ?visit_code rdfs:label ?visitcodelabelA.
                                    }}
                                    ?catProcessA rdf:type cmeo:categorization_process ;
                                                obi:has_specified_output ?cat_outputA .
                                    ?cat_outputA cmeo:has_value ?val .
                                    #FILTER(?val IN ("measurement", "drug_exposure"))

                                    ?stdProcessA rdf:type cmeo:data_standardization ;
                                                obi:has_specified_output ?codeA .
                                    ?codeA rdf:_1 ?primary_code_literal .
                                    ?primary_code_literal iao:denotes ?omop_id_uri ;
                                            cmeo:has_value ?code_value ;
                                            rdfs:label ?code_label .
                                    ?omop_id_uri rdf:type cmeo:omop_id ;
                                                cmeo:has_value ?omop_id .
                            }}
                        }}
                        GROUP BY ?omop_id ?code_label ?code_value ?val
                }}
            UNION
            {{
                    SELECT
                    ?omop_id ?code_label  ?code_value ?val
                    (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountB)
                    (GROUP_CONCAT(DISTINCT STR(?var_nameB); SEPARATOR=", ") AS ?varNameB)
                     (GROUP_CONCAT(CONCAT(STR(?var_nameB), "||", STR(?visitcodelabelB)); SEPARATOR=", ") AS ?visitsB)
                    ("{target}" AS ?target)
                        WHERE 
                        {{
                                GRAPH <{graph_repo}/{target}> 
                                {{
                                    ?dataElementB rdf:type cmeo:data_element ;
                                    dc:identifier ?var_nameB ;
                                    obi:is_specified_input_of ?catProcessB, ?stdProcessB.
                                    
                                    OPTIONAL {{
                                    ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
                                                iao:is_about ?dataElementB ;
                                                obi:is_specified_input_of ?vs_stdProcessAB .
                                    
                                    ?vs_stdProcessAB obi:has_specified_output ?visit_code.
                                    ?visit_code rdfs:label ?visitcodelabelB.
                                    
                                    }}
                                    ?catProcessB rdf:type cmeo:categorization_process ;
                                    obi:has_specified_output ?cat_outputB .
                                    ?cat_outputB cmeo:has_value ?val .
                                    #FILTER(?val IN ("measurement", "drug_exposure"))

                                    ?stdProcessB rdf:type cmeo:data_standardization ;
                                            obi:has_specified_output ?codeB .
                                    ?codeB rdf:_1 ?primary_code_literal .
                                    ?primary_code_literal iao:denotes ?omop_id_uri ;
                                    cmeo:has_value ?code_value;
                                    rdfs:label ?code_label.
                                    ?omop_id_uri rdf:type cmeo:omop_id ;
                                    cmeo:has_value ?omop_id.
                                }}
                        }}

                    GROUP BY ?omop_id  ?code_label ?code_value  ?val
                }}
            }}
            GROUP BY ?omop_id ?code_label ?code_value ?val
            #HAVING (COUNT(DISTINCT ?source) < 3)
            ORDER BY ?omop_id
    """


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

        assert len(src_vars) == len(src_visits), (
            f"Visit column Length mismatch with variable labels: {len(src_vars)} != {len(src_visits)}"
        )
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
            sv= sv if "date" not in sv else "baseline time"
            tv= tv if "date" not in tv else "baseline time"
            if sv == tv:
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
                        "mapping type": "exact match",
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

    mapping_type = "derived match" if ("derived" in source_varname.lower() or "derived" in target_varname.lower()) else "exact match"
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
            reachable = graph.bfs_bidirectional_reachable(sid, tgt_ids, max_depth=3)
        elif category in {"condition_occurrence", "condition_era", "procedure_occurrence",
                        "device_exposure", "drug_era"}:
            reachable = graph.bfs_bidirectional_reachable(sid, tgt_ids, max_depth=2)
        elif category == "measurement":
            reachable = graph.only_upward_or_downward(sid, tgt_ids, max_depth=1)

        if reachable:
            matched = set(reachable)
        else:
            score = 0.7 if category in {"drug_exposure", "drug_era"} else 0.85
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
                    if se["category"] != te["category"] or se["visit"] != te["visit"]:
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
                            "mapping type": "semantic match" if reachable else "text match",
                        }
                    )
    return final


def fetch_variables_statistics(var_names_list:list[str], study_name:str) -> pd.DataFrame:

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
            (GROUP_CONCAT(DISTINCT ?cate_code_value; separator="; ") AS ?all_cat_values)
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
                ?unit obi:is_specified_input_of ?mu_standardization .
                
                ?mu_standardization obi:has_specified_output ?unit_code_node .
                ?unit_code_node rdfs:label ?unit_label .

        }}
          # Optional: permissible values
            OPTIONAL {{

            ?dataElement obi:has_value_specification ?cat_val .
            ?cat_val a obi:categorical_value_specification;
                obi:is_specified_input_of ?mu_standardization;
                cmeo:has_value ?original_cat_val_value .
                
            ?mu_standardization obi:has_specified_output ?cat_codes .
            ?cat_codes rdfs:label  ?cate_code_value
            }}
        }}
        }}
        GROUP BY ?identifier ?stat_label ?unit_label ?data_type_val
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
                    "categories": result['all_cat_values']['value'] if 'all_cat_values' in result else None,
                    'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result else None
                })
    data_dict = pd.DataFrame.from_dict(data_dict)
    # print(f"head of data dict: {data_dict.head()}")
    return data_dict



def _attach_statistics(
    df: pd.DataFrame, source_vars: List[str], target_vars: List[str], src_study: str, tgt_study: str
) -> pd.DataFrame:
    src_stats = fetch_variables_statistics(source_vars, src_study)
    tgt_stats = fetch_variables_statistics(target_vars, tgt_study)

    if not src_stats.empty and "identifier" in src_stats.columns:
        df = df.merge(
            src_stats.rename(
                columns={
                    "identifier": "source",
                    "stat_label": "source_type",
                    "unit_label": "source_unit",
                    "data_type": "source_data_type",
                    "categories": "source_categories",
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
                "categories": "target_categories",
                "original_categories": "target_original_categories"
            }
        ),
        on="target",
        how="left",
    )
    return df



def _cross_category_matches(
    source_elements: List[Dict[str, Any]],
    target_elements: List[Dict[str, Any]]
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
    src_index: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for s in source_elements:
        src_index[(s["omop_id"], s["visit"])].append(s)

    for t in target_elements:
        key = (t["omop_id"], t["visit"])
        for s in src_index.get(key, []):
            print(s)
            if s['category'] in ["measurement", "observation", "condition_occurrence", "condition_era"] and t['category'] in ["measurement", "observation", "condition_occurrence", "condition_era"]:
                # tvisit= check_visit_string(t['visit'], visit_constraint)
                # svisit = check_visit_string(s['visit'], visit_constraint)
                tvisit = t['visit'] if "date" not in t['visit'] else "baseline time"
                svisit = s['visit'] if "date" not in s['visit'] else "baseline time"
                # print(f"source visit: {svisit} and target visit: {tvisit}")
                mapping_type = "cross category exact match" if s['category'] != t['category'] else "cross category approximate match"
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
                        "mapping type": mapping_type,
                        "source_visit": s['visit'],
                        "target_visit":  t['visit'],
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
    query = _build_alignment_query(source_study_name, target_study_name, graph_db_repo)
    bindings = _execute_query(query)

    source_elems, target_elems, matches = _parse_bindings(bindings)

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
        source_elems,
        target_elems
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

    df = _attach_statistics(
        df,
        df["source"].dropna().unique().tolist(),
        df["target"].dropna().unique().tolist(),
        source_study_name,
        target_study_name,
    )
   
    df["transformation_rule"] = df.apply(
        lambda row: apply_rules(
            domain=row.get("category", "") if "category" in row and pd.notna(row.get("category")) else "",
            src_info={
                "var_name": row.get("source", ""),
                "stats_type": row.get("source_type", ""),
                "unit": row.get("source_unit", ""),
                "data_type": row.get("source_data_type", ""),
                "categories": row.get("source_categories", ""),
            },
            tgt_info={
                "var_name": row.get("target", ""),
                "stats_type": row.get("target_type", ""),
                "unit": row.get("target_unit", ""),
                "data_type": row.get("target_data_type", ""),
                "categories": row.get("target_categories", ""),
            },
        ),
        axis=1,
    )
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].apply(json.dumps)
        elif df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(str)

    return df.drop_duplicates(keep="first")
