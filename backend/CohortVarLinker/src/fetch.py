
from SPARQLWrapper import SPARQLWrapper, JSON
from .config import settings
import pandas as pd

import time
from collections import defaultdict
from typing import List, Dict, Any
# from .embed import ModelEmbedding

from .utils import apply_rules, export_hierarchy_to_excel


    
def build_element_list(role, elements, visits, omop_id, code_id, code_label, domain):
    return [
        {
            'omop_id': omop_id,
            'code': code_id.strip(),
            'code_label': code_label,
            role: el,
            'category': domain,
            'visit': vis,
        }
        for el, vis in zip(elements, visits)
    ]
    
def get_exact_matches(
                src_elements: list[str],
                tgt_elements: list[str],
                src_visits: list[str],
                tgt_visits: list[str],
                code_id: str,
                code_label: str,
                omop_id: int,
                domain: str,
            ) -> list[dict]:
    """
    Cartesian‑product the source and target elements,
    keep only those whose visits match (after normalization),
    and return a list of exact‑match mapping records.
    """
    matches = []
    for i, s in enumerate(src_elements):
        for j, t in enumerate(tgt_elements):
          
            sv= src_visits[i] if "date" not in src_visits[i] else "baseline time"
            tv= tgt_visits[j] if "date" not in tgt_visits[j] else "baseline time"
            if sv == tv:   
                matches.append({
                    'source':  s,
                    'target':  t,
                    'somop_id': omop_id,
                    'tomop_id': omop_id,
                    'scode':   code_id.strip(),
                    'slabel':  code_label,
                    'tcode':   code_id.strip(),
                    'tlabel':  code_label,
                    'category': domain,
                    'mapping type': 'exact match',
                    'source_visit': src_visits[i],
                    'target_visit': tgt_visits[j],
                })
    return matches

def map_source_target(source_study_name:str , target_study_name:str, vector_db, embedding_model,graph_db_repo="https://w3id.org/CMEO/graph", collection_name="studies_metadata", graph:Any = None):
 
    from .vector_db import search_in_db 
    """
    The function is designed to identify data elements 
    that do not share the same OMOP ID in two different studies 
    and then check if they are still related through an 
    external relationship lookup (bulk_search_relationships).

    
    """
    # from .omop_graph import OmopGraphNX
    # graph = OmopGraphNX(csv_file_path=settings.concepts_file_path)
    start_time = time.time()
    # common_codes_df = find_common_codes(source_study_name, target_study_name)  # via common code, its first step

    query = f"""
        
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
                        ("{source_study_name}" AS ?source)
                        WHERE {{
                        GRAPH <{graph_db_repo}/{source_study_name}> 
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
                    ("{target_study_name}" AS ?target)
                        WHERE 
                        {{
                                GRAPH <{graph_db_repo}/{target_study_name}> 
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
    # print(query)
    
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
  
    final_dict = []
    # non_matching_dict = []
    source_elements = []
    target_elements = []
    # unmatched_dict = []
    for result in results["results"]["bindings"]:
        omop_id = int(result['omop_id']['value'])
        code_label = result['code_label']['value']
        source = result['source']['value']  
        target = result['target']['value']
        code_id = result['code_value']['value']
        source_data_elements = source.split(', ') if (source and source != '') else []
        target_data_elements = target.split(', ') if (target and target != '') else []
        
        source_data_elements_visit = (
                    [r.split("||")[-1].strip() for r in result['source_visit']['value'].split(', ')]
                    if (result['source_visit']['value'] and result['source_visit']['value'] != '')
                    else ["baseline time"] * len(source_data_elements)
                )
        target_data_elements_visit = (
            [r.split("||")[-1].strip() for r in result['target_visit']['value'].split(', ')]
            if (result['target_visit']['value'] and result['target_visit']['value'] != '')
            else ["baseline time"] * len(target_data_elements)
        )
        
        omop_domain = str(result['val']['value']).strip().lower()
        # print(f"{omop_id} == source visit: {source_data_elements_visit}")
        # print(f"{omop_id} == target visit: {target_data_elements_visit}")
        assert len(target_data_elements) == len(target_data_elements_visit), (
                f"Visit column Length mismatch with variable labels: {len(target_data_elements)} != {len(target_data_elements_visit)}"
            )

        if len(source_data_elements) > 0 and len(target_data_elements) > 0:
                final_dict += get_exact_matches(
                    src_elements=source_data_elements,
                    tgt_elements=target_data_elements,
                    src_visits=source_data_elements_visit,
                    tgt_visits=target_data_elements_visit,
                    code_id=code_id,
                    code_label=code_label,
                    omop_id=omop_id,
                    domain=omop_domain,
                )   
                source_elements += build_element_list(
                    role='source', elements=source_data_elements, visits=source_data_elements_visit, omop_id=omop_id, code_id=code_id, code_label=code_label, domain=omop_domain
                )
                target_elements += build_element_list(
                    role='target', elements=target_data_elements, visits=target_data_elements_visit, omop_id=omop_id, code_id=code_id, code_label=code_label, domain=omop_domain
                )
    
        else:

                if len(source_data_elements) > 0:
                    
                    source_elements += build_element_list(
                        role='source', elements=source_data_elements, visits=source_data_elements_visit, omop_id=omop_id, code_id=code_id, code_label=code_label, domain=omop_domain
                    )
        

                if len(target_data_elements) > 0:
                    target_elements += build_element_list(
                        role='target', elements=target_data_elements, visits=target_data_elements_visit, omop_id=omop_id, code_id=code_id, code_label=code_label, domain=omop_domain
                    )
           
    
    if len(target_elements) == 0 and len(final_dict) == 0:
        # return empty dataframe if no matches found
        print(f"No matches found for {source_study_name} and {target_study_name}.")
        empty_df = pd.DataFrame(columns=[f"{source_study_name}_variable", f"{target_study_name}_variable", "somop_id", "tomop_id", "scode", "slabel", "tcode", "tlabel", "category", "mapping type", "source_visit", "target_visit"])
        return empty_df

    single_source = {"source":source_elements, "target":target_elements, "mapped": final_dict}
    bmi_row  =  extend_with_dervied_variables(single_source, standard_derived_variable=("loinc:39156-5", "Body mass index (bmi) [ratio]", 3038553), parameters_omop_ids=[3036277, 3025315], variable_name="bmi")
    egfr_row = extend_with_dervied_variables(single_source, standard_derived_variable=("snomed:1556501000000100", "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula", 37169169), parameters_omop_ids=[3016723,3022304,46235213], variable_name="egfr")
    print(f"bmi row: {bmi_row}")
    print(f"egfr_row row: {egfr_row}")
    final_dict.append(bmi_row)
    final_dict.append(egfr_row)
    final_dict_new =  _cross_domain_matches(source_elements, target_elements)
    # print(f"final_dict_new: {len(final_dict_new)}")
    final_dict.extend(final_dict_new)

    unique_source_ids = {(s['omop_id'], s['category']) for s in source_elements}
    unique_target_ids = {(t['omop_id'], t['category'])  for t in target_elements}

   
    source_map = {(s['omop_id'], s['category']): [] for s in source_elements}
    target_map = {(t['omop_id'], t['category']): [] for t in target_elements}
    for s in source_elements:
        source_map[(s['omop_id'], s['category'])].append(s)
    for t in target_elements:
        target_map[(t['omop_id'], t['category'])].append(t)

    # Group all target OMOP IDs by category
    target_ids_by_domain = {}
    for tid, cat in unique_target_ids:
        target_ids_by_domain.setdefault(cat, set()).add(tid)
    for (source_id, omop_domain) in unique_source_ids:
        label_source_id = source_map[(source_id, omop_domain)][0]['code_label']
        target_ids_same_domain = target_ids_by_domain.get(omop_domain, set())
        target_ids_same_domain = {tid for tid in target_ids_same_domain if tid != source_id}

        # 🔁 Step 1: Graph-based reachability
        reachable_by_graph = None
        if omop_domain in ['drug_exposure', 'drug_era']:
            reachable_by_graph = graph.bfs_bidirectional_reachable(source_id, target_ids_same_domain, max_depth=3)
            # if source_id == 21601554: print(f"reachable_by_graph exist: {reachable_by_graph} and check omopid: {target_ids_same_domain}")
        elif omop_domain in ['condition_occurrence', 'condition_era', 'drug_era', 'procedure_occurrence', "device_exposure"]:
            # 2 or 3 depth not sure
            reachable_by_graph = graph.bfs_bidirectional_reachable(source_id, target_ids_same_domain, max_depth=2)
        if omop_domain in ['measurement']:
            reachable_by_graph = graph.only_upward_or_downward(source_id, target_ids_same_domain, max_depth=1)

        if reachable_by_graph:
            matched_targets = set(reachable_by_graph)
         
        else:
            # 🔁 Step 2: Fall back to vector similarity
            threshold = 0.7 if omop_domain in ['drug_exposure' or 'drug_era'] else 0.85
            #threshold = 0.8
            target_ids_by_similarity = search_in_db(
                vectordb=vector_db,
                embedding_model=embedding_model,
                query_text=label_source_id,
                target_study=target_study_name,
                limit=100,
                omop_domain=[omop_domain],
                min_score=threshold,
                collection_name=collection_name
            )
            matched_targets = set(target_ids_by_similarity)
            if  source_id == 21601554:
                print(f"21601554: {label_source_id} and target_ids_by_similarity: {matched_targets}")

        # 🔁 Step 3: Add to final_dict if target in target_map
        for target_id in matched_targets:
            key = (target_id, omop_domain)
            # if target_id == 3023103:
            #     print(f"3023103 key : {key}")
            if key not in target_map:
                continue
            for s in source_map[(source_id, omop_domain)]:
                for t in target_map[key]:

                    svisit = s['visit'] if "date" not in s['visit'] else "baseline time"
                    tvisit = t['visit'] if "date" not in t['visit'] else "baseline time"

                    if (s['category'] != t['category']) or svisit != tvisit:
                        
                        # non_matching_dict.append({
                        #     'source': s['source'],
                        #     'target': "",
                        #     'source_visit': svisit,
                        #     'target_visit': "",
                        #     'somop_id': source_id,
                        #     'tomop_id': "",
                        #     'scode': s['code'],
                        #     'slabel': s['code_label'],
                        #     'tcode': "",
                        #     'tlabel': "",
                        #     'category': s['category'],
                        #     'mapping type': 'Not applicable'
                        # })
                        # non_matching_dict.append({
                        #     'source': "",
                        #     'target': t['target'],
                        #     'source_visit': "",
                        #     'target_visit': tvisit,
                        #     'somop_id': "",
                        #     'tomop_id': target_id,
                        #     'scode': "",
                        #     'slabel': "",
                        #     'tcode': t['code'],
                        #     'tlabel': t['code_label'],
                        #     'category': t['category'],
                        #     'mapping type': 'Not applicable'
                        # })
                        continue
                        
                    final_dict.append({
                        'source': s['source'],
                        'target': t['target'],
                        'source_visit': s['visit'] if "date" not in s['visit'] else "baseline time",
                        'target_visit': t['visit'] if "date" not in t['visit'] else "baseline time",
                        'somop_id': source_id,
                        'tomop_id': target_id,
                        'scode': s['code'],
                        'slabel': s['code_label'],
                        'tcode': t['code'],
                        'tlabel': t['code_label'],
                        'category': omop_domain,
                        'mapping type': 'text match' if not reachable_by_graph else 'semantic match'
                    })

    final_df_frame = pd.DataFrame(final_dict)
    print(f"time taken: {time.time() - start_time}")    



    #     check transformation
    df_mapping =  final_df_frame.copy()
    source_vars = df_mapping["source"].dropna().unique().tolist()
    target_vars = df_mapping["target"].dropna().unique().tolist()

    # 3. Fetch statistics for both studies
    source_stats_df = fetch_variables_statistics(source_vars, source_study_name)
    target_stats_df = fetch_variables_statistics(target_vars, target_study_name)

    # 4. Merge stats into mapping DataFrame
    if not source_stats_df.empty and 'identifier' in source_stats_df.columns:
        df_mapping = df_mapping.merge(
            source_stats_df.rename(columns={
                "identifier": "source",
                "stat_label": "source_type",
                "unit_label": "source_unit",
                "data_type": "source_data_type"
            }),
            on="source",
            how="left"
        )
    else:
        print(f"Warning: No source statistics found for {source_study_name}")

    df_mapping = df_mapping.merge(
        target_stats_df.rename(columns={
            "identifier": "target",
            "stat_label": "target_type",
            "unit_label": "target_unit",
            "data_type": "target_data_type"
        }),
        on="target",
        how="left"
    )

    # 5. Remove NaN values and duplicates
  
    
    # non_matching_df = pd.DataFrame(non_matching_dict). # maybe we want to append unmatched variables to the mapping at later stage
    # remove duplicates
    df_mapping.fillna("", inplace=True)
    df_mapping.drop_duplicates(subset=["source", "target"], inplace=True)
    # df_mapping = pd.concat([df_mapping, non_matching_df], ignore_index=True)
    df_mapping.dropna(subset=["source", "target"], how="all", inplace=True)

        
    # 6. Add the transformation rules to the mapping DataFrame
    df_mapping["transformation_rule"] = df_mapping.apply(
        lambda row: apply_rules(
            domain=row.get("category", "") if "category" in row and pd.notna(row.get("category")) else "",
            src_info = {
                "var_name": row.get("source", "") if "source" in row and pd.notna(row.get("source")) else "",
                "stats_type": row.get("source_type", "") if "source_type" in row and pd.notna(row.get("source_type")) else "",
                "unit": row.get("source_unit", "") if "source_unit" in row and pd.notna(row.get("source_unit")) else "",
                "data_type": row.get("source_data_type", "") if "source_data_type" in row and pd.notna(row.get("source_data_type")) else "" },
            tgt_info= {
                "var_name": row.get("target", "") if "target" in row and pd.notna(row.get("target")) else "",
                "stats_type": row.get("target_type", "") if "target_type" in row and pd.notna(row.get("target_type")) else "",
                "unit": row.get("target_unit", "") if "target_unit" in row and pd.notna(row.get("target_unit")) else "",
                "data_type": row.get("target_data_type", "") if "target_data_type" in row and pd.notna(row.get("target_data_type")) else "",
            },
        ),
        axis=1
   )  
    
    # df_mapping.rename(columns={
    #     "source": f"{source_study_name}_variable",
    #     "target": f"{target_study_name}_variable",
    # })
    df_mapping = df_mapping.drop_duplicates(keep='first')
    return  df_mapping



def _cross_domain_matches(
    source_elements: List[Dict[str, Any]],
    target_elements: List[Dict[str, Any]],
    visit_constraint: bool = True,
) -> List[Dict[str, Any]]:
    """Generate cross‑domain pairings that share the same ``omop_id`` **and** visit.

    Parameters
    ----------
    source_elements / target_elements
        Flat lists produced in the first pass of ``build_mappings``.

    Returns
    -------
    List[Dict[str, Any]]
        New mapping dictionaries labelled ``cross‑domain exact match``.
    """
    final: List[Dict[str, Any]] = []
    src_index: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for s in source_elements:
        src_index[(s["omop_id"], s["visit"])].append(s)

    for t in target_elements:
        key = (t["omop_id"], t["visit"])
        for s in src_index.get(key, []):
            if s['category'] in ["measurement", "observation", "condition_occurrence", "condition_era"] and t['category'] in ["measurement", "observation", "condition_occurrence", "condition_era"]:
                tvisit = t['visit'] if "date" not in t['visit'] else "baseline time"
                svisit = s['visit'] if "date" not in s['visit'] else "baseline time"
                mapping_type = "cross domain exact match" if s['category'] != t['category'] else "cross domain approximate match"
                if svisit == tvisit:
                    final.append({
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
                    })
    return final

def extend_with_dervied_variables(single_source: List[Dict], 
                                  standard_derived_variable: tuple, 
                                  parameters_omop_ids: List[str], variable_name:str) -> Dict:
    """
   
     To create derived variables, we need to check if the source and target studies have the necessary parameters
        to derive the variable. If they do, we can create a new row in the data that represents the derived variable.
        The derived variable should have the same OMOP ID in both studies and the same LOINC code.
    provide standard_derived_variable as a tuple with the first element being the  code and the second element being the standard label and third as omop id .
    parameters_code is a list of OMOP ID for codes in (SNOMED, LOINC etc.) that are required to derive the variable.
    """

    single_source = single_source.copy()  # so we don't mutate the original in-place
    
    

    def find_omop_id_rows(data: List[Dict], omop_code: str, code_key:str ="omop_id") -> List[Dict]:
        """
        Returns a list of rows in 'data' where either 'somop_id' or 'tomop_id' (depending on 'side') 
        contains the specified loinc_code (assumes both are stored in lowercase strings).
        """
        # code_key = "code" if side == "source" else "code"
        found = []
        for row in data:
            # print(f"row: {row}")
            code_value = row.get(code_key, "") 
            # if code_value == omop_id and (temporal_info in['baseline time', 'follow-up 1 month']):
            if int(code_value) == int(omop_code):
                found.append(row)

        return found

    def can_produce_variable(data: List[Dict], parameters_codes:List[str], side: str = "source") -> bool:
        """
        Checks if the given side (source or target) either already has a bmi variable
        (loinc:39156-5) OR can derive it from height (loinc:8302-2) + weight (loinc:29463-7).
        """
        code_key = "somop_id" if side == "source" else "tomop_id"

 
        has_parameter_un_mapped   = all([len(find_omop_id_rows(data[side], omop_code=code,code_key="omop_id")) > 0 for code in parameters_codes])

        has_parameters_mapped = all([len(find_omop_id_rows(data['mapped'], omop_code =code, code_key=code_key)) > 0 for code in parameters_codes])

        return has_parameter_un_mapped or has_parameters_mapped



    # 3) Identify if there is an actual bmi variable in each side
    source_derviedvar_rows = find_omop_id_rows(single_source["source"], omop_code=standard_derived_variable[2], code_key="omop_id")
    target_derviedvar_rows = find_omop_id_rows(single_source["target"], omop_code=standard_derived_variable[2], code_key="omop_id")

    # If the source already has bmi, reuse its variable name; otherwise "bmi (derived)"
    if len(source_derviedvar_rows) < 1 and len(target_derviedvar_rows) < 1:
        # print(f"source or target both does not have bmi: {source_derviedvar_rows} and {target_derviedvar_rows}")
        return {}
    
        # 1) Check if source side can produce bmi
    source_can_bmi = can_produce_variable(single_source,  parameters_codes=parameters_omop_ids,side="source")

    # 2) Check if target side can produce bmi
    target_can_bmi = can_produce_variable(single_source, parameters_codes=parameters_omop_ids, side="target")

    # If either side cannot produce bmi, do nothing
    if not (source_can_bmi and target_can_bmi):
        return {}
    
    # print(f"source_rows: {source_derviedvar_rows} and target_rows: {target_derviedvar_rows}")
    if source_derviedvar_rows:
        # print(f"source already has bmi: {source_bmi_rows}")
        source_dervar_name = source_derviedvar_rows[0]["source"]
    else:
        source_dervar_name = f"{variable_name}(derived)"

    # If the target already has bmi, reuse its variable name; otherwise "bmi (derived)"
    if target_derviedvar_rows:
        # print(f"target already has bmi: {target_bmi_rows}")
        target_dervar_name = target_derviedvar_rows[0]["target"]
    else:
        target_dervar_name = f"{variable_name} (derived)"

    # 4) Construct a new row for the "derived match"
    if "derived" in source_dervar_name.lower() or "derived" in target_dervar_name.lower():
        mapping_type = "derived match"
    else:
        mapping_type = "exact match"
    new_row = {
        "source": source_dervar_name,
        "target": target_dervar_name,
        "somop_id": standard_derived_variable[2],
        "tomop_id": standard_derived_variable[2],
        "scode": standard_derived_variable[0],
        "slabel": standard_derived_variable[1],
        "tcode": standard_derived_variable[0],
        "tlabel": standard_derived_variable[1],
        "mapping type": mapping_type,
        "source_visit": "baseline time",
        "target_visit": "baseline time",
    }

    return new_row


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
                obi:is_specified_input_of ?mu_standardization.
                
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
                    'data_type': result['data_type_val']['value'] if 'data_type_val' in result else None
                })
    data_dict = pd.DataFrame.from_dict(data_dict)
    # print(f"head of data dict: {data_dict.head()}")
    return data_dict




def find_hierarchy_of_variables(study_name:str) -> List[Dict]:
    # Step 1: Query the triplestore to get variable_name, omop_id, code_label
    query = f"""
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
    PREFIX dc:   <http://purl.org/dc/elements/1.1/>
    PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX bfo:  <http://purl.obolibrary.org/obo/bfo.owl/>

    SELECT ?var_nameA 
           ?omop_id 
           ?code_label 
    WHERE {{
      GRAPH <https://w3id.org/CMEO/graph/{study_name}> {{
        ?dataElementA rdf:type cmeo:data_element ;
                      dc:identifier ?var_nameA ;
                      obi:is_specified_input_of ?catProcessA, ?stdProcessA .

        ?catProcessA rdf:type <https://w3id.org/CMEO/categorization_process> ;
                     obi:has_specified_output ?cat_outputA .

        ?stdProcessA rdf:type <https://w3id.org/CMEO/data_standardization> ;
                     obi:has_specified_output ?codeB .
        
        ?codeB rdf:_1 ?primary_code_literal .
        ?primary_code_literal obi:denotes ?omop_id_uri ;
                              cmeo:has_value ?code_value ;
                              rdfs:label ?code_label .
        ?omop_id_uri rdf:type cmeo:omop_id ;
                     cmeo:has_value ?omop_id .
      }}
    }}
    """
    print(query)
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data_dict = []
    for result in results["results"]["bindings"]:
        data_dict.append({
            'variable_name': result['var_nameA']['value'],
            'omop_id': result['omop_id']['value'],
            'code_label': result['code_label']['value']
        })

    # Step 2: Build hierarchy among the OMOP IDs using OmopGraphNX
    from .omop_graph import OmopGraphNX
    omop_nx = OmopGraphNX(csv_file_path=settings.concepts_file_path)
    omop_ids = [int(item["omop_id"]) for item in data_dict]

    # Create the basic parent/child mapping
    hierarchy_map = {}
    for concept_id in omop_ids:
        hierarchy_map[concept_id] = {"parents":set(), "children": set()}

    for concept_id in omop_ids:
        omop_ids_ = omop_ids.copy()
        omop_ids_.remove(concept_id)
        up = omop_nx.bfs_upward_with_equivalences(concept_id, omop_ids_, max_depth=10)
        down = omop_nx.bfs_downward_with_equivalences(concept_id, omop_ids_, max_depth=10)
        sibling_data_list = omop_nx.find_sibling_targets(concept_id, omop_ids_, max_depth=1)
        # print(f"sibling_data_list: {sibling_data_list}")    
        # siblings is the list with dictionary where  each dict has ({"sibling":tid, "parent":parents})
        for sdata in sibling_data_list:
            
            sibling_id = sdata["sibling"]
            shared_parents = list(sdata["parents"])  # set or list

            # Make sure sibling is in the hierarchy_map
            # if sibling_id not in hierarchy_map:
            #     hierarchy_map[sibling_id] = {"parents": set(), "children": set()}

            # For each shared parent, ensure it’s in the hierarchy,
            # then attach both cid and sibling_id as children of that parent.
            if sibling_id in hierarchy_map and sibling_id != concept_id:
    
                for p in shared_parents:
                    if p != sibling_id and p != concept_id:
                        if p not in hierarchy_map: 
                            print(f"{concept_id} --- {sibling_id} has parent {p}")
                            hierarchy_map[p] = {"parents": set(), "children": set()}

                        hierarchy_map[p]["children"].add(concept_id)
                        hierarchy_map[p]["children"].add(sibling_id)

                    # Also link the parent in each child's "parents" set
                        hierarchy_map[concept_id]["parents"].add(p)
                        hierarchy_map[sibling_id]["parents"].add(p)
            # else:
            #     print(f"{sibling_id} doesnot exist")
        
        hierarchy_map[concept_id]["parents"].update(up)
        hierarchy_map[concept_id]["children"].update(down)
        
                
    # Step 3: Create a final "hierarchy" dict so we can easily traverse it
    #   (concept_id → {parents: [...], children: [...]})
    hierarchy = {}
    for concept_id, relationships in hierarchy_map.items():
        if concept_id not in hierarchy:
            hierarchy[concept_id] = {
                "parents": relationships["parents"],
                "children": set()
            }
        for parent in relationships["parents"]:
            if parent not in hierarchy:
                hierarchy[parent] = {"parents": set(), "children": set()}
            hierarchy[parent]["children"].add(concept_id)

    # Step 4: Build a label lookup so we can print concept IDs with a friendly name
    #         e.g. "12345 -> Hypertension (var: sbp)"
    label_map = {}
    for item in data_dict:
        cid = int(item['omop_id'])
        label_map[cid] = f"{item['code_label']} (var={item['variable_name']})"

    # Optional: print the hierarchy in a tree-like format
    print("\n--- Hierarchy (Tree View) ---")
    print(f"length of hierarchy: {len(hierarchy)}")
    _print_tree(hierarchy, label_map)
    export_hierarchy_to_excel(hierarchy=hierarchy, label_map=label_map, output_file=f"/Users/komalgilani/Desktop/cmh/data/output/{study_name}_var_hierarchy.xlsx")
    # Return data_dict or the hierarchy, depending on your needs
    return data_dict


def _print_tree(hierarchy: dict, label_map: dict):
    """
    Print the hierarchy of concept IDs in a readable, indented tree structure.
    Any node with no parents is considered a top-level root.
    """

    # Identify root nodes (i.e. those that have no parents in this subset)
    root_nodes = set()
    for cid in hierarchy:
        parents = hierarchy[cid]["parents"]
        if not parents:  # no parents → root
            root_nodes.add(cid)

    visited = set()
    def dfs_print(node, indent_level=0):
        # Avoid re-printing cycles
        if node in visited:
            return
        visited.add(node)

        indent = "  " * indent_level
        label = label_map.get(node, f"CID={node}")
        print(f"{indent}- {label}")

        for child in hierarchy[node]["children"]:
            dfs_print(child, indent_level + 1)

    # Print each top-level root in a DFS manner
    for root in root_nodes:
        dfs_print(root)
    print("--- End of Hierarchy ---")




# def fetch_common_ids():
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     query = f"""
#         PREFIX vo: <http://purl.obolibrary.org/obo/VO_>
#         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#         PREFIX dc: <http://purl.org/dc/elements/1.1/>
#         PREFIX omop: <http://omop.org/>
#         PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#         PREFIX cmeo: <https://w3id.org/CMEO/>
#         PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

#         SELECT ?omop_id_value  ?label_value
#                (GROUP_CONCAT(DISTINCT STR(?var_name); SEPARATOR=", ") AS ?variable_name) 
#                (GROUP_CONCAT(DISTINCT STR(?studyGraph); SEPARATOR=", ") AS ?studies) 
#                (COUNT(DISTINCT ?studyGraph) AS ?studies_count)
#         WHERE {{
#           GRAPH ?studyGraph {{
#             ?variable dc:identifier ?var_name .
#             ?variable cmeo:has_standard_label ?label_uri.
#             ?label_uri rdfs:label ?label_value.
            
#             ?variable cmeo:has_omop_id ?omop_id_uri .
#             ?omop_id_uri cmeo:has_value ?omop_id_value .
#           }}
#           FILTER (?studyGraph != <https://w3id.org/CMEO/graph/studies_metadata>)
#         }}
#         GROUP BY ?omop_id_value ?label_value
#         HAVING (COUNT(DISTINCT ?studyGraph) > 1)
#     """

#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()

#     # Initialize an empty list to store the processed data
#     data = []

#     # Iterate over the results
#     for result in results["results"]["bindings"]:
#         omop_id_value = result['omop_id_value']['value']
#         label_value = result['label_value']['value']
#         variable_name = result['variable_name']['value'].split(', ')
#         studies = result['studies']['value'].split(', ')
#         studies_count = int(result['studies_count']['value'])

#         data.append({
#             'omop_id': omop_id_value,
#             'standard_label': label_value,
#             'labels_list': variable_name,
#             'graphs': studies,
#             'graphCount': studies_count
#         })

#     # Create a DataFrame
#     df = pd.DataFrame(data)

#     # Explode the lists in 'variable_names' and 'studies' columns
#     exploded_df = df.explode('labels_list').explode('graphs')

#     # Group and aggregate using named aggregation
#     grouped_df = exploded_df.groupby(['omop_id', 'standard_label', 'graphs']).agg(
#         variable_names=('labels_list', lambda x: list(set(x))),
#         variable_count=('labels_list', 'nunique')
#     ).reset_index()

#     return grouped_df

# def fetch_common_per_id(omop_id):
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     query = f"""    
#         PREFIX vo: <http://purl.obolibrary.org/obo/VO_>
#         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#         PREFIX dc: <http://purl.org/dc/elements/1.1/>
#         PREFIX omop: <http://omop.org/>
#         PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#         PREFIX cmeo: <https://w3id.org/CMEO/>
#         PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

#         SELECT ?omopId  ?label_value
#             (GROUP_CONCAT(DISTINCT STR(?var_name); SEPARATOR=", ") AS ?variable_name) 
#             (GROUP_CONCAT(DISTINCT STR(?studyGraph); SEPARATOR=", ") AS ?studies) 
#             (COUNT(DISTINCT ?studyGraph) AS ?studies_count) (COUNT(DISTINCT ?var_name) AS ?variable_count)
#         WHERE {{
#         GRAPH ?studyGraph {{
#             ?variable dc:identifier ?var_name .
#             ?variable cmeo:has_standard_label ?label_uri.
#             ?label_uri rdfs:label ?label_value.
            
#             ?variable cmeo:has_omop_id ?omop_id_uri .
#             ?omop_id_uri cmeo:has_value ?omopId .
#         }}
#         FILTER (?studyGraph != <https://w3id.org/variable/graph/metadata>)
#         FILTER(?omopId = {omop_id})
#         }}
#         GROUP BY ?omopId ?label_value
#         HAVING (COUNT(DISTINCT ?studyGraph) > 1)

#     """
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
    

#     # Initialize an empty list to store the processed data
#     data = []
#         # Iterate over the results
#     for result in results["results"]["bindings"]:
#         omopId = result['omopId']['value']
#         standard_label = result['standard_label']['value']
#         labels = result['variable_name']['value'].split(', ')  # variable names
#         graphs = result['studies']['value'].split(', ')
#         graphCount = int(result['studies_count']['value'])
#         variableCount = int(result['variable_count']['value'])
        
#         data.append({
#             'omopId': omopId,
#             'standard_label': standard_label,
#             'labels_list': labels,
#             'graphs': graphs,
#             'graphCount': graphCount,
#             'variableCount': variableCount
#         })

#     # Create a DataFrame
#     df = pd.DataFrame(data)
#     print(df.head())
#     print(f"columns: {df.columns}")
#     # Explode the lists in 'labels' and 'graphs' columns
#     exploded_df = df.explode('labels_list').explode('graphs')

#     grouped_df = exploded_df.groupby(
#         ['omop_id', 'standard_label', 'graphs']
#     ).agg(
#         variable_names=('labels_list', lambda x: list(set(x))),
#         variable_count=('labels_list', 'nunique')
#     ).reset_index()

#     # print(grouped_df.head())
#     return grouped_df



# def check_variables_with_common_categories():
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     query  = f"""
#     PREFIX vo: <http://purl.obolibrary.org/obo/VO_>
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX dc: <http://purl.org/dc/elements/1.1/>
#     PREFIX omop: <http://omop.org/>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX cmeo: <https://w3id.org/CMEO/>
#     PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

#     SELECT ?omop_id_value ?label_value 
#         (GROUP_CONCAT(DISTINCT STR(?category_label); SEPARATOR=", ") AS ?categories)
#         (COUNT(DISTINCT ?category_omop_id_value) AS ?category_count)
#         ?classification
#         (GROUP_CONCAT(DISTINCT STR(?studyGraph); SEPARATOR=", ") AS ?studies)
#         (COUNT(DISTINCT ?studyGraph) AS ?studies_count)
#     WHERE {{
#     GRAPH ?studyGraph {{
#         # Fetch variable and associated OMOP ID
#         ?variable dc:identifier ?var_name .
#         ?variable cmeo:has_standard_label ?label_uri .
#         ?label_uri rdfs:label ?label_value .

#         ?variable cmeo:has_omop_id ?omop_id_uri .
#         ?omop_id_uri cmeo:has_value ?omop_id_value .
        
#         # Ensure variable is either Dichotomous or Polychotomous
#         ?variable rdf:type ?classification .
#         FILTER (?classification IN (cmeo:dichotomous, cmeo:polychotomous))

#         # Fetch permissible values for the categorical variable
#         ?variable cmeo:has_permissible_value ?category_uri .
#         ?category_uri rdf:type cmeo:permissible_value .
#     # Optional attributes for permissible values
#         OPTIONAL {{ ?category_uri cmeo:has_categorical_label ?category_label. }}
#         OPTIONAL {{ ?category_label cmeo:has_value ?category_omop_id_value.  }}
#     }}

#     # Exclude metadata graph
#     FILTER (?studyGraph != <http://localhost:7200/repositories/icare4cvd/rdf-graphs/studies_metadata>)
#     }}
#     GROUP BY ?omop_id_value ?label_value ?classification
#     HAVING (COUNT(DISTINCT ?studyGraph) > 1) # Common across studies
#     ORDER BY ?omop_id_value ?label_value

#     """

#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
    
#     # Initialize an empty list to store the processed data
#     data = []
#     # Iterate over the results
#     for result in results["results"]["bindings"]:
#         omop_id_value = result['omop_id_value']['value']
#         label_value = result['label_value']['value']
#         categories = result['categories']['value'].split(', ')
#         category_count = int(result['category_count']['value'])
#         classification = result['classification']['value']
#         studies = result['studies']['value'].split(', ')
#         studies_count = int(result['studies_count']['value'])

#         data.append({
#             'omop_id': omop_id_value,
#             'standard_label': label_value,
#             'categories': categories,
#             'category_count': category_count,
#             'classification': classification,
#             'graphs': studies,
#             'graphCount': studies_count
#         })

#     # Create a DataFrame
#     df = pd.DataFrame(data)
#     return df
    

# def fetch_studies_with_disease(disease):
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     query = f"""
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX dc: <http://purl.org/dc/elements/1.1/>
#     PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
#     PREFIX cmeo: <https://w3id.org/CMEO/>
#     PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
#     PREFIX bfo: <http://purl.obolibrary.org/obo/bfo.owl/>

#     SELECT DISTINCT ?study_name ?purpose ?outcome 
#     WHERE {{
#         GRAPH <http://localhost:7200/repositories/icare4cvd/rdf-graphs/studies_metadata> {{
#             ?study a obi:study_design_execution ;
#                 obi:realizes ?plan .
#             ?plan obi:concretizes ?study_design .
#             ?study_design bfo:has_part ?protocol .
#             ?study dc:identifier ?study_name.
            
#             # Primary purpose specification
#             ?protocol bfo:has_part ?purposeInst .
#             ?purposeInst a cmeo:study_primary_purpose_specification .
#             ?purposeInst cmeo:has_value ?purpose .
            
#             # Outcome specification pattern (optional)
#             OPTIONAL {{
#                 ?protocol bfo:has_part ?outcomeInst .
#                 ?outcomeInst a cmeo:inclusion_criteria .
#                 ?outcomeInst cmeo:has_value ?outcome .
#             }}
            
#             # Filter: Either the purpose or outcome must contain "diabetes"
#             FILTER(
#                 CONTAINS(lcase(str(?purpose)), "{disease}") ||
#                 (BOUND(?outcome) && CONTAINS(lcase(str(?outcome)), "{disease}"))
#             )
#         }}
#     }}
#     """
#     # print(query)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
    
#     # Initialize an empty list to store the processed data
#     data = []
#     # Iterate over the results
#     for result in results["results"]["bindings"]:
#         study_name = result['study_name']['value']
#         # purpose = result['purpose']['value']
#         # outcome = result.get('outcome', {}).get('value', None)
        
#         data.append(study_name)
#     return data
#     # return list of study names only


# def fetch_data_elements_from_studyX_with_permissible_value(study_name, permissible_value, graph_db_repo="http://localhost:7200/repositories/icare4cvd/rdf-graphs"):

#     sparql = SPARQLWrapper(settings.query_endpoint)
#     query = f"""
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX dc: <http://purl.org/dc/elements/1.1/>
#     PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
#     PREFIX cmeo: <https://w3id.org/CMEO/>
#     PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
#     PREFIX bfo: <http://purl.obolibrary.org/obo/bfo.owl/>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
#     SELECT ?var_name ?s_var_val
#         WHERE {{
#             GRAPH <{graph_db_repo}/{study_name}> {{
#                 ?data_element a cmeo:data_element ;
#                             dc:identifier ?var_name;
#                             rdfs:label ?var_label;
#                             iao:is_denoted_by ?s_var;
#                             obi:has_value_specification ?permissiblevalues_part.
#                     ?s_var cmeo:has_value ?s_var_val.
#                     # Retrieve permissible categorical values FIRST
            
#                     ?permissiblevalues_part a obi:categorical_value_specification;
#                             obi:is_specified_input_of ?cat_data_standardization.
#                     ?cat_data_standardization a <https://w3id.org/CMEO/data_standardization>.
#                     ?cat_data_standardization obi:has_specified_output ?cat_standardized_code.
#                     ?cat_standardized_code rdfs:label ?label_value.
                
                
#                     FILTER (CONTAINS(?label_value, "{permissible_value}"))
#                 }}
#         }}
#         """
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
    
#     # Initialize an empty list to store the processed data
#     data = []
#     # Iterate over the results
#     for result in results["results"]["bindings"]:
#         var_name = result['var_name']['value']
#         data.append(var_name)
#     return data
#     # return list of variable names




# def study_x_with_data_elements_y(inclusion_criteria:str , data_element_criteria:str):
#     studies=fetch_studies_with_disease(inclusion_criteria)
#     data_elements = {}
#     for study in studies:
#         data_elements[study] = fetch_data_elements_from_studyX_with_permissible_value(study, data_element_criteria)
#     return data_elements


# def find_common_codes( source_study_name:str , target_study_name:str,graph_db_repo="http://localhost:7200/repositories/icare4cvd/rdf-graphs"):
#     query = f"""
#         PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#             PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#             PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
#             PREFIX dc:   <http://purl.org/dc/elements/1.1/>
#             PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
#             PREFIX cmeo: <https://w3id.org/CMEO/>
#             PREFIX bfo:  <http://purl.obolibrary.org/obo/bfo.owl/>
#             PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>

#             SELECT
#             ?omop_id ?code_label ?code_value ?val
#             (GROUP_CONCAT(DISTINCT ?varNameA; SEPARATOR=", ") AS ?source)
#             (GROUP_CONCAT(DISTINCT ?varNameB; SEPARATOR=", ") AS ?target)
#             (GROUP_CONCAT(DISTINCT STR(?visitsA); SEPARATOR=", ") AS ?source_visit)
#             (GROUP_CONCAT(DISTINCT STR(?visitsB); SEPARATOR=", ") AS ?target_visit)
            
#             WHERE 
#             {{
#              {{
#                         SELECT
#                         ?omop_id ?code_label ?code_value ?val

#                         (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountA)
#                         (GROUP_CONCAT(DISTINCT STR(?var_nameA); SEPARATOR=", ") AS ?varNameA)
#                         (GROUP_CONCAT(CONCAT(STR(?var_nameA), "||", STR(?visitcodelabelA)); SEPARATOR=", ") AS ?visitsA)
#                         ("{source_study_name}" AS ?source)
#                         WHERE {{
#                         GRAPH <{graph_db_repo}/{source_study_name}> 
#                         {{
#                                     ?dataElementA rdf:type cmeo:data_element ;
#                                                     dc:identifier ?var_nameA ;
#                                                     obi:is_specified_input_of ?catProcessA, ?stdProcessA .
#                                      OPTIONAL {{
#                                         ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
#                                                     iao:is_about ?dataElementA ;
#                                                     obi:is_specified_input_of ?vs_stdProcessA .
                                        
                                        
#                                         ?vs_stdProcessA obi:has_specified_output ?visit_code.
#                                         ?visit_code rdfs:label ?visitcodelabelA.
#                                     }}
#                                     ?catProcessA rdf:type cmeo:categorization_process ;
#                                                 obi:has_specified_output ?cat_outputA .
#                                     ?cat_outputA cmeo:has_value ?val .
#                                     #FILTER(?val IN ("measurement", "drug_exposure"))

#                                     ?stdProcessA rdf:type cmeo:data_standardization ;
#                                                 obi:has_specified_output ?codeA .
#                                     ?codeA rdf:_1 ?primary_code_literal .
#                                     ?primary_code_literal iao:denotes ?omop_id_uri ;
#                                             cmeo:has_value ?code_value ;
#                                             rdfs:label ?code_label .
#                                     ?omop_id_uri rdf:type cmeo:omop_id ;
#                                                 cmeo:has_value ?omop_id .
#                             }}
#                         }}
#                         GROUP BY ?omop_id ?code_label ?code_value ?val
#                 }}
#             UNION
#             {{
#                     SELECT
#                     ?omop_id ?code_label  ?code_value ?val
#                     (COUNT(DISTINCT ?primary_code_literal) AS ?codeCountB)
#                     (GROUP_CONCAT(DISTINCT STR(?var_nameB); SEPARATOR=", ") AS ?varNameB)
#                      (GROUP_CONCAT(CONCAT(STR(?var_nameB), "||", STR(?visitcodelabelB)); SEPARATOR=", ") AS ?visitsB)
#                     ("{target_study_name}" AS ?target)
#                         WHERE 
#                         {{
#                                 GRAPH <{graph_db_repo}/{target_study_name}> 
#                                 {{
#                                     ?dataElementB rdf:type cmeo:data_element ;
#                                     dc:identifier ?var_nameB ;
#                                     obi:is_specified_input_of ?catProcessB, ?stdProcessB.
                                    
#                                     OPTIONAL {{
#                                     ?visitdatum  rdf:type cmeo:visit_measurement_datum ;
#                                                 iao:is_about ?dataElementB ;
#                                                 obi:is_specified_input_of ?vs_stdProcessAB .
                                    
                                    
#                                     ?vs_stdProcessAB obi:has_specified_output ?visit_code.
#                                     ?visit_code rdfs:label ?visitcodelabelB.
#                                     }}
#                                     ?catProcessB rdf:type cmeo:categorization_process ;
#                                     obi:has_specified_output ?cat_outputB .
#                                     ?cat_outputB cmeo:has_value ?val .
#                                     #FILTER(?val IN ("measurement", "drug_exposure"))

#                                     ?stdProcessB rdf:type cmeo:data_standardization ;
#                                             obi:has_specified_output ?codeB .
#                                     ?codeB rdf:_1 ?primary_code_literal .
#                                     ?primary_code_literal iao:denotes ?omop_id_uri ;
#                                     cmeo:has_value ?code_value;
#                                     rdfs:label ?code_label.
#                                     ?omop_id_uri rdf:type cmeo:omop_id ;
#                                     cmeo:has_value ?omop_id.
#                                 }}
#                         }}

#                     GROUP BY ?omop_id  ?code_label ?code_value  ?val
#                 }}
#             }}
#             GROUP BY ?omop_id ?code_label ?code_value ?val
#             #HAVING (COUNT(DISTINCT ?source) < 3)
#             ORDER BY ?omop_id
    
#     """
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     data = []
#     for result in results["results"]["bindings"]:
#         omop_id = result['omop_id']['value']
#         code_label = result['code_label']['value']
#         vocab_code = result['code_value']['value']
#         # study_count = int(result['study_count']['value'])
#         # study_name = result['study_name']['value']
#         source = result['source']['value']
#         target = result['target']['value']
#         # source_code_type = result['source_code_type']['value']
#         # target_code_type = result['target_code_type']['value']
#         source_data_elements = source.split(', ') if (source and source != '') else []
#         target_data_elements = target.split(', ') if (target and target != '') else []
#         for si in source_data_elements:
#             for di in target_data_elements:
#                 data.append({
#                     'source': si,
#                     'target': di,
#                     'somop_id': omop_id,
#                     'tomop_id': omop_id,
#                     'scode': vocab_code,
#                     'slabel': code_label,
#                     'tcode': vocab_code,
#                     'tlabel': code_label,
#                     'path_depth': 'mapped'
#                 })
        
#     df = pd.DataFrame(data)
#     return df



def map_category_to_omop(category: str) -> str:
    """
    Maps a given category to its corresponding OMOP domain.
    :param category: The category to map.
    :return: The corresponding OMOP domain.
    """
    category_mapping = {
        "Etiology": "Observation",
        "Measurement": "Measurement",
        "Stages and scales": "Measurement",
        "Functional and Behavioral Assessment Score": "Measurement",
        "Blood measurement": "Measurement",
        "Cardiac Measurement": "Measurement",
        "Anthropometric Measurement": "Measurement",
        "Consumption Measurement": "Measurement",
        "Time to Event Measurement": "Measurement",
        "Clinical Finding": "Condition-Occurrence",
        "Disease or Disorder Finding": "Condition-Occurrence",
        "Functional Finding": "Observation",
        "Compliance Finding": "Observation",
        "Death": "Death",
        "Demographics": "Person",
        "Risk Factors": "Observation",
        "Sign or symptom": "Condition-Occurrence",
        "General sign": "Condition-Occurrence",
        "Cardiac Sign": "Condition-Occurrence",
        "Respiratory Sign": "Condition-Occurrence",
        "Neurological Sign": "Condition-Occurrence",
        "Vital Sign": "Measurement",
        "Symptom": "Condition-Occurrence",
        "Device Exposure": "Device-Exposure",
        "Medication Exposure": "Drug-Exposure",
        "Procedure": "Procedure",
        "Follow up Attrition": "Observation",
        "Medical History": "Observation",
        "Family History": "Observation",
        "Disease or Disorder History": "Observation",
        "Hospitalization History": "Observation",
        "Medication History": "Observation",
        "Procedure History": "Observation",
        "Dietary Intake": "Measurement",
        "Number of Occurrences": "Observation"
    }
    
    return category_mapping.get(category, "Unknown Category")
