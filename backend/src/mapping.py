from typing import Any

import requests
from fastapi import APIRouter, Depends, HTTPException, Query

from src.auth import get_current_user
from src.utils import curie_converter, run_query

router = APIRouter()

# NOTE: not really use now, Komal do the mappings in mapping_generation folder

from fastapi import Body
from fastapi.responses import StreamingResponse, JSONResponse
import io
import os
import sys
import json
import glob
from src.config import settings

# Import the CohortVarLinker function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CohortVarLinker')))
from CohortVarLinker.main import generate_mapping_csv

def get_latest_dictionary_timestamp(cohort_id: str) -> float | None:
    """Get the timestamp of the latest data dictionary file for a cohort"""
    try:
        cohort_folder = os.path.join(settings.cohort_folder, cohort_id)
        if not os.path.exists(cohort_folder):
            return None
            
        # Select most recent CSV file with 'datadictionary' in the name
        csv_candidates = [
            f for f in glob.glob(os.path.join(cohort_folder, "*.csv"))
            if ("datadictionary" in os.path.basename(f).lower()
            and "noheader" not in os.path.basename(f).lower())
        ]
        
        if csv_candidates:
            latest_file = max(csv_candidates, key=os.path.getmtime)
            return os.path.getmtime(latest_file)
        return None
    except Exception:
        return None

@router.post("/check-mapping-cache")
async def check_mapping_cache(
    source_study: str = Body(...),
    target_studies: list = Body(...)
):
    """
    Check cache status for mapping pairs without generating mappings.
    Returns cache information immediately with dictionary timestamps.
    """
    output_dir = "/app/CohortVarLinker/mapping_output"
    
    source_study = source_study.lower()
    target_studies_names = [t[0].lower() for t in target_studies]
    
    # Get dictionary timestamps for all involved cohorts
    all_cohorts = set([source_study] + target_studies_names)
    dictionary_timestamps = {}
    for cohort in all_cohorts:
        dict_timestamp = get_latest_dictionary_timestamp(cohort)
        if dict_timestamp:
            dictionary_timestamps[cohort] = dict_timestamp
    
    # Check cache status for each mapping pair
    cached_pairs = []
    uncached_pairs = []
    outdated_pairs = []
    
    for tstudy in target_studies_names:
        out_filename = f'{source_study}_{tstudy}_cross_mapping.csv'
        out_path = os.path.join(output_dir, out_filename)
        
        if os.path.exists(out_path):
            # Get file modification time
            cache_timestamp = os.path.getmtime(out_path)
            
            # Check if cache is outdated compared to dictionaries
            source_dict_time = dictionary_timestamps.get(source_study)
            target_dict_time = dictionary_timestamps.get(tstudy)
            
            is_outdated = False
            outdated_cohort = None
            
            if source_dict_time and source_dict_time > cache_timestamp:
                is_outdated = True
                outdated_cohort = source_study
            elif target_dict_time and target_dict_time > cache_timestamp:
                is_outdated = True
                outdated_cohort = tstudy
            
            pair_info = {
                'source': source_study,
                'target': tstudy,
                'timestamp': cache_timestamp
            }
            
            if is_outdated:
                pair_info['outdated_cohort'] = outdated_cohort
                outdated_pairs.append(pair_info)
            else:
                cached_pairs.append(pair_info)
        else:
            uncached_pairs.append({
                'source': source_study,
                'target': tstudy
            })
    
    return JSONResponse(content={
        'cached_pairs': cached_pairs,
        'uncached_pairs': uncached_pairs,
        'outdated_pairs': outdated_pairs,
        'dictionary_timestamps': dictionary_timestamps
    })

@router.post("/generate-mapping")
async def generate_mapping(
    source_study: str = Body(...),
    target_studies: list = Body(...)
):
    """
    Generate a mapping CSV for the given source and target studies and return it as a downloadable file.
    target_studies should be a list of [study_name, visit_constraint_bool]
    """
    # Call the backend function
    # The function writes CSVs to CohortVarLinker/mapping_output/{source}_{target}_full.csv
    # We'll return the first mapping file (for now, can be extended)
    
    target_studies = sorted(target_studies, key=lambda x: x[0])
    cache_info = generate_mapping_csv(source_study, target_studies)
    #output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CohortVarLinker', 'mapping_output')
    output_dir = "/app/CohortVarLinker/mapping_output"
    
     # Find the generated file(s)
    source_study = source_study.lower()
    target_str = "_".join([t[0].lower() for t in target_studies])
    
    filename = f"{source_study}_omop_id_grouped_{target_str}.json"
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        # Read file content
        with open(filepath, 'r') as f:
            file_content = f.read()
        
        # Return JSON response with both cache info and file content
        return JSONResponse(content={
            "cache_info": cache_info,
            "file_content": file_content,
            "filename": filename
        })
    return JSONResponse(status_code=404, content={"error": "Cache error. Mapping file not found."})

@router.get("/search-concepts")
async def search_concepts(
    query: str, domain: list[str] | None = Query(default=None), user: Any = Depends(get_current_user)
):
    """Search for concepts in the Athena API and check how many time those concepts are use in our KG."""
    if not domain:
        domain = []
    vocabs = ["LOINC", "ATC", "SNOMED"]  # "RxNorm"
    # TODO: Komal implement the search from her model
    try:
        response = requests.get(
            "https://athena.ohdsi.org/api/v1/concepts",
            params={
                "query": query,
                "domain": domain,
                "vocabulary": vocabs,
                "standardConcept": ["Standard", "Classification"],
                "pageSize": 15,
                "page": 1,
            },
            headers={
                # We need to fake the user agent to avoid a 403 error. What a bunch of douchebags, and incompetent with this! Try again losers!
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            },
            timeout=60,
        )
        response.raise_for_status()
        search_res = response.json().get("content", [])
    except Exception:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from OHDSI Athena API")

    # print(search_res)
    found_concepts = []
    for res in search_res:
        # Convert snomed CURIE to snomedct
        concept_id = f"{'snomedct' if res.get('vocabulary').lower() == 'snomed' else res.get('vocabulary').lower()}:{res.get('id')}"
        found_concepts.append(
            {
                "id": concept_id,
                "uri": curie_converter.expand(concept_id),
                "label": res.get("name"),
                "domain": res.get("domain"),
                "vocabulary": res.get("vocabulary"),
                "used_by": [],
            }
        )

    found_concepts_filter = " ".join([f"<{concept['uri']}>" for concept in found_concepts])
    sparql_query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX dcterms: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?cohortId ?varName ?varLabel ?omopDomain ?mappedId
    WHERE {{
        VALUES ?mappedId {{ {found_concepts_filter} }}
        GRAPH ?cohortMetadataGraph {{
            ?cohort a icare:Cohort ;
                dc:identifier ?cohortId .
        }}

        GRAPH ?cohortVarGraph {{
            ?cohort icare:hasVariable ?variable .
            ?variable a icare:Variable ;
                dc:identifier ?varName ;
                rdfs:label ?varLabel ;
                icare:varType ?varType ;
                icare:index ?index .
            OPTIONAL {{ ?variable icare:omop ?omopDomain }}
        }}

        {{
            GRAPH ?cohortMappingsGraph {{
                ?variable icare:mappedId ?mappedId .
            }}
        }} UNION {{
            GRAPH ?cohortVarGraph {{
                ?variable icare:categories ?category.
            }}
            GRAPH ?cohortMappingsGraph {{
                ?category icare:mappedId ?mappedId .
            }}
        }}
        OPTIONAL {{ ?mappedId rdfs:label ?mappedLabel }}
    }}
    ORDER BY ?cohort ?index
    """
    # TODO also get mappings from categories?
    # print(sparql_query)
    for row in run_query(sparql_query)["results"]["bindings"]:
        # print(row)
        # Find the concept in the list and add the cohort and variable to the used_by list
        for concept in found_concepts:
            if concept["uri"] == row["mappedId"]["value"]:
                used_by_entry = {
                    "cohort_id": row["cohortId"]["value"],
                    "var_name": row["varName"]["value"],
                    "var_label": row["varLabel"]["value"],
                    "omop_domain": row["omopDomain"]["value"] if "omopDomain" in row else None,
                }
                # NOTE: Normally the SPARQL query should note return duplicates, but in case it does:
                # existing_entries = [entry for entry in concept["used_by"] if entry["cohort_id"] == used_by_entry["cohort_id"] and entry["var_name"] == used_by_entry["var_name"]]
                # if not existing_entries:
                concept["used_by"].append(used_by_entry)
                break
    found_concepts.sort(key=lambda x: len(x["used_by"]), reverse=True)
    return found_concepts
