from typing import Any

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

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

# Import the CohortVarLinker function and settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CohortVarLinker')))
from CohortVarLinker.main import generate_mapping_csv
from CohortVarLinker.src.config import settings as cohort_linker_settings
from CohortVarLinker.src.utils import get_member_studies

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
    output_dir = cohort_linker_settings.output_dir
    
    source_study = source_study.lower()
    target_studies_names = [t[0].lower() for t in target_studies]
    
    # Add member studies (replicate the same logic as in generate_mapping_csv)
    new_studies = []
    for tstudy in target_studies_names:
        member_studies = get_member_studies(tstudy)
        if member_studies:
            new_studies.extend(member_studies)
    target_studies_names.extend(new_studies)
    
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
    # The function writes CSVs to CohortVarLinker/data/mapping_output/{source}_{target}_cross_mapping.csv
    # We'll return the combined JSON file
    
    target_studies = sorted(target_studies, key=lambda x: x[0])
    cache_info = generate_mapping_csv(source_study, target_studies)
    output_dir = cohort_linker_settings.output_dir
    
     # Find the generated file(s)
    source_study = source_study.lower()
    target_str = "_".join([t[0].lower() for t in target_studies])
    
    # Use the same naming convention as generate_mapping_csv: {source}_{targets}_{model}_{mode}.json
    model_name = "sapbert"
    mapping_mode = "ontology+embedding(concept)"
    filename = f"{source_study}_{target_str}_{model_name}_{mapping_mode}.json"
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


@router.get("/compare-eda/{source_cohort}/{source_var}/{target_cohort}/{target_var}")
async def compare_eda(
    source_cohort: str,
    source_var: str,
    target_cohort: str,
    target_var: str
):
    """
    Merge two EDA PNG files vertically (source on top, target on bottom) and return the merged image.
    """
    import io
    
    # Construct paths to the two EDA PNG files
    source_image_path = os.path.join(settings.data_folder, f"dcr_output_{source_cohort}", f"{source_var.lower()}.png")
    target_image_path = os.path.join(settings.data_folder, f"dcr_output_{target_cohort}", f"{target_var.lower()}.png")
    
    # Check if both files exist
    if not os.path.exists(source_image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Source EDA image not found: {source_image_path}"
        )
    
    if not os.path.exists(target_image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Target EDA image not found: {target_image_path}"
        )
    
    try:
        # Load both images
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)
        
        # Calculate dimensions for the merged image
        max_width = max(source_image.width, target_image.width)
        total_height = source_image.height + target_image.height
        
        # Create a new image with white background
        merged_image = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste source image on top (centered if narrower)
        source_x = (max_width - source_image.width) // 2
        merged_image.paste(source_image, (source_x, 0))
        
        # Paste target image on bottom (centered if narrower)
        target_x = (max_width - target_image.width) // 2
        merged_image.paste(target_image, (target_x, source_image.height))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        merged_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to merge EDA images: {str(e)}"
        )
