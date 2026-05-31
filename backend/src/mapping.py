from typing import Any
import logging

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from src.auth import get_current_user
from src.utils import curie_converter, run_query
from src.mapping_logger import log_main, log_detail, MappingRun, PROCESS_CVL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Add CohortVarLinker to path for lazy imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CohortVarLinker')))


def _read_or_generate_meta(json_filepath: str) -> dict | None:
    """Read the sidecar .meta.json for a mapping file.
    
    If the sidecar is missing or older than the mapping file, lazily
    parse the mapping JSON and write a fresh sidecar.
    Returns the stats dict or None on failure.
    """
    if json_filepath.endswith('.meta.json'):
        return None
    meta_path = json_filepath + ".meta.json"
    mapping_mtime = os.path.getmtime(json_filepath)
    
    # Check if sidecar exists and is up-to-date
    if os.path.exists(meta_path):
        meta_mtime = os.path.getmtime(meta_path)
        if meta_mtime >= mapping_mtime:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # fall through to regeneration
    
    # Sidecar missing or stale — regenerate from the mapping JSON
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            raw = f.read().replace("NaN", "null")
            data = json.loads(raw)
        
        total = 0
        harm_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}
        for entry in data.values():
            if not isinstance(entry, dict) or "mappings" not in entry:
                continue
            for m in entry["mappings"]:
                total += 1
                hs = str(m.get("harmonization_status") or "pending")
                mr = str(m.get("mapping_relation") or "")
                harm_counts[hs] = harm_counts.get(hs, 0) + 1
                rel_counts[mr] = rel_counts.get(mr, 0) + 1
        
        meta = {
            "total_mappings": total,
            "harmonization_status": harm_counts,
            "mapping_relation": rel_counts,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated sidecar stats: {meta_path}")
        return meta
    except Exception as e:
        logger.warning(f"Failed to generate sidecar for {json_filepath}: {e}")
        return None

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
    target_studies: list = Body(...),
    user: Any = Depends(get_current_user),
):
    """
    Check cache status for mapping pairs without generating mappings.
    Returns cache information immediately with dictionary timestamps.
    """
    # Lazy import to avoid module-level import errors
    from CohortVarLinker.src.config import settings as cohort_linker_settings
    from CohortVarLinker.src.utils import get_member_studies
    
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
    
    # Replicate the same naming convention as generate_mapping_csv
    model_name = "biolord"
    mapping_mode = "OEH"
    # LLM tag computation mirrors main.py logic
    from CohortVarLinker.src.config import settings as _cvl_settings
    from CohortVarLinker.src.data_model import MappingType as _MT
    _llm_model = _cvl_settings.llm_model
    if _llm_model and mapping_mode != _MT.OO.value:
        llm_tag = _llm_model.split("/")[-1].replace(":nitro", "")
    else:
        llm_tag = "no_llm"

    for tstudy in target_studies_names:
        out_filename = f'{source_study}_{tstudy}_{model_name}+{llm_tag}_{mapping_mode}_full.csv'
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


@router.post("/get-available-mapping-files")
async def get_available_mapping_files(
    cohort_ids: list[str] = Body(...),
    user: Any = Depends(get_current_user)
):
    """
    Get all available mapping files for the given cohort IDs.
    
    Mapping files are .json files with naming pattern:
    {cohort1}_{cohort2}_..._{cohortN}_{model}+{llm_tag}_{mode}.json
    e.g. time-chf_aachen-hf_biolord+no_llm_OEH.json

    All parts before the model token (e.g. 'biolord') are cohort names.
    A file is only included if ALL cohorts in its filename are among the
    selected cohorts.
    """
    logger.info(f"[DEBUG] get_available_mapping_files called with cohort_ids = {cohort_ids}")
    
    from CohortVarLinker.src.config import settings as cohort_linker_settings
    
    output_dir = cohort_linker_settings.output_dir
    logger.info(f"[DEBUG] get_available_mapping_files: output_dir = {output_dir}")
    logger.info(f"[DEBUG] get_available_mapping_files: os.path.exists(output_dir) = {os.path.exists(output_dir)}")
    
    logger.info(f"[DEBUG] get_available_mapping_files: received {len(cohort_ids)} cohort_ids")
    
    available_mappings = []
    
    # Scan directory for .json mapping files
    if os.path.exists(output_dir):
        all_files = os.listdir(output_dir)
        logger.info(f"[DEBUG] get_available_mapping_files: all files in output_dir = {all_files}")
        json_files = [f for f in all_files if f.endswith('.json') and not f.endswith('.meta.json')]
        logger.info(f"[DEBUG] get_available_mapping_files: json files = {json_files}")
        for filename in all_files:
            if not filename.endswith('.json') or filename.endswith('.meta.json'):
                continue
            
            # Parse cohort names from filename.
            # New pattern: {cohorts}_{model}+{llm}_{mode}.json
            # The model token contains '+' (e.g. 'biolord+no_llm').
            # We split on '+' first to find where the model info starts,
            # then extract cohort names from everything before it.
            stem = filename.replace('.json', '')
            plus_idx = stem.find('+')
            if plus_idx == -1:
                # Legacy format without '+', try old sapbert pattern
                parts = stem.split('_')
                try:
                    sep_idx = parts.index('sapbert')
                except ValueError:
                    try:
                        sep_idx = parts.index('biolord')
                    except ValueError:
                        continue
                file_cohorts = [p.lower() for p in parts[:sep_idx]]
            else:
                # New format: everything before the last '_' before '+' is cohorts+model
                before_plus = stem[:plus_idx]
                parts_before = before_plus.rsplit('_', 1)
                if len(parts_before) < 2:
                    continue
                cohort_part = parts_before[0]  # everything before model name
                file_cohorts = [p.lower() for p in cohort_part.split('_')]
            logger.info(f"[DEBUG] Parsed file '{filename}': cohorts = {file_cohorts}")
            
            if len(file_cohorts) < 2:
                # Need at least 2 cohorts for a mapping file
                logger.info(f"[DEBUG] Skipping '{filename}': less than 2 cohorts")
                continue
            
            filepath = os.path.join(output_dir, filename)
            file_size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            
            # Create display name showing all cohorts
            display_name = ' → '.join(file_cohorts)
            
            # Read sidecar stats (lazy-generate if missing/stale)
            stats = _read_or_generate_meta(filepath)
            
            # Skip files with 0 mappings (failed or empty runs)
            if stats and stats.get("total_mappings", 0) == 0:
                logger.info(f"[DEBUG] Skipping '{filename}': 0 total mappings")
                continue
            
            available_mappings.append({
                'cohorts': file_cohorts,
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'timestamp': mtime,
                'display_name': display_name,
                'stats': stats,
            })
    
    # De-duplicate: for each unique cohort set, keep only the most recent file
    seen: dict[str, int] = {}  # cohort_key -> index in available_mappings
    deduped = []
    # Sort by timestamp descending so we encounter newest first
    available_mappings.sort(key=lambda x: x['timestamp'], reverse=True)
    for entry in available_mappings:
        key = "_".join(sorted(entry['cohorts']))
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(entry)
    
    return JSONResponse(content={
        'available_mappings': deduped,
        'cohort_count': len(cohort_ids)
    })


@router.get("/get-cached-mapping-file/{filename}")
async def get_cached_mapping_file(
    filename: str,
    user: Any = Depends(get_current_user),
):
    """Return the JSON content of a cached mapping file by filename.
    
    Returns the raw JSON directly (not wrapped in another JSON envelope)
    to avoid double-serialization overhead on large files.
    The filename is returned via the X-Filename response header.
    """
    from CohortVarLinker.src.config import settings as cohort_linker_settings
    
    output_dir = cohort_linker_settings.output_dir
    # Security: prevent path traversal
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(output_dir, safe_filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Mapping file not found")
    
    if safe_filename.endswith('.meta.json'):
        raise HTTPException(status_code=400, detail="Cannot download sidecar meta files directly")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    # Sanitize NaN values server-side so clients don't have to
    file_content = file_content.replace("NaN", "null")
    
    return Response(
        content=file_content,
        media_type="application/json",
        headers={
            "X-Filename": safe_filename,
            "Access-Control-Expose-Headers": "X-Filename",
        },
    )


@router.post("/generate-mapping")
async def generate_mapping(
    source_study: str = Body(...),
    target_studies: list = Body(...),
    user: Any = Depends(get_current_user),
):
    """
    Generate a mapping CSV for the given source and target studies and return it as a downloadable file.
    target_studies should be a list of [study_name, visit_constraint_bool]
    """
    # Lazy import to avoid module-level import errors
    from CohortVarLinker.main_llm import generate_mapping_csv
    from CohortVarLinker.src.config import settings as cohort_linker_settings
    
    # Call the backend function
    # The function writes CSVs to CohortVarLinker/data/mapping_output/{source}_{target}_{model}+{llm}_{mode}_full.csv
    # We'll return the combined JSON file
    
    target_studies = sorted(target_studies, key=lambda x: x[0])
    target_names = [t[0] for t in target_studies]

    with MappingRun(PROCESS_CVL, source=source_study, targets=target_names,
                    user=user.get("email", "unknown")):
        cache_info = generate_mapping_csv(source_study, target_studies)

    output_dir = cohort_linker_settings.output_dir
    
    # generate_mapping_csv may expand target_studies with member/sub-studies,
    # so the JSON filename can differ from what we'd construct here.
    # Scan the output directory for the most recent JSON that starts with
    # the source study and contains all requested target cohorts.
    source_lower = source_study.lower()
    target_lowers = set(t[0].lower() for t in target_studies)
    
    best_file = None
    best_mtime = 0
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if not fname.endswith('.json') or fname.endswith('.meta.json'):
                continue
            if not fname.startswith(source_lower + '_'):
                continue
            fpath = os.path.join(output_dir, fname)
            mtime = os.path.getmtime(fpath)
            # Check that all requested targets appear in the filename
            stem = fname.replace('.json', '').lower()
            if all(t in stem for t in target_lowers) and mtime >= best_mtime:
                best_file = fpath
                best_mtime = mtime
    
    if best_file and os.path.exists(best_file):
        with open(best_file, 'r') as f:
            file_content = f.read()
        
        log_detail(PROCESS_CVL, "result_file_served",
                   f"Serving mapping result: {os.path.basename(best_file)}",
                   ctx={"filename": os.path.basename(best_file),
                        "file_size_bytes": os.path.getsize(best_file)})
        return JSONResponse(content={
            "cache_info": cache_info,
            "file_content": file_content,
            "filename": os.path.basename(best_file)
        })
    log_main(PROCESS_CVL, "result_not_found",
             f"No mapping output file found for {source_study}",
             ctx={"source": source_study, "targets": list(target_lowers)})
    return JSONResponse(status_code=404, content={"error": "Cache error. Mapping file not found."})


@router.get("/mapping-activity-log")
async def get_mapping_activity_log(
    limit: int = Query(default=200, ge=1, le=5000),
    level: str | None = Query(default=None, description="Filter by level: MAIN or DETAIL"),
    process: str | None = Query(default=None, description="Filter by process: cohort_var_linker or standard_code_mapping"),
    user: Any = Depends(get_current_user),
):
    """Return the mapping activity log (JSONL) as a JSON array for web display.
    
    Supports filtering by level and process, and returns the most recent `limit` entries.
    """
    from src.mapping_logger import _get_log_path
    import json as _json

    log_path = _get_log_path()
    if not os.path.exists(log_path):
        return JSONResponse(content={"entries": [], "total": 0})

    entries = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if level and entry.get("level") != level:
                continue
            if process and entry.get("process") != process:
                continue
            entries.append(entry)

    # Return only the most recent `limit` entries
    total = len(entries)
    entries = entries[-limit:]
    return JSONResponse(content={"entries": entries, "total": total})


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


def find_dcr_output_folder(cohort_id: str) -> str | None:
    """
    Find the actual dcr_output folder for a cohort, handling case-insensitive matching.
    Returns the actual folder name if found, None otherwise.
    """
    data_folder = settings.data_folder
    if not os.path.exists(data_folder):
        return None
    
    # Try exact match first
    exact_folder = f"dcr_output_{cohort_id}"
    if os.path.exists(os.path.join(data_folder, exact_folder)):
        return exact_folder
    
    # Try case-insensitive search
    target_prefix = f"dcr_output_{cohort_id.lower()}"
    for folder in os.listdir(data_folder):
        if folder.lower() == target_prefix and os.path.isdir(os.path.join(data_folder, folder)):
            return folder
    
    return None

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
    
    # Find the actual folder names (case-insensitive)
    source_folder = find_dcr_output_folder(source_cohort)
    target_folder = find_dcr_output_folder(target_cohort)
    
    # Collect detailed error messages
    errors = []
    
    # Check source cohort
    if not source_folder:
        errors.append(f"Source cohort '{source_cohort}': Exploratory Data Analysis has not yet been run on this cohort")
    else:
        source_image_path = os.path.join(settings.data_folder, source_folder, f"{source_var.lower()}.png")
        if not os.path.exists(source_image_path):
            errors.append(f"Source variable '{source_var}' in cohort '{source_cohort}': This variable was excluded from the EDA analysis")
    
    # Check target cohort
    if not target_folder:
        errors.append(f"Target cohort '{target_cohort}': Exploratory Data Analysis has not yet been run on this cohort")
    else:
        target_image_path = os.path.join(settings.data_folder, target_folder, f"{target_var.lower()}.png")
        if not os.path.exists(target_image_path):
            errors.append(f"Target variable '{target_var}' in cohort '{target_cohort}': This variable was excluded from the EDA analysis")
    
    # If any errors, raise with detailed message
    if errors:
        error_detail = "Cannot compare EDA images:\n" + "\n".join(f"• {err}" for err in errors)
        raise HTTPException(
            status_code=404,
            detail=error_detail
        )
    
    # Both files exist, construct paths
    source_image_path = os.path.join(settings.data_folder, source_folder, f"{source_var.lower()}.png")
    target_image_path = os.path.join(settings.data_folder, target_folder, f"{target_var.lower()}.png")
    
    try:
        # Load both images
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)
        
        # Convert images to RGB if they have transparency
        if source_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', source_image.size, 'white')
            if source_image.mode == 'P':
                source_image = source_image.convert('RGBA')
            background.paste(source_image, mask=source_image.split()[-1] if source_image.mode in ('RGBA', 'LA') else None)
            source_image = background
        elif source_image.mode != 'RGB':
            source_image = source_image.convert('RGB')
            
        if target_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', target_image.size, 'white')
            if target_image.mode == 'P':
                target_image = target_image.convert('RGBA')
            background.paste(target_image, mask=target_image.split()[-1] if target_image.mode in ('RGBA', 'LA') else None)
            target_image = background
        elif target_image.mode != 'RGB':
            target_image = target_image.convert('RGB')
        
        # Resize images to 75% of original size
        source_image = source_image.resize(
            (int(source_image.width * 0.75), int(source_image.height * 0.75)),
            Image.Resampling.LANCZOS
        )
        target_image = target_image.resize(
            (int(target_image.width * 0.75), int(target_image.height * 0.75)),
            Image.Resampling.LANCZOS
        )
        
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
