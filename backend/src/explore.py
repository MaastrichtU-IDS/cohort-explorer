import os
import logging
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata
from src.cohort_cache import get_cohorts_from_cache, is_cache_initialized

router = APIRouter()


@router.get("/cohorts-metadata")
def get_cohorts_metadata(user: Any = Depends(get_current_user)) -> dict[str, Cohort]:
    """Returns data dictionaries of all cohorts
    
    First tries to retrieve cohorts from the cache for fast access.
    Falls back to SPARQL queries if the cache is not initialized or is empty.
    """
    user_email = user["email"]
    
    # Try to get cohorts from the cache first
    if is_cache_initialized():
        logging.info("Retrieving cohorts from cache")
        cohorts = get_cohorts_from_cache(user_email)
        if cohorts:
            return cohorts
        logging.warning("Cache is initialized but empty, falling back to SPARQL queries")
    else:
        logging.info("Cache not initialized, falling back to SPARQL queries")
    
    # Fall back to SPARQL queries if cache is not available or empty
    cohorts = retrieve_cohorts_metadata(user_email)
    
    return cohorts


@router.get("/cohort-eda-output/{cohort_name}")
async def get_cohort_eda_output(
    cohort_name: str,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve the EDA output JSON file for a given cohort."""
    # Construct the directory and file paths
    dcr_output_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_name}")
    eda_file_path = os.path.join(dcr_output_dir, f"eda_output_{cohort_name}.json")
    
    # Check if directory exists
    if not os.path.exists(dcr_output_dir):
        raise HTTPException(
            status_code=404,
            detail=f"DCR output directory not found for cohort '{cohort_name}'"
        )
    
    # Check if file exists
    if not os.path.exists(eda_file_path):
        raise HTTPException(
            status_code=404,
            detail=f"EDA output file not found for cohort '{cohort_name}'"
        )
    
    # Read and return the JSON file
    try:
        with open(eda_file_path, 'r') as f:
            eda_data = json.load(f)
        return eda_data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing EDA output JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading EDA output file: {str(e)}"
        )


@router.get("/cohorts-metadata-sparql")
def get_cohorts_metadata_sparql(user: Any = Depends(get_current_user)) -> dict:
    """Returns data dictionaries of all cohorts using SPARQL queries only
    
    Always executes SPARQL queries directly against the triplestore,
    bypassing the cache completely. This provides real-time data but is slower.
    Returns both cohorts data and SPARQL execution metadata.
    """
    user_email = user["email"]
    
    logging.info("Retrieving cohorts using SPARQL queries (bypassing cache)")
    result = retrieve_cohorts_metadata(user_email, include_sparql_metadata=True)
    
    return result


@router.get("/cohort-spreadsheet/{cohort_id}")
async def download_cohort_spreasheet(cohort_id: str, user: Any = Depends(get_current_user)) -> FileResponse:
    """Download the data dictionary of a specified cohort as a spreadsheet."""
    # cohort_id = urllib.parse.unquote(cohort_id)
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)

    # List all CSV files excluding those with 'noHeader' in the name
    csv_files = [
        f for f in os.listdir(cohorts_folder)
        if f.endswith('.csv') and 'noHeader' not in f
    ]
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No metadata csv file found for cohort ID '{cohort_id}'")

    # Find the latest CSV file by modification time
    csv_files_full = [os.path.join(cohorts_folder, f) for f in csv_files]
    latest_file = max(csv_files_full, key=os.path.getmtime)
    latest_file_name = os.path.basename(latest_file)
    return FileResponse(path=latest_file, filename=latest_file_name, media_type="application/octet-stream")

