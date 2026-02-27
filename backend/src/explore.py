import os
import logging
import json
from typing import Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata
from src.cohort_cache import get_cohorts_from_cache, is_cache_initialized, cohort_to_dict

router = APIRouter()


@router.get("/cohorts-metadata")
def get_cohorts_metadata(summary: bool = False, user: Any = Depends(get_current_user)) -> dict:
    """Returns data dictionaries of all cohorts.
    
    - When summary=false (default): return full cohorts including variables.
    - When summary=true: return cohort headers only (no variables/categories) to keep payload small.
    
    First tries to retrieve cohorts from the cache for fast access.
    Falls back to SPARQL queries if the cache is not initialized or is empty.
    """
    user_email = user["email"]
    
    # Try to get cohorts from the cache first
    if is_cache_initialized():
        logging.info("Retrieving cohorts from cache")
        cohorts = get_cohorts_from_cache(user_email)
        if cohorts:
            result = cohorts if not summary else {cid: {k: v for k, v in cohort_to_dict(c).items() if k != "variables"} for cid, c in cohorts.items()}
            return {**result, "userEmail": user_email}
        logging.warning("Cache is initialized but empty, falling back to SPARQL queries")
    else:
        logging.info("Cache not initialized, falling back to SPARQL queries")
    
    # Fall back to SPARQL queries if cache is not available or empty
    cohorts = retrieve_cohorts_metadata(user_email)
    
    result = cohorts if not summary else {cid: {k: v for k, v in cohort_to_dict(c).items() if k != "variables"} for cid, c in cohorts.items()}
    return {**result, "userEmail": user_email}


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
def get_cohorts_metadata_sparql(summary: bool = False, user: Any = Depends(get_current_user)) -> dict:
    """Returns data dictionaries of all cohorts using SPARQL queries only.
    
    - When summary=false: return full cohorts (plus SPARQL execution metadata).
    - When summary=true: return only cohort headers (no variables) to keep payload small.
    
    Always executes SPARQL queries directly against the triplestore,
    bypassing the cache completely. This provides real-time data but is slower.
    Returns both cohorts data and SPARQL execution metadata.
    """
    user_email = user["email"]
    
    logging.info("Retrieving cohorts using SPARQL queries (bypassing cache)")
    result = retrieve_cohorts_metadata(user_email, include_sparql_metadata=True)
    
    # Add user email to result
    result["userEmail"] = user_email
    
    if not summary:
        return result
    
    # Convert cohorts payload to summary
    cohorts_full = result.get("cohorts", result)  # support both shapes if changed in future
    if isinstance(cohorts_full, dict) and all(hasattr(v, "cohort_id") for v in cohorts_full.values()):
        cohorts_summary = {cid: {k: v for k, v in cohort_to_dict(c).items() if k != "variables"} for cid, c in cohorts_full.items()}
        result["cohorts"] = cohorts_summary
        return result
    else:
        # If result is already a dict[str, Cohort]
        summary_result = {cid: {k: v for k, v in cohort_to_dict(c).items() if k != "variables"} for cid, c in cohorts_full.items()}
        summary_result["userEmail"] = user_email
        return summary_result


@router.get("/cohort-spreadsheet/{cohort_id}")
async def download_cohort_spreasheet(cohort_id: str, user: Any = Depends(get_current_user)) -> FileResponse:
    """Download the data dictionary of a specified cohort as a spreadsheet."""
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


@router.get("/download-cohorts-metadata-spreadsheet")
async def download_cohorts_metadata_spreadsheet(user: Any = Depends(get_current_user)) -> FileResponse:
    """Download the central cohorts metadata Excel and include last-modified info in headers."""
    # Import path constant lazily to avoid any potential circular import
    try:
        from src.upload import COHORTS_METADATA_FILEPATH  # type: ignore
        excel_path = COHORTS_METADATA_FILEPATH
    except Exception:
        excel_path = os.path.join(settings.data_folder, "iCARE4CVD_Cohorts.xlsx")

    if not os.path.exists(excel_path) or not os.path.isfile(excel_path):
        raise HTTPException(status_code=404, detail="Cohorts metadata Excel file not found on server")

    # Compute last modified metadata
    mtime = os.path.getmtime(excel_path)
    iso_ts = datetime.fromtimestamp(mtime).isoformat()
    headers = {
        "X-File-Last-Modified": iso_ts,
    }

    return FileResponse(
        path=excel_path,
        filename=os.path.basename(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


def _get_all_cohort_ids() -> list[str]:
    """Get all valid cohort IDs from the cache or triplestore."""
    if is_cache_initialized():
        cohorts = get_cohorts_from_cache("")
        return list(cohorts.keys()) if cohorts else []
    
    # Fallback to SPARQL
    cohorts = retrieve_cohorts_metadata("")
    return list(cohorts.keys()) if cohorts else []


def _normalize_cohort_name(name: str) -> str:
    """Normalize cohort name for comparison: lowercase and strip leading/trailing spaces."""
    return name.strip().lower()


def _find_cohort_id_by_name(cohort_name: str) -> str | None:
    """
    Find the actual cohort ID that matches the given name after normalization.
    Returns the original cohort ID if found, None otherwise.
    """
    normalized_input = _normalize_cohort_name(cohort_name)
    all_cohort_ids = _get_all_cohort_ids()
    
    for cohort_id in all_cohort_ids:
        if _normalize_cohort_name(cohort_id) == normalized_input:
            return cohort_id
    
    return None


def _get_cohorts_with_shuffled_samples() -> list[str]:
    """Get list of all cohort IDs that have shuffled sample files available."""
    cohorts_with_samples = []
    all_cohort_ids = _get_all_cohort_ids()
    
    for cohort_id in all_cohort_ids:
        storage_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
        shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
        if os.path.exists(shuffled_csv):
            cohorts_with_samples.append(cohort_id)
    
    return cohorts_with_samples


@router.get(
    "/get-cohorts-with-shuffled-samples",
    name="Get list of cohorts with shuffled samples",
    response_description="List of cohort IDs that have shuffled sample files available",
)
async def get_cohorts_with_shuffled_samples(user: Any = Depends(get_current_user)) -> dict[str, list[str]]:
    """
    Get a list of all cohorts that have shuffled sample files available.
    
    Returns:
    - cohorts_with_shuffled_samples: List of cohort IDs that have shuffled samples
    """
    cohorts_with_samples = _get_cohorts_with_shuffled_samples()
    return {"cohorts_with_shuffled_samples": sorted(cohorts_with_samples)}


@router.get(
    "/get-shuffled-sample/{cohort_name}",
    name="Get shuffled sample file for a cohort",
    response_description="CSV file containing the shuffled sample data",
)
async def get_shuffled_sample(cohort_name: str, user: Any = Depends(get_current_user)) -> FileResponse:
    """
    Download the shuffled sample CSV file for a given cohort.
    
    - **cohort_name**: Name of the cohort (case-insensitive, leading/trailing spaces ignored)
    
    Returns:
    - The shuffled_sample.csv file if available
    
    Errors:
    - 404 if cohort name is not valid
    - 404 if cohort exists but has no shuffled sample yet (includes list of cohorts that do have samples)
    """
    # Find the actual cohort ID matching the normalized name
    actual_cohort_id = _find_cohort_id_by_name(cohort_name)
    
    if actual_cohort_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{cohort_name}' is not a valid cohort name."
        )
    
    # Check if shuffled sample file exists
    storage_dir = os.path.join(settings.data_folder, f"dcr_output_{actual_cohort_id}")
    shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
    
    if not os.path.exists(shuffled_csv):
        cohorts_with_samples = _get_cohorts_with_shuffled_samples()
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Cohort '{actual_cohort_id}' does not have a shuffled sample file yet.",
                "cohorts_with_shuffled_samples": sorted(cohorts_with_samples)
            }
        )
    
    return FileResponse(
        path=shuffled_csv,
        filename=f"{actual_cohort_id}_shuffled_sample.csv",
        media_type="text/csv"
    )

