import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata

router = APIRouter()


@router.get("/cohorts-metadata")
def get_cohorts_metadata(user: Any = Depends(get_current_user)) -> dict[str, Cohort]:
    """Returns data dictionaries of all cohorts"""
    return retrieve_cohorts_metadata(user["email"])


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

