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

    # Search for a data dictionary file in the cohort's folder
    for file_name in os.listdir(cohorts_folder):
        if file_name.endswith("_datadictionary.csv") or file_name.endswith("_datadictionary.xlsx"):
            file_path = os.path.join(cohorts_folder, file_name)
            return FileResponse(path=file_path, filename=file_name, media_type="application/octet-stream")

    # If no file is found, return an error response
    raise HTTPException(status_code=404, detail=f"No data dictionary found for cohort ID '{cohort_id}'")
