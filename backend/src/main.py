import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.auth import get_current_user
from src.auth import router as auth_router
from src.config import settings
from src.decentriq import router as decentriq_router
from src.models import Cohort
from src.upload import init_triplestore
from src.upload import router as upload_router
from src.utils import retrieve_cohorts_metadata

init_triplestore()

app = FastAPI(
    title="iCARE4CVD API",
    description="""Upload and explore cohorts metadata files for the [iCARE4CVD project](https://icare4cvd.eu/).""",
)


@app.get("/cohorts-metadata", tags=["explore"])
def get_cohorts_metadata(user: Any = Depends(get_current_user)) -> dict[str, Cohort]:
    """Returns data dictionaries of all cohorts"""
    return retrieve_cohorts_metadata()


@app.get("/cohort-spreadsheet/{cohort_id}", tags=["explore"])
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

app.include_router(upload_router, tags=["upload"])
app.include_router(decentriq_router, tags=["upload"])
app.include_router(auth_router, tags=["authentication"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the route / to /docs"""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
