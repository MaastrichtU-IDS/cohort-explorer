"""TEMPORARY: manual maintenance of what the cross-study mapping reads.

Why this exists
---------------
Until August 2026 CohortVarLinker/main.py rebuilt its Qdrant vector index
(generate_studies_embeddings with recreate_db=True) before every mapping run.
The 10 Aug integration (commit ec49aef) switched to recreate_db=False, so the
index is whatever it was last built from and nothing in the app refreshes it.
(The same integration dropped the cohort-graph building step; that one is now
done by the upload itself, see upload.build_variables_graph.)

Remove this file, and its two lines in src/main.py, once CohortVarLinker
maintains its index on its own again.

How to run (all admin-only)
----------
  POST /api/temp/rebuild-vector-index              Qdrant collection, from the
                                                   current dictionary of each
                                                   cohort; returns at once, runs
                                                   in the background
  GET  /api/temp/rebuild-vector-index/status
  POST /api/temp/archive-old-dictionaries          one-off: move old dictionary
                                                   versions to
                                                   <cohort>/previous_versions_metadata/
                                                   (?dry_run=true to preview)

Run the vector rebuild again after a dictionary is re-uploaded. It holds
CohortVarLinker's mapping lock, so it waits for a running mapping and blocks
new ones while it runs (several minutes: it loads the embedding model and
embeds every variable and category).
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

router = APIRouter()

_COHORTS_DIR = os.path.join(settings.data_folder, "cohorts")
_EMPTY_STATUS = {"running": False, "started": None, "finished": None,
                 "done": [], "failed": {}, "pending": []}


# Status is kept in a file, not in memory: the API runs several uvicorn
# workers, and a GET would otherwise land on a worker that knows nothing
# about the job another one is running.
def _status_file(job: str) -> str:
    return os.path.join(settings.data_folder, f"temp_{job}_status.json")


def _read_status(job: str) -> dict[str, Any]:
    try:
        with open(_status_file(job)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(_EMPTY_STATUS)


def _write_status(job: str, status: dict[str, Any]) -> None:
    path = _status_file(job)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def _cohort_folders() -> list[str]:
    return sorted(f for f in os.listdir(_COHORTS_DIR)
                  if not f.startswith(".") and os.path.isdir(os.path.join(_COHORTS_DIR, f)))


def _require_admin(user: dict) -> None:
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


# ---------------------------------------------------------------------------
# Qdrant vector index
# ---------------------------------------------------------------------------

def rebuild_vector_index() -> dict[str, Any]:
    """Recreate the Qdrant collection the mapping searches, from the current
    dictionary of every cohort.

    generate_studies_embeddings() indexes the FIRST csv it finds in each folder,
    so the current dictionaries are first copied into a staging folder with
    one file per cohort. Collection name and modes come from CohortVarLinker's
    own defaults, so they cannot drift from what the mapping queries.
    """
    job = "vector_index"
    from CohortVarLinker.main import _exclusive_mapping_lock
    from CohortVarLinker.naming import DEFAULT_EMBED_MODEL, DEFAULT_EMBEDDING_MODE
    from CohortVarLinker.src.config import settings as cvl_settings
    from CohortVarLinker.src.vector_db import generate_studies_embeddings
    from src.upload import current_datadictionary

    status = {**_read_status(job), "running": True, "finished": None,
              "done": [], "failed": {}, "pending": []}
    staging = os.path.join(settings.data_folder, "temp_vector_index_staging")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        for folder in _cohort_folders():
            dictionary = current_datadictionary(os.path.join(_COHORTS_DIR, folder))
            if not dictionary:
                status["failed"][folder] = "no data dictionary"
                continue
            os.makedirs(os.path.join(staging, folder))
            shutil.copy2(dictionary, os.path.join(staging, folder, os.path.basename(dictionary)))
            status["pending"].append(folder)
        _write_status(job, status)

        collection = f"studies_metadata_{DEFAULT_EMBED_MODEL}_{DEFAULT_EMBEDDING_MODE}"
        t0 = time.time()
        with _exclusive_mapping_lock():
            vector_db, _ = generate_studies_embeddings(
                staging, cvl_settings.vector_db_path, collection,
                model_name=DEFAULT_EMBED_MODEL, embedding_mode=DEFAULT_EMBEDDING_MODE,
                recreate_db=True,
            )
        if vector_db is None:
            raise RuntimeError("generate_studies_embeddings returned no client")
        count = vector_db.count(collection).count
        logging.info("[temp_mapping_tools] vector index %s rebuilt: %d points in %.0fs",
                     collection, count, time.time() - t0)
        status["done"], status["pending"] = status["pending"], []
        status["points"] = count
    except Exception as e:
        logging.exception("[temp_mapping_tools] vector index rebuild failed")
        status["failed"]["*"] = str(e)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        status.update(running=False, finished=datetime.now().isoformat())
        _write_status(job, status)
    return status


@router.post("/temp/rebuild-vector-index")
async def start_vector_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    _require_admin(user)
    job = "vector_index"
    if _read_status(job).get("running"):
        raise HTTPException(status_code=409, detail="A vector index rebuild is already running.")
    _write_status(job, {**_EMPTY_STATUS, "running": True, "started": datetime.now().isoformat()})
    background.add_task(rebuild_vector_index)
    return {"started": True, "cohorts": _cohort_folders()}


@router.get("/temp/rebuild-vector-index/status")
async def vector_rebuild_status(user: Any = Depends(get_current_user)):
    return _read_status("vector_index")


# ---------------------------------------------------------------------------
# One-off: move old dictionary versions out of the cohort folders
# ---------------------------------------------------------------------------

@router.post("/temp/archive-old-dictionaries")
async def archive_old_dictionaries(dry_run: bool = False, user: Any = Depends(get_current_user)):
    """Apply upload.archive_previous_dictionaries() to every cohort folder: all
    dictionary files except the current one and its _noHeader twin move to
    <cohort>/previous_versions_metadata/. Uploads do this on their own from
    now on; this catches up the folders that predate that.
    With ?dry_run=true nothing moves; the response shows what would."""
    _require_admin(user)
    from src.upload import archive_previous_dictionaries
    result = {}
    for folder in _cohort_folders():
        r = archive_previous_dictionaries(os.path.join(_COHORTS_DIR, folder), dry_run=dry_run)
        if r["moved"]:
            result[folder] = r
    return {"dry_run": dry_run, "cohorts": result,
            "files": sum(len(r["moved"]) for r in result.values())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(rebuild_vector_index())
