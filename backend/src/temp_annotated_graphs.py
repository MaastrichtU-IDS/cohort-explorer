"""TEMPORARY: manual rebuilds of what the cross-study mapping reads.

Why this exists
---------------
The mapping (CohortVarLinker) aligns variables through their OMOP concept
codes in the triplestore: data_element -> skos:closeMatch -> code set ->
cmeo:omop_id. Until August 2026 CohortVarLinker/main.py built those graphs
itself (create_cohort_specific_metadata_graph -> process_variables_metadata_file)
and rebuilt its Qdrant vector index (generate_studies_embeddings with
recreate_db=True) before every mapping run. Both steps were dropped in the
10 Aug integration (commit ec49aef): the graphs on the assumption that the
upload publishes an equivalent graph (it does not: backend/src/upload.py stores
the OMOP columns as flat literals, which no mapping query reads), the index by
switching to recreate_db=False. Result: zero annotated variables in every
cohort graph, and a vector index frozen at whatever dictionaries it was last
built from.

This module restores both steps as side functions on the backend side of the
code base, without touching CohortVarLinker. It reuses its untouched builders
and publishes the graph INTO the cohort's existing upload graph, so the two
shapes coexist in one named graph.

Remove this file, and its two lines in src/main.py, once CohortVarLinker
builds or consumes its graphs and index on its own again.

How to run (all admin-only; the POSTs return at once and run in the background)
----------
  POST /api/temp/rebuild-annotated-graphs          every cohort's graph
  GET  /api/temp/rebuild-annotated-graphs/status
  POST /api/temp/rebuild-vector-index              Qdrant collection, from the
                                                   current dictionary of each cohort
  GET  /api/temp/rebuild-vector-index/status
  POST /api/temp/archive-old-dictionaries          one-off: move old dictionary
                                                   versions to previous_versions_metadata/

Run the two rebuilds again after a dictionary is re-uploaded (the upload
replaces the whole cohort graph) and after the triplestore is rebuilt from
scratch. The vector rebuild holds CohortVarLinker's mapping lock, so it waits
for a running mapping and blocks new ones while it runs (several minutes).
"""

import glob
import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from rdflib import Dataset, URIRef

from src.auth import get_current_user
from src.config import settings

router = APIRouter()

CMEO = "https://w3id.org/CMEO/"
_COHORTS_DIR = os.path.join(settings.data_folder, "cohorts")

_EMPTY_STATUS = {"running": False, "started": None, "finished": None,
                 "done": [], "failed": {}, "pending": []}


# Status is kept in files, not in memory: the API runs several uvicorn
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


def _start(job: str, background: BackgroundTasks, fn) -> dict[str, Any]:
    if _read_status(job).get("running"):
        raise HTTPException(status_code=409, detail=f"{job} is already running.")
    _write_status(job, {**_EMPTY_STATUS, "running": True,
                        "started": datetime.now().isoformat()})
    background.add_task(fn)
    return {"started": True, "cohorts": _cohort_folders()}


# ---------------------------------------------------------------------------
# 1. Annotated cohort graphs
# ---------------------------------------------------------------------------

def _eda_file(folder: str) -> Optional[str]:
    # Same lookup as the old create_cohort_specific_metadata_graph: the EDA json
    # from the cohort's DCR run, if there is one (adds min/max statistics).
    eda_dir = os.path.join(settings.data_folder, f"dcr_output_{folder}")
    candidates = glob.glob(os.path.join(eda_dir, "eda*.json"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _delete_previous_annotations(graph_uri: str, cohort_key: str) -> None:
    """Drop what an earlier run of this module put in the graph (subjects under
    CMEO/<cohort>/...), leaving the upload's own ICARE-based triples alone."""
    from SPARQLWrapper import SPARQLWrapper
    update = SPARQLWrapper(settings.update_endpoint)
    update.setMethod("POST")
    update.setRequestMethod("urlencoded")
    update.setQuery(f"""
        DELETE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
        WHERE  {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }}
                  FILTER(STRSTARTS(STR(?s), "{CMEO}{cohort_key}/")) }}
    """)
    update.query()


def rebuild_one_graph(folder: str) -> int:
    """Build the annotated graph for one cohort folder and publish it into the
    cohort's existing named graph. Returns the number of triples published."""
    # Lazy: CohortVarLinker pulls in heavy dependencies at import time.
    from CohortVarLinker.src.query_builder import SPARQLQueryBuilder
    from CohortVarLinker.src.utils import normalize_text
    from CohortVarLinker.src.variables_kg import process_variables_metadata_file
    from src.upload import get_latest_datadictionary, publish_graph_to_endpoint

    dictionary = get_latest_datadictionary(os.path.join(_COHORTS_DIR, folder))
    if not dictionary:
        raise FileNotFoundError(f"no data dictionary in cohorts/{folder}")

    graphs_dir = os.path.join(settings.data_folder, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    result = process_variables_metadata_file(
        dictionary,
        cohort_name=folder,
        eda_file_path=_eda_file(folder),
        study_metadata_graph_file_path=os.path.join(graphs_dir, "studies_metadata.trig"),
    )
    # The builder returns the exception object (not raises) on some malformed rows.
    if not isinstance(result, tuple):
        raise RuntimeError(f"builder failed: {result!r}")
    built, cohort_key = result
    if not built or len(built) == 0:
        raise RuntimeError("builder produced an empty graph")

    # Publish under the graph the upload created (e.g. .../graph/GISSI-HF), found
    # case-insensitively; if the cohort was never uploaded, CohortVarLinker's own
    # lower-cased IRI is used instead.
    target = SPARQLQueryBuilder.graph_iri(folder)
    merged = Dataset(store="Oxigraph")
    for s, p, o, _ in built.quads():
        merged.add((s, p, o, URIRef(target)))

    _delete_previous_annotations(target, normalize_text(cohort_key))
    time.sleep(1)
    if not publish_graph_to_endpoint(merged):
        raise RuntimeError("triplestore rejected the graph")
    return len(merged)


def rebuild_annotated_graphs() -> dict[str, Any]:
    job = "annotated_graphs"
    folders = _cohort_folders()
    status = {**_read_status(job), "running": True, "finished": None,
              "done": [], "failed": {}, "pending": list(folders)}
    _write_status(job, status)
    for folder in folders:
        t0 = time.time()
        try:
            n = rebuild_one_graph(folder)
            logging.info("[temp_annotated_graphs] %s: %d triples in %.0fs", folder, n, time.time() - t0)
            status["done"].append(folder)
        except Exception as e:
            logging.exception("[temp_annotated_graphs] %s failed", folder)
            status["failed"][folder] = str(e)
        status["pending"].remove(folder)
        _write_status(job, status)
    # The mapping code caches the list of graph names; forget it so a cohort
    # published for the first time here is found by the next mapping run.
    try:
        from CohortVarLinker.src import query_builder
        with query_builder._graph_names_lock:
            query_builder._graph_names.clear()
    except Exception:
        pass
    status.update(running=False, finished=datetime.now().isoformat())
    _write_status(job, status)
    return status


@router.post("/temp/rebuild-annotated-graphs")
async def start_graph_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    _require_admin(user)
    return _start("annotated_graphs", background, rebuild_annotated_graphs)


@router.get("/temp/rebuild-annotated-graphs/status")
async def graph_rebuild_status(user: Any = Depends(get_current_user)):
    return _read_status("annotated_graphs")


# ---------------------------------------------------------------------------
# 2. Qdrant vector index
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
    from src.upload import get_latest_datadictionary

    status = {**_read_status(job), "running": True, "finished": None,
              "done": [], "failed": {}, "pending": []}
    staging = os.path.join(settings.data_folder, "temp_vector_index_staging")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        for folder in _cohort_folders():
            dictionary = get_latest_datadictionary(os.path.join(_COHORTS_DIR, folder))
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
        logging.info("[temp_annotated_graphs] vector index %s rebuilt: %d points in %.0fs",
                     collection, count, time.time() - t0)
        status["done"], status["pending"] = status["pending"], []
        status["points"] = count
    except Exception as e:
        logging.exception("[temp_annotated_graphs] vector index rebuild failed")
        status["failed"]["*"] = str(e)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        status.update(running=False, finished=datetime.now().isoformat())
        _write_status(job, status)
    return status


@router.post("/temp/rebuild-vector-index")
async def start_vector_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    _require_admin(user)
    return _start("vector_index", background, rebuild_vector_index)


@router.get("/temp/rebuild-vector-index/status")
async def vector_rebuild_status(user: Any = Depends(get_current_user)):
    return _read_status("vector_index")


# ---------------------------------------------------------------------------
# 3. One-off: move old dictionary versions out of the cohort folders
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
    import sys
    logging.basicConfig(level=logging.INFO)
    what = sys.argv[1] if len(sys.argv) > 1 else "graphs"
    print({"graphs": rebuild_annotated_graphs, "vectors": rebuild_vector_index}[what]())
