"""TEMPORARY: rebuild the cohort graphs the cross-study mapping needs.

Why this exists
---------------
The mapping (CohortVarLinker) aligns variables through their OMOP concept
codes in the triplestore: data_element -> skos:closeMatch -> code set ->
cmeo:omop_id. Until August 2026 CohortVarLinker/main.py built those graphs
itself (create_cohort_specific_metadata_graph -> process_variables_metadata_file)
before every mapping run. That step was dropped in the 10 Aug integration
(commit ec49aef) on the assumption that the upload already publishes an
equivalent graph. It does not: backend/src/upload.py stores the OMOP columns as
flat literals, which no mapping query reads, so every cohort graph in the store
has zero annotated variables and every mapping pair comes back empty.

This module restores the dropped step as a side function on the backend side
of the code base, without touching CohortVarLinker. It reuses its untouched
builder (process_variables_metadata_file) and publishes the result INTO the
cohort's existing upload graph, so the two shapes coexist in one named graph.

Remove this file, and its two lines in src/main.py, once CohortVarLinker
builds or consumes its graphs on its own again.

How to run
----------
  POST /api/temp/rebuild-annotated-graphs            (admin; all cohorts)
  GET  /api/temp/rebuild-annotated-graphs/status
or from a shell inside the backend container:
  python -m src.temp_annotated_graphs

Run it again after a dictionary is re-uploaded (the upload replaces the whole
cohort graph) and after the triplestore is rebuilt from scratch.
"""

import glob
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from rdflib import Dataset, URIRef

from src.auth import get_current_user
from src.config import settings

router = APIRouter()

CMEO = "https://w3id.org/CMEO/"

# Status is kept in a file, not in memory: the API runs several uvicorn
# workers, and the GET would otherwise land on a worker that knows nothing
# about the job another one is running.
_STATUS_FILE = os.path.join(settings.data_folder, "temp_annotated_graphs_status.json")
_EMPTY_STATUS = {"running": False, "started": None, "finished": None,
                 "done": [], "failed": {}, "pending": []}


def _read_status() -> dict[str, Any]:
    try:
        with open(_STATUS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(_EMPTY_STATUS)


def _write_status(status: dict[str, Any]) -> None:
    tmp = f"{_STATUS_FILE}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, _STATUS_FILE)


def _cohort_folders() -> list[str]:
    root = os.path.join(settings.data_folder, "cohorts")
    return sorted(f for f in os.listdir(root)
                  if not f.startswith(".") and os.path.isdir(os.path.join(root, f)))


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


def rebuild_one(folder: str) -> int:
    """Build the annotated graph for one cohort folder and publish it into the
    cohort's existing named graph. Returns the number of triples published."""
    # Lazy: CohortVarLinker pulls in heavy dependencies at import time.
    from CohortVarLinker.src.query_builder import SPARQLQueryBuilder
    from CohortVarLinker.src.utils import normalize_text
    from CohortVarLinker.src.variables_kg import process_variables_metadata_file
    from src.upload import get_latest_datadictionary, publish_graph_to_endpoint

    dictionary = get_latest_datadictionary(os.path.join(settings.data_folder, "cohorts", folder))
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
    folders = _cohort_folders()
    status = {**_EMPTY_STATUS, "running": True, "started": datetime.now().isoformat(),
              "done": [], "failed": {}, "pending": list(folders)}
    _write_status(status)
    for folder in folders:
        t0 = time.time()
        try:
            n = rebuild_one(folder)
            logging.info("[temp_annotated_graphs] %s: %d triples in %.0fs", folder, n, time.time() - t0)
            status["done"].append(folder)
        except Exception as e:
            logging.exception("[temp_annotated_graphs] %s failed", folder)
            status["failed"][folder] = str(e)
        status["pending"].remove(folder)
        _write_status(status)
    # The mapping code caches the list of graph names; forget it so a cohort
    # published for the first time here is found by the next mapping run.
    try:
        from CohortVarLinker.src import query_builder
        with query_builder._graph_names_lock:
            query_builder._graph_names.clear()
    except Exception:
        pass
    status.update(running=False, finished=datetime.now().isoformat())
    _write_status(status)
    return status


@router.post("/temp/rebuild-annotated-graphs")
async def start_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")
    if _read_status().get("running"):
        raise HTTPException(status_code=409, detail="A rebuild is already running.")
    background.add_task(rebuild_annotated_graphs)
    return {"started": True, "cohorts": _cohort_folders()}


@router.get("/temp/rebuild-annotated-graphs/status")
async def rebuild_status(user: Any = Depends(get_current_user)):
    return _read_status()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(rebuild_annotated_graphs())
