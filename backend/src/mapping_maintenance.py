"""Maintenance of what the cross-study mapping (CohortVarLinker) reads.

A cohort's dictionary feeds two derived stores the mapping queries:

- its graph in the triplestore (variables with their OMOP concept chains,
  built by CohortVarLinker's own builder via upload.build_variables_graph);
- the Qdrant vector index (one embedding per variable and category).

Both are rebuilt here, in the background, and the time of each rebuild is
recorded so mapping.py can tell when a cached mapping CSV predates its inputs.

Triggers
--------
- dictionary upload           -> that cohort's graph, then the whole vector index
- new EDA output for a cohort -> that cohort's graph (the builder adds the
                                 min/max statistics from dcr_output_<cohort>)
- startup from an empty store -> graphs (synchronously, in init_triplestore)
- admin endpoints below       -> everything, e.g. after restoring the store
                                 or after a CohortVarLinker change

Endpoints (admin)
----------
  POST /api/mapping-maintenance/graphs                  rebuild every cohort graph
  GET  /api/mapping-maintenance/graphs/status           per-cohort publish times
  POST /api/mapping-maintenance/vector-index            rebuild the Qdrant collection
  GET  /api/mapping-maintenance/vector-index/status     build time, points, errors
  POST /api/mapping-maintenance/archive-old-dictionaries[?dry_run=true]

State lives in files under <data>/mapping_maintenance/ because the API runs
several uvicorn workers; nothing here is kept in process memory.
"""

import glob
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

_COHORTS_DIR = os.path.join(settings.data_folder, "cohorts")
_STATE_DIR = os.path.join(settings.data_folder, "mapping_maintenance")
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------

def _state_path(name: str) -> str:
    return os.path.join(_STATE_DIR, f"{name}.json")


def _read_state(name: str) -> dict[str, Any]:
    try:
        with open(_state_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_state(name: str, state: dict[str, Any]) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    path = _state_path(name)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _update_state(name: str, **fields: Any) -> dict[str, Any]:
    with _write_lock:
        state = _read_state(name)
        state.update(fields)
        _write_state(name, state)
        return state


def _cohort_folders() -> list[str]:
    if not os.path.isdir(_COHORTS_DIR):
        return []
    return sorted(f for f in os.listdir(_COHORTS_DIR)
                  if not f.startswith(".") and os.path.isdir(os.path.join(_COHORTS_DIR, f)))


def _resolve_cohort_folder(cohort_id: str) -> Optional[str]:
    """Folder name for a cohort id, tolerating case and the '-' for ' ' swap
    DCR node names use."""
    wanted = {cohort_id.lower(), cohort_id.lower().replace("-", " ")}
    for folder in _cohort_folders():
        if folder.lower() in wanted or folder.lower().replace(" ", "-") in wanted:
            return folder
    return None


def _require_admin(user: dict) -> None:
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


# ---------------------------------------------------------------------------
# Cohort graphs
# ---------------------------------------------------------------------------

def graph_published_at(cohort_id: str) -> Optional[float]:
    """Epoch time the cohort's graph was last published, or None if unknown."""
    state = _read_state("graphs")
    entry = next((v for k, v in state.items() if k.lower() == cohort_id.lower()), None)
    return entry.get("published_at") if entry else None


def record_graph_published(cohort_id: str, triples: int) -> None:
    with _write_lock:
        state = _read_state("graphs")
        state[cohort_id] = {"published_at": time.time(), "published_iso": datetime.now().isoformat(),
                            "triples": triples, "error": None}
        _write_state("graphs", state)


def publish_cohort_graph(cohort_id: str) -> int:
    """Build the cohort's graph from its current dictionary (and EDA output, if
    any), replace it in the triplestore and record the time. Returns the
    number of triples published. Runs in the background after an upload or a
    DCR computation; safe to call again at any time."""
    from rdflib import Literal
    from rdflib.namespace import DC, RDF, RDFS
    from src.upload import (
        OntologyNamespaces, build_variables_graph, current_datadictionary, delete_existing_triples,
        get_cohort_graph_uri, get_cohort_uri, init_graph, publish_graph_to_endpoint,
    )

    folder = _resolve_cohort_folder(cohort_id) or cohort_id
    try:
        dictionary = current_datadictionary(os.path.join(_COHORTS_DIR, folder))
        if not dictionary:
            raise FileNotFoundError(f"no data dictionary in cohorts/{folder}")
        cohort_uri = get_cohort_uri(folder)
        graph_uri = get_cohort_graph_uri(folder)
        g = init_graph()
        g.add((cohort_uri, RDF.type, OntologyNamespaces.CMEO.value.study, graph_uri))
        g.add((cohort_uri, DC.identifier, Literal(folder), graph_uri))
        g.add((cohort_uri, RDFS.label, Literal(folder), graph_uri))
        build_variables_graph(g, dictionary, folder, graph_uri)
        delete_existing_triples(graph_uri)
        if not publish_graph_to_endpoint(g):
            raise RuntimeError("triplestore rejected the graph")
        record_graph_published(folder, len(g))
        logger.info("[mapping_maintenance] graph for %s published: %d triples", folder, len(g))
        return len(g)
    except Exception as e:
        logger.exception("[mapping_maintenance] graph for %s failed", folder)
        with _write_lock:
            state = _read_state("graphs")
            state[folder] = {**(state.get(folder) or {}), "error": str(e),
                             "failed_iso": datetime.now().isoformat()}
            _write_state("graphs", state)
        raise


def rebuild_all_graphs() -> dict[str, Any]:
    folders = _cohort_folders()
    _update_state("graphs_job", running=True, started=datetime.now().isoformat(),
                  finished=None, done=[], failed={}, pending=list(folders))
    done, failed = [], {}
    for folder in folders:
        try:
            publish_cohort_graph(folder)
            done.append(folder)
        except Exception as e:
            failed[folder] = str(e)
        _update_state("graphs_job", done=done, failed=failed,
                      pending=[f for f in folders if f not in done and f not in failed])
    try:  # the mapping caches the graph list; a cohort published for the first time must be found
        from CohortVarLinker.src import query_builder
        with query_builder._graph_names_lock:
            query_builder._graph_names.clear()
    except Exception:
        pass
    return _update_state("graphs_job", running=False, finished=datetime.now().isoformat())


@router.post("/mapping-maintenance/graphs")
async def start_graphs_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    _require_admin(user)
    if _read_state("graphs_job").get("running"):
        raise HTTPException(status_code=409, detail="A graph rebuild is already running.")
    _update_state("graphs_job", running=True, started=datetime.now().isoformat(), finished=None)
    background.add_task(rebuild_all_graphs)
    return {"started": True, "cohorts": _cohort_folders()}


@router.get("/mapping-maintenance/graphs/status")
async def graphs_status(user: Any = Depends(get_current_user)):
    return {"job": _read_state("graphs_job"), "cohorts": _read_state("graphs")}


# ---------------------------------------------------------------------------
# Qdrant vector index
# ---------------------------------------------------------------------------

def vector_index_built_at() -> Optional[float]:
    return _read_state("vector_index").get("built_at")


def vector_index_collection() -> str:
    from CohortVarLinker.naming import DEFAULT_EMBED_MODEL, DEFAULT_EMBEDDING_MODE
    return f"studies_metadata_{DEFAULT_EMBED_MODEL}_{DEFAULT_EMBEDDING_MODE}"


def rebuild_vector_index() -> dict[str, Any]:
    """Recreate the Qdrant collection from the current dictionary of every
    cohort. Loads the embedding model and embeds every variable and category,
    so it takes minutes; holds CohortVarLinker's mapping lock meanwhile.

    generate_studies_embeddings() indexes the FIRST csv it finds per folder, so
    the current dictionaries are staged one per cohort. If another rebuild was
    requested while this one ran (an upload during the rebuild), it runs again
    so the index never misses the newest dictionary.
    """
    from CohortVarLinker.main import _exclusive_mapping_lock
    from CohortVarLinker.naming import DEFAULT_EMBED_MODEL, DEFAULT_EMBEDDING_MODE
    from CohortVarLinker.src.config import settings as cvl_settings
    from CohortVarLinker.src.vector_db import generate_studies_embeddings
    from src.upload import current_datadictionary

    staging = os.path.join(settings.data_folder, "vector_index_staging")
    while True:
        _update_state("vector_index", running=True, started=datetime.now().isoformat(),
                      finished=None, rerun_requested=False, error=None)
        t0 = time.time()
        try:
            shutil.rmtree(staging, ignore_errors=True)
            skipped = {}
            for folder in _cohort_folders():
                dictionary = current_datadictionary(os.path.join(_COHORTS_DIR, folder))
                if not dictionary:
                    skipped[folder] = "no data dictionary"
                    continue
                os.makedirs(os.path.join(staging, folder))
                shutil.copy2(dictionary, os.path.join(staging, folder, os.path.basename(dictionary)))
            collection = vector_index_collection()
            with _exclusive_mapping_lock():
                vector_db, _ = generate_studies_embeddings(
                    staging, cvl_settings.vector_db_path, collection,
                    model_name=DEFAULT_EMBED_MODEL, embedding_mode=DEFAULT_EMBEDDING_MODE,
                    recreate_db=True,
                )
            if vector_db is None:
                raise RuntimeError("generate_studies_embeddings returned no client")
            points = vector_db.count(collection).count
            logger.info("[mapping_maintenance] vector index %s rebuilt: %d points in %.0fs",
                        collection, points, time.time() - t0)
            _update_state("vector_index", built_at=time.time(), built_iso=datetime.now().isoformat(),
                          collection=collection, points=points, skipped=skipped,
                          duration_seconds=round(time.time() - t0))
        except Exception as e:
            logger.exception("[mapping_maintenance] vector index rebuild failed")
            _update_state("vector_index", error=str(e))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        state = _update_state("vector_index", running=False, finished=datetime.now().isoformat())
        if not state.get("rerun_requested"):
            return state


def request_vector_index_rebuild(background: BackgroundTasks) -> str:
    """Start a rebuild, or flag the running one to go again when it finishes."""
    with _write_lock:
        state = _read_state("vector_index")
        if state.get("running"):
            state["rerun_requested"] = True
            _write_state("vector_index", state)
            return "queued"
        state.update(running=True, started=datetime.now().isoformat(), finished=None)
        _write_state("vector_index", state)
    background.add_task(rebuild_vector_index)
    return "started"


@router.post("/mapping-maintenance/vector-index")
async def start_vector_rebuild(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    _require_admin(user)
    return {"status": request_vector_index_rebuild(background), "cohorts": _cohort_folders()}


@router.get("/mapping-maintenance/vector-index/status")
async def vector_index_status(user: Any = Depends(get_current_user)):
    return _read_state("vector_index")


# ---------------------------------------------------------------------------
# Staleness of cached mapping CSVs
# ---------------------------------------------------------------------------

def inputs_changed_at(cohorts: list[str]) -> float:
    """Newest change among everything a mapping between these cohorts reads:
    their dictionaries, their graphs and the shared vector index."""
    from src.mapping import get_latest_dictionary_timestamp
    newest = vector_index_built_at() or 0.0
    for cohort in cohorts:
        for ts in (get_latest_dictionary_timestamp(cohort), graph_published_at(cohort)):
            if ts and ts > newest:
                newest = ts
    return newest


def maintenance_running() -> dict[str, bool]:
    return {"graphs": bool(_read_state("graphs_job").get("running")),
            "vector_index": bool(_read_state("vector_index").get("running"))}


def invalidate_stale_mapping_caches(source: str, targets: list[str]) -> list[str]:
    """Retire cached per-pair CSVs that predate their inputs (a cohort graph or
    the vector index rebuilt after them), so CohortVarLinker (which only
    compares against dictionary mtimes) recomputes them. The files are not
    deleted: they move to the output folder's replaced_due_to_new_dictionary
    subfolder, since those rebuilds follow a dictionary upload or new EDA
    output. Returns the retired file names."""
    from CohortVarLinker.main import archive_replaced_csv, find_cached_csv, study_family
    from CohortVarLinker.src.config import settings as cvl_settings
    out_dir = cvl_settings.output_dir
    source_family = study_family(source.lower())
    retired = []
    for target in targets:
        for member in study_family(target.lower()):
            cutoff = inputs_changed_at(source_family + [member])
            while True:
                path = find_cached_csv(source, member, out_dir)
                if not path or os.path.getmtime(path) >= cutoff:
                    break
                archive_replaced_csv(path, "new_dictionary")
                retired.append(os.path.basename(path))
    if retired:
        logger.info("[mapping_maintenance] moved stale mapping caches to replaced_due_to_new_dictionary/: %s", retired)
    return retired


# ---------------------------------------------------------------------------
# One-off: move old dictionary versions out of the cohort folders
# ---------------------------------------------------------------------------

@router.post("/mapping-maintenance/archive-old-dictionaries")
async def archive_old_dictionaries(dry_run: bool = False, user: Any = Depends(get_current_user)):
    """Apply upload.archive_previous_dictionaries() to every cohort folder: all
    dictionary files except the current one and its _noHeader twin move to
    <cohort>/previous_versions_metadata/. Uploads do this on their own; this
    catches up folders that predate that or were changed by hand."""
    _require_admin(user)
    from src.upload import archive_previous_dictionaries
    result = {}
    for folder in _cohort_folders():
        r = archive_previous_dictionaries(os.path.join(_COHORTS_DIR, folder), dry_run=dry_run)
        if r["moved"]:
            result[folder] = r
    return {"dry_run": dry_run, "cohorts": result,
            "files": sum(len(r["moved"]) for r in result.values())}
