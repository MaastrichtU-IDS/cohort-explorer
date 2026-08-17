"""API-facing entry point for cross-study variable mapping.
Called by backend/src/mapping.py -> POST /api/generate-mapping.
"""

import glob
import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import pandas as pd

try:
    import fcntl
except ImportError:  # non-POSIX host; the in-process thread lock still applies
    fcntl = None

from .naming import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBEDDING_MODE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAPPING_MODE,
    config_tag,
    csv_name,
    json_name,
)
from .src.config import settings
from .src.constraints import CategoryMapper
from .src.data_model import MappingType
from .src.graph_similarity import _EMBED_CACHE
from .src.omop_graph_nx import OmopGraphNX
from .src.run import StudyMapper
from .src.utils import (
    get_member_studies,
    get_cohort_mapping_uri,
    graph_exists,
    OntologyNamespaces,
)
from .src.vector_db import _embed_cache, generate_studies_embeddings

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_EMPTY_MAPPING_COLS = ["source_study", "target_study", "source", "target",
                       "harmonization_status"]

API_EMBED_MODEL = DEFAULT_EMBED_MODEL
API_EMBEDDING_MODE = DEFAULT_EMBEDDING_MODE
API_MAPPING_MODE = DEFAULT_MAPPING_MODE
API_LLM_MODEL = DEFAULT_LLM_MODEL


# Guards construction of the singletons only — cheap, held for seconds.
_mapper_lock = threading.Lock()
# Keyed by configuration: two different configs need two different StudyMappers
# (different Qdrant collection, different OMOP graph). In practice callers use
# the defaults, so this holds one entry — but a second config would otherwise
# have silently reused the first one's mapper.
_mapper_state: dict[tuple, StudyMapper] = {}

# Guards the pipeline against other THREADS in this process. Every request
# shares one StudyMapper, and run_pipeline() stores per-pair state on it
# (last_source_variables, last_target_variables, and
# matcher.llm_matcher._study_context). Two threads running concurrently would
# interleave on that state and silently feed one pair's study context into the
# other pair's LLM prompts.
_pipeline_lock = threading.Lock()

# Guards the pipeline against other PROCESSES on this host. uvicorn runs with
# --workers 6, and each worker gets its own copy of _pipeline_lock and of the
# StudyMapper singleton — so the thread lock alone lets six workers map the same
# pair at once, burning six times the GPU and LLM spend for one result. flock()
# is arbitrated by the kernel on the underlying file, so every worker queues on
# the same lock. Lives in output_dir, which is a local bind mount; flock over
# NFS is not reliable, so keep it on local storage.
_LOCKFILE_NAME = ".mapping.lock"

@contextmanager
def _exclusive_mapping_lock():
    """Hold the mapping lock across every thread and every worker on this host.

    Always acquired thread-lock first, then file-lock — one consistent order, so
    there is nothing to deadlock against.
    """
    os.makedirs(settings.output_dir, exist_ok=True)
    lock_path = os.path.join(settings.output_dir, _LOCKFILE_NAME)
    with _pipeline_lock:
        if fcntl is None:
            yield
            return
        with open(lock_path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)   # blocks until every other worker is done
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def clear_all_caches() -> None:
    """Drop the embedding caches. Stale entries survive a model or KG change."""
    _embed_cache.clear()
    _EMBED_CACHE.clear()
    CategoryMapper._label_embedding_cache.clear()
    CategoryMapper._label_omop_cache.clear()
    CategoryMapper._alignment_cache.clear()


def study_family(study: str) -> list[str]:
    """Every study reachable from `study` through obi:has_member, itself first.

    get_member_studies() already looks both ways, so the family is the same set
    whichever member is named. The walk is a closure rather than a single hop so
    a chain (A has_member B has_member C) still collapses to one family.
    """
    family, queue = [study], [study]
    while queue:
        for member in get_member_studies(queue.pop()):
            if member not in family:
                family.append(member)
                queue.append(member)
    return family


def _resolve_concepts_file() -> str:
    """settings.concepts_file_path is relative to a checkout; an API worker has a
    different cwd, so fall back to the package-local data dir."""
    path = settings.concepts_file_path
    for candidate in (path, os.path.join(_BASE_PATH, "data", os.path.basename(path))):
        if os.path.exists(candidate):
            return candidate
    return path  # let OmopGraphNX raise against the configured path




def _get_mapper(embed_model: str, embedding_mode: str, mapping_mode: str,
                llm_model: Optional[str]) -> StudyMapper:
    """Build the StudyMapper once per process, per configuration. Loading the
    embedding model and the OMOP graph costs minutes — a request must not pay
    that every time."""
    key = (embed_model, embedding_mode, mapping_mode, llm_model)
    with _mapper_lock:
        if key in _mapper_state:
            return _mapper_state[key]

        collection_name = f"studies_metadata_{embed_model}_{embedding_mode}"
        vector_db, embedding_model = generate_studies_embeddings(
            os.path.join(_BASE_PATH, "data", "cross_mapping"),
            settings.vector_db_path,
            collection_name,
            model_name=embed_model,
            embedding_mode=embedding_mode,
            recreate_db=False,
        )
        if vector_db is None or embedding_model is None:
            raise RuntimeError(
                f"Vector DB collection '{collection_name}' not available at "
                f"{settings.vector_db_path}. Build it once with recreate_db=True."
            )

        # Prefer the prebuilt graph pickle (settings.omop_graph_pickle_path,
        # which resolves to <CohortVarLinker>/data/graph_nx.pkl.gz and honours
        # the OMOP_GRAPH_PICKLE_PATH env override); only build from the concepts
        # CSV when no pickle is available.
        if mapping_mode == MappingType.NE.value:
            omop_graph = None
        else:
            graph_file = settings.omop_graph_pickle_path
            if os.path.exists(graph_file):
                logging.info("Loading prebuilt OMOP graph from %s", graph_file)
                omop_graph = OmopGraphNX(output_file=graph_file)
            else:
                logging.info(
                    "No prebuilt OMOP graph pickle at %s; building from concepts CSV",
                    graph_file,
                )
                omop_graph = OmopGraphNX(
                    csv_file_path=_resolve_concepts_file(), output_file=graph_file
                )

        _mapper_state[key] = StudyMapper(
            vector_db=vector_db,
            vector_collection=collection_name,
            embedding_model=embedding_model,
            omop_graph=omop_graph,
            mapping_mode=mapping_mode,
            llm_model=llm_model,
            list_of_var=[],   # empty -> no benchmark scoping, map every source variable
        )
        return _mapper_state[key]


def _latest_dictionary_mtime(cohort_id: str) -> Optional[float]:
    """mtime of the newest data dictionary uploaded for a cohort, or None.

    Mirrors get_latest_dictionary_timestamp() in backend/src/mapping.py so the
    cache-check endpoint and this module agree on when a CSV has gone stale.
    """
    folder = os.path.join(settings.data_folder, "cohorts", cohort_id)
    if not os.path.isdir(folder):
        return None
    candidates = [
        f for f in glob.glob(os.path.join(folder, "*.csv"))
        if "datadictionary" in os.path.basename(f).lower()
        and "noheader" not in os.path.basename(f).lower()
    ]
    return max((os.path.getmtime(f) for f in candidates), default=None)


def _cross_mapping_csv_path(source_study: str, target_study: str, cfg: str) -> str:
    """cfg keeps runs under different settings in separate files, so a cached
    CSV from one configuration is never mistaken for another's."""
    return os.path.join(settings.output_dir, csv_name(source_study, target_study, cfg))


def _is_stale(source_study: str, target: str, cache_mtime: float,
              source_dict_mtime: Optional[float],
              target_dict_mtime: Optional[float]) -> Optional[str]:
    """Name of the cohort whose dictionary is newer than the cached CSV, or None."""
    if source_dict_mtime and source_dict_mtime > cache_mtime:
        return source_study
    if target_dict_mtime and target_dict_mtime > cache_mtime:
        return target
    return None


def _write_pair_csv(mapper: StudyMapper, source_study: str, source_family: list[str],
                    target: str, embed_model: str, mapping_mode: str, cfg: str) -> int:
    """Map every source-family member onto `target` and write one CSV.

    Source-family rows share a single file named for the study the caller asked
    for; the source_study/target_study columns keep each row self-describing.
    """
    parts = []
    for src in source_family:
        print(f"[api] mapping {src} -> {target} ({embed_model}, {mapping_mode})")
        df = mapper.run_pipeline(src_study=src, tgt_study=target,
                                 mapping_mode=mapping_mode)
        if df.empty:
            continue
        if "harmonization_status" in df.columns:
            df = df[df["harmonization_status"].str.strip().str.lower()
                    != "not applicable"].copy()
        if df.empty:
            continue
        df.insert(0, "source_study", src)
        df.insert(1, "target_study", target)
        parts.append(df)

    combined = (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=_EMPTY_MAPPING_COLS))

    # Write to a temp file and rename: os.replace is atomic on the same
    # filesystem, so a concurrent reader never sees a half-written CSV.
    final_path = _cross_mapping_csv_path(source_study, target, cfg)
    tmp_path = f"{final_path}.tmp{os.getpid()}"
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)

    print(f"[api] {len(combined)} rows -> {os.path.basename(final_path)}")
    return len(combined)


def _combine_cross_mapping_json(source_study: str, target_studies: list[str],
                                json_path: str, cfg: str) -> int:
    """Fold the per-target CSVs into the single JSON the frontend downloads,
    keyed by source variable.

    NaN becomes null so the payload is valid JSON — the frontend currently
    strips NaN with a regex, which this makes unnecessary.
    """
    mappings = defaultdict(list)
    for target in target_studies:
        csv_path = _cross_mapping_csv_path(source_study, target, cfg)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.astype(object).where(pd.notna(df), None)
        for row in df.to_dict(orient="records"):
            src_var = str(row.get("source") or "").strip()
            if not src_var:
                continue
            mappings[src_var].append(
                {"target_study": target,
                 **{k: v for k, v in row.items() if k != "source"}}
            )

    final_json = {k: {"from": source_study, "mappings": v}
                  for k, v in mappings.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False, default=str)
    return len(final_json)


def generate_mapping_csv(source_study: str, target_studies: list,
                         force: bool = False,
                         embed_model: str = API_EMBED_MODEL,
                         embedding_mode: str = API_EMBEDDING_MODE,
                         mapping_mode: str = API_MAPPING_MODE,
                         llm_model: Optional[str] = API_LLM_MODEL) -> dict:
    """Map `source_study` onto every requested target, caching per pair.
    """
    started = time.time()
    out_dir = settings.output_dir
    os.makedirs(out_dir, exist_ok=True)

    source_study = source_study.lower()
    requested: list[str] = []
    for t in target_studies:
        name = str(t[0] if isinstance(t, (list, tuple)) else t).strip().lower()
        if name and name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("generate_mapping_csv called with no target studies")

    # One fingerprint for this run, stamped into every filename it writes. The
    # JSON is named for the REQUESTED targets, not the expanded families, so
    # backend/src/mapping.py can predict the name from the request alone.
    cfg = config_tag(embed_model, mapping_mode, llm_model)

    source_family = study_family(source_study)

    # Expand each target to its full family so members get their own CSV, and
    # drop any target sharing a family with the source — nothing to map there.
    effective_targets: list[str] = []
    skipped: list[str] = []
    for tstudy in requested:
        family = study_family(tstudy)
        if any(t in source_family for t in family):
            skipped.append(tstudy)
            continue
        for member in family:
            if member not in effective_targets:
                effective_targets.append(member)

    dictionary_timestamps = {}
    for cohort in dict.fromkeys(source_family + effective_targets):
        mtime = _latest_dictionary_mtime(cohort)
        if mtime is not None:
            dictionary_timestamps[cohort] = mtime
    source_dict_mtime = max(
        (dictionary_timestamps[s] for s in source_family if s in dictionary_timestamps),
        default=None,
    )

    cached_pairs, uncached_pairs, outdated_pairs = [], [], []
    to_compute: list[str] = []

    for tgt in effective_targets:
        pair = {"source": source_study, "target": tgt}
        csv_path = _cross_mapping_csv_path(source_study, tgt, cfg)

        if not os.path.exists(csv_path):
            uncached_pairs.append(pair)
            to_compute.append(tgt)
            continue

        cache_mtime = os.path.getmtime(csv_path)
        stale_by = _is_stale(source_study, tgt, cache_mtime, source_dict_mtime,
                             dictionary_timestamps.get(tgt))

        if stale_by or force:
            entry = {**pair, "timestamp": cache_mtime}
            if stale_by:
                entry["outdated_cohort"] = stale_by
            outdated_pairs.append(entry)
            to_compute.append(tgt)
        else:
            cached_pairs.append({**pair, "timestamp": cache_mtime})

    computed, reused_after_wait = [], []
    if to_compute:
        # One mapping at a time across the whole host. Requests that only hit
        # cache never reach this block, so they are never made to wait.
        with _exclusive_mapping_lock():
            mapper = _get_mapper(embed_model, embedding_mode, mapping_mode, llm_model)
            clear_all_caches()

            for tgt in to_compute:
                # Re-check under the lock. If we queued behind a request that
                # wanted the same pair — in this worker or any other — its CSV is
                # on disk now and fresh. Take it instead of spending another 15
                # minutes producing the identical file.
                csv_path = _cross_mapping_csv_path(source_study, tgt, cfg)
                if not force and os.path.exists(csv_path):
                    cache_mtime = os.path.getmtime(csv_path)
                    if not _is_stale(source_study, tgt, cache_mtime, source_dict_mtime,
                                     dictionary_timestamps.get(tgt)):
                        reused_after_wait.append(tgt)
                        continue

                _write_pair_csv(mapper, source_study, source_family, tgt,
                                embed_model, mapping_mode, cfg)
                computed.append(tgt)

    out_name = json_name(source_study, requested, cfg)
    n_source_vars = _combine_cross_mapping_json(
        source_study, effective_targets, os.path.join(out_dir, out_name), cfg)

    # Every per-pair CSV this run left on disk, so the caller can offer them for
    # download alongside the combined JSON. Named for the expanded targets — one
    # requested target with a family produces one CSV per member.
    pair_files = [
        {"source": source_study, "target": tgt,
         "filename": csv_name(source_study, tgt, cfg)}
        for tgt in effective_targets
        if os.path.exists(_cross_mapping_csv_path(source_study, tgt, cfg))
    ]

    return {
        "cached_pairs": cached_pairs,
        "uncached_pairs": uncached_pairs,
        "outdated_pairs": outdated_pairs,
        "pair_files": pair_files,
        "skipped_pairs": [
            {"source": source_study, "target": t,
             "reason": "same study family as source"}
            for t in skipped
        ],
        "dictionary_timestamps": dictionary_timestamps,
        "output_file": out_name,
        "source_variables": n_source_vars,
        "computed_pairs": computed,
        "reused_after_wait": reused_after_wait,
        "duration_seconds": round(time.time() - started, 2),
        # What actually ran, so a cached result can be told apart from a rerun
        # under different settings without decoding the filename.
        "config": {
            "embed_model": embed_model,
            "embedding_mode": embedding_mode,
            "mapping_mode": mapping_mode,
            "llm_model": llm_model,
        },
    }


# ---------------------------------------------------------------------------
# Cache-status helpers grafted from main's CohortVarLinker/main.py.
#
# backend/src/mapping.py's /check-mapping-cache endpoint imports these two
# functions directly. Komal's rewritten module (above) computes cache status
# internally inside generate_mapping_csv, but the separate cache-check endpoint
# still needs these. find_cached_csv's "{source}_{target}_*.csv" glob matches
# the config-tagged names produced by naming.csv_name(). Study metadata graphs
# are published at upload time (backend/src/upload.py), so _check_graphs_need_recreate
# remains valid for reporting whether the triplestore graphs are current.
# ---------------------------------------------------------------------------

def find_cached_csv(source_study, target_study, output_dir):
    """Find the most recent cached CSV file for a source→target pair.

    Normalizes both names to lowercase (preserving dashes) and searches
    only in output_dir for .csv files matching either:
      - {source}_{target}.csv       (exact, no suffix)
      - {source}_{target}_*.csv     (with config-tag suffix from naming.csv_name)
    Returns the path to the most recent match, or None if not found.
    """
    source = source_study.lower()
    target = target_study.lower()
    matches = glob.glob(os.path.join(output_dir, f"{source}_{target}.csv"))
    matches += glob.glob(os.path.join(output_dir, f"{source}_{target}_*.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _check_graphs_need_recreate(cohort_ids, cohort_file_path) -> bool:
    """Check if any cohort dictionaries are newer than their existing graph files,
    or if the graphs are missing from the triplestore.

    Returns True if:
      - Any dictionary is newer than its graph file on disk, OR
      - Any graph file is missing on disk, OR
      - Any cohort graph is missing from the triplestore.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(base_path, "data", "graphs")

    # Check the shared studies_metadata graph first
    studies_graph_uri = OntologyNamespaces.CMEO.value["graph/studies_metadata"]
    try:
        if not graph_exists(studies_graph_uri):
            print(f"Studies metadata graph missing in triplestore ({studies_graph_uri}), need recreate")
            return True
    except Exception as e:
        print(f"Error checking triplestore for studies_metadata: {e}, assuming recreate needed")
        return True

    for cid in cohort_ids:
        # Check triplestore — if the graph isn't there, we must recreate
        cohort_graph_uri = get_cohort_mapping_uri(cid)
        try:
            if not graph_exists(cohort_graph_uri):
                print(f"Graph missing in triplestore for {cid} ({cohort_graph_uri}), need recreate")
                return True
        except Exception as e:
            print(f"Error checking triplestore for {cid}: {e}, assuming recreate needed")
            return True

        cohort_dir = os.path.join(cohort_file_path, cid)
        if not os.path.isdir(cohort_dir):
            continue
        dict_candidates = [
            f for f in glob.glob(os.path.join(cohort_dir, "*.csv"))
            if ("datadictionary" in os.path.basename(f).lower()
            and "noheader" not in os.path.basename(f).lower())
        ]
        if not dict_candidates:
            continue
        latest_dict_mtime = max(os.path.getmtime(f) for f in dict_candidates)
        graph_file = os.path.join(graphs_dir, f"{cid}_metadata.trig")
        if not os.path.exists(graph_file):
            return True
        graph_mtime = os.path.getmtime(graph_file)
        if latest_dict_mtime > graph_mtime:
            return True
    return False
