import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import requests

GraphSnapshot = TypeVar("GraphSnapshot")
CacheSnapshot = TypeVar("CacheSnapshot")
StagedGraph = TypeVar("StagedGraph")


class MetadataRollbackError(RuntimeError):
    def __init__(self, cause: BaseException, rollback_errors: list[BaseException]) -> None:
        self.cause = cause
        self.rollback_errors = tuple(rollback_errors)
        details = "; ".join(str(error) for error in rollback_errors)
        super().__init__(f"Metadata replacement failed ({cause}); rollback also failed: {details}")


def _graph_store_url(sparql_endpoint: str) -> str:
    return f"{sparql_endpoint.rstrip('/')}/store"


def snapshot_named_graph(sparql_endpoint: str, graph_uri: str) -> bytes | None:
    """Return a named graph serialization, or ``None`` when it does not exist."""
    response = requests.get(
        _graph_store_url(sparql_endpoint),
        params={"graph": graph_uri},
        headers={"Accept": "text/turtle"},
        timeout=120,
    )
    if response.status_code in {204, 404}:
        return None
    response.raise_for_status()
    return response.content or None


def replace_named_graph(sparql_endpoint: str, graph_uri: str, graph: Any | None) -> None:
    """Replace a named graph through the SPARQL Graph Store protocol."""
    url = _graph_store_url(sparql_endpoint)
    if graph is None:
        response = requests.delete(url, params={"graph": graph_uri}, timeout=120)
        if response.status_code != 404:
            response.raise_for_status()
        return

    if isinstance(graph, bytes):
        content = graph
    elif isinstance(graph, str):
        content = graph.encode("utf-8")
    else:
        if hasattr(graph, "graph"):
            from rdflib import URIRef

            graph = graph.graph(URIRef(graph_uri))
        serialized = graph.serialize(format="turtle", encoding="utf-8")
        content = serialized if isinstance(serialized, bytes) else serialized.encode("utf-8")
    response = requests.put(
        url,
        params={"graph": graph_uri},
        headers={"Content-Type": "text/turtle"},
        data=content,
        timeout=120,
    )
    response.raise_for_status()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def replace_metadata_transactionally(
    *,
    canonical_path: Path,
    content: bytes,
    staged_graph: StagedGraph,
    snapshot_graph: Callable[[], GraphSnapshot],
    replace_graph: Callable[[Any], None],
    snapshot_cache: Callable[[], CacheSnapshot],
    replace_cache: Callable[[Path], None],
    restore_cache: Callable[[CacheSnapshot], None],
    restore_graph: Callable[[GraphSnapshot], None] | None = None,
) -> None:
    """Replace file, graph, and cache, restoring all snapshots on any failure."""
    file_existed = canonical_path.exists()
    previous_file = canonical_path.read_bytes() if file_existed else None
    previous_graph = snapshot_graph()
    previous_cache = snapshot_cache()
    graph_restorer = restore_graph or replace_graph

    try:
        _atomic_write(canonical_path, content)
        replace_graph(staged_graph)
        replace_cache(canonical_path)
    except BaseException as cause:
        rollback_errors: list[BaseException] = []
        for restore in (
            lambda: restore_cache(previous_cache),
            lambda: graph_restorer(previous_graph),
            lambda: _atomic_write(canonical_path, previous_file)
            if previous_file is not None
            else canonical_path.unlink(missing_ok=True),
        ):
            try:
                restore()
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise MetadataRollbackError(cause, rollback_errors) from cause
        raise
