"""Running DCR compute nodes from My DCRs, and the Results Gallery.

My DCRs: a participant can run (or re-run) any compute node of a DCR they are
an analyst of, and view or download each file of its output. The run goes
through the service account (an analyst of the same nodes in every DCR the
explorer creates) in a background thread, since a run that pulls a merge
chain can take minutes; its state lives in a run.json on disk so the four
uvicorn workers all see it. Each node keeps its latest successful output:
a re-run replaces it only once it succeeds.

  {data_folder}/dcr_results/{dcr_id}/{node}/run.json   status + file list
  {data_folder}/dcr_results/{dcr_id}/{node}/files/     extracted output

Results Gallery: an analyst can share single output files with every
logged-in user, with a title, description and details. Sharing copies the
file, so a later re-run never changes what was shared.

  {data_folder}/results_gallery/items.json
  {data_folder}/results_gallery/files/{item_id}/{file name}

Who may run, view or share is decided from the DCR records of /my-dcrs (the
participants and their analyst_of nodes as read from Decentriq).
"""

import fcntl
import io
import json
import logging
import mimetypes
import os
import shutil
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings

router = APIRouter(tags=["dcr-results"])
logger = logging.getLogger(__name__)

# A run still marked "running" after this long was cut off (worker restart).
STALE_RUN_AFTER = timedelta(hours=3)
MAX_TITLE = 200
MAX_DESCRIPTION = 2000
MAX_DETAILS = 10000
MAX_NAME = 120

# Airlock (preview) nodes cannot be run on their own; everything else that
# computes can.
_NOT_RUNNABLE = {"PreviewComputeNodeDefinition"}


def _is_compute(node_type: str) -> bool:
    return node_type.endswith("ComputeNodeDefinition") and node_type not in _NOT_RUNNABLE


def _email(user: Any) -> str:
    email = user.get("email") if isinstance(user, dict) else None
    if not email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return email.strip().lower()


def _is_admin(email: str) -> bool:
    return email in (getattr(settings, "admins_list", []) or [])


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: str, data: Any) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except Exception as exc:
        logger.warning("Unreadable JSON %s: %s", path, exc)
        return default


@contextmanager
def _file_lock(path: str) -> Iterator[None]:
    """Cross-process lock (the API runs several workers)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# DCR records and permissions
# ---------------------------------------------------------------------------

def _user_dcr(email: str, dcr_id: str) -> dict[str, Any]:
    from src.decentriq import get_dcrs_for_participant
    for record in get_dcrs_for_participant(email):
        if record.get("id") == dcr_id:
            return record
    raise HTTPException(status_code=404, detail="DCR not found among your Data Clean Rooms")


def _runnable_nodes(record: dict[str, Any], email: str) -> list[str]:
    """Compute nodes of the DCR this user is an analyst of, in DCR order."""
    analyst_of: set[str] = set()
    for p in record.get("participants") or []:
        if (p.get("email") or "").strip().lower() == email:
            analyst_of.update(p.get("analyst_of") or [])
    return [
        n["name"] for n in record.get("nodes") or []
        if n.get("name") in analyst_of and _is_compute(str(n.get("type") or ""))
    ]


def _require_node(email: str, dcr_id: str, node_name: str) -> dict[str, Any]:
    record = _user_dcr(email, dcr_id)
    if node_name not in _runnable_nodes(record, email):
        raise HTTPException(status_code=403, detail="You are not an analyst of this compute node")
    return record


def _dcr_cohorts(record: dict[str, Any]) -> list[str]:
    """Catalog cohort ids of a DCR: matched against its metadata-derived cohort
    names and its node names (node names use '-' for spaces; no-code rooms
    have no dictionary nodes, so their data nodes are what names them)."""
    names = set(record.get("cohorts") or [])
    names.update(n.get("name") for n in record.get("nodes") or [] if n.get("name"))
    try:
        from src.cohort_cache import get_cached_cohort_ids
        catalog = get_cached_cohort_ids()
    except Exception as exc:
        logger.warning("Could not list catalog cohorts: %s", exc)
        catalog = []
    found = [c for c in catalog if c in names or c.replace(" ", "-") in names]
    return sorted(found) or sorted(record.get("cohorts") or [])


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _node_dir(dcr_id: str, node_name: str) -> str:
    # Both come from the participant's DCR record, but keep them path-safe anyway.
    safe_dcr = os.path.basename(dcr_id)
    safe_node = os.path.basename(node_name).replace("..", "_")
    if not safe_dcr or not safe_node:
        raise HTTPException(status_code=400, detail="Invalid DCR or node name")
    return os.path.join(settings.data_folder, "dcr_results", safe_dcr, safe_node)


def _list_files(files_dir: str) -> list[dict[str, Any]]:
    out = []
    for root, _, names in os.walk(files_dir):
        for name in names:
            full = os.path.join(root, name)
            out.append({"path": os.path.relpath(full, files_dir), "size": os.path.getsize(full)})
    return sorted(out, key=lambda f: f["path"])


def _run_state(dcr_id: str, node_name: str) -> dict[str, Any]:
    state = _read_json(os.path.join(_node_dir(dcr_id, node_name), "run.json"), {})
    if state.get("status") == "running":
        try:
            started = datetime.fromisoformat(state.get("started_at") or "")
            if datetime.now() - started > STALE_RUN_AFTER:
                state["status"] = "failed"
                state["error"] = "The run was interrupted (no result after 3 hours). Run it again."
        except ValueError:
            pass
    return state


def _execute_run(dcr_id: str, node_name: str, email: str) -> None:
    """Background thread: run the node and swap its output in on success."""
    import decentriq_platform as dq

    node_dir = _node_dir(dcr_id, node_name)
    state_path = os.path.join(node_dir, "run.json")
    new_dir = os.path.join(node_dir, f"files.{uuid.uuid4().hex}")
    try:
        client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
        dcr = client.retrieve_analytics_dcr(dcr_id)
        node = dcr.get_node(node_name)
        if node is None:
            raise RuntimeError(f"Node '{node_name}' not found in the DCR")
        result = node.run_computation_and_get_results_as_zip()
        os.makedirs(new_dir)
        if result is not None:
            zf = result if isinstance(result, zipfile.ZipFile) else zipfile.ZipFile(io.BytesIO(result))
            base = os.path.realpath(new_dir)
            for member in zf.namelist():
                target = os.path.realpath(os.path.join(base, member))
                if not target.startswith(base + os.sep):
                    logger.warning("Skipping unsafe path %r in output of %s/%s", member, dcr_id, node_name)
                    continue
                zf.extract(member, base)
        files_dir = os.path.join(node_dir, "files")
        old_dir = f"{files_dir}.old.{uuid.uuid4().hex}"
        with _file_lock(os.path.join(node_dir, ".lock")):
            if os.path.isdir(files_dir):
                os.replace(files_dir, old_dir)
            os.replace(new_dir, files_dir)
            state = _read_json(state_path, {})
            state.update({
                "status": "succeeded", "finished_at": _now(), "error": None,
                "result_run_by": email, "result_generated_at": _now(),
                "files": _list_files(files_dir),
            })
            _write_json(state_path, state)
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info("DCR node run succeeded: %s/%s by %s", dcr_id, node_name, email)
    except Exception as exc:
        logger.warning("DCR node run failed: %s/%s by %s: %s", dcr_id, node_name, email, exc)
        shutil.rmtree(new_dir, ignore_errors=True)
        with _file_lock(os.path.join(node_dir, ".lock")):
            state = _read_json(state_path, {})
            state.update({"status": "failed", "finished_at": _now(), "error": f"{type(exc).__name__}: {exc}"})
            _write_json(state_path, state)


@router.get("/my-dcrs/{dcr_id}/results", name="Compute nodes of a DCR with their latest results")
def list_dcr_results(dcr_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    email = _email(user)
    record = _user_dcr(email, dcr_id)
    nodes = []
    for name in _runnable_nodes(record, email):
        state = _run_state(dcr_id, name)
        nodes.append({
            "name": name,
            "status": state.get("status") or "never_run",
            "started_at": state.get("started_at"),
            "started_by": state.get("started_by"),
            "finished_at": state.get("finished_at"),
            "error": state.get("error"),
            "result_generated_at": state.get("result_generated_at"),
            "result_run_by": state.get("result_run_by"),
            "files": state.get("files") or [],
        })
    return {"dcr_id": dcr_id, "cohorts": _dcr_cohorts(record), "nodes": nodes}


@router.post("/my-dcrs/{dcr_id}/nodes/{node_name}/run", name="Run or re-run a DCR compute node")
def run_dcr_node(dcr_id: str, node_name: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    email = _email(user)
    _require_node(email, dcr_id, node_name)
    node_dir = _node_dir(dcr_id, node_name)
    os.makedirs(node_dir, exist_ok=True)
    with _file_lock(os.path.join(node_dir, ".lock")):
        if _run_state(dcr_id, node_name).get("status") == "running":
            raise HTTPException(status_code=409, detail="This node is already running")
        state_path = os.path.join(node_dir, "run.json")
        state = _read_json(state_path, {})
        state.update({"status": "running", "started_at": _now(), "started_by": email,
                      "finished_at": None, "error": None})
        _write_json(state_path, state)
    threading.Thread(target=_execute_run, args=(dcr_id, node_name, email), daemon=True).start()
    return {"status": "running", "started_at": state["started_at"]}


def _result_file(dcr_id: str, node_name: str, path: str) -> str:
    base = os.path.realpath(os.path.join(_node_dir(dcr_id, node_name), "files"))
    full = os.path.realpath(os.path.join(base, path))
    if not full.startswith(base + os.sep) or not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    return full


def _serve(full: str, filename: str) -> FileResponse:
    # Always an attachment with nosniff: the page fetches the bytes and renders
    # them itself (HTML only inside a sandboxed iframe), so a result file never
    # runs as a page on the API's origin.
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(full, media_type=media_type, filename=filename,
                        headers={"X-Content-Type-Options": "nosniff"})


@router.get("/my-dcrs/{dcr_id}/nodes/{node_name}/files/{path:path}", name="One output file of a DCR compute node")
def get_dcr_result_file(dcr_id: str, node_name: str, path: str, user: Any = Depends(get_current_user)):
    email = _email(user)
    _require_node(email, dcr_id, node_name)
    full = _result_file(dcr_id, node_name, path)
    return _serve(full, os.path.basename(full))


# ---------------------------------------------------------------------------
# Results Gallery
# ---------------------------------------------------------------------------

def _gallery_dir() -> str:
    return os.path.join(settings.data_folder, "results_gallery")


def _gallery_items() -> list[dict[str, Any]]:
    items = _read_json(os.path.join(_gallery_dir(), "items.json"), [])
    return items if isinstance(items, list) else []


def _clean(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


@router.post("/results-gallery", name="Share a DCR result file to the Results Gallery")
def share_result(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict[str, Any]:
    email = _email(user)
    dcr_id = str(body.get("dcr_id") or "")
    node_name = str(body.get("node_name") or "")
    path = str(body.get("file_path") or "")
    record = _require_node(email, dcr_id, node_name)
    src = _result_file(dcr_id, node_name, path)

    title = _clean(body.get("title"), MAX_TITLE)
    description = _clean(body.get("description"), MAX_DESCRIPTION)
    sharer_name = _clean(body.get("sharer_name"), MAX_NAME)
    if not title or not description or not sharer_name:
        raise HTTPException(status_code=400, detail="Title, description and your name are required")
    dcr_cohorts = _dcr_cohorts(record)
    requested = body.get("cohorts")
    cohorts = [c for c in requested if c in dcr_cohorts] if isinstance(requested, list) else dcr_cohorts

    state = _run_state(dcr_id, node_name)
    item_id = uuid.uuid4().hex
    file_name = os.path.basename(src)
    dest_dir = os.path.join(_gallery_dir(), "files", item_id)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_dir, file_name))
    item = {
        "id": item_id,
        "title": title,
        "description": description,
        "details": _clean(body.get("details"), MAX_DETAILS),
        "sharer_name": sharer_name,
        "shared_by": email,
        "shared_at": _now(),
        "dcr_id": dcr_id,
        "dcr_title": record.get("title") or "",
        "dcr_created_at": record.get("createdAt"),
        "cohorts": cohorts,
        "node_name": node_name,
        "file_path": path,
        "file_name": file_name,
        "size": os.path.getsize(src),
        "result_generated_at": state.get("result_generated_at"),
    }
    with _file_lock(os.path.join(_gallery_dir(), ".lock")):
        items = _gallery_items()
        items.append(item)
        _write_json(os.path.join(_gallery_dir(), "items.json"), items)
    logger.info("Result shared to gallery: %s (%s/%s/%s) by %s", item_id, dcr_id, node_name, path, email)
    return item


@router.get("/results-gallery", name="List the shared results (any logged-in user)")
def list_gallery(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    email = _email(user)
    items = sorted(_gallery_items(), key=lambda i: str(i.get("shared_at") or ""), reverse=True)
    admin = _is_admin(email)
    for item in items:
        item["can_delete"] = admin or item.get("shared_by") == email
    return {"items": items}


def _gallery_item(item_id: str) -> dict[str, Any]:
    for item in _gallery_items():
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail="Shared result not found")


@router.get("/results-gallery/{item_id}/file", name="The file of a shared result (any logged-in user)")
def get_gallery_file(item_id: str, user: Any = Depends(get_current_user)):
    _email(user)
    item = _gallery_item(item_id)
    full = os.path.join(_gallery_dir(), "files", os.path.basename(item["id"]), os.path.basename(item["file_name"]))
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    return _serve(full, item["file_name"])


@router.delete("/results-gallery/{item_id}", name="Remove a shared result (its sharer or an admin)")
def delete_gallery_item(item_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    email = _email(user)
    with _file_lock(os.path.join(_gallery_dir(), ".lock")):
        items = _gallery_items()
        item = next((i for i in items if i.get("id") == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Shared result not found")
        if not (_is_admin(email) or item.get("shared_by") == email):
            raise HTTPException(status_code=403, detail="Only the sharer or an admin can remove this result")
        _write_json(os.path.join(_gallery_dir(), "items.json"), [i for i in items if i.get("id") != item_id])
    shutil.rmtree(os.path.join(_gallery_dir(), "files", os.path.basename(item_id)), ignore_errors=True)
    logger.info("Shared result %s removed by %s", item_id, email)
    return {"deleted": item_id}
