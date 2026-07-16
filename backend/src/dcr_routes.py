"""Provider-neutral routes for Cohort Explorer DCR operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form

from src.auth import get_current_user
from src.dcr_backends.contracts import DcrBackend
from src.dcr_backends.factory import get_dcr_backend

router = APIRouter()


def get_backend() -> DcrBackend:
    """FastAPI dependency returning the configured lazy backend."""
    return get_dcr_backend()


def _serialize(payload: Any) -> Any:
    to_dict = getattr(payload, "to_dict", None)
    return to_dict() if callable(to_dict) else payload


@router.get("/api/dcr/provider")
async def provider_contract(
    _user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> dict[str, Any]:
    """Return the configured provider before a user starts a creation flow."""
    return {
        "provider": backend.provider_name,
        "capabilities": _serialize(backend.capabilities),
    }


@router.post("/create-provision-dcr")
async def create_provision_room(
    cohort_id: str = Form(...),
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.create_provision_room({"cohort_id": cohort_id}, user))


@router.post("/get-compute-dcr-definition")
async def preview_definition(
    request: dict[str, Any],
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.preview_definition(request, user))


@router.post("/create-live-compute-dcr")
async def create_live_room(
    request: dict[str, Any],
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.create_live_room(request, user))


@router.get("/dcr-log/{dcr_id}")
async def audit_log(
    dcr_id: str,
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.audit_log(dcr_id, user, main_only=False))


@router.get("/dcr-log-main/{dcr_id}")
async def main_audit_log(
    dcr_id: str,
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.audit_log(dcr_id, user, main_only=True))


@router.get("/compute-get-output/{dcr_id}")
async def computation_output(
    dcr_id: str,
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.computation_output(dcr_id, user))


@router.get("/shuffle-get-output/{dcr_id}")
async def shuffle_output(
    dcr_id: str,
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.shuffle_output(dcr_id, user))


@router.get("/my-dcrs")
async def list_rooms(
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.list_rooms(user, refresh=False))


@router.post("/my-dcrs/refresh")
async def refresh_rooms(
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> Any:
    return _serialize(await backend.list_rooms(user, refresh=True))


@router.get("/my-dcrs/last-modified")
async def rooms_last_modified(
    user: dict[str, Any] = Depends(get_current_user),
    backend: DcrBackend = Depends(get_backend),
) -> dict[str, str | None]:
    modified = await backend.rooms_last_modified(user)
    return {"last_modified": modified.isoformat() if modified else None}
