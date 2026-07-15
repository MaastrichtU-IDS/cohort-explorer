"""Select a DCR adapter without importing unselected provider SDKs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.config import Settings, settings
from src.dcr_backends.contracts import (
    AuditEntry,
    ComputeRoomRequest,
    CurrentUser,
    DcrBackend,
    ProvisionRoomRequest,
    ResponsePayload,
)
from src.dcr_backends.models import DcrCapabilities, DcrListResult, LiveCreateResult


@dataclass(frozen=True)
class AdapterSpec:
    module: str
    class_name: str
    capabilities: DcrCapabilities


ADAPTERS = {
    "decentriq": AdapterSpec(
        module="src.dcr_backends.decentriq_backend",
        class_name="DecentriqBackend",
        capabilities=DcrCapabilities(
            supports_provisioning=True,
            supports_definition_preview=True,
            supports_live_creation=True,
            supports_room_refresh=True,
            supports_audit_log=True,
            supports_computation_output=True,
            supports_shuffle_output=True,
        ),
    ),
    "aadcrv2": AdapterSpec(
        module="src.dcr_backends.aadcr_backend",
        class_name="AadcrBackend",
        capabilities=DcrCapabilities(
            supports_provisioning=True,
            supports_definition_preview=True,
            supports_live_creation=True,
            supports_room_refresh=True,
            supports_audit_log=True,
            supports_computation_output=True,
            supports_shuffle_output=False,
            synthetic_data_only=True,
            local_simulation=True,
        ),
    ),
}


class LazyDcrBackend:
    """Resolve the selected adapter only when an operation is first invoked."""

    def __init__(self, app_settings: Settings, provider_name: str, spec: AdapterSpec):
        self.settings = app_settings
        self.provider_name = provider_name
        self.capabilities = spec.capabilities
        self._spec = spec
        self._resolved: DcrBackend | None = None

    def _resolve(self) -> DcrBackend:
        if self._resolved is None:
            module = importlib.import_module(self._spec.module)
            adapter_class = getattr(module, self._spec.class_name)
            self._resolved = adapter_class(self.settings)
        return self._resolved

    async def create_provision_room(
        self,
        request: ProvisionRoomRequest,
        user: CurrentUser,
    ) -> dict[str, Any]:
        return await self._resolve().create_provision_room(request, user)

    async def preview_definition(
        self,
        request: ComputeRoomRequest,
        user: CurrentUser,
    ) -> ResponsePayload:
        return await self._resolve().preview_definition(request, user)

    async def create_live_room(
        self,
        request: ComputeRoomRequest,
        user: CurrentUser,
    ) -> LiveCreateResult:
        return await self._resolve().create_live_room(request, user)

    async def list_rooms(
        self,
        user: CurrentUser,
        refresh: bool = False,
    ) -> DcrListResult:
        return await self._resolve().list_rooms(user, refresh=refresh)

    async def rooms_last_modified(self, user: CurrentUser) -> datetime | None:
        return await self._resolve().rooms_last_modified(user)

    async def audit_log(
        self,
        dcr_id: str,
        user: CurrentUser,
        main_only: bool = True,
    ) -> list[AuditEntry]:
        return await self._resolve().audit_log(dcr_id, user, main_only=main_only)

    async def computation_output(self, dcr_id: str, user: CurrentUser) -> ResponsePayload:
        return await self._resolve().computation_output(dcr_id, user)

    async def shuffle_output(self, dcr_id: str, user: CurrentUser) -> ResponsePayload:
        return await self._resolve().shuffle_output(dcr_id, user)


def get_dcr_backend(app_settings: Settings | None = None) -> DcrBackend:
    """Return a lazy adapter for the configured provider, failing closed."""
    configured = app_settings or settings
    provider_name = configured.dcr_backend.strip().lower()
    spec = ADAPTERS.get(provider_name)
    if spec is None:
        raise ValueError(f"Unsupported DCR_BACKEND: {configured.dcr_backend}")
    return LazyDcrBackend(configured, provider_name, spec)
