"""The application-facing contract implemented by every DCR provider."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from src.dcr_backends.models import DcrCapabilities, DcrListResult, LiveCreateResult

CurrentUser = dict[str, Any]
ProvisionRoomRequest = dict[str, Any]
ComputeRoomRequest = dict[str, Any]
AuditEntry = dict[str, Any]
ResponsePayload = Any


@runtime_checkable
class DcrBackend(Protocol):
    """Operations exposed through Cohort Explorer's existing DCR routes."""

    provider_name: str
    capabilities: DcrCapabilities

    async def create_provision_room(
        self,
        request: ProvisionRoomRequest,
        user: CurrentUser,
    ) -> dict[str, Any]: ...

    async def preview_definition(
        self,
        request: ComputeRoomRequest,
        user: CurrentUser,
    ) -> ResponsePayload: ...

    async def create_live_room(
        self,
        request: ComputeRoomRequest,
        user: CurrentUser,
    ) -> LiveCreateResult: ...

    async def list_rooms(
        self,
        user: CurrentUser,
        refresh: bool = False,
    ) -> DcrListResult: ...

    async def rooms_last_modified(self, user: CurrentUser) -> datetime | None: ...

    async def audit_log(
        self,
        dcr_id: str,
        user: CurrentUser,
        main_only: bool = True,
    ) -> list[AuditEntry]: ...

    async def computation_output(
        self,
        dcr_id: str,
        user: CurrentUser,
    ) -> ResponsePayload: ...

    async def shuffle_output(
        self,
        dcr_id: str,
        user: CurrentUser,
    ) -> ResponsePayload: ...
