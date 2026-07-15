"""AADCR v2 adapter for Cohort Explorer's provider-neutral DCR contract."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import Response

from src.config import Settings
from src.dcr_backends.aadcr_client import AadcrClient
from src.dcr_backends.aadcr_translation import (
    DcrOperationError,
    build_room_plan,
    create_aadcr_room,
    participants_response,
)
from src.dcr_backends.definition_archive import build_definition_archive
from src.dcr_backends.factory import ADAPTERS
from src.dcr_backends.models import DcrListResult, LiveCreateResult

ClientFactory = Callable[[Settings, dict[str, Any]], Any]
CohortLoader = Callable[[str], dict[str, Any]]


class AadcrBackend:
    """Translate existing Cohort Explorer operations to native AADCR HTTP calls."""

    provider_name = "aadcrv2"

    def __init__(
        self,
        app_settings: Settings,
        *,
        client_factory: ClientFactory = AadcrClient,
        cohort_loader: CohortLoader | None = None,
        merge_poll_attempts: int | None = None,
        merge_poll_interval: float = 0.25,
    ):
        self.settings = app_settings
        self.capabilities = ADAPTERS[self.provider_name].capabilities.model_copy(deep=True)
        self._client_factory = client_factory
        self._cohort_loader = cohort_loader
        self._merge_poll_interval = merge_poll_interval
        self._merge_poll_attempts = (
            max(
                1,
                math.ceil(app_settings.aadcrv2_timeout_seconds / max(merge_poll_interval, 0.25)),
            )
            if merge_poll_attempts is None
            else merge_poll_attempts
        )

    def _load_cohorts(self, email: str) -> dict[str, Any]:
        if self._cohort_loader is not None:
            return self._cohort_loader(email)
        from src.cohort_cache import get_cohorts_from_cache

        return get_cohorts_from_cache(email)

    def _build_plan(self, request: dict[str, Any], user: dict[str, Any]):
        email = str(user.get("email") or "")
        return build_room_plan(request, user, self._load_cohorts(email), self.settings)

    async def create_provision_room(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> dict[str, Any]:
        cohort_id = str(request.get("cohort_id") or "").strip()
        if not cohort_id:
            raise DcrOperationError(detail="A cohort ID is required", failed_step="validate request")
        user_email = str(user.get("email") or "")
        cohort = self._load_cohorts(user_email).get(cohort_id)
        if cohort is None or not bool(getattr(cohort, "can_edit", False)):
            raise DcrOperationError(
                detail=f"User {user_email} cannot publish cohort {cohort_id}",
                failed_step="authorize provision room",
                status_code=403,
            )
        result = await self.create_live_room(
            {
                "cohorts": {cohort_id: []},
                "dcr_name": f"{cohort_id} provisioning",
                "include_shuffled_samples": False,
            },
            user,
        )
        return result.to_dict()

    async def preview_definition(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> Response:
        archive = build_definition_archive(self._build_plan(request, user))
        return Response(
            content=archive,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="dcr_config_package.zip"'},
        )

    async def create_live_room(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> LiveCreateResult:
        plan = self._build_plan(request, user)
        async with self._client_factory(self.settings, user) as client:
            outcome = await create_aadcr_room(
                client,
                plan,
                merge_poll_attempts=self._merge_poll_attempts,
                merge_poll_interval=self._merge_poll_interval,
            )
        dcr_url = self.settings.aadcrv2_room_url_template.format(dcr_id=outcome.dcr_id)
        return LiveCreateResult(
            message="AADCR v2 room created, merged, and provisioned",
            dcr_id=outcome.dcr_id,
            dcr_url=dcr_url,
            dcr_title=plan.title,
            cohort_ids=plan.cohort_ids,
            num_cohorts=len(plan.cohort_ids),
            metadata_upload_results=outcome.metadata_upload_results,
            metadata_uploads_successful=sum(status == "success" for status in outcome.metadata_upload_results.values()),
            shuffled_upload_results=outcome.shuffled_upload_results,
            shuffled_uploads_successful=sum(status == "success" for status in outcome.shuffled_upload_results.values()),
            mapping_upload_results=outcome.mapping_upload_results,
            mapping_uploads_successful=sum(status == "success" for status in outcome.mapping_upload_results.values()),
            participants=participants_response(plan),
            provider=self.provider_name,
            capabilities=self.capabilities,
            merge_request_id=outcome.merge_request_id,
            row_upload_results=outcome.row_upload_results,
            row_uploads_successful=sum(status == "success" for status in outcome.row_upload_results.values()),
            aggregate_computation_node_id=outcome.prod_node_ids.computation_nodes["aggregate-summary-local-simulation"],
        )

    async def list_rooms(
        self,
        user: dict[str, Any],
        refresh: bool = False,
    ) -> DcrListResult:
        raise DcrOperationError(
            detail="AADCR room reads are implemented by the resumable read adapter",
            failed_step="list rooms",
        )

    async def rooms_last_modified(self, user: dict[str, Any]) -> datetime | None:
        journal = Path(self.settings.aadcrv2_operation_journal)
        if not journal.exists():
            return None
        return datetime.fromtimestamp(journal.stat().st_mtime).astimezone()

    async def audit_log(
        self,
        dcr_id: str,
        user: dict[str, Any],
        main_only: bool = True,
    ) -> list[dict[str, Any]]:
        raise DcrOperationError(
            detail="AADCR audit reads are implemented by the resumable read adapter",
            failed_step="read audit log",
            dcr_id=dcr_id,
        )

    async def computation_output(
        self,
        dcr_id: str,
        user: dict[str, Any],
    ) -> Any:
        raise DcrOperationError(
            detail="AADCR computation reads are implemented by the resumable read adapter",
            failed_step="read computation output",
            dcr_id=dcr_id,
        )

    async def shuffle_output(
        self,
        dcr_id: str,
        user: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "detail": "AADCR v2 does not expose the Decentriq shuffle computation",
            "dcr_id": dcr_id,
            "provider": self.provider_name,
        }
