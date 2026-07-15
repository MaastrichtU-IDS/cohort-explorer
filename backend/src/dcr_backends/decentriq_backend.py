"""Adapter for the existing Decentriq implementation.

The legacy module remains the source of truth for Decentriq-specific behavior.
This adapter keeps those imports and SDK calls behind the provider contract so
selecting another backend does not initialize Decentriq clients.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from src.config import Settings
from src.dcr_backends.factory import ADAPTERS
from src.dcr_backends.models import DcrListResult, LiveCreateResult


class DecentriqBackend:
    """Expose the current Decentriq workflow through the DCR backend contract."""

    provider_name = "decentriq"

    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        self.capabilities = ADAPTERS[self.provider_name].capabilities.model_copy(deep=True)

    @staticmethod
    def _legacy_module():
        # Import lazily so an AADCR-only process never imports the Decentriq SDK.
        from src import decentriq

        return decentriq

    async def create_provision_room(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> dict[str, Any]:
        from src.cohort_cache import get_cohorts_from_cache

        cohort_id = str(request.get("cohort_id", ""))
        user_email = user.get("email")
        cohorts = get_cohorts_from_cache(user_email)
        cohort = cohorts.get(cohort_id)
        if cohort is None:
            raise HTTPException(status_code=403, detail=f"Cohort ID {cohort_id} does not exists")
        if not cohort.can_edit:
            raise HTTPException(
                status_code=403,
                detail=f"User {user_email} cannot publish cohort {cohort_id}",
            )

        legacy = self._legacy_module()
        try:
            return await asyncio.to_thread(legacy.create_provision_dcr, user, cohort)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"There was an issue when uploading the cohort {cohort_id} to Decentriq: {exc}",
            ) from exc

    async def preview_definition(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> Any:
        return await self._legacy_module().api_get_compute_dcr_definition(request, user)

    async def create_live_room(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
    ) -> LiveCreateResult:
        result = await self._legacy_module().api_create_live_compute_dcr(request, user)
        normalized = dict(result)
        normalized["provider"] = self.provider_name
        normalized["capabilities"] = self.capabilities
        return LiveCreateResult(**normalized)

    async def list_rooms(
        self,
        user: dict[str, Any],
        refresh: bool = False,
    ) -> DcrListResult:
        legacy = self._legacy_module()
        user_email = user.get("email")
        if not user_email:
            raise HTTPException(status_code=401, detail="Not authenticated")

        refresh_summary = None
        if refresh:
            # Seed the in-memory index from the persisted history first. This
            # preserves the incremental startup behavior of the legacy path
            # and prevents existing rooms from being appended a second time.
            await asyncio.to_thread(legacy.load_dcr_history_from_disk)
            refresh_summary = await asyncio.to_thread(legacy.refresh_dcrs_in_memory_only)
        records = await asyncio.to_thread(legacy.get_dcrs_for_participant, user_email)

        normalized_records: list[dict[str, Any]] = []
        for record in records:
            normalized = dict(record)
            normalized["provider"] = self.provider_name
            normalized["capabilities"] = self.capabilities.to_dict()
            if normalized.get("id") and not normalized.get("dcr_url"):
                normalized["dcr_url"] = (
                    "https://platform.decentriq.com/datarooms/p/" + str(normalized["id"])
                )
            normalized_records.append(normalized)

        return DcrListResult(
            dcrs=normalized_records,
            count=len(normalized_records),
            email=user_email,
            provider=self.provider_name,
            capabilities=self.capabilities,
            refresh_summary=refresh_summary,
        )

    async def rooms_last_modified(self, user: dict[str, Any]) -> datetime | None:
        if not user.get("email"):
            raise HTTPException(status_code=401, detail="Not authenticated")
        legacy = self._legacy_module()
        try:
            timestamp = await asyncio.to_thread(
                legacy.os.path.getmtime,
                legacy._dcr_history_path(),
            )
        except FileNotFoundError:
            return None
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get last modified: {exc}",
            ) from exc
        return datetime.fromtimestamp(timestamp, timezone.utc)

    async def audit_log(
        self,
        dcr_id: str,
        user: dict[str, Any],
        main_only: bool = True,
    ) -> list[dict[str, Any]]:
        legacy = self._legacy_module()
        log_reader = legacy.get_dcr_log_main if main_only else legacy.get_dcr_log
        return await asyncio.to_thread(log_reader, dcr_id, user)

    async def computation_output(
        self,
        dcr_id: str,
        user: dict[str, Any],
    ) -> Any:
        return await asyncio.to_thread(
            self._legacy_module().run_computation_get_output,
            dcr_id,
            user,
        )

    async def shuffle_output(
        self,
        dcr_id: str,
        user: dict[str, Any],
    ) -> Any:
        return await asyncio.to_thread(
            self._legacy_module().run_shuffle_get_output,
            dcr_id,
            user,
        )
