"""AADCR v2 adapter for Cohort Explorer's provider-neutral DCR contract."""

from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import Response

from src.config import Settings
from src.dcr_backends.aadcr_client import AadcrClient, AadcrUpstreamError
from src.dcr_backends.aadcr_translation import (
    AGGREGATE_NODE_NAME,
    DcrOperationError,
    build_room_plan,
    create_aadcr_room,
    decode_result_archive,
    normalize_aadcr_audit,
    normalize_aadcr_room,
    participants_response,
    resolve_computation_node_id,
)
from src.dcr_backends.definition_archive import build_definition_archive
from src.dcr_backends.factory import ADAPTERS
from src.dcr_backends.models import DcrListResult, LiveCreateResult
from src.dcr_backends.operation_journal import (
    JournalCorruptionError,
    OperationJournal,
    normalize_request_metadata,
    request_fingerprint,
    validate_session_id,
)

ClientFactory = Callable[[Settings, dict[str, Any]], Any]
CohortLoader = Callable[[str], dict[str, Any]]
_NO_COMPUTATION_RESULT = object()


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
        result_poll_attempts: int | None = None,
        result_poll_interval: float = 0.25,
        result_archive_max_bytes: int = 10 * 1024 * 1024,
        result_archive_max_members: int = 64,
    ):
        if result_poll_attempts is not None and result_poll_attempts < 0:
            raise ValueError("result_poll_attempts cannot be negative")
        if result_archive_max_bytes <= 0 or result_archive_max_members <= 0:
            raise ValueError("result archive limits must be positive")
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
        self._result_poll_interval = result_poll_interval
        self._result_poll_attempts = (
            max(
                1,
                math.ceil(app_settings.aadcrv2_timeout_seconds / max(result_poll_interval, 0.25)),
            )
            if result_poll_attempts is None
            else result_poll_attempts
        )
        self._result_archive_max_bytes = result_archive_max_bytes
        self._result_archive_max_members = result_archive_max_members
        configured_journal = Path(app_settings.aadcrv2_operation_journal).expanduser()
        if not configured_journal.is_absolute():
            configured_journal = Path(app_settings.data_folder).expanduser() / configured_journal.name
        self._journal = OperationJournal(configured_journal.resolve())

    def _load_cohorts(self, email: str) -> dict[str, Any]:
        if self._cohort_loader is not None:
            return self._cohort_loader(email)
        from src.cohort_cache import get_cohorts_from_cache

        return get_cohorts_from_cache(email)

    def _build_plan(self, request: dict[str, Any], user: dict[str, Any]):
        email = str(user.get("email") or "")
        return build_room_plan(request, user, self._load_cohorts(email), self.settings)

    @staticmethod
    def _require_email(user: dict[str, Any]) -> str:
        email = str(user.get("email") or "").strip().lower()
        if not email:
            raise DcrOperationError(
                detail="Not authenticated",
                failed_step="authenticate request",
                status_code=401,
            )
        return email

    @staticmethod
    def _safe_dcr_id(value: object, *, failed_step: str) -> str:
        dcr_id = str(value or "").strip()
        if (
            not dcr_id
            or len(dcr_id) > 255
            or dcr_id in {".", ".."}
            or any(character in dcr_id for character in "/\\?#")
            or any(ord(character) < 32 for character in dcr_id)
        ):
            raise DcrOperationError(
                detail="AADCR returned an invalid room identifier",
                failed_step=failed_step,
            )
        return dcr_id

    @staticmethod
    def _from_upstream(exc: AadcrUpstreamError, *, dcr_id: str | None = None) -> DcrOperationError:
        return DcrOperationError(
            detail=exc.detail,
            failed_step=exc.failed_step,
            dcr_id=dcr_id,
            retryable=exc.retryable,
            status_code=exc.status_code,
        )

    def _live_result(self, plan: Any, outcome: Any) -> LiveCreateResult:
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
            aggregate_computation_node_id=outcome.prod_node_ids.computation_nodes[AGGREGATE_NODE_NAME],
        )

    def _journal_metadata(
        self,
        request: dict[str, Any],
        user: dict[str, Any],
        plan: Any,
    ) -> dict[str, Any]:
        return normalize_request_metadata(
            {
                "dcr_name": str(request.get("dcr_name") or "").strip(),
                "research_question": str(request.get("research_question") or "").strip(),
                "creator_email": self._require_email(user),
                "selected_variables": plan.selected_variables,
                "include_shuffled_samples": request.get("include_shuffled_samples", True),
                "include_mapping_upload_slot": bool(request.get("include_mapping_upload_slot", False)),
                "selected_mapping_files": request.get("selected_mapping_files", []),
                "additional_analysts": request.get("additional_analysts", []),
                "excluded_data_owners": request.get("excluded_data_owners", []),
                "synthetic_demo": bool(plan.synthetic_demo),
            }
        )

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
        raw_session_id = request.get("session_id")
        if raw_session_id is None:
            async with self._client_factory(self.settings, user) as client:
                outcome = await create_aadcr_room(
                    client,
                    plan,
                    merge_poll_attempts=self._merge_poll_attempts,
                    merge_poll_interval=self._merge_poll_interval,
                )
            return self._live_result(plan, outcome)

        try:
            session_id = validate_session_id(raw_session_id)
        except ValueError as exc:
            raise DcrOperationError(
                detail=str(exc),
                failed_step="validate session",
            ) from None
        metadata = self._journal_metadata(request, user, plan)
        fingerprint = request_fingerprint(metadata, plan.cohort_ids)

        async with self._journal.session_lock(session_id):
            try:
                state = self._journal.load(session_id)
            except (JournalCorruptionError, OSError, ValueError):
                raise DcrOperationError(
                    detail="The AADCR operation journal could not be read safely",
                    failed_step="resume operation",
                    status_code=500,
                ) from None
            if state is not None and state.request_fingerprint != fingerprint:
                raise DcrOperationError(
                    detail="This session_id is already associated with a different room request",
                    failed_step="resume operation",
                    dcr_id=state.dcr_id,
                    status_code=409,
                )
            if state is not None and "completed" in state.confirmed_steps:
                if state.final_response is None:
                    raise DcrOperationError(
                        detail="The completed operation journal entry omitted its final response",
                        failed_step="resume operation",
                        dcr_id=state.dcr_id,
                        status_code=500,
                    )
                return LiveCreateResult(**state.final_response)
            if state is None:
                try:
                    state = self._journal.append(
                        session_id,
                        request_metadata=metadata,
                        request_fingerprint=fingerprint,
                        cohort_ids=plan.cohort_ids,
                        dcr_id=None,
                        confirmed_steps=[],
                        final_response=None,
                    )
                except (OSError, ValueError):
                    raise DcrOperationError(
                        detail="The AADCR operation journal could not be updated safely",
                        failed_step="start operation",
                        status_code=500,
                    ) from None

            async def confirm_step(step: str, dcr_id: str) -> None:
                nonlocal state
                confirmed = [*state.confirmed_steps]
                if step not in confirmed:
                    confirmed.append(step)
                try:
                    state = self._journal.append(
                        session_id,
                        request_metadata=metadata,
                        request_fingerprint=fingerprint,
                        cohort_ids=plan.cohort_ids,
                        dcr_id=dcr_id,
                        confirmed_steps=confirmed,
                        final_response=None,
                    )
                except (OSError, ValueError):
                    raise DcrOperationError(
                        detail="The AADCR operation journal could not confirm progress safely",
                        failed_step="record operation progress",
                        dcr_id=dcr_id,
                        status_code=500,
                    ) from None

            async with self._client_factory(self.settings, user) as client:
                outcome = await create_aadcr_room(
                    client,
                    plan,
                    merge_poll_attempts=self._merge_poll_attempts,
                    merge_poll_interval=self._merge_poll_interval,
                    resume_dcr_id=state.dcr_id,
                    confirmed_steps=set(state.confirmed_steps),
                    on_confirm=confirm_step,
                    creation_key=hashlib.sha256(f"{session_id}\0{fingerprint}".encode()).hexdigest(),
                    expected_creator_email=self._require_email(user),
                )
            result = self._live_result(plan, outcome)
            completed_steps = [*state.confirmed_steps]
            if "completed" not in completed_steps:
                completed_steps.append("completed")
            try:
                self._journal.append(
                    session_id,
                    request_metadata=metadata,
                    request_fingerprint=fingerprint,
                    cohort_ids=plan.cohort_ids,
                    dcr_id=outcome.dcr_id,
                    confirmed_steps=completed_steps,
                    final_response=result.to_dict(),
                )
            except (OSError, ValueError):
                raise DcrOperationError(
                    detail="The AADCR operation journal could not persist the completed response",
                    failed_step="complete operation",
                    dcr_id=outcome.dcr_id,
                    status_code=500,
                ) from None
            return result

    async def list_rooms(
        self,
        user: dict[str, Any],
        refresh: bool = False,
    ) -> DcrListResult:
        email = self._require_email(user)
        try:
            async with self._client_factory(self.settings, user) as client:
                native_rooms = await client.request_json(
                    "GET",
                    "/api/dcr/",
                    failed_step="list rooms",
                )
                if not isinstance(native_rooms, list):
                    raise DcrOperationError(
                        detail="AADCR room list was not an array",
                        failed_step="list rooms",
                    )
                normalized_rooms: list[dict[str, Any]] = []
                for native_room in native_rooms:
                    if not isinstance(native_room, dict):
                        raise DcrOperationError(
                            detail="AADCR room list contained an invalid entry",
                            failed_step="list rooms",
                        )
                    dcr_id = self._safe_dcr_id(
                        native_room.get("id"),
                        failed_step="list rooms",
                    )
                    prod_view = await client.request_json(
                        "GET",
                        f"/api/dcr/{dcr_id}/prod/view",
                        failed_step="read PROD view",
                    )
                    provisioned = await client.request_json(
                        "GET",
                        f"/api/dcr/{dcr_id}/provisioned-datasets",
                        failed_step="read provisioned datasets",
                    )
                    try:
                        journal_state = self._journal.find_by_dcr_id(dcr_id)
                    except (JournalCorruptionError, OSError, ValueError):
                        raise DcrOperationError(
                            detail="The AADCR operation journal could not be read safely",
                            failed_step="normalize room",
                            dcr_id=dcr_id,
                            status_code=500,
                        ) from None
                    normalized_rooms.append(
                        normalize_aadcr_room(
                            native_room,
                            prod_view,
                            provisioned,
                            journal_state,
                            dcr_url=self.settings.aadcrv2_room_url_template.format(dcr_id=dcr_id),
                            capabilities=self.capabilities.to_dict(),
                        )
                    )
        except DcrOperationError:
            raise
        except AadcrUpstreamError as exc:
            raise self._from_upstream(exc) from None
        return DcrListResult(
            dcrs=normalized_rooms,
            count=len(normalized_rooms),
            email=email,
            provider=self.provider_name,
            capabilities=self.capabilities,
            refresh_summary={"refreshed": bool(refresh), "count": len(normalized_rooms)},
        )

    async def rooms_last_modified(self, user: dict[str, Any]) -> datetime | None:
        self._require_email(user)
        try:
            return self._journal.latest_timestamp()
        except (JournalCorruptionError, OSError, ValueError):
            raise DcrOperationError(
                detail="The AADCR operation journal could not be read safely",
                failed_step="read rooms last modified",
                status_code=500,
            ) from None

    async def audit_log(
        self,
        dcr_id: str,
        user: dict[str, Any],
        main_only: bool = True,
    ) -> list[dict[str, Any]]:
        self._require_email(user)
        safe_dcr_id = self._safe_dcr_id(dcr_id, failed_step="read audit log")
        try:
            async with self._client_factory(self.settings, user) as client:
                payload = await client.request_json(
                    "GET",
                    f"/api/dcr/{safe_dcr_id}/audit-logs",
                    failed_step="read audit log",
                )
            return normalize_aadcr_audit(
                payload,
                dcr_id=safe_dcr_id,
                main_only=main_only,
            )
        except DcrOperationError:
            raise
        except AadcrUpstreamError as exc:
            raise self._from_upstream(exc, dcr_id=safe_dcr_id) from None

    def _computation_terminal_response(
        self,
        payload: Any,
        *,
        dcr_id: str,
    ) -> Response | None:
        if not isinstance(payload, dict):
            raise DcrOperationError(
                detail="AADCR computation result was not an object",
                failed_step="poll computation result",
                dcr_id=dcr_id,
            )
        status = str(payload.get("status") or "").upper()
        if status in {"QUEUED", "RUNNING"}:
            return None
        if status == "COMPLETED":
            archive = decode_result_archive(
                payload,
                dcr_id=dcr_id,
                max_archive_bytes=self._result_archive_max_bytes,
                max_members=self._result_archive_max_members,
            )
            return Response(
                content=archive,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="aggregate-result.zip"'},
            )
        if status in {"FAILED", "CANCELLED"}:
            raise DcrOperationError(
                detail="The aggregate computation did not complete successfully",
                failed_step="run aggregate computation",
                dcr_id=dcr_id,
            )
        raise DcrOperationError(
            detail="AADCR returned an unknown aggregate computation status",
            failed_step="poll computation result",
            dcr_id=dcr_id,
        )

    async def computation_output(
        self,
        dcr_id: str,
        user: dict[str, Any],
    ) -> Any:
        self._require_email(user)
        safe_dcr_id = self._safe_dcr_id(
            dcr_id,
            failed_step="read computation output",
        )
        try:
            async with self._client_factory(self.settings, user) as client:
                prod_view = await client.request_json(
                    "GET",
                    f"/api/dcr/{safe_dcr_id}/prod/view",
                    failed_step="read PROD view",
                )
                computation_node_id = resolve_computation_node_id(
                    prod_view,
                    AGGREGATE_NODE_NAME,
                    safe_dcr_id,
                )
                request_body = {
                    "computationNodeId": computation_node_id,
                    "environment": "PROD",
                }
                results_path = f"/api/dcr/{safe_dcr_id}/computation-nodes/results"
                run_path = f"/api/dcr/{safe_dcr_id}/computation-nodes/run"
                initial_result: Any = _NO_COMPUTATION_RESULT
                try:
                    initial_result = await client.request_json(
                        "POST",
                        results_path,
                        failed_step="read computation result",
                        json_body=request_body,
                    )
                except AadcrUpstreamError as exc:
                    no_execution = exc.status_code == 404 and "no execution found" in exc.detail.casefold()
                    if not no_execution:
                        raise
                    started = await client.request_json(
                        "POST",
                        run_path,
                        failed_step="start aggregate computation",
                        json_body=request_body,
                    )
                    if not isinstance(started, dict) or str(started.get("status") or "").upper() != "QUEUED":
                        raise DcrOperationError(
                            detail="AADCR did not confirm that the aggregate computation was queued",
                            failed_step="start aggregate computation",
                            dcr_id=safe_dcr_id,
                        )

                result_calls = 0
                if initial_result is not _NO_COMPUTATION_RESULT:
                    result_calls = 1
                    completed = self._computation_terminal_response(
                        initial_result,
                        dcr_id=safe_dcr_id,
                    )
                    if completed is not None:
                        return completed

                while result_calls < self._result_poll_attempts:
                    if result_calls > 0 and self._result_poll_interval > 0:
                        await asyncio.sleep(self._result_poll_interval)
                    payload = await client.request_json(
                        "POST",
                        results_path,
                        failed_step="poll computation result",
                        json_body=request_body,
                    )
                    result_calls += 1
                    completed = self._computation_terminal_response(
                        payload,
                        dcr_id=safe_dcr_id,
                    )
                    if completed is not None:
                        return completed
                raise DcrOperationError(
                    detail=(
                        "Aggregate computation did not reach a terminal state "
                        f"after {self._result_poll_attempts} result polls"
                    ),
                    failed_step="poll computation result",
                    dcr_id=safe_dcr_id,
                    retryable=True,
                )
        except DcrOperationError:
            raise
        except AadcrUpstreamError as exc:
            raise self._from_upstream(exc, dcr_id=safe_dcr_id) from None

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
