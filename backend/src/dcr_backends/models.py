"""Serializable, provider-neutral DCR response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DcrResponseModel(BaseModel):
    """Base model that emits the JSON keys used by the existing frontend."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class DcrCapabilities(DcrResponseModel):
    supports_provisioning: bool
    supports_definition_preview: bool
    supports_live_creation: bool
    supports_room_refresh: bool
    supports_audit_log: bool
    supports_computation_output: bool
    supports_shuffle_output: bool
    synthetic_data_only: bool = False
    local_simulation: bool = False


class ProviderError(DcrResponseModel):
    detail: str
    provider: str
    failed_step: str | None = None
    dcr_id: str | None = None
    retryable: bool = False
    status_code: int | None = None


class DcrRoom(DcrResponseModel):
    id: str | None = None
    title: str | None = None
    description: str | None = None
    created_at: datetime | str | None = Field(default=None, alias="createdAt")
    owner: dict[str, Any] | None = None
    participants: list[dict[str, Any]] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    cohorts: list[str] = Field(default_factory=list)
    provisioned_datasets: list[dict[str, Any]] = Field(default_factory=list)
    dcr_url: str | None = None
    provider: str
    capabilities: DcrCapabilities
    error: ProviderError | str | None = None


class DcrListResult(DcrResponseModel):
    dcrs: list[DcrRoom | dict[str, Any]] = Field(default_factory=list)
    count: int
    email: str
    provider: str
    capabilities: DcrCapabilities


class LiveCreateResult(DcrResponseModel):
    message: str
    dcr_id: str
    dcr_url: str
    dcr_title: str
    cohort_ids: list[str]
    num_cohorts: int
    metadata_upload_results: dict[str, str] = Field(default_factory=dict)
    metadata_uploads_successful: int = 0
    shuffled_upload_results: dict[str, str] = Field(default_factory=dict)
    shuffled_uploads_successful: int = 0
    mapping_upload_results: dict[str, str] = Field(default_factory=dict)
    mapping_uploads_successful: int = 0
    row_upload_results: dict[str, str] = Field(default_factory=dict)
    row_uploads_successful: int = 0
    participants: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    provider: str
    capabilities: DcrCapabilities
    handoff_mode: str | None = None
    environment: str | None = None
    data_node_ids: dict[str, str] | None = None
    merge_request_id: str | None = None
    aggregate_computation_node_id: str | None = None
