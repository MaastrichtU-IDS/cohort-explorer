"""Translate Cohort Explorer room requests to AADCR v2's native API graph."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import stat
import struct
import zipfile
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from fastapi import HTTPException

from src.config import Settings
from src.dcr_backends.aadcr_client import AadcrUpstreamError

METADATA_PREVIEW_NODE_NAME = "metadata-preview-local-simulation"
AGGREGATE_NODE_NAME = "aggregate-summary-local-simulation"
PROVIDER_NAME = "aadcrv2"

ConfirmationCallback = Callable[[str, str], Awaitable[None]]


class DcrOperationError(HTTPException):
    """Provider-normalized error that keeps the failed step and partial room ID."""

    def __init__(
        self,
        *,
        detail: str,
        failed_step: str,
        dcr_id: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ):
        self.safe_detail = detail
        self.failed_step = failed_step
        self.dcr_id = dcr_id
        self.retryable = retryable
        self.upstream_status_code = status_code
        response_status = status_code if status_code is not None else (503 if retryable else 422)
        room_suffix = f" for room {dcr_id}" if dcr_id else ""
        self.message = f"DCR operation failed during {failed_step}{room_suffix}: {detail}"
        super().__init__(status_code=response_status, detail=self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.safe_detail,
            "provider": PROVIDER_NAME,
            "failed_step": self.failed_step,
            "dcr_id": self.dcr_id,
            "retryable": self.retryable,
            "status_code": self.upstream_status_code,
        }

    def __str__(self) -> str:
        return self.message


@dataclass
class ParticipantRoles:
    """Provider-neutral node roles for one participant."""

    data_owner_of: set[str] = field(default_factory=set)
    analyst_of: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DataNodeSpec:
    name: str
    type: str = "FILE"


@dataclass(frozen=True)
class ComputationNodeSpec:
    name: str
    code: str
    data_dependencies: tuple[str, ...]
    computation_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssetSpec:
    kind: str
    key: str
    node_name: str
    path: Path
    archive_path: str | None


@dataclass(frozen=True)
class PermissionSpec:
    type: str
    email: str
    node_name: str


@dataclass
class RoomPlan:
    title: str
    description: str
    cohort_ids: list[str]
    selected_variables: dict[str, list[str]]
    participants: dict[str, ParticipantRoles]
    participant_emails: list[str]
    data_nodes: list[DataNodeSpec]
    computation_nodes: list[ComputationNodeSpec]
    permissions: list[PermissionSpec]
    assets: list[AssetSpec]
    synthetic_demo: bool

    def definition_config(self) -> dict[str, Any]:
        """Return the path-free, stable definition used by preview archives."""
        return {
            "dataScienceDataRoom": {
                "cohorts": self.cohort_ids,
                "computation_nodes": [
                    {
                        "code": node.code,
                        "computation_dependencies": list(node.computation_dependencies),
                        "data_dependencies": list(node.data_dependencies),
                        "name": node.name,
                    }
                    for node in self.computation_nodes
                ],
                "confidential_boundary": False,
                "data_nodes": [{"name": node.name, "type": node.type} for node in self.data_nodes],
                "description": self.description,
                "local_simulation": True,
                "name": self.title,
                "participants": [
                    {
                        "analyst_of": sorted(self.participants[email].analyst_of),
                        "data_owner_of": sorted(self.participants[email].data_owner_of),
                        "email": email,
                    }
                    for email in self.participant_emails
                ],
                "permissions": [
                    {"email": permission.email, "node": permission.node_name, "type": permission.type}
                    for permission in self.permissions
                ],
                "provider": PROVIDER_NAME,
                "selected_variables": self.selected_variables,
                "synthetic_demo": self.synthetic_demo,
            }
        }


@dataclass(frozen=True)
class ProdNodeIds:
    data_nodes: dict[str, str]
    computation_nodes: dict[str, str]


@dataclass
class CreationOutcome:
    dcr_id: str
    merge_request_id: str
    prod_node_ids: ProdNodeIds
    metadata_upload_results: dict[str, str] = field(default_factory=dict)
    shuffled_upload_results: dict[str, str] = field(default_factory=dict)
    mapping_upload_results: dict[str, str] = field(default_factory=dict)
    row_upload_results: dict[str, str] = field(default_factory=dict)


def _normalize_email(value: object) -> str:
    return str(value or "").strip().lower()


def _node_base(cohort_id: str) -> str:
    return cohort_id.replace(" ", "-")


def _mapping_node_name(mapping_file: dict[str, Any], filename: str) -> str:
    cohorts = mapping_file.get("cohorts")
    if isinstance(cohorts, list) and len(cohorts) >= 2:
        return f"{'_'.join(str(cohort) for cohort in cohorts)}_mapping"
    base_name = filename
    for suffix in (".json", ".csv"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
    base_name = base_name.replace(" ", "-").replace("(", "").replace(")", "").replace("+", "_")
    return f"mapping_{base_name}"


def _selected_sample(request_value: object, cohort_id: str) -> bool:
    if isinstance(request_value, dict):
        return bool(request_value.get(cohort_id, False))
    return bool(request_value)


def _resolve_dictionary(cohort_id: str, cohort: Any, settings: Settings) -> Path:
    try:
        candidate = Path(cohort.metadata_filepath).expanduser()
    except (AttributeError, FileNotFoundError, TypeError):
        candidate = Path(settings.demo_pack_dir) / "cohorts" / cohort_id / f"{cohort_id}_datadictionary.csv"
    if not candidate.is_file():
        fallback = Path(settings.demo_pack_dir) / "cohorts" / cohort_id / f"{cohort_id}_datadictionary.csv"
        if fallback.is_file():
            candidate = fallback
    if not candidate.is_file():
        raise DcrOperationError(
            detail=f"Metadata dictionary is unavailable for cohort {cohort_id}",
            failed_step="validate request",
        )
    return candidate.resolve()


def _resolve_sample(cohort_id: str, settings: Settings) -> Path | None:
    candidates = (
        Path(settings.data_folder) / f"dcr_output_{cohort_id}" / "shuffled_sample.csv",
        Path(settings.demo_pack_dir) / f"dcr_output_{cohort_id}" / "shuffled_sample.csv",
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _metadata_preview_code(node_names: list[str]) -> str:
    names_literal = json.dumps(node_names, ensure_ascii=True)
    return f"""# Local simulation for deterministic metadata preview; not a confidential-computing boundary.
import csv
import json
from pathlib import Path

node_names = {names_literal}
preview = {{"confidential_boundary": False, "local_simulation": True, "nodes": {{}}}}
for node_name in node_names:
    source = Path("input") / node_name / "file"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for _row in reader)
    preview["nodes"][node_name] = {{"columns": header, "rows": row_count}}
Path("output").mkdir(exist_ok=True)
Path("output/metadata-preview.json").write_text(
    json.dumps(preview, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
"""


def _aggregate_code(node_names: list[str]) -> str:
    names_literal = json.dumps(node_names, ensure_ascii=True)
    return f"""# Local simulation only: aggregate output, not a confidential-computing or clean-room boundary.
import csv
import json
from pathlib import Path

node_names = {names_literal}
result = {{"confidential_boundary": False, "local_simulation": True, "nodes": {{}}}}
for node_name in node_names:
    source = Path("input") / node_name / "file"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row_count = 0
        non_empty = {{column: 0 for column in (reader.fieldnames or [])}}
        numeric = {{
            column: {{"count": 0, "maximum": None, "minimum": None, "sum": 0.0}}
            for column in (reader.fieldnames or [])
        }}
        for row in reader:
            row_count += 1
            for column, value in row.items():
                if value is None or not value.strip():
                    continue
                non_empty[column] += 1
                try:
                    numeric_value = float(value)
                except ValueError:
                    pass
                else:
                    state = numeric[column]
                    state["count"] += 1
                    state["sum"] += numeric_value
                    if state["minimum"] is None or numeric_value < state["minimum"]:
                        state["minimum"] = numeric_value
                    if state["maximum"] is None or numeric_value > state["maximum"]:
                        state["maximum"] = numeric_value
    numeric_summary = {{}}
    for column, state in numeric.items():
        if state["count"]:
            numeric_summary[column] = {{
                "count": state["count"],
                "maximum": state["maximum"],
                "mean": state["sum"] / state["count"],
                "minimum": state["minimum"],
            }}
    result["nodes"][node_name] = {{
        "non_empty": non_empty,
        "numeric": numeric_summary,
        "rows": row_count,
    }}
Path("output").mkdir(exist_ok=True)
Path("output/aggregate-summary.json").write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
"""


def build_room_plan(
    request: dict[str, Any],
    user: dict[str, Any],
    all_cohorts: dict[str, Any],
    settings: Settings,
) -> RoomPlan:
    """Validate one wizard request and build a deterministic native AADCR plan.

    Participant resolution mirrors ``build_dcr_participants`` without importing
    ``src.decentriq``. Importing that legacy module would initialize the
    Decentriq SDK on the explicitly isolated AADCR provider path.
    """
    user_email = _normalize_email(user.get("email"))
    if not user_email:
        raise DcrOperationError(detail="An authenticated user email is required", failed_step="validate request")

    requested_cohorts = request.get("cohorts")
    if not isinstance(requested_cohorts, dict) or not requested_cohorts:
        raise DcrOperationError(detail="At least one cohort must be selected", failed_step="validate request")
    cohort_ids = sorted(str(cohort_id) for cohort_id in requested_cohorts)
    for cohort_id in cohort_ids:
        if cohort_id not in all_cohorts:
            raise DcrOperationError(
                detail=f"Selected cohort {cohort_id} is not available",
                failed_step="validate request",
            )
        if not isinstance(requested_cohorts[cohort_id], list):
            raise DcrOperationError(
                detail=f"Selected variables for cohort {cohort_id} must be a list",
                failed_step="validate request",
            )

    base_title = str(request.get("dcr_name") or "").strip()
    if not base_title:
        base_title = f"iCARE4CVD AADCR compute: {', '.join(cohort_ids)}"
    title = f"{base_title} - created by {user_email}"[:255]
    research_question = str(request.get("research_question") or "").strip()
    description = f"RESEARCH QUESTION: {research_question or 'no research question specified'}."

    excluded = {_normalize_email(email) for email in request.get("excluded_data_owners", []) if _normalize_email(email)}
    participants: dict[str, ParticipantRoles] = {}
    participant_emails: list[str] = []

    def ensure_participant(email: object) -> str:
        normalized = _normalize_email(email)
        if normalized and normalized not in participants:
            participants[normalized] = ParticipantRoles()
            participant_emails.append(normalized)
        return normalized

    ensure_participant(user_email)
    if not settings.dev_mode:
        for cohort_id in cohort_ids:
            cohort = all_cohorts[cohort_id]
            for owner in getattr(cohort, "cohort_email", []) or []:
                normalized_owner = _normalize_email(owner)
                if normalized_owner and normalized_owner not in excluded:
                    ensure_participant(normalized_owner)
            administrator = _normalize_email(getattr(cohort, "administrator_email", None))
            if administrator and administrator not in excluded:
                ensure_participant(administrator)

    service_email = ensure_participant(settings.decentriq_email)
    for analyst in request.get("additional_analysts", []) or []:
        ensure_participant(analyst)

    data_nodes: list[DataNodeSpec] = []
    metadata_assets: list[AssetSpec] = []
    shuffled_assets: list[AssetSpec] = []
    synthetic_assets: list[AssetSpec] = []
    row_nodes: list[str] = []
    metadata_nodes: list[str] = []
    include_samples = request.get("include_shuffled_samples", True)

    for cohort_id in cohort_ids:
        cohort = all_cohorts[cohort_id]
        base = _node_base(cohort_id)
        row_node = base
        metadata_node = f"{base}_metadata_dictionary"
        row_nodes.append(row_node)
        metadata_nodes.append(metadata_node)
        data_nodes.extend((DataNodeSpec(row_node), DataNodeSpec(metadata_node)))

        dictionary = _resolve_dictionary(cohort_id, cohort, settings)
        metadata_assets.append(
            AssetSpec(
                kind="metadata",
                key=cohort_id,
                node_name=metadata_node,
                path=dictionary,
                archive_path=f"metadata_dictionaries/{dictionary.name}",
            )
        )

        if settings.dev_mode:
            participants[user_email].data_owner_of.update((row_node, metadata_node))
        else:
            owner_emails = [*(getattr(cohort, "cohort_email", []) or [])]
            administrator = getattr(cohort, "administrator_email", None)
            if administrator:
                owner_emails.append(administrator)
            for owner in owner_emails:
                normalized_owner = _normalize_email(owner)
                if normalized_owner in participants and normalized_owner not in excluded:
                    participants[normalized_owner].data_owner_of.update((row_node, metadata_node))

        participants[user_email].data_owner_of.add(metadata_node)
        if service_email:
            participants[service_email].data_owner_of.add(metadata_node)

        if _selected_sample(include_samples, cohort_id):
            sample = _resolve_sample(cohort_id, settings)
            if sample is not None:
                sample_node = f"{base}_shuffled_sample"
                data_nodes.append(DataNodeSpec(sample_node))
                shuffled_assets.append(
                    AssetSpec(
                        kind="shuffled",
                        key=cohort_id,
                        node_name=sample_node,
                        path=sample,
                        archive_path=f"shuffled_samples/{base}_shuffled_sample.csv",
                    )
                )
                for roles in participants.values():
                    if row_node in roles.data_owner_of:
                        roles.data_owner_of.add(sample_node)
                participants[user_email].data_owner_of.add(sample_node)
                if service_email:
                    participants[service_email].data_owner_of.add(sample_node)

        if settings.aadcrv2_synthetic_demo:
            raw = Path(settings.demo_pack_dir) / "dcr-input" / f"{cohort_id}.csv"
            if not raw.is_file():
                raise DcrOperationError(
                    detail=f"Synthetic row-level CSV is unavailable for cohort {cohort_id}",
                    failed_step="validate request",
                )
            participants[user_email].data_owner_of.add(row_node)
            synthetic_assets.append(
                AssetSpec(
                    kind="synthetic",
                    key=cohort_id,
                    node_name=row_node,
                    path=raw.resolve(),
                    archive_path=None,
                )
            )

    # Legacy additional analysts copy the requester's initial cohort ownership,
    # before automatic metadata/mapping upload ownership is granted below.
    if settings.dev_mode:
        for analyst in request.get("additional_analysts", []) or []:
            normalized_analyst = _normalize_email(analyst)
            if normalized_analyst and normalized_analyst != user_email:
                participants[normalized_analyst].data_owner_of.update(
                    node for node in row_nodes + metadata_nodes if node in participants[user_email].data_owner_of
                )

    mapping_root = Path(settings.mapping_output_dir).expanduser().resolve()
    mapping_specs: list[tuple[DataNodeSpec, AssetSpec]] = []
    for raw_mapping in request.get("selected_mapping_files", []) or []:
        if not isinstance(raw_mapping, dict):
            raise DcrOperationError(
                detail="Each selected mapping must be an object",
                failed_step="validate mapping files",
            )
        raw_path = raw_mapping.get("filepath")
        try:
            mapping_path = Path(str(raw_path or "")).expanduser().resolve(strict=True)
            mapping_path.relative_to(mapping_root)
        except (FileNotFoundError, RuntimeError, ValueError):
            raise DcrOperationError(
                detail="Selected mapping file must exist inside the configured mapping output directory",
                failed_step="validate mapping files",
            ) from None
        if not mapping_path.is_file():
            raise DcrOperationError(
                detail="Selected mapping path is not a file",
                failed_step="validate mapping files",
            )
        mapping_name = mapping_path.name.casefold()
        if not mapping_name.endswith(".csv"):
            raise DcrOperationError(
                detail="Selected file must be a CSV mapping artifact, not JSON, a sidecar, or a log",
                failed_step="validate mapping files",
            )
        requested_filename = str(raw_mapping.get("filename") or mapping_path.name)
        if Path(requested_filename).name != mapping_path.name:
            raise DcrOperationError(
                detail="Selected mapping filename does not match its canonical path",
                failed_step="validate mapping files",
            )
        node_name = _mapping_node_name(raw_mapping, mapping_path.name)
        mapping_specs.append(
            (
                DataNodeSpec(node_name),
                AssetSpec(
                    kind="mapping",
                    key=mapping_path.name,
                    node_name=node_name,
                    path=mapping_path,
                    archive_path=f"mapping_files/{mapping_path.name}",
                ),
            )
        )
    mapping_specs.sort(key=lambda pair: (pair[0].name, pair[1].path.name))
    mapping_assets = [asset for _node, asset in mapping_specs]
    data_nodes.extend(node for node, _asset in mapping_specs)

    mapping_node_names = [node.name for node, _asset in mapping_specs]
    if request.get("include_mapping_upload_slot", False):
        data_nodes.append(DataNodeSpec("CrossStudyMappings"))
        mapping_node_names.append("CrossStudyMappings")

    duplicate_names = sorted(
        name for name in {node.name for node in data_nodes} if sum(node.name == name for node in data_nodes) > 1
    )
    if duplicate_names:
        raise DcrOperationError(
            detail=f"Duplicate AADCR data node names: {', '.join(duplicate_names)}",
            failed_step="validate request",
        )

    for node_name in mapping_node_names:
        for roles in participants.values():
            roles.data_owner_of.add(node_name)

    metadata_dependencies = list(metadata_nodes)
    computation_nodes = [
        ComputationNodeSpec(
            name=METADATA_PREVIEW_NODE_NAME,
            code=_metadata_preview_code(metadata_dependencies),
            data_dependencies=tuple(metadata_dependencies),
        ),
        ComputationNodeSpec(
            name=AGGREGATE_NODE_NAME,
            code=_aggregate_code(row_nodes),
            data_dependencies=tuple(row_nodes),
        ),
    ]
    computation_names = {node.name for node in computation_nodes}
    for roles in participants.values():
        roles.analyst_of.update(computation_names)

    permissions: list[PermissionSpec] = []
    data_node_names = [node.name for node in data_nodes]
    computation_node_names = [node.name for node in computation_nodes]
    for email in participant_emails:
        roles = participants[email]
        permissions.extend(
            PermissionSpec("DATA_OWNER", email, node_name)
            for node_name in data_node_names
            if node_name in roles.data_owner_of
        )
    for email in participant_emails:
        roles = participants[email]
        permissions.extend(
            PermissionSpec("DATA_ANALYST", email, node_name)
            for node_name in computation_node_names
            if node_name in roles.analyst_of
        )

    selected_variables = {
        cohort_id: sorted(str(variable) for variable in requested_cohorts[cohort_id]) for cohort_id in cohort_ids
    }
    return RoomPlan(
        title=title,
        description=description,
        cohort_ids=cohort_ids,
        selected_variables=selected_variables,
        participants=participants,
        participant_emails=participant_emails,
        data_nodes=data_nodes,
        computation_nodes=computation_nodes,
        permissions=permissions,
        assets=[*metadata_assets, *mapping_assets, *shuffled_assets, *synthetic_assets],
        synthetic_demo=settings.aadcrv2_synthetic_demo,
    )


def _required_string(payload: Any, key: str, *, failed_step: str, dcr_id: str | None = None) -> str:
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value:
        raise DcrOperationError(
            detail=f"AADCR response did not include {key}",
            failed_step=failed_step,
            dcr_id=dcr_id,
        )
    return value


def _safe_api_identifier(value: str, *, failed_step: str, dcr_id: str | None = None) -> str:
    if (
        not value
        or len(value) > 255
        or value in {".", ".."}
        or any(character in value for character in "/\\?#")
        or any(ord(character) < 32 for character in value)
    ):
        raise DcrOperationError(
            detail="AADCR returned an invalid API identifier",
            failed_step=failed_step,
            dcr_id=dcr_id,
        )
    return value


def _required_api_identifier(
    payload: Any,
    key: str,
    *,
    failed_step: str,
    dcr_id: str | None = None,
) -> str:
    return _safe_api_identifier(
        _required_string(payload, key, failed_step=failed_step, dcr_id=dcr_id),
        failed_step=failed_step,
        dcr_id=dcr_id,
    )


def _collect_dev_change_ids(dev_view: Any, expected_count: int, dcr_id: str) -> list[str]:
    if not isinstance(dev_view, dict):
        raise DcrOperationError(
            detail="AADCR DEV view was not an object",
            failed_step="collect DEV changes",
            dcr_id=dcr_id,
        )
    change_ids: list[str] = []
    for collection_name in ("participants", "data_nodes", "computation_nodes", "permissions"):
        collection = dev_view.get(collection_name)
        if not isinstance(collection, list):
            raise DcrOperationError(
                detail=f"AADCR DEV view omitted {collection_name}",
                failed_step="collect DEV changes",
                dcr_id=dcr_id,
            )
        for item in collection:
            if not isinstance(item, dict):
                continue
            change_id = item.get("changeId")
            node_status = item.get("nodeStatus")
            if (
                isinstance(change_id, str)
                and change_id
                and node_status in {"DEV_ADDED", "DEV_PROD_REMOVED"}
                and change_id not in change_ids
            ):
                change_ids.append(change_id)
    if len(change_ids) != expected_count:
        raise DcrOperationError(
            detail=f"Expected {expected_count} new DEV change IDs but received {len(change_ids)}",
            failed_step="collect DEV changes",
            dcr_id=dcr_id,
        )
    return change_ids


async def wait_for_merge(
    client: Any,
    dcr_id: str,
    merge_request_id: str,
    *,
    max_attempts: int,
    poll_interval: float,
) -> dict[str, Any]:
    """Poll a merge request without ever invoking the approval endpoint."""
    path = f"/api/dcr/{dcr_id}/merge-requests/{merge_request_id}"
    for attempt in range(max_attempts):
        payload = await client.request_json("GET", path, failed_step="poll merge request")
        if not isinstance(payload, dict):
            raise DcrOperationError(
                detail="AADCR merge request response was not an object",
                failed_step="poll merge request",
                dcr_id=dcr_id,
            )
        status = str(payload.get("status") or "").upper()
        if status == "MERGED":
            return payload
        if status == "PENDING":
            approvals = payload.get("approvals")
            if isinstance(approvals, list) and any(
                isinstance(approval, dict)
                and str(approval.get("status") or "").upper() == "PENDING"
                and not bool(approval.get("isImplicit"))
                for approval in approvals
            ):
                raise DcrOperationError(
                    detail="Merge request is pending data-owner approval; no approval was bypassed",
                    failed_step="await merge approval",
                    dcr_id=dcr_id,
                    retryable=True,
                )
        elif status not in {"APPROVED"}:
            raise DcrOperationError(
                detail=f"Merge request entered terminal status {status or 'UNKNOWN'}",
                failed_step="merge DEV changes",
                dcr_id=dcr_id,
            )
        if attempt + 1 < max_attempts and poll_interval > 0:
            await asyncio.sleep(poll_interval)
    raise DcrOperationError(
        detail=f"Merge request did not reach MERGED after {max_attempts} polls",
        failed_step="poll merge request",
        dcr_id=dcr_id,
        retryable=True,
    )


def _index_prod_nodes(items: Any, kind: str, dcr_id: str) -> dict[str, str]:
    if not isinstance(items, list):
        raise DcrOperationError(
            detail=f"AADCR PROD view omitted {kind}",
            failed_step="resolve PROD nodes",
            dcr_id=dcr_id,
        )
    indexed: dict[str, str] = {}
    duplicates: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        node_id = item.get("id")
        if not isinstance(name, str) or not isinstance(node_id, str) or not node_id:
            continue
        if name in indexed:
            duplicates.add(name)
        indexed[name] = node_id
    if duplicates:
        raise DcrOperationError(
            detail=f"AADCR PROD view contains duplicate node names: {', '.join(sorted(duplicates))}",
            failed_step="resolve PROD nodes",
            dcr_id=dcr_id,
        )
    return indexed


def resolve_prod_node_ids(plan: RoomPlan, prod_view: Any, dcr_id: str) -> ProdNodeIds:
    """Resolve immutable PROD IDs by exact node name and fail closed on drift."""
    if not isinstance(prod_view, dict):
        raise DcrOperationError(
            detail="AADCR PROD view was not an object",
            failed_step="resolve PROD nodes",
            dcr_id=dcr_id,
        )
    data_nodes = _index_prod_nodes(prod_view.get("data_nodes"), "data_nodes", dcr_id)
    computation_nodes = _index_prod_nodes(prod_view.get("computation_nodes"), "computation_nodes", dcr_id)
    expected_data = {node.name for node in plan.data_nodes}
    expected_computations = {node.name for node in plan.computation_nodes}
    missing = sorted((expected_data - data_nodes.keys()) | (expected_computations - computation_nodes.keys()))
    if missing:
        raise DcrOperationError(
            detail=f"AADCR PROD view is missing expected nodes: {', '.join(missing)}",
            failed_step="resolve PROD nodes",
            dcr_id=dcr_id,
        )
    return ProdNodeIds(
        data_nodes={name: data_nodes[name] for name in sorted(expected_data)},
        computation_nodes={name: computation_nodes[name] for name in sorted(expected_computations)},
    )


def _content_type(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "application/json"
    return "text/csv"


def _confirmed_merge_request_id(confirmed_steps: set[str]) -> str | None:
    prefix = "merge_requested:"
    matches = [step[len(prefix) :] for step in confirmed_steps if step.startswith(prefix)]
    if len(matches) > 1 or (matches and not matches[0]):
        raise DcrOperationError(
            detail="The operation journal contains ambiguous merge state",
            failed_step="resume operation",
        )
    return _safe_api_identifier(matches[0], failed_step="resume operation") if matches else None


def _index_dev_items(
    items: Any,
    *,
    collection_name: str,
    identity_key: str,
    failed_step: str,
    dcr_id: str,
) -> dict[str, str]:
    if not isinstance(items, list):
        raise DcrOperationError(
            detail=f"AADCR DEV view omitted {collection_name}",
            failed_step=failed_step,
            dcr_id=dcr_id,
        )
    indexed: dict[str, str] = {}
    duplicates: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        identity = item.get(identity_key)
        item_id = item.get("id")
        if identity_key == "userEmail":
            identity = _normalize_email(identity)
        if not isinstance(identity, str) or not identity or not isinstance(item_id, str) or not item_id:
            continue
        if identity in indexed:
            duplicates.add(identity)
        indexed[identity] = item_id
    if duplicates:
        raise DcrOperationError(
            detail=f"AADCR DEV view contains duplicate {collection_name}: {', '.join(sorted(duplicates))}",
            failed_step=failed_step,
            dcr_id=dcr_id,
        )
    return indexed


def _existing_permission_keys(
    permissions: Any,
    *,
    participant_ids: dict[str, str],
    data_node_ids: dict[str, str],
    computation_node_ids: dict[str, str],
    dcr_id: str,
) -> set[tuple[str, str, str]]:
    if not isinstance(permissions, list):
        raise DcrOperationError(
            detail="AADCR DEV view omitted permissions",
            failed_step="resume DEV graph",
            dcr_id=dcr_id,
        )
    participants_by_id = {participant_id: email for email, participant_id in participant_ids.items()}
    nodes_by_id = {
        **{node_id: name for name, node_id in data_node_ids.items()},
        **{node_id: name for name, node_id in computation_node_ids.items()},
    }
    keys: set[tuple[str, str, str]] = set()
    for permission in permissions:
        if not isinstance(permission, dict):
            continue
        permission_type = permission.get("type") or permission.get("permissionType")
        email = permission.get("userEmail")
        node_name = permission.get("resourceName")
        if not isinstance(email, str):
            email = participants_by_id.get(str(permission.get("participantId") or ""))
        if not isinstance(node_name, str):
            node_name = nodes_by_id.get(str(permission.get("nodeId") or ""))
        normalized_email = _normalize_email(email)
        if (
            isinstance(permission_type, str)
            and permission_type
            and normalized_email
            and isinstance(node_name, str)
            and node_name
        ):
            keys.add((permission_type, normalized_email, node_name))
    return keys


async def _confirm_step(
    callback: ConfirmationCallback | None,
    step: str,
    dcr_id: str,
) -> None:
    if callback is not None:
        await callback(step, dcr_id)


async def create_aadcr_room(
    client: Any,
    plan: RoomPlan,
    *,
    merge_poll_attempts: int,
    merge_poll_interval: float,
    resume_dcr_id: str | None = None,
    confirmed_steps: set[str] | frozenset[str] = frozenset(),
    on_confirm: ConfirmationCallback | None = None,
    creation_key: str | None = None,
    expected_creator_email: str | None = None,
) -> CreationOutcome:
    """Execute or resume one native create -> DEV -> merge -> provision flow."""
    dcr_id = _safe_api_identifier(resume_dcr_id, failed_step="resume operation") if resume_dcr_id is not None else None
    durable_steps = set(confirmed_steps)
    try:
        if dcr_id is None:
            pending_name = plan.title
            room = None
            if creation_key is not None:
                marker = f" [ce-operation:{creation_key}]"
                pending_name = f"{plan.title[: 255 - len(marker)]}{marker}"
                rooms = await client.request_json(
                    "GET",
                    "/api/dcr/",
                    failed_step="reconcile DCR creation",
                )
                if not isinstance(rooms, list):
                    raise DcrOperationError(
                        detail="AADCR room list was not an array",
                        failed_step="reconcile DCR creation",
                    )
                creator = _normalize_email(expected_creator_email)
                matches = [
                    candidate
                    for candidate in rooms
                    if isinstance(candidate, dict)
                    and candidate.get("name") == pending_name
                    and _normalize_email(candidate.get("creatorEmail")) == creator
                ]
                if len(matches) > 1:
                    raise DcrOperationError(
                        detail="AADCR returned multiple rooms for the same creation operation",
                        failed_step="reconcile DCR creation",
                        status_code=409,
                    )
                room = matches[0] if matches else None
            if room is None:
                room = await client.request_json(
                    "POST",
                    "/api/dcr/",
                    failed_step="create DCR",
                    json_body={"name": pending_name},
                )
            dcr_id = _required_api_identifier(room, "id", failed_step="create DCR")
            await _confirm_step(on_confirm, "room_created", dcr_id)
            durable_steps.add("room_created")
        if creation_key is not None:
            await client.request_json(
                "PATCH",
                f"/api/dcr/{dcr_id}",
                failed_step="finalize DCR name",
                json_body={"name": plan.title},
            )
        dev_root = f"/api/dcr/{dcr_id}/dev"

        merge_request_id = _confirmed_merge_request_id(durable_steps)
        merge_title = f"Create {plan.title}"[:255]
        if creation_key is not None:
            merge_marker = f" [ce-operation:{creation_key}]"
            merge_prefix = "Create "
            merge_title = f"{merge_prefix}{plan.title[: 255 - len(merge_prefix) - len(merge_marker)]}{merge_marker}"
        if (
            "merged" not in durable_steps
            and merge_request_id is None
            and creation_key is not None
            and resume_dcr_id is not None
        ):
            merge_requests_payload = await client.request_json(
                "GET",
                f"/api/dcr/{dcr_id}/merge-requests/",
                failed_step="reconcile merge request creation",
            )
            merge_requests = (
                merge_requests_payload.get("mergeRequests")
                if isinstance(merge_requests_payload, dict)
                else None
            )
            if not isinstance(merge_requests, list):
                raise DcrOperationError(
                    detail="AADCR merge-request list omitted mergeRequests",
                    failed_step="reconcile merge request creation",
                    dcr_id=dcr_id,
                )
            creator = _normalize_email(expected_creator_email)
            matches = [
                candidate
                for candidate in merge_requests
                if isinstance(candidate, dict)
                and candidate.get("title") == merge_title
                and _normalize_email(candidate.get("createdBy")) == creator
            ]
            if len(matches) > 1:
                raise DcrOperationError(
                    detail="AADCR returned multiple merge requests for the same creation operation",
                    failed_step="reconcile merge request creation",
                    dcr_id=dcr_id,
                    status_code=409,
                )
            if matches:
                merge_request_id = _required_api_identifier(
                    matches[0],
                    "id",
                    failed_step="reconcile merge request creation",
                    dcr_id=dcr_id,
                )
                merge_step = f"merge_requested:{merge_request_id}"
                await _confirm_step(on_confirm, merge_step, dcr_id)
                durable_steps.add(merge_step)
        if "merged" not in durable_steps and merge_request_id is None:
            if resume_dcr_id is None:
                participant_ids: dict[str, str] = {}
                dev_data_ids: dict[str, str] = {}
                dev_computation_ids: dict[str, str] = {}
                existing_permissions: set[tuple[str, str, str]] = set()
            else:
                existing_view = await client.request_json(
                    "GET",
                    f"{dev_root}/view",
                    failed_step="resume DEV graph",
                )
                if not isinstance(existing_view, dict):
                    raise DcrOperationError(
                        detail="AADCR DEV view was not an object",
                        failed_step="resume DEV graph",
                        dcr_id=dcr_id,
                    )
                participant_ids = _index_dev_items(
                    existing_view.get("participants"),
                    collection_name="participants",
                    identity_key="userEmail",
                    failed_step="resume DEV graph",
                    dcr_id=dcr_id,
                )
                dev_data_ids = _index_dev_items(
                    existing_view.get("data_nodes"),
                    collection_name="data nodes",
                    identity_key="name",
                    failed_step="resume DEV graph",
                    dcr_id=dcr_id,
                )
                dev_computation_ids = _index_dev_items(
                    existing_view.get("computation_nodes"),
                    collection_name="computation nodes",
                    identity_key="name",
                    failed_step="resume DEV graph",
                    dcr_id=dcr_id,
                )
                existing_permissions = _existing_permission_keys(
                    existing_view.get("permissions"),
                    participant_ids=participant_ids,
                    data_node_ids=dev_data_ids,
                    computation_node_ids=dev_computation_ids,
                    dcr_id=dcr_id,
                )

            for email in plan.participant_emails:
                if email in participant_ids:
                    continue
                response = await client.request_json(
                    "POST",
                    f"{dev_root}/participants",
                    failed_step="add DEV participant",
                    json_body={"userEmail": email},
                )
                participant_ids[email] = _required_string(
                    response,
                    "id",
                    failed_step="add DEV participant",
                    dcr_id=dcr_id,
                )

            for node in plan.data_nodes:
                if node.name in dev_data_ids:
                    continue
                response = await client.request_json(
                    "POST",
                    f"{dev_root}/data-nodes",
                    failed_step="add DEV data node",
                    json_body={"name": node.name, "type": node.type},
                )
                dev_data_ids[node.name] = _required_string(
                    response,
                    "id",
                    failed_step="add DEV data node",
                    dcr_id=dcr_id,
                )

            for node in plan.computation_nodes:
                if node.name in dev_computation_ids:
                    continue
                try:
                    data_dependencies = [dev_data_ids[name] for name in node.data_dependencies]
                    computation_dependencies = [dev_computation_ids[name] for name in node.computation_dependencies]
                except KeyError as exc:
                    raise DcrOperationError(
                        detail=f"Computation dependency {exc.args[0]} was not created in DEV",
                        failed_step="add DEV computation node",
                        dcr_id=dcr_id,
                    ) from None
                response = await client.request_json(
                    "POST",
                    f"{dev_root}/computation-nodes",
                    failed_step="add DEV computation node",
                    json_body={
                        "name": node.name,
                        "code": node.code,
                        "dataDependencies": data_dependencies,
                        "computationDependencies": computation_dependencies,
                    },
                )
                dev_computation_ids[node.name] = _required_string(
                    response,
                    "id",
                    failed_step="add DEV computation node",
                    dcr_id=dcr_id,
                )

            for permission in plan.permissions:
                permission_key = (permission.type, permission.email, permission.node_name)
                if permission_key in existing_permissions:
                    continue
                node_ids = dev_data_ids if permission.type == "DATA_OWNER" else dev_computation_ids
                await client.request_json(
                    "POST",
                    f"{dev_root}/permissions",
                    failed_step="add DEV permission",
                    json_body={
                        "type": permission.type,
                        "participantId": participant_ids[permission.email],
                        "nodeId": node_ids[permission.node_name],
                    },
                )
                existing_permissions.add(permission_key)

            dev_view = await client.request_json(
                "GET",
                f"{dev_root}/view",
                failed_step="read DEV view",
            )
            expected_change_count = (
                len(plan.participant_emails)
                + len(plan.data_nodes)
                + len(plan.computation_nodes)
                + len(plan.permissions)
            )
            change_ids = _collect_dev_change_ids(dev_view, expected_change_count, dcr_id)
            merge = await client.request_json(
                "POST",
                f"/api/dcr/{dcr_id}/merge-requests/",
                failed_step="create merge request",
                json_body={
                    "title": merge_title,
                    "description": "Initial Cohort Explorer graph for the AADCR local simulation.",
                    "change_ids": change_ids,
                },
            )
            merge_request_id = _required_api_identifier(
                merge,
                "id",
                failed_step="create merge request",
                dcr_id=dcr_id,
            )
            merge_step = f"merge_requested:{merge_request_id}"
            await _confirm_step(on_confirm, merge_step, dcr_id)
            durable_steps.add(merge_step)

        if merge_request_id is None:
            raise DcrOperationError(
                detail="The operation journal omitted the merge request identifier",
                failed_step="resume operation",
                dcr_id=dcr_id,
            )
        if "merged" not in durable_steps:
            await wait_for_merge(
                client,
                dcr_id,
                merge_request_id,
                max_attempts=merge_poll_attempts,
                poll_interval=merge_poll_interval,
            )
            await _confirm_step(on_confirm, "merged", dcr_id)
            durable_steps.add("merged")

        prod_view = await client.request_json(
            "GET",
            f"/api/dcr/{dcr_id}/prod/view",
            failed_step="read PROD view",
        )
        prod_node_ids = resolve_prod_node_ids(plan, prod_view, dcr_id)
        outcome = CreationOutcome(
            dcr_id=dcr_id,
            merge_request_id=merge_request_id,
            prod_node_ids=prod_node_ids,
        )
        result_maps = {
            "metadata": outcome.metadata_upload_results,
            "mapping": outcome.mapping_upload_results,
            "shuffled": outcome.shuffled_upload_results,
            "synthetic": outcome.row_upload_results,
        }
        for asset in plan.assets:
            checkpoint = f"provisioned:{asset.kind}:{asset.node_name}"
            if checkpoint in durable_steps:
                result_maps[asset.kind][asset.key] = "success"
                continue
            try:
                content = asset.path.read_bytes()
            except OSError:
                raise DcrOperationError(
                    detail=f"Upload asset {asset.path.name} is no longer available",
                    failed_step=f"read {asset.kind} asset",
                    dcr_id=dcr_id,
                ) from None
            uploaded = await client.upload_file(
                "/api/upload",
                filename=asset.path.name,
                content=content,
                content_type=_content_type(asset.path),
                form={"dataset_name": asset.node_name},
                failed_step=f"upload {asset.kind} asset",
            )
            dataset_id = _required_string(
                uploaded,
                "dataset_id",
                failed_step=f"upload {asset.kind} asset",
                dcr_id=dcr_id,
            )
            await client.request_json(
                "POST",
                f"/api/dcr/{dcr_id}/provision-dataset",
                failed_step=f"provision {asset.kind} asset",
                json_body={
                    "dataset_id": dataset_id,
                    "dataset_node_id": prod_node_ids.data_nodes[asset.node_name],
                    "provision_type": "PROD",
                },
            )
            result_maps[asset.kind][asset.key] = "success"
            await _confirm_step(on_confirm, checkpoint, dcr_id)
            durable_steps.add(checkpoint)
        return outcome
    except DcrOperationError:
        raise
    except AadcrUpstreamError as exc:
        raise DcrOperationError(
            detail=exc.detail,
            failed_step=exc.failed_step,
            dcr_id=dcr_id,
            retryable=exc.retryable,
            status_code=exc.status_code,
        ) from None


def participants_response(plan: RoomPlan) -> dict[str, dict[str, list[str]]]:
    """Serialize participant role sets in the legacy live-create shape."""
    return {
        email: {
            "data_owner_of": sorted(plan.participants[email].data_owner_of),
            "analyst_of": sorted(plan.participants[email].analyst_of),
        }
        for email in plan.participant_emails
    }


def normalize_aadcr_room(
    room: Any,
    prod_view: Any,
    provisioned_payload: Any,
    journal_state: Any,
    *,
    dcr_url: str,
    capabilities: Any,
) -> dict[str, Any]:
    """Project native room state into the provider-neutral My DCRs shape."""
    if not isinstance(room, dict) or not isinstance(prod_view, dict):
        raise DcrOperationError(
            detail="AADCR room or PROD state was not an object",
            failed_step="normalize room",
        )
    dcr_id = _required_string(room, "id", failed_step="normalize room")
    collections: dict[str, list[Any]] = {}
    for collection_name in ("participants", "data_nodes", "computation_nodes", "permissions"):
        collection = prod_view.get(collection_name)
        if not isinstance(collection, list):
            raise DcrOperationError(
                detail=f"AADCR PROD view omitted {collection_name}",
                failed_step="normalize room",
                dcr_id=dcr_id,
            )
        collections[collection_name] = collection

    participant_roles: dict[str, dict[str, set[str]]] = {}
    participant_order: list[str] = []
    for participant in collections["participants"]:
        if not isinstance(participant, dict):
            continue
        email = _normalize_email(participant.get("userEmail"))
        if email and email not in participant_roles:
            participant_roles[email] = {"data_owner_of": set(), "analyst_of": set()}
            participant_order.append(email)
    for permission in collections["permissions"]:
        if not isinstance(permission, dict):
            continue
        email = _normalize_email(permission.get("userEmail"))
        resource_name = permission.get("resourceName")
        permission_type = permission.get("permissionType")
        if not email or not isinstance(resource_name, str) or not resource_name:
            continue
        if email not in participant_roles:
            participant_roles[email] = {"data_owner_of": set(), "analyst_of": set()}
            participant_order.append(email)
        if permission_type == "DATA_OWNER":
            participant_roles[email]["data_owner_of"].add(resource_name)
        elif permission_type == "DATA_ANALYST":
            participant_roles[email]["analyst_of"].add(resource_name)

    participants = [
        {
            "email": email,
            "data_owner_of": sorted(participant_roles[email]["data_owner_of"]),
            "analyst_of": sorted(participant_roles[email]["analyst_of"]),
        }
        for email in participant_order
    ]

    nodes: list[dict[str, Any]] = []
    data_node_names: dict[str, str] = {}
    for node in collections["data_nodes"]:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        name = node.get("name")
        if not isinstance(node_id, str) or not node_id or not isinstance(name, str) or not name:
            continue
        if node_id in data_node_names:
            raise DcrOperationError(
                detail="AADCR PROD view contains duplicate data-node identifiers",
                failed_step="normalize room",
                dcr_id=dcr_id,
            )
        data_node_names[node_id] = name
        normalized_node: dict[str, Any] = {
            "id": node_id,
            "name": name,
            "type": "RawDataNodeDefinition",
            "provider_type": str(node.get("type") or "FILE"),
        }
        dataset_status = node.get("datasetStatus")
        if isinstance(dataset_status, str) and dataset_status:
            normalized_node["status"] = dataset_status
        nodes.append(normalized_node)
    for node in collections["computation_nodes"]:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        name = node.get("name")
        if not isinstance(node_id, str) or not node_id or not isinstance(name, str) or not name:
            continue
        normalized_node = {
            "id": node_id,
            "name": name,
            "type": "PythonComputeNodeDefinition",
        }
        computation_status = node.get("computationStatus")
        if isinstance(computation_status, str) and computation_status:
            normalized_node["status"] = computation_status
        nodes.append(normalized_node)

    provisioned = provisioned_payload.get("provisioned_datasets") if isinstance(provisioned_payload, dict) else None
    if not isinstance(provisioned, list):
        raise DcrOperationError(
            detail="AADCR provisioned-dataset response omitted provisioned_datasets",
            failed_step="normalize room",
            dcr_id=dcr_id,
        )
    provisioned_datasets: list[dict[str, Any]] = []
    safe_provision_fields = (
        "dataset_id",
        "dataset_node_id",
        "dataset_name",
        "provision_type",
        "provisioned_at",
        "provisioned_by",
        "uploader",
        "uploaded_at",
        "data_size_kb",
    )
    for dataset in provisioned:
        if not isinstance(dataset, dict):
            continue
        node_id = dataset.get("dataset_node_id")
        if not isinstance(node_id, str) or node_id not in data_node_names:
            raise DcrOperationError(
                detail="AADCR provisioned dataset references an unknown PROD data node",
                failed_step="normalize room",
                dcr_id=dcr_id,
            )
        normalized_dataset = {
            field_name: dataset[field_name]
            for field_name in safe_provision_fields
            if field_name in dataset and dataset[field_name] is not None
        }
        normalized_dataset["node_name"] = data_node_names[node_id]
        normalized_dataset["status"] = "provisioned"
        provisioned_datasets.append(normalized_dataset)

    metadata = getattr(journal_state, "request_metadata", {}) if journal_state else {}
    cohort_ids = list(getattr(journal_state, "cohort_ids", ())) if journal_state else []
    research_question = metadata.get("research_question") if isinstance(metadata, dict) else None
    if isinstance(research_question, str):
        description = f"RESEARCH QUESTION: {research_question or 'no research question specified'}."
    else:
        native_description = room.get("description")
        description = native_description if isinstance(native_description, str) else None

    normalized_room: dict[str, Any] = {
        "id": dcr_id,
        "title": str(room.get("name") or ""),
        "description": description,
        "owner": {"email": str(room.get("creatorEmail") or "")},
        "participants": participants,
        "nodes": nodes,
        "cohorts": cohort_ids,
        "provisioned_datasets": provisioned_datasets,
        "dcr_url": dcr_url,
        "provider": PROVIDER_NAME,
        "capabilities": capabilities,
    }
    for native_name, normalized_name in (
        ("createdAt", "createdAt"),
        ("updatedAt", "updatedAt"),
        ("version", "version"),
    ):
        if room.get(native_name) is not None:
            normalized_room[normalized_name] = room[native_name]
    return normalized_room


def normalize_aadcr_audit(payload: Any, *, dcr_id: str, main_only: bool) -> list[dict[str, Any]]:
    """Return safe audit aliases without native request bodies or response output."""
    audit_rows = payload.get("audit_logs") if isinstance(payload, dict) else None
    if not isinstance(audit_rows, list):
        raise DcrOperationError(
            detail="AADCR audit response omitted audit_logs",
            failed_step="read audit log",
            dcr_id=dcr_id,
        )
    normalized: list[dict[str, Any]] = []
    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        method = str(row.get("http_method") or "").upper()
        path = str(row.get("path") or "")
        if main_only and method == "GET" and path.rstrip("/").endswith("/audit-logs"):
            continue
        entry = {
            "timestamp": str(row.get("timestamp") or ""),
            "user": str(row.get("user_email") or ""),
            "desc": str(row.get("action_label") or "")[:300],
            "provider": PROVIDER_NAME,
        }
        source = {
            key: value
            for key, value in {
                "id": row.get("id"),
                "method": method,
                "path": path,
            }.items()
            if isinstance(value, str) and value
        }
        if source:
            entry["source"] = source
        normalized.append(entry)
    return normalized


def resolve_computation_node_id(prod_view: Any, node_name: str, dcr_id: str) -> str:
    """Resolve one exact PROD computation node without accepting name drift."""
    if not isinstance(prod_view, dict):
        raise DcrOperationError(
            detail="AADCR PROD view was not an object",
            failed_step="resolve aggregate computation",
            dcr_id=dcr_id,
        )
    indexed = _index_prod_nodes(prod_view.get("computation_nodes"), "computation_nodes", dcr_id)
    node_id = indexed.get(node_name)
    if node_id is None:
        raise DcrOperationError(
            detail="AADCR PROD view is missing the aggregate computation node",
            failed_step="resolve aggregate computation",
            dcr_id=dcr_id,
        )
    return node_id


def decode_result_archive(
    payload: Any,
    *,
    dcr_id: str,
    max_archive_bytes: int,
    max_members: int,
) -> bytes:
    """Validate a bounded base64 ZIP result without extracting untrusted paths."""

    def invalid(detail: str) -> DcrOperationError:
        return DcrOperationError(
            detail=detail,
            failed_step="decode computation result",
            dcr_id=dcr_id,
        )

    encoded = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(encoded, str) or not encoded:
        raise invalid("Completed computation did not include a base64 result archive")
    max_encoded_length = ((max_archive_bytes + 2) // 3) * 4
    if len(encoded) > max_encoded_length:
        raise invalid("Computation result exceeds the archive size limit")
    try:
        archive_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise invalid("Computation result was not valid base64") from None
    if len(archive_bytes) > max_archive_bytes:
        raise invalid("Computation result exceeds the archive size limit")

    eocd_signature = b"PK\x05\x06"
    search_start = max(0, len(archive_bytes) - (65_535 + 22))
    search_end = len(archive_bytes)
    eocd_offset = -1
    while search_end > search_start:
        candidate = archive_bytes.rfind(eocd_signature, search_start, search_end)
        if candidate < 0:
            break
        if candidate + 22 <= len(archive_bytes):
            comment_length = struct.unpack_from("<H", archive_bytes, candidate + 20)[0]
            if candidate + 22 + comment_length == len(archive_bytes):
                eocd_offset = candidate
                break
        search_end = candidate
    if eocd_offset < 0:
        raise invalid("Computation result was not a valid ZIP archive")
    (
        _signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        declared_entries,
        central_directory_size,
        central_directory_offset,
        _comment_length,
    ) = struct.unpack_from("<4s4H2LH", archive_bytes, eocd_offset)
    if (
        disk_number != 0
        or central_directory_disk != 0
        or entries_on_disk != declared_entries
        or declared_entries == 0xFFFF
        or central_directory_size == 0xFFFFFFFF
        or central_directory_offset == 0xFFFFFFFF
    ):
        raise invalid("Computation result ZIP uses an unsupported archive structure")
    if declared_entries > max_members:
        raise invalid("Computation result ZIP contains too many members")
    central_directory_end = central_directory_offset + central_directory_size
    if central_directory_end != eocd_offset:
        raise invalid("Computation result was not a valid ZIP archive")
    cursor = central_directory_offset
    counted_entries = 0
    while cursor < central_directory_end:
        if (
            counted_entries >= max_members
            or cursor + 46 > central_directory_end
            or archive_bytes[cursor : cursor + 4] != b"PK\x01\x02"
        ):
            raise invalid("Computation result ZIP contains too many or invalid members")
        filename_length, extra_length, member_comment_length = struct.unpack_from("<HHH", archive_bytes, cursor + 28)
        cursor += 46 + filename_length + extra_length + member_comment_length
        if cursor > central_directory_end:
            raise invalid("Computation result was not a valid ZIP archive")
        counted_entries += 1
    if cursor != central_directory_end or counted_entries != declared_entries:
        raise invalid("Computation result was not a valid ZIP archive")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise invalid("Computation result ZIP contains too many members")
            seen_paths: set[str] = set()
            advertised_size = 0
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                path_parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
                if (
                    not name
                    or "\\" in name
                    or "\x00" in name
                    or path.is_absolute()
                    or not path_parts
                    or any(part in {"", ".", ".."} for part in path_parts)
                    or ":" in path_parts[0]
                ):
                    raise invalid("Computation result ZIP contains an unsafe member path")
                canonical_path = path.as_posix()
                if canonical_path in seen_paths:
                    raise invalid("Computation result ZIP contains duplicate member paths")
                seen_paths.add(canonical_path)
                if member.flag_bits & 0x1:
                    raise invalid("Computation result ZIP contains an encrypted member")
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                if stat.S_ISLNK(unix_mode) or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise invalid("Computation result ZIP contains an unsafe member type")
                if member.file_size < 0 or member.file_size > max_archive_bytes:
                    raise invalid("Computation result exceeds the archive size limit")
                advertised_size += member.file_size
                if advertised_size > max_archive_bytes:
                    raise invalid("Computation result exceeds the archive size limit")

            actual_size = 0
            for member in members:
                if member.is_dir():
                    continue
                with archive.open(member, "r") as member_file:
                    while chunk := member_file.read(64 * 1024):
                        actual_size += len(chunk)
                        if actual_size > max_archive_bytes:
                            raise invalid("Computation result exceeds the archive size limit")
    except DcrOperationError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        raise invalid("Computation result was not a valid ZIP archive") from None
    return archive_bytes
