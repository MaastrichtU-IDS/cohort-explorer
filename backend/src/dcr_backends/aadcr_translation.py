"""Translate Cohort Explorer room requests to AADCR v2's native API graph."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.config import Settings
from src.dcr_backends.aadcr_client import AadcrUpstreamError

METADATA_PREVIEW_NODE_NAME = "metadata-preview-local-simulation"
AGGREGATE_NODE_NAME = "aggregate-summary-local-simulation"
PROVIDER_NAME = "aadcrv2"


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


async def create_aadcr_room(
    client: Any,
    plan: RoomPlan,
    *,
    merge_poll_attempts: int,
    merge_poll_interval: float,
) -> CreationOutcome:
    """Execute one native create -> DEV -> merge -> PROD -> provision flow."""
    dcr_id: str | None = None
    try:
        room = await client.request_json(
            "POST",
            "/api/dcr/",
            failed_step="create DCR",
            json_body={"name": plan.title},
        )
        dcr_id = _required_string(room, "id", failed_step="create DCR")
        dev_root = f"/api/dcr/{dcr_id}/dev"

        participant_ids: dict[str, str] = {}
        for email in plan.participant_emails:
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

        dev_data_ids: dict[str, str] = {}
        for node in plan.data_nodes:
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

        dev_computation_ids: dict[str, str] = {}
        for node in plan.computation_nodes:
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

        dev_view = await client.request_json("GET", f"{dev_root}/view", failed_step="read DEV view")
        expected_change_count = (
            len(plan.participant_emails) + len(plan.data_nodes) + len(plan.computation_nodes) + len(plan.permissions)
        )
        change_ids = _collect_dev_change_ids(dev_view, expected_change_count, dcr_id)
        merge = await client.request_json(
            "POST",
            f"/api/dcr/{dcr_id}/merge-requests/",
            failed_step="create merge request",
            json_body={
                "title": f"Create {plan.title}"[:255],
                "description": "Initial Cohort Explorer graph for the AADCR local simulation.",
                "change_ids": change_ids,
            },
        )
        merge_request_id = _required_string(
            merge,
            "id",
            failed_step="create merge request",
            dcr_id=dcr_id,
        )
        await wait_for_merge(
            client,
            dcr_id,
            merge_request_id,
            max_attempts=merge_poll_attempts,
            poll_interval=merge_poll_interval,
        )

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
