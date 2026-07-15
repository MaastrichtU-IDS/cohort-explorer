import asyncio
import base64
import io
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.dcr_backends.aadcr_backend import AadcrBackend
from src.dcr_backends.aadcr_client import AadcrUpstreamError
from src.dcr_backends.aadcr_translation import AGGREGATE_NODE_NAME, DcrOperationError
from src.dcr_backends.operation_journal import OperationJournal

TEST_AADCR_SECRET = "task-five-test-secret"


def run(coroutine):
    return asyncio.run(coroutine)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _upstream_error(path: str, failed_step: str, *, status_code: int, detail: str):
    return AadcrUpstreamError(
        method="POST" if "computation-nodes" in path or path == "/api/upload" else "GET",
        path=path,
        detail=detail,
        failed_step=failed_step,
        status_code=status_code,
        retryable=status_code >= 500,
    )


class CreationService:
    """Stateful native AADCR double; state survives short-lived client contexts."""

    def __init__(
        self,
        *,
        fail_prod_once: bool = False,
        fail_upload_number_once: int | None = None,
        fail_participant_after_mutation_once: bool = False,
    ):
        self.calls: list[dict] = []
        self.participants: list[dict] = []
        self.data_nodes: list[dict] = []
        self.computation_nodes: list[dict] = []
        self.permissions: list[dict] = []
        self.room_count = 0
        self.merge_count = 0
        self.fail_prod_once = fail_prod_once
        self.failed_prod = False
        self.fail_upload_number_once = fail_upload_number_once
        self.failed_upload = False
        self.upload_count = 0
        self.fail_participant_after_mutation_once = fail_participant_after_mutation_once
        self.failed_participant = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    async def request_json(self, method, path, *, failed_step, json_body=None, **_kwargs):
        self.calls.append(
            {
                "kind": "json",
                "method": method,
                "path": path,
                "failed_step": failed_step,
                "body": json_body,
            }
        )
        if method == "POST" and path == "/api/dcr/":
            self.room_count += 1
            return {
                "id": f"room-{self.room_count}",
                "name": json_body["name"],
                "creatorEmail": "creator@example.test",
                "version": 1,
                "createdAt": "2026-07-15T09:00:00Z",
                "updatedAt": "2026-07-15T09:00:00Z",
            }
        if path.endswith("/dev/participants"):
            response = {
                "id": f"dev-participant-{len(self.participants)}",
                "userEmail": json_body["userEmail"],
                "changeId": f"change-participant-{len(self.participants)}",
            }
            self.participants.append(response)
            if self.fail_participant_after_mutation_once and not self.failed_participant:
                self.failed_participant = True
                raise _upstream_error(
                    path,
                    failed_step,
                    status_code=503,
                    detail="participant response was interrupted",
                )
            return response
        if path.endswith("/dev/data-nodes"):
            response = {
                **json_body,
                "id": f"dev-data-{len(self.data_nodes)}",
                "changeId": f"change-data-{len(self.data_nodes)}",
            }
            self.data_nodes.append(response)
            return response
        if path.endswith("/dev/computation-nodes"):
            response = {
                **json_body,
                "id": f"dev-computation-{len(self.computation_nodes)}",
                "changeId": f"change-computation-{len(self.computation_nodes)}",
            }
            self.computation_nodes.append(response)
            return response
        if path.endswith("/dev/permissions"):
            response = {
                **json_body,
                "id": f"dev-permission-{len(self.permissions)}",
                "changeId": f"change-permission-{len(self.permissions)}",
            }
            self.permissions.append(response)
            return response
        if method == "GET" and path.endswith("/dev/view"):

            def changed(values):
                return [{**value, "nodeStatus": "DEV_ADDED", "currentUser": "creator@example.test"} for value in values]

            return {
                "participants": changed(self.participants),
                "data_nodes": changed(self.data_nodes),
                "computation_nodes": changed(self.computation_nodes),
                "permissions": changed(self.permissions),
            }
        if method == "POST" and path.endswith("/merge-requests/"):
            self.merge_count += 1
            return {"id": f"merge-{self.merge_count}", "status": "PENDING"}
        if method == "GET" and "/merge-requests/" in path:
            return {
                "id": path.rsplit("/", 1)[-1],
                "status": "MERGED",
                "approvals": [],
            }
        if method == "GET" and path.endswith("/prod/view"):
            if self.fail_prod_once and not self.failed_prod:
                self.failed_prod = True
                raise _upstream_error(
                    path,
                    failed_step,
                    status_code=503,
                    detail="temporary PROD read failure",
                )
            return self._prod_view()
        if method == "POST" and path.endswith("/provision-dataset"):
            return {
                "success": True,
                **json_body,
                "created_at": "2026-07-15T09:02:00Z",
                "created_by": "creator@example.test",
            }
        raise AssertionError(f"Unexpected native request: {method} {path}")

    async def upload_file(
        self,
        path,
        *,
        filename,
        content,
        content_type,
        failed_step,
        form=None,
        field_name="file",
    ):
        self.upload_count += 1
        self.calls.append(
            {
                "kind": "upload",
                "method": "POST",
                "path": path,
                "filename": filename,
                "content": content,
                "content_type": content_type,
                "failed_step": failed_step,
                "form": form,
                "field_name": field_name,
            }
        )
        if self.fail_upload_number_once == self.upload_count and not self.failed_upload:
            self.failed_upload = True
            raise _upstream_error(
                path,
                failed_step,
                status_code=503,
                detail="temporary upload failure",
            )
        return {
            "success": True,
            "dataset_id": f"dataset-{filename}",
            "dataset_name": form["dataset_name"],
            "data_size_kb": max(1, len(content) // 1024),
            "created_at": "2026-07-15T09:01:00Z",
        }

    def _prod_view(self):
        return {
            "participants": [
                {
                    "id": f"prod-participant-{index}",
                    "userEmail": participant["userEmail"],
                    "currentUser": "creator@example.test",
                    "nodeStatus": "PROD",
                    "changeId": participant["changeId"],
                }
                for index, participant in enumerate(self.participants)
            ],
            "data_nodes": [
                {
                    "id": f"prod::{node['name']}",
                    "name": node["name"],
                    "type": "FILE",
                    "datasetName": None,
                    "datasetId": None,
                    "datasetStatus": "NOT_ADDED",
                    "dataOwners": [],
                    "currentUser": "creator@example.test",
                    "nodeStatus": "PROD",
                    "changeId": node["changeId"],
                    "schema": None,
                }
                for node in self.data_nodes
            ],
            "computation_nodes": [
                {
                    "id": f"prod::{node['name']}",
                    "name": node["name"],
                    "code": node["code"],
                    "dataAnalysts": ["creator@example.test"],
                    "currentUser": "creator@example.test",
                    "lastRun": None,
                    "computationStatus": "INITIAL",
                    "dataDependencies": [
                        {"id": dependency, "name": dependency} for dependency in node["dataDependencies"]
                    ],
                    "computationDependencies": node["computationDependencies"],
                    "nodeStatus": "PROD",
                    "changeId": node["changeId"],
                }
                for node in self.computation_nodes
            ],
            "permissions": [],
        }


def _creation_fixture(settings_factory, tmp_path, service: CreationService):
    runtime = tmp_path / "runtime"
    pack = tmp_path / "pack"
    dictionary = _write(
        tmp_path / "dictionaries" / "TIME-CHF_datadictionary.csv",
        "VARIABLENAME\nAGE\n",
    )
    _write(pack / "dcr-input" / "TIME-CHF.csv", "subject_id,AGE\npatient-1,72\n")
    cohort = SimpleNamespace(
        cohort_id="TIME-CHF",
        cohort_email=["owner@example.test"],
        administrator_email=None,
        metadata_filepath=str(dictionary),
        variables={},
        can_edit=True,
    )
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_synthetic_demo=True,
        aadcrv2_operation_journal=str(tmp_path / "operations.jsonl"),
        aadcrv2_room_url_template="http://rooms.test/dcr/{dcr_id}",
        data_folder=str(runtime),
        demo_pack_dir=str(pack),
        mapping_output_dir=str(tmp_path / "mappings"),
        decentriq_email="service@example.test",
        dev_mode=False,
    )
    backend = AadcrBackend(
        settings,
        client_factory=lambda _settings, _user: service,
        cohort_loader=lambda _email: {"TIME-CHF": cohort},
        merge_poll_attempts=3,
        merge_poll_interval=0,
        result_poll_interval=0,
    )
    request = {
        "cohorts": {"TIME-CHF": ["AGE"]},
        "dcr_name": "Resumable synthetic study",
        "research_question": "What does the aggregate fixture show?",
        "include_shuffled_samples": False,
        "session_id": "session-resume-123",
    }
    return backend, settings, request


def _mutation_calls(service: CreationService):
    return [
        call
        for call in service.calls
        if call["method"] == "POST"
        and call["kind"] == "json"
        and (call["path"] == "/api/dcr/" or "/dev/" in call["path"] or call["path"].endswith("/merge-requests/"))
    ]


def test_completed_session_replays_identical_response_without_a_second_room(settings_factory, tmp_path):
    service = CreationService()
    backend, settings, request = _creation_fixture(settings_factory, tmp_path, service)

    first = run(backend.create_live_room(request, {"email": "creator@example.test"}))
    calls_after_first = list(service.calls)
    replayed = run(backend.create_live_room(request, {"email": "creator@example.test"}))

    assert replayed.to_dict() == first.to_dict()
    assert service.room_count == 1
    assert service.calls == calls_after_first
    state = OperationJournal(settings.aadcrv2_operation_journal).load("session-resume-123")
    assert state.dcr_id == "room-1"
    assert state.confirmed_steps[-1] == "completed"
    assert state.final_response == first.to_dict()


def test_completed_session_rejects_changed_request_without_contacting_native(settings_factory, tmp_path):
    service = CreationService()
    backend, _settings, request = _creation_fixture(settings_factory, tmp_path, service)
    run(backend.create_live_room(request, {"email": "creator@example.test"}))
    calls_after_completion = list(service.calls)

    with pytest.raises(DcrOperationError) as caught:
        run(
            backend.create_live_room(
                request | {"dcr_name": "A different study"},
                {"email": "creator@example.test"},
            )
        )

    assert caught.value.status_code == 409
    assert caught.value.failed_step == "resume operation"
    assert service.calls == calls_after_completion


def test_retry_reconciles_a_native_dev_mutation_that_failed_before_response(
    settings_factory,
    tmp_path,
):
    service = CreationService(fail_participant_after_mutation_once=True)
    backend, settings, request = _creation_fixture(settings_factory, tmp_path, service)

    with pytest.raises(DcrOperationError) as interrupted:
        run(backend.create_live_room(request, {"email": "creator@example.test"}))
    state = OperationJournal(settings.aadcrv2_operation_journal).load("session-resume-123")

    assert interrupted.value.failed_step == "add DEV participant"
    assert state.confirmed_steps == ("room_created",)

    result = run(backend.create_live_room(request, {"email": "creator@example.test"}))
    participant_calls = [call for call in service.calls if call["path"].endswith("/dev/participants")]

    assert result.dcr_id == "room-1"
    assert service.room_count == 1
    assert [call["body"]["userEmail"] for call in participant_calls].count("creator@example.test") == 1


def test_retry_after_confirmed_merge_resumes_at_prod_without_repeating_mutations(
    settings_factory,
    tmp_path,
):
    service = CreationService(fail_prod_once=True)
    backend, settings, request = _creation_fixture(settings_factory, tmp_path, service)

    with pytest.raises(DcrOperationError) as interrupted:
        run(backend.create_live_room(request, {"email": "creator@example.test"}))
    mutation_calls = list(_mutation_calls(service))
    state = OperationJournal(settings.aadcrv2_operation_journal).load("session-resume-123")

    assert interrupted.value.dcr_id == "room-1"
    assert interrupted.value.failed_step == "read PROD view"
    assert "merged" in state.confirmed_steps

    result = run(backend.create_live_room(request, {"email": "creator@example.test"}))

    assert result.dcr_id == "room-1"
    assert _mutation_calls(service) == mutation_calls
    assert service.room_count == 1
    assert service.merge_count == 1


def test_retry_after_first_provision_skips_every_confirmed_upload_and_provision(
    settings_factory,
    tmp_path,
):
    service = CreationService(fail_upload_number_once=2)
    backend, settings, request = _creation_fixture(settings_factory, tmp_path, service)

    with pytest.raises(DcrOperationError) as interrupted:
        run(backend.create_live_room(request, {"email": "creator@example.test"}))
    state = OperationJournal(settings.aadcrv2_operation_journal).load("session-resume-123")

    assert interrupted.value.failed_step == "upload synthetic asset"
    assert any(step.startswith("provisioned:metadata:") for step in state.confirmed_steps)

    result = run(backend.create_live_room(request, {"email": "creator@example.test"}))
    upload_filenames = [call["filename"] for call in service.calls if call["kind"] == "upload"]
    provision_calls = [call for call in service.calls if call["path"].endswith("/provision-dataset")]

    assert result.dcr_id == "room-1"
    assert upload_filenames.count("TIME-CHF_datadictionary.csv") == 1
    assert upload_filenames.count("TIME-CHF.csv") == 2
    assert len(provision_calls) == 2
    assert service.room_count == 1
    assert service.merge_count == 1


def _prod_view():
    return {
        "participants": [
            {
                "id": "prod-participant-creator",
                "userEmail": "creator@example.test",
                "currentUser": "creator@example.test",
                "nodeStatus": "PROD",
                "changeId": "change-creator",
            },
            {
                "id": "prod-participant-owner",
                "userEmail": "owner@example.test",
                "currentUser": "creator@example.test",
                "nodeStatus": "PROD",
                "changeId": "change-owner",
            },
        ],
        "data_nodes": [
            {
                "id": "prod-row",
                "name": "TIME-CHF",
                "type": "FILE",
                "datasetName": None,
                "datasetId": None,
                "datasetStatus": "NOT_ADDED",
                "dataOwners": ["owner@example.test"],
                "currentUser": "creator@example.test",
                "schema": None,
                "nodeStatus": "PROD",
                "changeId": "change-row",
            },
            {
                "id": "prod-metadata",
                "name": "TIME-CHF_metadata_dictionary",
                "type": "FILE",
                "datasetName": "TIME-CHF_metadata_dictionary",
                "datasetId": "dataset-metadata",
                "datasetStatus": "ADDED",
                "dataOwners": ["creator@example.test"],
                "currentUser": "creator@example.test",
                "schema": None,
                "nodeStatus": "PROD",
                "changeId": "change-metadata",
            },
        ],
        "computation_nodes": [
            {
                "id": "prod-aggregate",
                "name": AGGREGATE_NODE_NAME,
                "code": "# aggregate local simulation",
                "dataAnalysts": ["creator@example.test"],
                "currentUser": "creator@example.test",
                "lastRun": None,
                "computationStatus": "INITIAL",
                "dataDependencies": [{"id": "prod-row", "name": "TIME-CHF"}],
                "computationDependencies": [],
                "nodeStatus": "PROD",
                "changeId": "change-aggregate",
            }
        ],
        "permissions": [
            {
                "id": "permission-owner",
                "userEmail": "owner@example.test",
                "permissionType": "DATA_OWNER",
                "resourceName": "TIME-CHF",
                "currentUser": "creator@example.test",
                "nodeStatus": "PROD",
                "changeId": "change-permission-owner",
            },
            {
                "id": "permission-analyst",
                "userEmail": "creator@example.test",
                "permissionType": "DATA_ANALYST",
                "resourceName": AGGREGATE_NODE_NAME,
                "currentUser": "creator@example.test",
                "nodeStatus": "PROD",
                "changeId": "change-permission-analyst",
            },
        ],
    }


class ReadService:
    def __init__(self):
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    async def request_json(self, method, path, *, failed_step, json_body=None, **_kwargs):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "failed_step": failed_step,
                "body": json_body,
            }
        )
        if method == "GET" and path == "/api/dcr/":
            return [
                {
                    "id": "room-read",
                    "name": "Native room",
                    "creatorEmail": "creator@example.test",
                    "version": 1,
                    "createdAt": "2026-07-15T09:00:00Z",
                    "updatedAt": "2026-07-15T09:30:00Z",
                }
            ]
        if method == "GET" and path == "/api/dcr/room-read/prod/view":
            return _prod_view()
        if method == "GET" and path == "/api/dcr/room-read/provisioned-datasets":
            return {
                "provisioned_datasets": [
                    {
                        "dataset_id": "dataset-metadata",
                        "dataset_node_id": "prod-metadata",
                        "provisioned_at": "2026-07-15T09:20:00Z",
                        "provisioned_by": "creator@example.test",
                        "provision_type": "PROD",
                        "dataset_name": "TIME-CHF dictionary",
                        "uploader": "creator@example.test",
                        "uploaded_at": "2026-07-15T09:19:00Z",
                        "data_size_kb": 4,
                    }
                ]
            }
        if method == "GET" and path == "/api/dcr/room-read/audit-logs":
            return {
                "audit_logs": [
                    {
                        "id": "audit-2",
                        "timestamp": "2026-07-15T09:30:00Z",
                        "user_email": "creator@example.test",
                        "action_label": "GET audit-logs",
                        "http_method": "GET",
                        "path": "/api/dcr/room-read/audit-logs",
                        "request_payload": None,
                    },
                    {
                        "id": "audit-1",
                        "timestamp": "2026-07-15T09:20:00Z",
                        "user_email": "creator@example.test",
                        "action_label": "Provision dataset",
                        "http_method": "POST",
                        "path": "/api/dcr/room-read/provision-dataset",
                        "request_payload": {
                            "authorization": "Bearer audit-secret",
                            "rows": [{"subject_id": "patient-secret"}],
                        },
                    },
                ]
            }
        raise AssertionError(f"Unexpected read request: {method} {path}")


def _read_backend(settings_factory, tmp_path, service, **backend_options):
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_operation_journal=str(tmp_path / "read-operations.jsonl"),
        aadcrv2_room_url_template="http://rooms.test/dcr/{dcr_id}",
    )
    backend = AadcrBackend(
        settings,
        client_factory=lambda _settings, _user: service,
        result_poll_interval=0,
        **backend_options,
    )
    return backend, settings


def _seed_room_journal(settings):
    return OperationJournal(settings.aadcrv2_operation_journal).append(
        "session-room-read",
        request_metadata={
            "dcr_name": "Native room",
            "research_question": "Can this room be normalized?",
            "selected_variables": {"TIME-CHF": ["AGE"]},
            "include_shuffled_samples": False,
            "include_mapping_upload_slot": False,
            "selected_mapping_files": [],
            "synthetic_demo": True,
        },
        request_fingerprint="a" * 64,
        cohort_ids=["TIME-CHF"],
        dcr_id="room-read",
        confirmed_steps=["room_created", "merged"],
        final_response=None,
    )


def test_room_list_refresh_and_last_modified_normalize_native_state(settings_factory, tmp_path):
    service = ReadService()
    backend, settings = _read_backend(settings_factory, tmp_path, service)
    durable_state = _seed_room_journal(settings)

    listed = run(backend.list_rooms({"email": "creator@example.test"}, refresh=False)).to_dict()
    refreshed = run(backend.list_rooms({"email": "creator@example.test"}, refresh=True)).to_dict()
    modified = run(backend.rooms_last_modified({"email": "creator@example.test"}))

    assert listed["count"] == 1
    assert listed["email"] == "creator@example.test"
    assert listed["provider"] == "aadcrv2"
    room = listed["dcrs"][0]
    assert room["id"] == "room-read"
    assert room["title"] == "Native room"
    assert room["description"] == "RESEARCH QUESTION: Can this room be normalized?."
    assert room["owner"] == {"email": "creator@example.test"}
    assert room["cohorts"] == ["TIME-CHF"]
    assert room["dcr_url"] == "http://rooms.test/dcr/room-read"
    assert room["provider"] == "aadcrv2"
    assert {participant["email"] for participant in room["participants"]} == {
        "creator@example.test",
        "owner@example.test",
    }
    owner = next(participant for participant in room["participants"] if participant["email"] == "owner@example.test")
    assert owner["data_owner_of"] == ["TIME-CHF"]
    assert room["provisioned_datasets"] == [
        {
            "dataset_id": "dataset-metadata",
            "dataset_node_id": "prod-metadata",
            "dataset_name": "TIME-CHF dictionary",
            "node_name": "TIME-CHF_metadata_dictionary",
            "status": "provisioned",
            "provision_type": "PROD",
            "provisioned_at": "2026-07-15T09:20:00Z",
            "provisioned_by": "creator@example.test",
            "uploader": "creator@example.test",
            "uploaded_at": "2026-07-15T09:19:00Z",
            "data_size_kb": 4,
        }
    ]
    assert listed["refresh_summary"] == {"refreshed": False, "count": 1}
    assert refreshed["refresh_summary"] == {"refreshed": True, "count": 1}
    assert modified == durable_state.timestamp


def test_audit_aliases_map_native_rows_without_payload_or_sensitive_stdout(settings_factory, tmp_path):
    service = ReadService()
    backend, _settings = _read_backend(settings_factory, tmp_path, service)

    full = run(backend.audit_log("room-read", {"email": "creator@example.test"}, main_only=False))
    main = run(backend.audit_log("room-read", {"email": "creator@example.test"}, main_only=True))

    assert [entry["desc"] for entry in full] == ["GET audit-logs", "Provision dataset"]
    assert [entry["desc"] for entry in main] == ["Provision dataset"]
    assert main[0] == {
        "timestamp": "2026-07-15T09:20:00Z",
        "user": "creator@example.test",
        "desc": "Provision dataset",
        "provider": "aadcrv2",
        "source": {
            "id": "audit-1",
            "method": "POST",
            "path": "/api/dcr/room-read/provision-dataset",
        },
    }
    assert "audit-secret" not in repr(full)
    assert "patient-secret" not in repr(full)


def _result(status: str, *, results: str | None = None, stdout: str = "", message: str | None = None):
    return {
        "status": status,
        "message": message or f"Execution is {status.lower()}",
        "stdout": stdout,
        "files": ["untrusted/../upstream-name.json"],
        "results": results,
    }


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class ComputationService:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.last_result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    async def request_json(self, method, path, *, failed_step, json_body=None, **_kwargs):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "failed_step": failed_step,
                "body": json_body,
            }
        )
        if method == "GET" and path.endswith("/prod/view"):
            return _prod_view()
        if method == "POST" and path.endswith("/computation-nodes/run"):
            return {"status": "QUEUED"}
        if method == "POST" and path.endswith("/computation-nodes/results"):
            outcome = self.results.pop(0) if self.results else self.last_result
            if isinstance(outcome, Exception):
                raise outcome
            self.last_result = outcome
            return outcome
        raise AssertionError(f"Unexpected computation request: {method} {path}")


def _result_calls(service: ComputationService, suffix: str):
    return [call for call in service.calls if call["path"].endswith(suffix)]


def test_no_execution_starts_once_then_polls_queued_running_and_completed_zip(
    settings_factory,
    tmp_path,
):
    archive = _zip_bytes({"aggregate-summary.json": b'{"nodes":{"TIME-CHF":{"rows":2}}}'})
    results_path = "/api/dcr/room-read/computation-nodes/results"
    service = ComputationService(
        [
            _upstream_error(
                results_path,
                "read computation result",
                status_code=404,
                detail="No execution found for this computation node in PROD environment",
            ),
            _result("QUEUED"),
            _result("RUNNING"),
            _result(
                "COMPLETED",
                results=base64.b64encode(archive).decode("ascii"),
                stdout="patient-secret must never be returned",
            ),
        ]
    )
    backend, _settings = _read_backend(
        settings_factory,
        tmp_path,
        service,
        result_poll_attempts=3,
    )

    response = run(backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert isinstance(response, Response)
    assert response.media_type == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="aggregate-result.zip"'
    assert response.body == archive
    assert len(_result_calls(service, "/computation-nodes/run")) == 1
    assert len(_result_calls(service, "/computation-nodes/results")) == 4
    bodies = [call["body"] for call in service.calls if "/computation-nodes/" in call["path"]]
    assert all(body == {"computationNodeId": "prod-aggregate", "environment": "PROD"} for body in bodies)


def test_completed_result_is_fetched_without_starting_a_duplicate_run(settings_factory, tmp_path):
    archive = _zip_bytes({"aggregate-summary.json": b"{}"})
    service = ComputationService([_result("COMPLETED", results=base64.b64encode(archive).decode("ascii"))])
    backend, _settings = _read_backend(settings_factory, tmp_path, service)

    response = run(backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert response.body == archive
    assert _result_calls(service, "/computation-nodes/run") == []
    assert len(_result_calls(service, "/computation-nodes/results")) == 1


def test_null_result_response_fails_closed_without_starting_or_polling(settings_factory, tmp_path):
    service = ComputationService([None])
    backend, _settings = _read_backend(settings_factory, tmp_path, service)

    with pytest.raises(DcrOperationError) as caught:
        run(backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert caught.value.failed_step == "poll computation result"
    assert _result_calls(service, "/computation-nodes/run") == []
    assert len(_result_calls(service, "/computation-nodes/results")) == 1


def test_failed_execution_and_timeout_are_safe_and_never_expose_native_stdout(
    settings_factory,
    tmp_path,
):
    failed_service = ComputationService(
        [
            _result(
                "FAILED",
                stdout="patient-secret-from-stdout",
                message="failure included patient-secret-from-message",
            )
        ]
    )
    failed_backend, _settings = _read_backend(settings_factory, tmp_path, failed_service)

    with pytest.raises(DcrOperationError) as failed:
        run(failed_backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert failed.value.failed_step == "run aggregate computation"
    assert "patient-secret" not in str(failed.value)
    assert "patient-secret" not in repr(failed.value.to_dict())

    queued_service = ComputationService([_result("QUEUED")])
    queued_backend, _settings = _read_backend(
        settings_factory,
        tmp_path / "queued",
        queued_service,
        result_poll_attempts=3,
    )
    with pytest.raises(DcrOperationError) as timed_out:
        run(queued_backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert timed_out.value.failed_step == "poll computation result"
    assert timed_out.value.retryable is True
    assert timed_out.value.status_code == 503
    assert len(_result_calls(queued_service, "/computation-nodes/results")) == 3
    assert _result_calls(queued_service, "/computation-nodes/run") == []


@pytest.mark.parametrize(
    ("encoded", "expected_detail"),
    [
        ("%%%not-base64%%%", "base64"),
        (base64.b64encode(b"not-a-zip").decode("ascii"), "ZIP"),
        (
            base64.b64encode(_zip_bytes({"../outside.json": b"secret"})).decode("ascii"),
            "unsafe member path",
        ),
    ],
)
def test_invalid_or_path_traversing_result_archives_fail_closed(
    settings_factory,
    tmp_path,
    encoded,
    expected_detail,
):
    service = ComputationService(
        [
            _result(
                "COMPLETED",
                results=encoded,
                stdout="patient-secret-from-stdout",
            )
        ]
    )
    backend, _settings = _read_backend(settings_factory, tmp_path, service)

    with pytest.raises(DcrOperationError) as caught:
        run(backend.computation_output("room-read", {"email": "creator@example.test"}))

    assert caught.value.failed_step == "decode computation result"
    assert expected_detail in caught.value.safe_detail
    assert "patient-secret" not in str(caught.value)


def test_oversized_or_overpopulated_result_archives_are_rejected_before_download(
    settings_factory,
    tmp_path,
):
    oversized = base64.b64encode(b"x" * 129).decode("ascii")
    oversized_service = ComputationService([_result("COMPLETED", results=oversized)])
    oversized_backend, _settings = _read_backend(
        settings_factory,
        tmp_path / "oversized",
        oversized_service,
        result_archive_max_bytes=128,
    )
    with pytest.raises(DcrOperationError, match="size limit"):
        run(oversized_backend.computation_output("room-read", {"email": "creator@example.test"}))

    many_members = _zip_bytes({"one.json": b"{}", "two.json": b"{}"})
    members_service = ComputationService([_result("COMPLETED", results=base64.b64encode(many_members).decode("ascii"))])
    members_backend, _settings = _read_backend(
        settings_factory,
        tmp_path / "members",
        members_service,
        result_archive_max_members=1,
    )
    with pytest.raises(DcrOperationError, match="too many members"):
        run(members_backend.computation_output("room-read", {"email": "creator@example.test"}))


def test_result_archive_rejects_uncompressed_bombs_and_symlink_members(
    settings_factory,
    tmp_path,
):
    expanded = _zip_bytes({"aggregate-summary.json": b"x" * 1024})
    assert len(expanded) < 256
    expanded_service = ComputationService([_result("COMPLETED", results=base64.b64encode(expanded).decode("ascii"))])
    expanded_backend, _settings = _read_backend(
        settings_factory,
        tmp_path / "expanded",
        expanded_service,
        result_archive_max_bytes=256,
    )
    with pytest.raises(DcrOperationError, match="size limit"):
        run(expanded_backend.computation_output("room-read", {"email": "creator@example.test"}))

    symlink_buffer = io.BytesIO()
    with zipfile.ZipFile(symlink_buffer, "w") as archive:
        link = zipfile.ZipInfo("aggregate-summary.json")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "outside.json")
    symlink_service = ComputationService(
        [
            _result(
                "COMPLETED",
                results=base64.b64encode(symlink_buffer.getvalue()).decode("ascii"),
            )
        ]
    )
    symlink_backend, _settings = _read_backend(
        settings_factory,
        tmp_path / "symlink",
        symlink_service,
    )
    with pytest.raises(DcrOperationError, match="unsafe member type"):
        run(symlink_backend.computation_output("room-read", {"email": "creator@example.test"}))


def test_shuffle_capability_response_never_contacts_native_or_decentriq(settings_factory, tmp_path):
    service = ComputationService([])
    backend, _settings = _read_backend(settings_factory, tmp_path, service)

    result = run(backend.shuffle_output("room-read", {"email": "creator@example.test"}))

    assert result == {
        "status": "unsupported",
        "detail": "AADCR v2 does not expose the Decentriq shuffle computation",
        "dcr_id": "room-read",
        "provider": "aadcrv2",
    }
    assert service.calls == []
