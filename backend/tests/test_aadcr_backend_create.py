import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import get_current_user
from src.dcr_backends.aadcr_backend import AadcrBackend
from src.dcr_backends.aadcr_client import AadcrUpstreamError
from src.dcr_backends.aadcr_translation import DcrOperationError
from src.dcr_routes import get_backend, router

TEST_AADCR_SECRET = "backend-create-secret"


def run(coroutine):
    return asyncio.run(coroutine)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _cohort(cohort_id: str, dictionary: Path, owner: str, *, can_edit: bool = True):
    return SimpleNamespace(
        cohort_id=cohort_id,
        cohort_email=[owner],
        administrator_email=None,
        metadata_filepath=str(dictionary),
        variables={},
        can_edit=can_edit,
    )


class RecordingClient:
    def __init__(self, *, missing_prod_node: str | None = None, fail_upload: bool = False):
        self.calls: list[dict] = []
        self.participants: list[dict] = []
        self.data_nodes: list[dict] = []
        self.computation_nodes: list[dict] = []
        self.permissions: list[dict] = []
        self.rooms: list[dict] = []
        self.missing_prod_node = missing_prod_node
        self.fail_upload = fail_upload

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
        if method == "GET" and path == "/api/dcr/":
            return list(self.rooms)
        if method == "POST" and path == "/api/dcr/":
            room = {
                "id": "room-123",
                "name": json_body["name"],
                "creatorEmail": "creator@example.test",
            }
            self.rooms.append(room)
            return room
        if method == "PATCH" and path == "/api/dcr/room-123":
            self.rooms[0]["name"] = json_body["name"]
            return self.rooms[0]
        if path.endswith("/dev/participants"):
            response = {
                "id": f"dev-participant-{len(self.participants)}",
                "userEmail": json_body["userEmail"],
                "changeId": f"change-participant-{len(self.participants)}",
            }
            self.participants.append(response)
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
                return [{**value, "nodeStatus": "DEV_ADDED"} for value in values]

            return {
                "participants": changed(self.participants),
                "data_nodes": changed(self.data_nodes),
                "computation_nodes": changed(self.computation_nodes),
                "permissions": changed(self.permissions),
            }
        if method == "POST" and path.endswith("/merge-requests/"):
            return {"id": "merge-1", "status": "MERGED"}
        if method == "GET" and path.endswith("/merge-requests/merge-1"):
            return {"id": "merge-1", "status": "MERGED", "approvals": []}
        if method == "GET" and path.endswith("/prod/view"):
            return {
                "participants": [
                    {"id": f"prod-participant-{index}", "userEmail": participant["userEmail"]}
                    for index, participant in enumerate(self.participants)
                ],
                "data_nodes": [
                    {"id": f"prod::{node['name']}", "name": node["name"], "type": "FILE"}
                    for node in self.data_nodes
                    if node["name"] != self.missing_prod_node
                ],
                "computation_nodes": [
                    {"id": f"prod::{node['name']}", "name": node["name"]} for node in self.computation_nodes
                ],
                "permissions": [],
            }
        if method == "POST" and path.endswith("/provision-dataset"):
            return {"success": True, **json_body}
        raise AssertionError(f"Unexpected request: {method} {path}")

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
        if self.fail_upload:
            raise AadcrUpstreamError(
                method="POST",
                path=path,
                detail="fixture upload failed",
                failed_step=failed_step,
                status_code=400,
                retryable=False,
            )
        return {
            "success": True,
            "dataset_id": f"dataset-{filename}",
            "dataset_name": form["dataset_name"],
        }


def _fixture(
    settings_factory,
    tmp_path: Path,
    *,
    synthetic: bool = True,
    fail_upload: bool = False,
    merge_poll_attempts: int = 3,
):
    runtime = tmp_path / "runtime"
    pack = tmp_path / "pack"
    mappings = tmp_path / "mappings"
    time_dictionary = _write(tmp_path / "dictionaries" / "TIME-CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    gissi_dictionary = _write(
        tmp_path / "dictionaries" / "GISSI-HF_datadictionary.csv",
        "VARIABLENAME\nWEIGHT\n",
    )
    mapping = _write(mappings / "gissi-time.csv", "from,to\nWEIGHT,weight\n")
    _write(runtime / "dcr_output_TIME-CHF" / "shuffled_sample.csv", "AGE\n70\n")
    _write(pack / "dcr-input" / "TIME-CHF.csv", "AGE\n72\n")
    _write(pack / "dcr-input" / "GISSI-HF.csv", "WEIGHT\n81\n")
    cohorts = {
        "TIME-CHF": _cohort("TIME-CHF", time_dictionary, "time-owner@example.test"),
        "GISSI-HF": _cohort("GISSI-HF", gissi_dictionary, "gissi-owner@example.test"),
    }
    request = {
        "cohorts": {"TIME-CHF": ["AGE"], "GISSI-HF": ["WEIGHT"]},
        "dcr_name": "Local AADCR study",
        "research_question": "What do the aggregate synthetic summaries show?",
        "include_shuffled_samples": {"TIME-CHF": True, "GISSI-HF": False},
        "additional_analysts": ["analyst@example.test"],
        "selected_mapping_files": [
            {
                "filename": mapping.name,
                "filepath": str(mapping),
                "cohorts": ["GISSI-HF", "TIME-CHF"],
            }
        ],
        "include_mapping_upload_slot": True,
        "session_id": "session-123",
    }
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_url="http://aadcr.test",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_synthetic_demo=synthetic,
        aadcrv2_room_url_template="http://rooms.test/dcr/{dcr_id}",
        data_folder=str(runtime),
        demo_pack_dir=str(pack),
        mapping_output_dir=str(mappings),
        decentriq_email="service@example.test",
        dev_mode=False,
    )
    client = RecordingClient(fail_upload=fail_upload)
    backend = AadcrBackend(
        settings,
        client_factory=lambda _settings, _user: client,
        cohort_loader=lambda _email: cohorts,
        merge_poll_attempts=merge_poll_attempts,
        merge_poll_interval=0,
    )
    return backend, client, request


def test_explicit_zero_merge_poll_budget_is_not_silently_replaced(settings_factory, tmp_path):
    backend, client, request = _fixture(settings_factory, tmp_path, merge_poll_attempts=0)

    with pytest.raises(DcrOperationError) as caught:
        run(backend.create_live_room(request, {"email": "creator@example.test"}))

    assert caught.value.failed_step == "poll merge request"
    assert "after 0 polls" in caught.value.safe_detail
    assert all(not call["path"].endswith("/merge-requests/merge-1") for call in client.calls)


def test_create_live_room_uses_exact_native_dev_merge_prod_and_provision_sequence(settings_factory, tmp_path):
    backend, client, request = _fixture(settings_factory, tmp_path)

    result = run(backend.create_live_room(request, {"email": "creator@example.test"}))

    calls = client.calls
    assert calls[0] == {
        "kind": "json",
        "method": "GET",
        "path": "/api/dcr/",
        "failed_step": "reconcile DCR creation",
        "body": None,
    }
    assert calls[1]["method"] == "POST"
    assert calls[1]["path"] == "/api/dcr/"
    pending_name = calls[1]["body"]["name"]
    assert re.fullmatch(
        r"Local AADCR study - created by creator@example\.test \[ce-operation:[0-9a-f]{64}\]",
        pending_name,
    )
    assert request["session_id"] not in pending_name
    assert calls[2] == {
        "kind": "json",
        "method": "PATCH",
        "path": "/api/dcr/room-123",
        "failed_step": "finalize DCR name",
        "body": {"name": "Local AADCR study - created by creator@example.test"},
    }

    participant_calls = [call for call in calls if call["path"].endswith("/dev/participants")]
    assert [call["body"]["userEmail"] for call in participant_calls] == [
        "creator@example.test",
        "gissi-owner@example.test",
        "time-owner@example.test",
        "service@example.test",
        "analyst@example.test",
    ]

    data_calls = [call for call in calls if call["path"].endswith("/dev/data-nodes")]
    assert [call["body"] for call in data_calls] == [
        {"name": "GISSI-HF", "type": "FILE"},
        {"name": "GISSI-HF_metadata_dictionary", "type": "FILE"},
        {"name": "TIME-CHF", "type": "FILE"},
        {"name": "TIME-CHF_metadata_dictionary", "type": "FILE"},
        {"name": "TIME-CHF_shuffled_sample", "type": "FILE"},
        {"name": "GISSI-HF_TIME-CHF_mapping", "type": "FILE"},
        {"name": "CrossStudyMappings", "type": "FILE"},
    ]

    computation_calls = [call for call in calls if call["path"].endswith("/dev/computation-nodes")]
    assert [call["body"]["name"] for call in computation_calls] == [
        "metadata-preview-local-simulation",
        "aggregate-summary-local-simulation",
    ]
    dev_data_ids = {node["id"] for node in client.data_nodes}
    assert all(set(call["body"]["dataDependencies"]) <= dev_data_ids for call in computation_calls)
    assert "not a confidential" in computation_calls[1]["body"]["code"].lower()

    permission_calls = [call for call in calls if call["path"].endswith("/dev/permissions")]
    permission_types = [call["body"]["type"] for call in permission_calls]
    first_analyst = permission_types.index("DATA_ANALYST")
    assert set(permission_types[:first_analyst]) == {"DATA_OWNER"}
    assert set(permission_types[first_analyst:]) == {"DATA_ANALYST"}

    dev_view_index = next(index for index, call in enumerate(calls) if call["path"].endswith("/dev/view"))
    merge_create_index = next(
        index
        for index, call in enumerate(calls)
        if call["method"] == "POST" and call["path"].endswith("/merge-requests/")
    )
    merge_get_index = next(
        index for index, call in enumerate(calls) if call["path"].endswith("/merge-requests/merge-1")
    )
    prod_view_index = next(index for index, call in enumerate(calls) if call["path"].endswith("/prod/view"))
    first_upload_index = next(index for index, call in enumerate(calls) if call["kind"] == "upload")
    assert dev_view_index < merge_create_index < merge_get_index < prod_view_index < first_upload_index

    merge_body = calls[merge_create_index]["body"]
    expected_change_ids = [
        value["changeId"]
        for values in (client.participants, client.data_nodes, client.computation_nodes, client.permissions)
        for value in values
    ]
    assert merge_body["change_ids"] == expected_change_ids
    assert len(set(merge_body["change_ids"])) == len(merge_body["change_ids"])
    assert all("/approve" not in call["path"] for call in calls)

    uploads = [call for call in calls if call["kind"] == "upload"]
    assert [call["filename"] for call in uploads] == [
        "GISSI-HF_datadictionary.csv",
        "TIME-CHF_datadictionary.csv",
        "gissi-time.csv",
        "shuffled_sample.csv",
        "GISSI-HF.csv",
        "TIME-CHF.csv",
    ]
    provisions = [call for call in calls if call["path"].endswith("/provision-dataset")]
    assert len(provisions) == len(uploads)
    assert [call["body"]["dataset_node_id"] for call in provisions] == [
        "prod::GISSI-HF_metadata_dictionary",
        "prod::TIME-CHF_metadata_dictionary",
        "prod::GISSI-HF_TIME-CHF_mapping",
        "prod::TIME-CHF_shuffled_sample",
        "prod::GISSI-HF",
        "prod::TIME-CHF",
    ]
    assert all(call["body"]["provision_type"] == "PROD" for call in provisions)

    assert result.dcr_id == "room-123"
    assert result.dcr_url == "http://rooms.test/dcr/room-123"
    assert result.provider == "aadcrv2"
    assert result.metadata_uploads_successful == 2
    assert result.mapping_uploads_successful == 1
    assert result.shuffled_uploads_successful == 1
    assert result.row_upload_results == {"GISSI-HF": "success", "TIME-CHF": "success"}
    assert result.capabilities.local_simulation is True


def test_upload_failure_is_normalized_with_created_room_id(settings_factory, tmp_path):
    backend, _client, request = _fixture(settings_factory, tmp_path, fail_upload=True)

    with pytest.raises(DcrOperationError) as caught:
        run(backend.create_live_room(request, {"email": "creator@example.test"}))

    assert caught.value.failed_step == "upload metadata asset"
    assert caught.value.dcr_id == "room-123"
    assert caught.value.status_code == 400
    assert caught.value.to_dict() == {
        "detail": "fixture upload failed",
        "provider": "aadcrv2",
        "failed_step": "upload metadata asset",
        "dcr_id": "room-123",
        "retryable": False,
        "status_code": 400,
    }


def test_provider_error_reaches_existing_route_as_structured_safe_detail(settings_factory, tmp_path):
    backend, _client, request = _fixture(settings_factory, tmp_path, fail_upload=True)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"email": "creator@example.test"}
    app.dependency_overrides[get_backend] = lambda: backend

    with TestClient(app, raise_server_exceptions=False) as route_client:
        response = route_client.post("/create-live-compute-dcr", json=request)

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "detail": "fixture upload failed",
            "provider": "aadcrv2",
            "failed_step": "upload metadata asset",
            "dcr_id": "room-123",
            "retryable": False,
            "status_code": 400,
        }
    }


def test_live_creation_with_synthetic_mode_disabled_omits_row_csvs(settings_factory, tmp_path):
    backend, client, request = _fixture(settings_factory, tmp_path, synthetic=False)

    result = run(backend.create_live_room(request, {"email": "creator@example.test"}))

    uploaded_names = [call["filename"] for call in client.calls if call["kind"] == "upload"]
    assert "GISSI-HF.csv" not in uploaded_names
    assert "TIME-CHF.csv" not in uploaded_names
    assert result.row_upload_results == {}


def test_single_cohort_provision_room_preserves_existing_edit_authorization(settings_factory, tmp_path):
    backend, client, _request = _fixture(settings_factory, tmp_path, synthetic=False)
    dictionary = Path(tmp_path / "dictionaries" / "TIME-CHF_datadictionary.csv")
    backend._cohort_loader = lambda _email: {
        "TIME-CHF": _cohort(
            "TIME-CHF",
            dictionary,
            "time-owner@example.test",
            can_edit=False,
        )
    }

    with pytest.raises(DcrOperationError) as caught:
        run(
            backend.create_provision_room(
                {"cohort_id": "TIME-CHF"},
                {"email": "creator@example.test"},
            )
        )

    assert caught.value.failed_step == "authorize provision room"
    assert caught.value.status_code == 403
    assert client.calls == []
