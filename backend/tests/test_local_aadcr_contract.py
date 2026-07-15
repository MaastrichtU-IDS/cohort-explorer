"""Contract tests for the local Cohort Explorer to AADCR smoke runner."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from jose import JWTError

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "backend" / "scripts" / "smoke_local_aadcr.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_local_aadcr", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    return buffer.getvalue()


def test_smoke_http_boundary_rejects_non_loopback_and_sanitizes_failures():
    module = _load_smoke_module()

    with pytest.raises(module.SmokeError, match="loopback"):
        module.SmokeHttp("https://example.com")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            text='Bearer super-secret; Cookie=session-secret; {"patient_row":"raw"}',
        )

    with (
        module.SmokeHttp(
            "http://127.0.0.1:3000",
            transport=httpx.MockTransport(handler),
        ) as client,
        pytest.raises(module.SmokeError) as failure,
    ):
        client.json(
            "GET",
            "/failing-step",
            step="read room",
            expected_status={200},
            required={"id": str},
        )

    message = str(failure.value)
    assert message == "read room failed with HTTP 502"
    assert "super-secret" not in message
    assert "session-secret" not in message
    assert "patient_row" not in message


def test_smoke_http_boundary_fails_closed_on_response_shape():
    module = _load_smoke_module()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 123, "token": "must-not-leak"})

    with (
        module.SmokeHttp(
            "http://localhost:3000",
            transport=httpx.MockTransport(handler),
        ) as client,
        pytest.raises(module.SmokeError) as failure,
    ):
        client.json(
            "GET",
            "/wrong-shape",
            step="read room",
            expected_status={200},
            required={"id": str},
        )

    assert str(failure.value) == "read room returned an invalid id"
    assert "must-not-leak" not in str(failure.value)


def test_smoke_http_boundary_ignores_proxy_environment():
    module = _load_smoke_module()

    with module.SmokeHttp(
        "http://127.0.0.1:3000",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
    ) as client:
        assert client._client._trust_env is False


def test_evidence_writer_accepts_only_bounded_ids_hashes_counts_and_statuses(tmp_path):
    module = _load_smoke_module()
    evidence_path = tmp_path / "smoke-evidence.json"
    evidence = {
        "schema_version": 1,
        "room_id": "room-123",
        "merge_request_id": "merge-123",
        "preview_sha256": "a" * 64,
        "result_sha256": "b" * 64,
        "cohort_count": 2,
        "room_count": 1,
        "computation_status": "COMPLETED",
        "idempotent": True,
    }

    module.write_evidence(evidence_path, evidence)

    assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
    assert evidence_path.stat().st_mode & 0o777 == 0o600

    for forbidden in (
        {**evidence, "token": "secret"},
        {**evidence, "cookie": "secret"},
        {**evidence, "authorization": "Bearer secret"},
        {**evidence, "rows": [{"patient_id": 1}]},
    ):
        with pytest.raises(module.SmokeError, match="unsafe evidence"):
            module.write_evidence(evidence_path, forbidden)


def test_room_request_is_deterministic_and_uses_manifest_mapping_semantics(tmp_path):
    module = _load_smoke_module()
    (tmp_path / "manifest.json").write_bytes(b'{"schema_version":1}\n')
    manifest = SimpleNamespace(
        root=tmp_path,
        selected_mapping_rows=(
            {"source": "AGE", "target": "age_years"},
            {"source": "SEX", "target": "sex_at_birth"},
            {"source": "AGE", "target": "age_years"},
        ),
    )
    mapping = {
        "filename": "time-chf_gissi-hf_full.csv",
        "filepath": "/demo-runtime/mappings/time-chf_gissi-hf_full.csv",
        "display_name": "TIME-CHF to GISSI-HF",
        "cohorts": ["TIME-CHF", "GISSI-HF"],
        "ignored": "not forwarded",
    }

    first = module._build_room_request(manifest, mapping)
    second = module._build_room_request(manifest, mapping)

    assert first == second
    assert first["cohorts"] == {
        "GISSI-HF": ["age_years", "sex_at_birth"],
        "TIME-CHF": ["AGE", "SEX"],
    }
    assert first["session_id"].startswith("local-demo-")
    assert len(first["session_id"]) == len("local-demo-") + 24
    assert first["selected_mapping_files"] == [
        {
            "filename": mapping["filename"],
            "filepath": mapping["filepath"],
            "display_name": mapping["display_name"],
            "cohorts": mapping["cohorts"],
        }
    ]
    assert first["include_shuffled_samples"] == {
        "GISSI-HF": True,
        "TIME-CHF": True,
    }
    assert first["airlock_settings"] == {"GISSI-HF": 0, "TIME-CHF": 0}


def test_tampered_token_changes_signed_bytes_and_is_rejected():
    module = _load_smoke_module()
    token = module._mint_aadcr_token("owner@example.test", "s" * 64)

    tampered = module._tamper_token_signature(token)

    assert tampered != token
    options = {
        "algorithms": [module.JWT_ALGORITHM],
        "audience": "aadcrv2-local",
        "issuer": "cohort-explorer-local",
    }
    assert module.jwt.decode(token, "s" * 64, **options)["email"] == "owner@example.test"
    with pytest.raises(JWTError):
        module.jwt.decode(tampered, "s" * 64, **options)


def test_explicit_run_nonce_forces_a_fresh_room_while_remaining_replayable(tmp_path):
    module = _load_smoke_module()
    (tmp_path / "manifest.json").write_bytes(b'{"schema_version":1}\n')
    manifest = SimpleNamespace(
        root=tmp_path,
        selected_mapping_rows=(
            {"source": "AGE", "target": "age_years"},
            {"source": "SEX", "target": "sex_at_birth"},
        ),
    )
    mapping = {
        "filename": "time-chf_gissi-hf_full.csv",
        "filepath": "/demo-runtime/mappings/time-chf_gissi-hf_full.csv",
        "display_name": "TIME-CHF to GISSI-HF",
        "cohorts": ["TIME-CHF", "GISSI-HF"],
    }

    first = module._build_room_request(manifest, mapping, run_nonce="a" * 24)
    replay = module._build_room_request(manifest, mapping, run_nonce="a" * 24)
    next_run = module._build_room_request(manifest, mapping, run_nonce="b" * 24)

    assert first == replay
    assert first["session_id"] != next_run["session_id"]


def test_definition_inspector_requires_exact_metadata_fixture_members():
    module = _load_smoke_module()
    owner = "nikolas.molyndris@decentriq.ch"
    data_names = sorted(module.EXPECTED_DATA_NODES)
    computation_names = sorted(module.EXPECTED_COMPUTATION_NODES)
    selected_variables = {"GISSI-HF": ["age_years"], "TIME-CHF": ["AGE"]}
    assets = {
        "mapping_files/time-chf_gissi-hf_full.csv": ("mapping", b"from,to\nAGE,age_years\n"),
        "metadata_dictionaries/GISSI-HF_datadictionary.csv": ("metadata", b"VARIABLENAME\nage_years\n"),
        "metadata_dictionaries/TIME-CHF_datadictionary.csv": ("metadata", b"VARIABLENAME\nAGE\n"),
        "shuffled_samples/GISSI-HF_shuffled_sample.csv": ("shuffled", b"age_years\n70\n"),
        "shuffled_samples/TIME-CHF_shuffled_sample.csv": ("shuffled", b"AGE\n72\n"),
    }
    provenance = {
        "files": [
            {
                "archive_path": path,
                "kind": kind,
                "sha256": module._sha256(content),
            }
            for path, (kind, content) in sorted(assets.items())
        ],
        "format_version": 1,
        "provider": "aadcrv2",
        "synthetic_fixture": True,
    }
    members = {
        "dcr_config.json": json.dumps(
            {
                "dataScienceDataRoom": {
                    "provider": "aadcrv2",
                    "synthetic_demo": True,
                    "local_simulation": True,
                    "confidential_boundary": False,
                    "cohorts": ["GISSI-HF", "TIME-CHF"],
                    "data_nodes": [{"name": name, "type": "FILE"} for name in data_names],
                    "computation_nodes": [{"name": name} for name in computation_names],
                    "participants": [
                        {
                            "email": owner,
                            "data_owner_of": data_names,
                            "analyst_of": computation_names,
                        }
                    ],
                    "permissions": [{"email": owner, "node": name, "type": "DATA_OWNER"} for name in data_names]
                    + [{"email": owner, "node": name, "type": "DATA_ANALYST"} for name in computation_names],
                    "selected_variables": selected_variables,
                }
            }
        ).encode(),
        "fixture-provenance.json": json.dumps(provenance).encode(),
        **{path: content for path, (_kind, content) in assets.items()},
    }

    expected_hashes = {path: module._sha256(content) for path, (_kind, content) in assets.items()}
    digest, count = module._inspect_definition(
        _zip_bytes(members),
        expected_asset_hashes=expected_hashes,
        owner_email=owner,
        expected_selected_variables=selected_variables,
    )

    assert len(digest) == 64
    assert count == 7
    with pytest.raises(module.SmokeError, match="exact metadata fixture"):
        module._inspect_definition(
            _zip_bytes({name: content for name, content in members.items() if "mapping_files/" not in name}),
            expected_asset_hashes=expected_hashes,
            owner_email=owner,
            expected_selected_variables=selected_variables,
        )


def test_result_inspector_requires_exact_fixture_aggregate_values(tmp_path):
    module = _load_smoke_module()
    row_paths = {}
    for cohort, content in {
        "GISSI-HF": "AGE,LABEL\n70,alpha\n,beta\n72,\n",
        "TIME-CHF": "AGE,LABEL\n10,alpha\noops,beta\n20,\n",
    }.items():
        path = tmp_path / f"{cohort}.csv"
        path.write_text(content, encoding="utf-8")
        row_paths[cohort] = path
    manifest = SimpleNamespace(rows=lambda cohort: row_paths[cohort])
    summary = {
        "confidential_boundary": False,
        "local_simulation": True,
        "nodes": module._expected_aggregate_nodes(manifest),
    }
    content = _zip_bytes({"aggregate-summary.json": json.dumps(summary).encode()})

    digest, count = module._inspect_result(
        content,
        manifest=manifest,
    )

    assert len(digest) == 64
    assert count == 1
    with pytest.raises(module.SmokeError, match="aggregate-only"):
        module._inspect_result(
            _zip_bytes(
                {
                    "aggregate-summary.json": json.dumps(summary).encode(),
                    "raw.csv": b"patient_id\n1\n",
                }
            ),
            manifest=manifest,
        )
    wrong_values = deepcopy(summary)
    wrong_values["nodes"]["GISSI-HF"]["non_empty"] = {"AGE": 0, "LABEL": 0}
    wrong_values["nodes"]["GISSI-HF"]["numeric"] = {}
    with pytest.raises(module.SmokeError, match="fixture aggregate"):
        module._inspect_result(
            _zip_bytes({"aggregate-summary.json": json.dumps(wrong_values).encode()}),
            manifest=manifest,
        )


def test_smoke_origins_must_match_the_selected_gateway_bindings():
    module = _load_smoke_module()
    bindings = {
        "18000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18000"}],
        "3000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3000"}],
        "3001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3001"}],
    }

    origins = module._assert_requested_origins_match_gateway(
        bindings,
        base_url="http://127.0.0.1:3000",
        aadcr_url="http://localhost:18000",
    )

    assert origins == {
        "aadcr": "http://localhost:18000",
        "backend": "http://127.0.0.1:3000",
        "frontend": "http://127.0.0.1:3001",
    }
    with pytest.raises(module.SmokeError, match="selected gateway"):
        module._assert_requested_origins_match_gateway(
            bindings,
            base_url="http://127.0.0.1:3999",
            aadcr_url="http://localhost:18000",
        )


def test_native_room_inspector_requires_exact_graph_and_asset_provisioning():
    module = _load_smoke_module()
    owner = "nikolas.molyndris@decentriq.ch"
    room_id = "room-1"
    merge_id = "merge-1"
    computation_id = "aggregate-1"
    data_names = (
        "GISSI-HF",
        "GISSI-HF_metadata_dictionary",
        "GISSI-HF_shuffled_sample",
        "TIME-CHF",
        "TIME-CHF_metadata_dictionary",
        "TIME-CHF_shuffled_sample",
        "time-chf_gissi-hf_mapping",
        "CrossStudyMappings",
    )
    data_nodes = [{"id": f"node-{index}", "name": name} for index, name in enumerate(data_names)]
    permissions = [
        {"userEmail": owner, "permissionType": "DATA_OWNER", "resourceName": name} for name in data_names
    ] + [
        {
            "userEmail": owner,
            "permissionType": "DATA_ANALYST",
            "resourceName": name,
        }
        for name in (
            "metadata-preview-local-simulation",
            "aggregate-summary-local-simulation",
        )
    ]
    prod = {
        "participants": [{"userEmail": owner}],
        "data_nodes": data_nodes,
        "computation_nodes": [
            {"id": "preview-1", "name": "metadata-preview-local-simulation"},
            {"id": computation_id, "name": "aggregate-summary-local-simulation"},
        ],
        "permissions": permissions,
    }
    expected_asset_names = set(data_names) - {"CrossStudyMappings"}
    provisioned = [
        {
            "dataset_id": f"dataset-{index}",
            "dataset_node_id": node["id"],
            "dataset_name": node["name"],
            "provision_type": "PROD",
        }
        for index, node in enumerate(data_nodes)
        if node["name"] in expected_asset_names
    ]
    responses = {
        f"/api/dcr/{room_id}": {"id": room_id, "name": "Synthetic demo"},
        f"/api/dcr/{room_id}/merge-requests/{merge_id}": {
            "id": merge_id,
            "status": "MERGED",
        },
        f"/api/dcr/{room_id}/dev/view": {
            "participants": [],
            "data_nodes": [],
            "computation_nodes": [],
            "permissions": [],
        },
        f"/api/dcr/{room_id}/prod/view": prod,
        f"/api/dcr/{room_id}/provisioned-datasets": {
            "provisioned_datasets": provisioned,
        },
        f"/api/dcr/{room_id}/audit-logs": {"audit_logs": [{"event": "merged"}]},
    }

    class FakeClient:
        def __init__(self, payloads):
            self.payloads = payloads

        def json(self, _method, path, **_kwargs):
            return deepcopy(self.payloads[path])

    dataset_id, dataset_count, audit_count, native_provisions = module._inspect_native_room(
        FakeClient(responses),
        headers={"Authorization": "redacted"},
        room_id=room_id,
        merge_id=merge_id,
        computation_id=computation_id,
        owner_email=owner,
    )

    assert (dataset_id, dataset_count, audit_count) == ("dataset-0", 7, 1)
    assert {item["dataset_name"] for item in native_provisions} == expected_asset_names
    broken = deepcopy(responses)
    broken[f"/api/dcr/{room_id}/provisioned-datasets"]["provisioned_datasets"][0]["dataset_name"] = "wrong-node"
    with pytest.raises(module.SmokeError, match="asset coverage"):
        module._inspect_native_room(
            FakeClient(broken),
            headers={"Authorization": "redacted"},
            room_id=room_id,
            merge_id=merge_id,
            computation_id=computation_id,
            owner_email=owner,
        )


def test_run_smoke_wires_the_full_fresh_preview_create_compute_replay_contract(
    tmp_path,
    monkeypatch,
):
    module = _load_smoke_module()
    owner = "nikolas.molyndris@decentriq.ch"
    preview_bytes = b"preview-archive"
    result_bytes = b"result-archive"
    manifest = SimpleNamespace(
        cohorts={
            "GISSI-HF": SimpleNamespace(row_count=2500),
            "TIME-CHF": SimpleNamespace(row_count=2500),
        }
    )
    calls: list[tuple[str, object]] = []
    preview_requests: list[dict] = []

    class FakeHttp:
        def __init__(self, base_url, **_kwargs):
            self.base_url = base_url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def request(self, method, path, **kwargs):
            calls.append((f"request:{method}:{path}", self.base_url))
            if path == "/get-compute-dcr-definition":
                preview_requests.append(kwargs["json"])
                return SimpleNamespace(content=preview_bytes)
            if path.startswith("/compute-get-output/"):
                return SimpleNamespace(content=result_bytes)
            raise AssertionError(path)

        def json(self, method, path, **_kwargs):
            calls.append((f"json:{method}:{path}", self.base_url))
            if path == "/create-live-compute-dcr":
                return {"created": True}
            if path.endswith("/prod/view"):
                return {"data_nodes": [{"id": "node-1"}]}
            if path.endswith("/computation-nodes/results"):
                return {
                    "status": "COMPLETED",
                    "files": ["aggregate-summary.json"],
                    "results": module.base64.b64encode(result_bytes).decode(),
                }
            raise AssertionError(path)

    room_counts = iter((0, 1))
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "SmokeHttp", FakeHttp)
    monkeypatch.setattr(module, "validate_demo_pack", lambda _path: manifest)
    monkeypatch.setattr(
        module,
        "_runtime_values",
        lambda _path: {"LOCAL_AUTH_EMAIL": owner, "AADCRV2_JWT_SECRET": "s" * 64},
    )
    monkeypatch.setattr(module, "_login_and_seed", lambda *_args: calls.append(("seed", None)))
    monkeypatch.setattr(
        module,
        "_generate_mapping",
        lambda *_args: {"filename": "mapping.csv"},
    )
    monkeypatch.setattr(
        module,
        "_build_room_request",
        lambda _manifest, _mapping, **kwargs: (
            captured.setdefault("run_nonce", kwargs["run_nonce"])
            and {"cohorts": {"GISSI-HF": ["age"], "TIME-CHF": ["AGE"]}}
        ),
    )
    monkeypatch.setattr(
        module,
        "_native_room_count",
        lambda *_args, **_kwargs: next(room_counts),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "a" * 24)
    monkeypatch.setattr(module, "_expected_definition_asset_hashes", lambda _manifest: {"asset": "hash"})

    def inspect_definition(content, **kwargs):
        captured["definition"] = (content, kwargs)
        return "a" * 64, 7

    def inspect_live(payload, **kwargs):
        captured.setdefault("live", []).append((payload, kwargs))
        return "room-1", "merge-1", "compute-1"

    def inspect_native(*_args, **kwargs):
        captured["native"] = kwargs
        return "dataset-1", 7, 3, ({"dataset_name": "GISSI-HF"},)

    def inspect_result(content, **kwargs):
        captured["result"] = (content, kwargs)
        return "b" * 64, 1

    def inspect_ce(*_args, **kwargs):
        captured["ce"] = kwargs
        return 1, 2

    def adversarial(*_args, **kwargs):
        captured["adversarial"] = kwargs
        return 4

    monkeypatch.setattr(module, "_inspect_definition", inspect_definition)
    monkeypatch.setattr(module, "_inspect_live_creation", inspect_live)
    monkeypatch.setattr(module, "_inspect_native_room", inspect_native)
    monkeypatch.setattr(module, "_inspect_result", inspect_result)
    monkeypatch.setattr(module, "_inspect_ce_reads", inspect_ce)
    monkeypatch.setattr(module, "_run_adversarial_checks", adversarial)
    monkeypatch.setattr(
        module,
        "_assert_internal_network_and_no_public_egress",
        lambda **kwargs: (
            captured.setdefault("network", kwargs)
            and {
                "aadcr": "http://127.0.0.1:18000",
                "backend": "http://127.0.0.1:3000",
                "frontend": "http://127.0.0.1:3001",
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "write_evidence",
        lambda path, evidence: captured.update(evidence_path=path, evidence=evidence),
    )
    args = SimpleNamespace(
        base_url="http://127.0.0.1:3000",
        aadcr_url="http://127.0.0.1:18000",
        pack=tmp_path / "pack",
        runtime_env=tmp_path / "runtime.env",
        project_name="smoke-test",
        compose_file=[tmp_path / "compose.yml"],
        evidence=tmp_path / "evidence.json",
        timeout=5.0,
        upload_limit_bytes=25 * 1024 * 1024,
    )

    evidence = module.run_smoke(args)

    assert preview_requests[0] is preview_requests[1]
    assert len(preview_requests) == 2
    assert captured["run_nonce"] == "a" * 24
    assert captured["definition"][0] == preview_bytes
    assert captured["definition"][1]["owner_email"] == owner
    assert captured["native"]["owner_email"] == owner
    assert captured["result"][1]["manifest"] is manifest
    assert captured["live"][0][1]["frontend_url"] == "http://127.0.0.1:3001"
    assert captured["ce"]["frontend_url"] == "http://127.0.0.1:3001"
    assert captured["ce"]["expected_room_count"] == 1
    assert captured["adversarial"]["expected_room_count"] == 1
    assert len(captured["live"]) == 2
    assert evidence["room_count"] == 1
    assert evidence["idempotent"] is True
    assert evidence["adversarial_check_count"] == 5
    assert captured["evidence"] == evidence
