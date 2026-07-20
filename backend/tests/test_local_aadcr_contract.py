"""Contract tests for the metadata-only CE to AADCR v2 handoff smoke."""

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

    assert str(failure.value) == "read room failed with HTTP 502"
    assert "super-secret" not in str(failure.value)
    assert "session-secret" not in str(failure.value)
    assert "patient_row" not in str(failure.value)


def test_evidence_writer_accepts_only_handoff_ids_hashes_counts_and_statuses(tmp_path):
    module = _load_smoke_module()
    evidence_path = tmp_path / "smoke-evidence.json"
    evidence = {
        "schema_version": 1,
        "room_id": "room-123",
        "preview_sha256": "a" * 64,
        "cohort_count": 2,
        "data_node_count": 8,
        "room_count": 1,
        "audit_count": 9,
        "definition_member_count": 7,
        "handoff_mode": "bootstrap",
        "environment": "DEV",
        "idempotent": True,
        "adversarial_check_count": 4,
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
    assert first["selected_mapping_files"] == [
        {
            "filename": mapping["filename"],
            "filepath": mapping["filepath"],
            "display_name": mapping["display_name"],
            "cohorts": mapping["cohorts"],
        }
    ]


def test_tampered_token_changes_signed_bytes_and_is_rejected():
    module = _load_smoke_module()
    token = module._mint_aadcr_token("owner@example.test", "s" * 64)

    tampered = module._tamper_token_signature(token)

    options = {
        "algorithms": [module.JWT_ALGORITHM],
        "audience": "aadcrv2-local",
        "issuer": "cohort-explorer-local",
    }
    assert module.jwt.decode(token, "s" * 64, **options)["email"] == "owner@example.test"
    with pytest.raises(JWTError):
        module.jwt.decode(tampered, "s" * 64, **options)


def test_definition_inspector_requires_exact_metadata_fixture_members():
    module = _load_smoke_module()
    owner = "nikolas.molyndris@decentriq.ch"
    data_names = sorted(module.EXPECTED_DATA_NODES)
    computation_names = sorted(module.EXPECTED_COMPUTATION_NODES)
    selected_variables = {"GISSI-HF": ["age_years"], "TIME-CHF": ["AGE"]}
    assets = {
        "mapping_files/time-chf_gissi-hf_full.csv": ("mapping", b"from,to\nAGE,age_years\n"),
        "metadata_dictionaries/GISSI-HF_datadictionary.csv": (
            "metadata",
            b"VARIABLENAME\nage_years\n",
        ),
        "metadata_dictionaries/TIME-CHF_datadictionary.csv": (
            "metadata",
            b"VARIABLENAME\nAGE\n",
        ),
        "shuffled_samples/GISSI-HF_shuffled_sample.csv": ("shuffled", b"age_years\n70\n"),
        "shuffled_samples/TIME-CHF_shuffled_sample.csv": ("shuffled", b"AGE\n72\n"),
    }
    provenance = {
        "files": [
            {"archive_path": path, "kind": kind, "sha256": module._sha256(content)}
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
                    "permissions": [
                        {"email": owner, "node": name, "type": "DATA_OWNER"}
                        for name in data_names
                    ]
                    + [
                        {"email": owner, "node": name, "type": "DATA_ANALYST"}
                        for name in computation_names
                    ],
                    "selected_variables": selected_variables,
                }
            }
        ).encode(),
        "fixture-provenance.json": json.dumps(provenance).encode(),
        **{path: content for path, (_kind, content) in assets.items()},
    }
    expected_hashes = {
        path: module._sha256(content) for path, (_kind, content) in assets.items()
    }

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
            _zip_bytes(
                {
                    name: content
                    for name, content in members.items()
                    if "mapping_files/" not in name
                }
            ),
            expected_asset_hashes=expected_hashes,
            owner_email=owner,
            expected_selected_variables=selected_variables,
        )


def test_bootstrap_creation_requires_real_aadcr_handoff_and_no_automation():
    module = _load_smoke_module()
    node_ids = {name: f"node-{index}" for index, name in enumerate(sorted(module.EXPECTED_DATA_NODES))}
    payload = {
        "dcr_id": "room-1",
        "dcr_url": "http://localhost:3002/aadcrv2/dcr/room-1",
        "cohort_ids": ["TIME-CHF", "GISSI-HF"],
        "num_cohorts": 2,
        "provider": "aadcrv2",
        "handoff_mode": "bootstrap",
        "environment": "DEV",
        "participants": {},
        "data_node_ids": node_ids,
        "metadata_upload_results": {},
        "row_upload_results": {},
        "shuffled_upload_results": {},
        "mapping_upload_results": {},
        "metadata_uploads_successful": 0,
        "row_uploads_successful": 0,
        "shuffled_uploads_successful": 0,
        "mapping_uploads_successful": 0,
        "capabilities": {"local_simulation": True, "synthetic_data_only": True},
    }

    assert module._inspect_bootstrap_creation(
        payload,
        aadcr_ui_url="http://127.0.0.1:3002",
    ) == ("room-1", node_ids)

    for forbidden_key, forbidden_value in (
        ("merge_request_id", "merge-1"),
        ("aggregate_computation_node_id", "compute-1"),
        ("participants", {"owner@example.test": {}}),
        ("metadata_uploads_successful", 1),
    ):
        broken = deepcopy(payload)
        broken[forbidden_key] = forbidden_value
        with pytest.raises(module.SmokeError, match="bootstrap contract"):
            module._inspect_bootstrap_creation(
                broken,
                aadcr_ui_url="http://127.0.0.1:3002",
            )


def test_native_bootstrap_inspector_requires_only_dev_data_slots():
    module = _load_smoke_module()
    room_id = "room-1"
    node_ids = {name: f"node-{index}" for index, name in enumerate(sorted(module.EXPECTED_DATA_NODES))}
    responses = {
        f"/api/dcr/{room_id}": {"id": room_id, "name": "Synthetic demo"},
        f"/api/dcr/{room_id}/dev/view": {
            "participants": [],
            "data_nodes": [
                {"id": node_id, "name": name, "nodeStatus": "DEV_ADDED"}
                for name, node_id in node_ids.items()
            ],
            "computation_nodes": [],
            "permissions": [],
        },
        f"/api/dcr/{room_id}/prod/view": {
            "participants": [],
            "data_nodes": [],
            "computation_nodes": [],
            "permissions": [],
        },
        f"/api/dcr/{room_id}/provisioned-datasets": {"provisioned_datasets": []},
        f"/api/dcr/{room_id}/merge-requests/": {"mergeRequests": []},
        f"/api/dcr/{room_id}/audit-logs": {"audit_logs": [{"event": "created"}]},
    }

    class FakeClient:
        def json(self, _method, path, **_kwargs):
            return deepcopy(responses[path])

    assert module._inspect_native_bootstrap(
        FakeClient(),
        headers={"Authorization": "redacted"},
        room_id=room_id,
        expected_node_ids=node_ids,
    ) == 1

    responses[f"/api/dcr/{room_id}/dev/view"]["computation_nodes"] = [
        {"id": "compute-1", "name": "too-soon"}
    ]
    with pytest.raises(module.SmokeError, match="handoff boundary"):
        module._inspect_native_bootstrap(
            FakeClient(),
            headers={"Authorization": "redacted"},
            room_id=room_id,
            expected_node_ids=node_ids,
        )


def test_smoke_origins_must_match_all_selected_gateway_bindings():
    module = _load_smoke_module()
    bindings = {
        "18000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18000"}],
        "3000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3000"}],
        "3001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3001"}],
        "3002/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3002"}],
    }

    origins = module._assert_requested_origins_match_gateway(
        bindings,
        base_url="http://127.0.0.1:3000",
        aadcr_url="http://localhost:18000",
        aadcr_ui_url="http://localhost:3002",
    )

    assert origins == {
        "aadcr": "http://localhost:18000",
        "aadcr_ui": "http://localhost:3002",
        "backend": "http://127.0.0.1:3000",
        "frontend": "http://127.0.0.1:3001",
    }
    with pytest.raises(module.SmokeError, match="selected gateway"):
        module._assert_requested_origins_match_gateway(
            bindings,
            base_url="http://127.0.0.1:3999",
            aadcr_url="http://localhost:18000",
            aadcr_ui_url="http://localhost:3002",
        )
