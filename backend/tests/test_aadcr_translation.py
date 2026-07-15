import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dcr_backends.aadcr_translation import (
    AGGREGATE_NODE_NAME,
    DcrOperationError,
    build_room_plan,
    resolve_prod_node_ids,
    wait_for_merge,
)

TEST_AADCR_SECRET = "translation-test-secret"


def run(coroutine):
    return asyncio.run(coroutine)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _cohort(cohort_id: str, dictionary: Path, owner: str, administrator: str | None = None):
    return SimpleNamespace(
        cohort_id=cohort_id,
        cohort_email=[owner],
        administrator_email=administrator,
        metadata_filepath=str(dictionary),
        variables={},
    )


def _settings(settings_factory, tmp_path: Path, *, synthetic: bool = True):
    return settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_synthetic_demo=synthetic,
        data_folder=str(tmp_path / "runtime"),
        demo_pack_dir=str(tmp_path / "pack"),
        mapping_output_dir=str(tmp_path / "mappings"),
        decentriq_email="service@example.test",
        dev_mode=False,
    )


def _request(mapping: Path) -> dict:
    return {
        "cohorts": {"TIME CHF": ["age"]},
        "dcr_name": "Synthetic heart failure study",
        "research_question": "Can two synthetic cohorts be summarized?",
        "include_shuffled_samples": {"TIME CHF": True},
        "additional_analysts": ["analyst@example.test"],
        "excluded_data_owners": ["excluded@example.test"],
        "selected_mapping_files": [
            {
                "filename": mapping.name,
                "filepath": str(mapping),
                "cohorts": ["TIME CHF", "GISSI-HF"],
            }
        ],
        "include_mapping_upload_slot": True,
    }


def test_room_plan_preserves_legacy_participant_roles_and_marks_local_simulation(
    settings_factory,
    tmp_path,
):
    settings = _settings(settings_factory, tmp_path)
    dictionary = _write(tmp_path / "dictionary" / "TIME CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    sample = _write(
        tmp_path / "runtime" / "dcr_output_TIME CHF" / "shuffled_sample.csv",
        "AGE\n71\n",
    )
    raw = _write(tmp_path / "pack" / "dcr-input" / "TIME CHF.csv", "AGE\n73\n")
    mapping = _write(tmp_path / "mappings" / "time-gissi.csv", "from,to\nAGE,age\n")
    cohorts = {
        "TIME CHF": _cohort(
            "TIME CHF",
            dictionary,
            "owner@example.test",
            administrator="excluded@example.test",
        )
    }

    plan = build_room_plan(
        _request(mapping),
        {"email": "creator@example.test"},
        cohorts,
        settings,
    )

    assert plan.title == "Synthetic heart failure study - created by creator@example.test"
    assert plan.participant_emails == [
        "creator@example.test",
        "owner@example.test",
        "service@example.test",
        "analyst@example.test",
    ]
    assert "excluded@example.test" not in plan.participants
    assert plan.participants["owner@example.test"].data_owner_of >= {
        "TIME-CHF",
        "TIME-CHF_metadata_dictionary",
    }
    assert plan.participants["analyst@example.test"].data_owner_of == {
        "TIME CHF_GISSI-HF_mapping",
        "CrossStudyMappings",
    }
    assert plan.participants["creator@example.test"].data_owner_of >= {
        "TIME-CHF",
        "TIME-CHF_metadata_dictionary",
        "TIME-CHF_shuffled_sample",
        "TIME CHF_GISSI-HF_mapping",
        "CrossStudyMappings",
    }
    assert all(
        roles.analyst_of == {"metadata-preview-local-simulation", AGGREGATE_NODE_NAME}
        for roles in plan.participants.values()
    )
    assert [node.name for node in plan.data_nodes] == [
        "TIME-CHF",
        "TIME-CHF_metadata_dictionary",
        "TIME-CHF_shuffled_sample",
        "TIME CHF_GISSI-HF_mapping",
        "CrossStudyMappings",
    ]
    assert all(node.type == "FILE" for node in plan.data_nodes)
    aggregate = next(node for node in plan.computation_nodes if node.name == AGGREGATE_NODE_NAME)
    assert "numeric[column].append" not in aggregate.code
    assert "local simulation" in aggregate.code.lower()
    assert "not a confidential" in aggregate.code.lower()
    assert [asset.path for asset in plan.assets] == [dictionary, mapping, sample, raw]
    assert [asset.kind for asset in plan.assets] == ["metadata", "mapping", "shuffled", "synthetic"]


def test_non_synthetic_plan_never_contains_row_level_uploads(settings_factory, tmp_path):
    settings = _settings(settings_factory, tmp_path, synthetic=False)
    dictionary = _write(tmp_path / "dictionary" / "TIME CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    mapping = _write(tmp_path / "mappings" / "time-gissi.csv", "from,to\nAGE,age\n")
    _write(tmp_path / "pack" / "dcr-input" / "TIME CHF.csv", "AGE\n73\n")
    cohorts = {"TIME CHF": _cohort("TIME CHF", dictionary, "owner@example.test")}

    plan = build_room_plan(
        _request(mapping) | {"include_shuffled_samples": False},
        {"email": "creator@example.test"},
        cohorts,
        settings,
    )

    assert all(asset.kind != "synthetic" for asset in plan.assets)
    assert "TIME-CHF" not in plan.participants["creator@example.test"].data_owner_of


def test_generated_local_computations_execute_and_emit_aggregate_only_outputs(settings_factory, tmp_path):
    settings = _settings(settings_factory, tmp_path, synthetic=False)
    dictionary = _write(tmp_path / "dictionary" / "TIME CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    mapping = _write(tmp_path / "mappings" / "time-gissi.csv", "from,to\nAGE,age\n")
    cohorts = {"TIME CHF": _cohort("TIME CHF", dictionary, "owner@example.test")}
    plan = build_room_plan(
        _request(mapping) | {"include_shuffled_samples": False},
        {"email": "creator@example.test"},
        cohorts,
        settings,
    )
    metadata = next(node for node in plan.computation_nodes if node.name == "metadata-preview-local-simulation")
    aggregate = next(node for node in plan.computation_nodes if node.name == AGGREGATE_NODE_NAME)

    metadata_work = tmp_path / "metadata-computation"
    _write(
        metadata_work / "input" / "TIME-CHF_metadata_dictionary" / "file",
        "VARIABLENAME,LABEL\nAGE,Age\n",
    )
    metadata_run = subprocess.run(  # noqa: S603 - controlled generated script under test
        [sys.executable, "-c", metadata.code],
        cwd=metadata_work,
        check=False,
        capture_output=True,
        text=True,
    )
    assert metadata_run.returncode == 0, metadata_run.stderr
    preview = json.loads((metadata_work / "output" / "metadata-preview.json").read_text())
    assert preview["nodes"]["TIME-CHF_metadata_dictionary"] == {
        "columns": ["VARIABLENAME", "LABEL"],
        "rows": 1,
    }

    aggregate_work = tmp_path / "aggregate-computation"
    _write(
        aggregate_work / "input" / "TIME-CHF" / "file",
        "subject_id,AGE\npatient-1,70\npatient-2,74\n",
    )
    aggregate_run = subprocess.run(  # noqa: S603 - controlled generated script under test
        [sys.executable, "-c", aggregate.code],
        cwd=aggregate_work,
        check=False,
        capture_output=True,
        text=True,
    )
    assert aggregate_run.returncode == 0, aggregate_run.stderr
    aggregate_bytes = (aggregate_work / "output" / "aggregate-summary.json").read_bytes()
    assert b"patient-1" not in aggregate_bytes
    assert b"patient-2" not in aggregate_bytes
    summary = json.loads(aggregate_bytes)
    assert summary["confidential_boundary"] is False
    assert summary["nodes"]["TIME-CHF"]["numeric"]["AGE"] == {
        "count": 2,
        "maximum": 74.0,
        "mean": 72.0,
        "minimum": 70.0,
    }


def test_request_validation_rejects_missing_cohort_and_mapping_escape(settings_factory, tmp_path):
    settings = _settings(settings_factory, tmp_path, synthetic=False)
    dictionary = _write(tmp_path / "dictionary" / "TIME CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    valid_mapping = _write(tmp_path / "mappings" / "valid.csv", "from,to\nAGE,age\n")
    cohorts = {"TIME CHF": _cohort("TIME CHF", dictionary, "owner@example.test")}

    with pytest.raises(DcrOperationError) as missing:
        build_room_plan(
            _request(valid_mapping) | {"cohorts": {"UNKNOWN": []}},
            {"email": "creator@example.test"},
            cohorts,
            settings,
        )
    assert missing.value.failed_step == "validate request"
    assert missing.value.dcr_id is None
    assert "UNKNOWN" in missing.value.safe_detail

    escaped_mapping = _write(tmp_path / "outside.csv", "from,to\nAGE,age\n")
    with pytest.raises(DcrOperationError) as escaped:
        build_room_plan(
            _request(escaped_mapping),
            {"email": "creator@example.test"},
            cohorts,
            settings,
        )
    assert escaped.value.failed_step == "validate mapping files"
    assert "configured mapping output directory" in escaped.value.safe_detail

    sidecar = _write(tmp_path / "mappings" / "valid.csv.meta.json", '{"private": "provenance"}\n')
    with pytest.raises(DcrOperationError) as unsupported:
        build_room_plan(
            _request(sidecar),
            {"email": "creator@example.test"},
            cohorts,
            settings,
        )
    assert unsupported.value.failed_step == "validate mapping files"
    assert "CSV mapping artifact" in unsupported.value.safe_detail

    json_mapping = _write(tmp_path / "mappings" / "valid.json", '{"AGE": "age"}\n')
    with pytest.raises(DcrOperationError) as unsupported_json:
        build_room_plan(
            _request(json_mapping),
            {"email": "creator@example.test"},
            cohorts,
            settings,
        )
    assert unsupported_json.value.status_code == 422
    assert unsupported_json.value.failed_step == "validate mapping files"
    assert "CSV mapping artifact" in unsupported_json.value.safe_detail


def test_prod_resolution_requires_every_expected_node_name(settings_factory, tmp_path):
    settings = _settings(settings_factory, tmp_path, synthetic=False)
    dictionary = _write(tmp_path / "dictionary" / "TIME CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    mapping = _write(tmp_path / "mappings" / "time-gissi.csv", "from,to\nAGE,age\n")
    cohorts = {"TIME CHF": _cohort("TIME CHF", dictionary, "owner@example.test")}
    plan = build_room_plan(
        _request(mapping) | {"include_shuffled_samples": False},
        {"email": "creator@example.test"},
        cohorts,
        settings,
    )
    prod_view = {
        "data_nodes": [
            {"id": f"prod-{node.name}", "name": node.name}
            for node in plan.data_nodes
            if node.name != "TIME-CHF_metadata_dictionary"
        ],
        "computation_nodes": [{"id": f"prod-{node.name}", "name": node.name} for node in plan.computation_nodes],
    }

    with pytest.raises(DcrOperationError) as caught:
        resolve_prod_node_ids(plan, prod_view, "room-123")

    assert caught.value.failed_step == "resolve PROD nodes"
    assert caught.value.dcr_id == "room-123"
    assert "TIME-CHF_metadata_dictionary" in caught.value.safe_detail


class MergeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def request_json(self, method, path, *, failed_step, **_kwargs):
        self.calls.append((method, path, failed_step))
        return self.responses.pop(0)


def test_pending_approval_is_reported_without_governance_bypass():
    client = MergeClient(
        [
            {
                "status": "PENDING",
                "approvals": [{"status": "PENDING", "isImplicit": False, "approverCandidates": ["owner@example.test"]}],
            }
        ]
    )

    with pytest.raises(DcrOperationError) as caught:
        run(wait_for_merge(client, "room-123", "merge-1", max_attempts=4, poll_interval=0))

    assert caught.value.failed_step == "await merge approval"
    assert caught.value.dcr_id == "room-123"
    assert caught.value.retryable is True
    assert [call[0] for call in client.calls] == ["GET"]
    assert all("approve" not in call[1] for call in client.calls)


def test_merge_rejection_and_polling_timeout_are_bounded():
    rejected = MergeClient([{"status": "REJECTED", "approvals": []}])
    with pytest.raises(DcrOperationError) as merge_failed:
        run(wait_for_merge(rejected, "room-123", "merge-1", max_attempts=3, poll_interval=0))
    assert merge_failed.value.failed_step == "merge DEV changes"
    assert merge_failed.value.retryable is False

    pending = MergeClient([{"status": "PENDING", "approvals": []}] * 3)
    with pytest.raises(DcrOperationError) as timed_out:
        run(wait_for_merge(pending, "room-456", "merge-2", max_attempts=3, poll_interval=0))
    assert timed_out.value.failed_step == "poll merge request"
    assert timed_out.value.dcr_id == "room-456"
    assert timed_out.value.retryable is True
    assert len(pending.calls) == 3
