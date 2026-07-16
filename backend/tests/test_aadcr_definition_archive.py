import asyncio
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dcr_backends.aadcr_backend import AadcrBackend
from src.dcr_backends.aadcr_translation import build_room_plan
from src.dcr_backends.definition_archive import build_definition_archive

TEST_AADCR_SECRET = "archive-secret"


def run(coroutine):
    return asyncio.run(coroutine)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_plan(settings_factory, root: Path, *, synthetic_demo: bool = False):
    dictionary = _write(root / "source" / "TIME-CHF_datadictionary.csv", "VARIABLENAME\nAGE\n")
    mapping = _write(root / "mappings" / "time-gissi.csv", "from,to\nAGE,age\n")
    _write(root / "pack" / "dcr-input" / "TIME-CHF.csv", "AGE\n70\n")
    _write(root / "runtime" / "dcr_output_TIME-CHF" / "shuffled_sample.csv", "AGE\n70\n")
    cohort = SimpleNamespace(
        cohort_id="TIME-CHF",
        cohort_email=["owner@example.test"],
        administrator_email=None,
        metadata_filepath=str(dictionary),
        variables={},
    )
    request = {
        "cohorts": {"TIME-CHF": ["AGE"]},
        "dcr_name": "Deterministic preview",
        "research_question": "Can metadata be previewed deterministically?",
        "include_shuffled_samples": True,
        "selected_mapping_files": [
            {
                "filename": mapping.name,
                "filepath": str(mapping),
                "cohorts": ["TIME-CHF", "GISSI-HF"],
            }
        ],
    }
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_synthetic_demo=synthetic_demo,
        data_folder=str(root / "runtime"),
        demo_pack_dir=str(root / "pack"),
        mapping_output_dir=str(root / "mappings"),
        decentriq_email="service@example.test",
        dev_mode=False,
    )
    cohorts = {"TIME-CHF": cohort}
    plan = build_room_plan(request, {"email": "creator@example.test"}, cohorts, settings)
    return plan, settings, cohorts, request


def test_definition_archive_has_stable_hash_members_metadata_and_no_absolute_paths(settings_factory, tmp_path):
    plan_a, _settings_a, _cohorts_a, _request_a = _make_plan(settings_factory, tmp_path / "one")
    plan_b, _settings_b, _cohorts_b, _request_b = _make_plan(settings_factory, tmp_path / "two")

    archive_a = build_definition_archive(plan_a)
    archive_b = build_definition_archive(plan_b)

    assert hashlib.sha256(archive_a).hexdigest() == hashlib.sha256(archive_b).hexdigest()
    with zipfile.ZipFile(io.BytesIO(archive_a)) as package:
        assert package.namelist() == sorted(
            [
                "dcr_config.json",
                "fixture-provenance.json",
                "mapping_files/time-gissi.csv",
                "metadata_dictionaries/TIME-CHF_datadictionary.csv",
                "shuffled_samples/TIME-CHF_shuffled_sample.csv",
            ]
        )
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in package.infolist())
        assert all((info.external_attr >> 16) & 0o777 == 0o644 for info in package.infolist())

        config_bytes = package.read("dcr_config.json")
        config = json.loads(config_bytes)
        assert list(config) == ["dataScienceDataRoom"]
        room = config["dataScienceDataRoom"]
        assert room["provider"] == "aadcrv2"
        assert room["local_simulation"] is True
        assert room["confidential_boundary"] is False
        assert [node["name"] for node in room["data_nodes"]] == [
            "TIME-CHF",
            "TIME-CHF_metadata_dictionary",
            "TIME-CHF_shuffled_sample",
            "TIME-CHF_GISSI-HF_mapping",
        ]
        assert str(tmp_path).encode() not in config_bytes

        provenance = json.loads(package.read("fixture-provenance.json"))
        assert provenance == {
            "files": [
                {
                    "archive_path": "mapping_files/time-gissi.csv",
                    "kind": "mapping",
                    "sha256": hashlib.sha256(b"from,to\nAGE,age\n").hexdigest(),
                },
                {
                    "archive_path": "metadata_dictionaries/TIME-CHF_datadictionary.csv",
                    "kind": "metadata",
                    "sha256": hashlib.sha256(b"VARIABLENAME\nAGE\n").hexdigest(),
                },
                {
                    "archive_path": "shuffled_samples/TIME-CHF_shuffled_sample.csv",
                    "kind": "shuffled",
                    "sha256": hashlib.sha256(b"AGE\n70\n").hexdigest(),
                },
            ],
            "format_version": 1,
            "provider": "aadcrv2",
            "synthetic_fixture": False,
        }


@pytest.mark.parametrize("synthetic_demo", [False, True])
def test_definition_archive_provenance_matches_room_plan_synthetic_mode(
    settings_factory,
    tmp_path,
    synthetic_demo,
):
    plan, _settings, _cohorts, _request = _make_plan(
        settings_factory,
        tmp_path / str(synthetic_demo).lower(),
        synthetic_demo=synthetic_demo,
    )

    with zipfile.ZipFile(io.BytesIO(build_definition_archive(plan))) as package:
        provenance = json.loads(package.read("fixture-provenance.json"))

    assert plan.synthetic_demo is synthetic_demo
    assert provenance["synthetic_fixture"] is synthetic_demo


def test_preview_returns_existing_zip_contract_without_loading_decentriq(settings_factory, tmp_path):
    plan, settings, cohorts, request = _make_plan(settings_factory, tmp_path / "preview")
    sys.modules.pop("src.decentriq", None)

    def should_not_create_client(_settings, _user):
        raise AssertionError("definition preview must not contact AADCR")

    backend = AadcrBackend(
        settings,
        client_factory=should_not_create_client,
        cohort_loader=lambda _email: cohorts,
    )
    response = run(backend.preview_definition(request, {"email": "creator@example.test"}))

    assert response.media_type == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="dcr_config_package.zip"'
    assert response.body == build_definition_archive(plan)
    assert "src.decentriq" not in sys.modules
