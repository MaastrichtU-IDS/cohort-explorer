import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
TRACKED_MAPPING = (
    BACKEND_ROOT / "CohortVarLinker" / "mapping_output" / "time-chf_gissi-hf_full.csv"
)
PROFILE = BACKEND_ROOT / "demo" / "metadata-fixtures" / "mapping-profile.json"


def _seed_dictionary(cohorts_root: Path, cohort_id: str) -> Path:
    folder = cohorts_root / cohort_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{cohort_id}_datadictionary.csv"
    path.write_text("VARIABLENAME,VARIABLELABEL\nhr,Heart rate\n", encoding="utf-8")
    return path


def _provider(tmp_path: Path):
    from src.metadata_providers.fixture_mapping import FixtureMappingGenerationProvider

    cohorts_root = tmp_path / "cohorts"
    _seed_dictionary(cohorts_root, "TIME-CHF")
    _seed_dictionary(cohorts_root, "GISSI-HF")
    provider = FixtureMappingGenerationProvider(
        output_dir=tmp_path / "mapping-output",
        cohorts_root=cohorts_root,
        tracked_mapping_path=TRACKED_MAPPING,
        profile_path=PROFILE,
    )
    return provider, cohorts_root


def test_fixture_provider_materializes_csv_json_provenance_and_activity(
    tmp_path: Path,
) -> None:
    provider, _cohorts_root = _provider(tmp_path)

    result = provider.generate("TIME-CHF", [("GISSI-HF", False)])

    output_dir = tmp_path / "mapping-output"
    csv_path = output_dir / "time-chf_gissi-hf_full.csv"
    json_path = output_dir / "time-chf_gissi-hf_fixture.json"
    meta_path = output_dir / "time-chf_gissi-hf_fixture.json.meta.json"
    assert csv_path.read_bytes() == TRACKED_MAPPING.read_bytes()
    mapping_json = json.loads(json_path.read_text(encoding="utf-8"))
    first_source = next(iter(mapping_json))
    first_mapping = mapping_json[first_source]["mappings"][0]
    assert mapping_json[first_source]["from"] == "time-chf"
    assert first_mapping["target_study"] == "gissi-hf"
    assert first_mapping["mapping_relation"]
    assert first_mapping["harmonization_status"] == "pending"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(TRACKED_MAPPING.read_bytes()).hexdigest()
    assert meta["provider"] == "fixture"
    assert meta["synthetic"] is True
    assert meta["source_commit"] == "ecf8ca97d84576d9b605b4c93e529fc90d32049b"
    assert meta["source_sha256"] == source_sha256
    assert meta["parameters"] == {
        "source_study": "time-chf",
        "target_studies": [["gissi-hf", False]],
    }
    assert meta["output_sha256"] == {
        csv_path.name: hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        json_path.name: hashlib.sha256(json_path.read_bytes()).hexdigest(),
    }
    assert meta["total_mappings"] == 1289
    assert result.cache_info["cached_pairs"][0]["source"] == "time-chf"
    assert result.cache_info["uncached_pairs"] == []

    entries = [
        json.loads(line)
        for line in (output_dir / "mapping_activity.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["event"] == "fixture_materialized"
    assert entries[-1]["ctx"]["provider"] == "fixture"
    assert entries[-1]["ctx"]["synthetic"] is True


def test_fixture_provider_rejects_unsupported_pairs_with_422_compatible_error(
    tmp_path: Path,
) -> None:
    from src.metadata_providers.fixture_mapping import MappingProviderError

    provider, _cohorts_root = _provider(tmp_path)

    with pytest.raises(MappingProviderError, match="TIME-CHF.*OTHER") as error:
        provider.generate("TIME-CHF", [("OTHER", False)])

    assert error.value.status_code == 422


def test_fixture_provider_regenerates_instead_of_reusing_stale_output(tmp_path: Path) -> None:
    provider, cohorts_root = _provider(tmp_path)
    provider.generate("TIME-CHF", [("GISSI-HF", False)])
    output = tmp_path / "mapping-output" / "time-chf_gissi-hf_full.csv"
    first_mtime = output.stat().st_mtime_ns
    source_dictionary = cohorts_root / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    newer = first_mtime + 1_000_000_000
    os.utime(source_dictionary, ns=(newer, newer))

    provider.generate("TIME-CHF", [("GISSI-HF", False)])

    assert output.stat().st_mtime_ns > newer


def test_mapping_routes_share_the_configured_store_and_preserve_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import mapping

    provider, cohorts_root = _provider(tmp_path)
    test_settings = SimpleNamespace(
        mapping_output_dir=str(tmp_path / "mapping-output"),
        cohort_folder=str(cohorts_root),
        mapping_generation_backend="fixture",
    )
    monkeypatch.setattr(mapping, "settings", test_settings)
    monkeypatch.setattr(mapping, "get_mapping_generation_provider", lambda _settings: provider)
    unrelated = tmp_path / "mapping-output" / "notes.json"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text('{"not": "a mapping"}\n', encoding="utf-8")

    app = FastAPI()
    app.include_router(mapping.router)
    app.dependency_overrides[mapping.get_current_user] = lambda: {"email": "admin@example.test"}

    with TestClient(app) as client:
        generated = client.post(
            "/generate-mapping",
            json={
                "source_study": "TIME-CHF",
                "target_studies": [["GISSI-HF", False]],
            },
        )
        assert generated.status_code == 200
        generated_payload = generated.json()
        assert set(generated_payload) == {"cache_info", "file_content", "filename"}
        assert generated_payload["filename"] == "time-chf_gissi-hf_fixture.json"
        assert json.loads(generated_payload["file_content"])

        listed = client.post(
            "/get-available-mapping-files",
            json=["TIME-CHF", "GISSI-HF"],
        )
        assert listed.status_code == 200
        assert listed.json()["cohort_count"] == 2
        assert [item["filename"] for item in listed.json()["available_mappings"]] == [
            "time-chf_gissi-hf_full.csv"
        ]

        retrieved = client.get("/get-cached-mapping-file/time-chf_gissi-hf_full.csv")
        assert retrieved.status_code == 200
        assert retrieved.content == TRACKED_MAPPING.read_bytes()
        assert retrieved.headers["x-filename"] == "time-chf_gissi-hf_full.csv"

        blocked_sidecar = client.get(
            "/get-cached-mapping-file/time-chf_gissi-hf_fixture.json.meta.json"
        )
        assert blocked_sidecar.status_code == 400
        assert client.get("/get-cached-mapping-file/notes.json").status_code == 404

        cache = client.post(
            "/check-mapping-cache",
            json={
                "source_study": "TIME-CHF",
                "target_studies": [["GISSI-HF", False]],
            },
        )
        assert cache.status_code == 200
        assert cache.json()["cached_pairs"][0]["target"] == "gissi-hf"
        assert cache.json()["outdated_pairs"] == []

        activity = client.get("/mapping-activity-log")
        assert activity.status_code == 200
        events = {entry["event"] for entry in activity.json()["entries"]}
        assert {
            "run_started",
            "fixture_materialized",
            "result_file_served",
            "run_completed",
        }.issubset(events)

        unsupported = client.post(
            "/generate-mapping",
            json={
                "source_study": "TIME-CHF",
                "target_studies": [["OTHER", False]],
            },
        )
        assert unsupported.status_code == 422
        assert "Unsupported fixture mapping pair" in unsupported.json()["detail"]


def test_cache_route_reports_stale_canonical_dictionary(tmp_path: Path, monkeypatch) -> None:
    from src import mapping

    provider, cohorts_root = _provider(tmp_path)
    provider.generate("TIME-CHF", [("GISSI-HF", False)])
    output = tmp_path / "mapping-output" / "time-chf_gissi-hf_full.csv"
    source_dictionary = cohorts_root / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    newer = output.stat().st_mtime_ns + 1_000_000_000
    os.utime(source_dictionary, ns=(newer, newer))
    monkeypatch.setattr(
        mapping,
        "settings",
        SimpleNamespace(
            mapping_output_dir=str(tmp_path / "mapping-output"),
            cohort_folder=str(cohorts_root),
        ),
    )

    app = FastAPI()
    app.include_router(mapping.router)
    app.dependency_overrides[mapping.get_current_user] = lambda: {"email": "admin@example.test"}
    with TestClient(app) as client:
        response = client.post(
            "/check-mapping-cache",
            json={
                "source_study": "TIME-CHF",
                "target_studies": [["GISSI-HF", False]],
            },
        )
        stale_download = client.get(
            "/get-cached-mapping-file/time-chf_gissi-hf_full.csv"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cached_pairs"] == []
    assert payload["outdated_pairs"][0]["outdated_cohort"] == "time-chf"
    assert stale_download.status_code == 409
    assert "outdated" in stale_download.json()["detail"].lower()
