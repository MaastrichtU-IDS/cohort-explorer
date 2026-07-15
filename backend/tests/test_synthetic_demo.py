import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.demo.generator import generate_demo_pack
from src.demo.manifest import DemoPackError, validate_demo_pack
from src.demo.profiles import COHORT_PROFILES
from src.dictionary_validation import REQUIRED_DICTIONARY_COLUMNS, validate_dictionary_schema
from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

COHORTS = ("TIME-CHF", "GISSI-HF")


@pytest.fixture(scope="module")
def generated_pack(tmp_path_factory: pytest.TempPathFactory):
    return generate_demo_pack(
        tmp_path_factory.mktemp("synthetic-demo") / "pack",
        seed=42,
        rows=1500,
        force=False,
    )


def test_same_seed_produces_identical_manifest_hashes(tmp_path: Path) -> None:
    first = generate_demo_pack(tmp_path / "one", seed=42, rows=180, force=False)
    second = generate_demo_pack(tmp_path / "two", seed=42, rows=180, force=False)

    assert first.files == second.files
    assert (first.root / "manifest.json").read_bytes() == (
        second.root / "manifest.json"
    ).read_bytes()
    assert first.source_commit == second.source_commit


def test_manifest_contract_and_validation_detect_corruption(
    generated_pack,
    tmp_path: Path,
) -> None:
    payload = json.loads((generated_pack.root / "manifest.json").read_text(encoding="utf-8"))
    assert list(payload) == [
        "schema_version",
        "generator_version",
        "seed",
        "source_commit",
        "mapping_source",
        "workbook",
        "cohorts",
        "selected_mapping_rows",
        "files",
    ]
    assert payload["schema_version"] == 1
    assert payload["generator_version"] == "1.0.0"
    assert len(payload["source_commit"]) == 40
    assert set(payload["source_commit"]) <= set("0123456789abcdef")
    assert payload["workbook"] == "iCARE4CVD_Cohorts.xlsx"
    assert payload["mapping_source"]["repository_path"] == (
        "backend/CohortVarLinker/mapping_output/time-chf_gissi-hf_full.csv"
    )
    assert len(payload["mapping_source"]["sha256"]) == 64
    assert set(payload["cohorts"]) == set(COHORTS)
    assert "manifest.json" not in payload["files"]
    validated = validate_demo_pack(generated_pack.root)
    assert validated.files == generated_pack.files
    assert list(validated.selected_mapping_rows[0]) == [
        "source",
        "target",
        "mapping_type",
        "source_visit",
        "target_visit",
    ]

    copied = tmp_path / "corrupt"
    generate_demo_pack(copied, seed=7, rows=80, force=False)
    rows_path = copied / "dcr-input" / "TIME-CHF.csv"
    original_rows = rows_path.read_bytes()
    corrupted = bytearray(original_rows)
    corrupted[-2] = ord("X") if corrupted[-2] != ord("X") else ord("Y")
    rows_path.write_bytes(corrupted)
    with pytest.raises(DemoPackError, match="SHA-256 mismatch"):
        validate_demo_pack(copied)

    rows_path.write_bytes(original_rows)
    unexpected = copied / "unexpected.txt"
    unexpected.write_text("not in manifest", encoding="utf-8")
    with pytest.raises(DemoPackError, match="not listed in the manifest"):
        validate_demo_pack(copied)

    unexpected.unlink()
    manifest_path = copied / "manifest.json"
    invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_manifest["source_commit"] = "unknown"
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(DemoPackError, match="source_commit"):
        validate_demo_pack(copied)


def test_dictionary_columns_match_rows_and_validate(
    generated_pack,
    tmp_path: Path,
) -> None:
    concept_validator = FixtureConceptValidationProvider()
    for cohort in COHORTS:
        dictionary = generated_pack.dictionary_frame(cohort)
        rows = generated_pack.rows_frame(cohort)
        dictionary_path = generated_pack.dictionary(cohort)

        assert tuple(dictionary.columns) == REQUIRED_DICTIONARY_COLUMNS
        assert list(rows.columns) == list(dictionary["VARIABLENAME"])
        assert len(rows) == 1500
        assert validate_dictionary_schema(dictionary_path.read_text(encoding="utf-8")) == []
        assert concept_validator.validate(
            dictionary_path,
            tmp_path / f"{cohort}-concept-validation.csv",
        )
        assert dictionary["COUNT"].astype(int).le(1500).all()
        assert (dictionary["COUNT"].astype(int) + dictionary["NA"].astype(int) == 1500).all()

    workbook = pd.read_excel(generated_pack.workbook, sheet_name="Descriptions")
    assert set(workbook["Study name"]) == set(COHORTS)
    assert set(workbook["Administrator email address"]) == {
        "nikolas.molyndris@decentriq.ch"
    }


@pytest.mark.parametrize("cohort", COHORTS)
def test_generated_clinical_directions_and_exposure_invariants(
    generated_pack,
    cohort: str,
) -> None:
    rows = generated_pack.rows_frame(cohort)
    profile = COHORT_PROFILES[cohort]
    nyha = profile.column("nyha_class")

    assert rows[nyha].corr(rows[profile.column("nt_pro_bnp")], method="spearman") > 0.35
    assert rows[nyha].corr(rows[profile.column("ejection_fraction")], method="spearman") < -0.25
    assert (
        rows[nyha].corr(
            rows[profile.column("heart_failure_hospitalization")],
            method="spearman",
        )
        > 0.15
    )
    assert (rows[profile.column("furosemide_dose")] == 0).equals(
        rows[profile.column("furosemide_exposed")] == 0
    )


@pytest.mark.parametrize("cohort", COHORTS)
def test_followup_dropout_is_monotone_and_trajectories_are_bounded(
    generated_pack,
    cohort: str,
) -> None:
    rows = generated_pack.rows_frame(cohort)
    profile = COHORT_PROFILES[cohort]
    weight_3m = profile.column("weight", "3m")
    weight_1y = profile.column("weight", "1y")
    creatinine_3m = profile.column("creatinine", "3m")
    creatinine_1y = profile.column("creatinine", "1y")

    assert not (rows[weight_1y].notna() & rows[weight_3m].isna()).any()
    assert not (rows[creatinine_1y].notna() & rows[creatinine_3m].isna()).any()
    assert rows[profile.column("patient_id")].is_unique
    weight_columns = [
        profile.column("weight", visit) for visit in ("baseline", "3m", "1y")
    ]
    assert rows[weight_columns].stack().between(35, 220).all()
    assert rows[profile.column("ejection_fraction")].between(15, 75).all()


def test_selected_mapping_rows_resolve_to_both_emitted_dictionaries(
    generated_pack,
) -> None:
    source = set(generated_pack.dictionary_frame("TIME-CHF")["VARIABLENAME"])
    target = set(generated_pack.dictionary_frame("GISSI-HF")["VARIABLENAME"])
    mappings = generated_pack.selected_mapping_frame()

    assert {"nbnp", "nyha_class"} <= set(mappings["source"])
    assert {"v1_nt_probnp", "nyha"} <= set(mappings["target"])
    assert set(mappings["source"]) <= source
    assert set(mappings["target"]) <= target
    assert mappings[["source", "target"]].duplicated().sum() == 0


def test_pack_paths_assets_and_file_digests_are_complete(generated_pack) -> None:
    expected_static = {
        "iCARE4CVD_Cohorts.xlsx",
        "cohorts/TIME-CHF/TIME-CHF_datadictionary.csv",
        "cohorts/GISSI-HF/GISSI-HF_datadictionary.csv",
        "dcr-input/TIME-CHF.csv",
        "dcr-input/GISSI-HF.csv",
    }
    assert expected_static <= set(generated_pack.files)

    for cohort in COHORTS:
        record = generated_pack.cohorts[cohort]
        assert record.eda == f"dcr_output_{cohort}/eda_output_{cohort}.json"
        assert record.shuffled_sample == f"dcr_output_{cohort}/shuffled_sample.csv"
        assert record.shuffle_summary == f"dcr_output_{cohort}/shuffle_summary.txt"
        assert record.images
        assert all(Path(image).name == Path(image).name.lower() for image in record.images)
        assert all((generated_pack.root / image).is_file() for image in record.images)

    for relative_path, digest in generated_pack.files.items():
        content = (generated_pack.root / relative_path).read_bytes()
        assert digest.sha256 == hashlib.sha256(content).hexdigest()
        assert digest.bytes == len(content)


def test_asset_paths_reject_unsafe_components_and_symlink_escape(
    tmp_path: Path,
) -> None:
    from src.demo.assets import contained_asset_path, validate_asset_component

    for value in ("", ".", "..", "bad/name", "bad\\name", "bad\x00name", "bad\nname"):
        with pytest.raises(ValueError, match="asset component"):
            validate_asset_component(value)

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes the asset root"):
        contained_asset_path(root, "escape", "outside.png")


def test_offline_routes_read_pack_without_falling_back_to_runtime(
    generated_pack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import explore, mapping

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    test_settings = SimpleNamespace(
        offline_demo=True,
        demo_pack_dir=str(generated_pack.root),
        data_folder=str(runtime),
    )
    monkeypatch.setattr(explore, "settings", test_settings)
    monkeypatch.setattr(mapping, "settings", test_settings)
    monkeypatch.setattr(explore, "_get_all_cohort_ids", lambda: list(COHORTS))

    app = FastAPI()
    app.include_router(explore.router)
    app.include_router(mapping.router, prefix="/api")
    app.dependency_overrides[explore.get_current_user] = lambda: {
        "email": "nikolas.molyndris@decentriq.ch"
    }
    app.dependency_overrides[mapping.get_current_user] = lambda: {
        "email": "nikolas.molyndris@decentriq.ch"
    }

    source_var = COHORT_PROFILES["TIME-CHF"].column("nt_pro_bnp")
    target_var = COHORT_PROFILES["GISSI-HF"].column("nt_pro_bnp")
    with TestClient(app) as client:
        assert client.head("/cohort-eda-output/TIME-CHF").status_code == 200
        assert client.get("/cohort-eda-output/TIME-CHF").status_code == 200
        assert client.get("/get-cohorts-with-shuffled-samples").json() == {
            "cohorts_with_shuffled_samples": ["GISSI-HF", "TIME-CHF"]
        }
        sample = client.get("/get-shuffled-sample/TIME-CHF")
        assert sample.status_code == 200
        assert sample.content == generated_pack.shuffled_sample("TIME-CHF").read_bytes()
        comparison = client.get(
            f"/api/compare-eda/TIME-CHF/{source_var}/GISSI-HF/{target_var}"
        )
        assert comparison.status_code == 200
        assert comparison.headers["content-type"] == "image/png"

        encoded_separator_requests = (
            "/cohort-eda-output/bad%5Cname",
            "/cohort-eda-output/bad%00name",
            f"/api/compare-eda/TIME-CHF/bad%5Cname/GISSI-HF/{target_var}",
        )
        for request_path in encoded_separator_requests:
            response = client.get(request_path)
            assert response.status_code == 400

        encoded_dot_or_slash_requests = (
            "/cohort-eda-output/%2E%2E",
            f"/api/compare-eda/TIME-CHF/bad%2Fname/GISSI-HF/{target_var}",
        )
        for request_path in encoded_dot_or_slash_requests:
            response = client.get(request_path)
            assert response.status_code in {400, 404}

    test_settings.offline_demo = False
    with TestClient(app) as client:
        assert client.head("/cohort-eda-output/TIME-CHF").status_code == 404


def test_generator_refuses_overwrite_and_cli_can_validate_without_writing(
    tmp_path: Path,
) -> None:
    from scripts.seed_synthetic_data import main

    output = tmp_path / "pack"
    output.mkdir()
    (output / "owner.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(DemoPackError, match="non-empty"):
        generate_demo_pack(output, seed=42, rows=10, force=False)
    assert (output / "owner.txt").read_text(encoding="utf-8") == "preserve me"

    generated = tmp_path / "generated"
    assert main(["--seed", "9", "--rows", "40", "--output", str(generated)]) == 0
    before = {
        path.relative_to(generated): path.read_bytes()
        for path in generated.rglob("*")
        if path.is_file()
    }
    assert main(["--validate", "--output", str(generated)]) == 0
    after = {
        path.relative_to(generated): path.read_bytes()
        for path in generated.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_force_refuses_unrelated_directory_and_excessive_row_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.demo import generator

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    owner_file = unrelated / "owner.txt"
    owner_file.write_text("must survive", encoding="utf-8")
    with pytest.raises(DemoPackError, match="generated demo pack"):
        generate_demo_pack(unrelated, seed=42, rows=10, force=True)
    assert owner_file.read_text(encoding="utf-8") == "must survive"

    generated = tmp_path / "generated"
    generate_demo_pack(generated, seed=1, rows=10, force=False)
    replaced = generate_demo_pack(generated, seed=2, rows=12, force=True)
    assert replaced.seed == 2
    assert {record.row_count for record in replaced.cohorts.values()} == {12}

    monkeypatch.setattr(
        generator,
        "_populate_pack",
        lambda *_args, **_kwargs: pytest.fail("allocation started before row limit"),
    )
    with pytest.raises(DemoPackError, match="at most 100000"):
        generate_demo_pack(tmp_path / "too-large", seed=42, rows=100_001, force=False)


def test_failed_force_install_restores_the_prior_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.demo import generator

    output = tmp_path / "pack"
    generate_demo_pack(output, seed=3, rows=40, force=False)
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    real_replace = os.replace
    failed_install = False

    def fail_new_pack_install(source, destination):
        nonlocal failed_install
        if Path(destination) == output and not failed_install:
            failed_install = True
            raise OSError("simulated install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_new_pack_install)
    with pytest.raises(OSError, match="simulated install failure"):
        generate_demo_pack(output, seed=4, rows=50, force=True)

    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert validate_demo_pack(output).seed == 3
    assert not list(tmp_path.glob(".pack.transaction-*"))
