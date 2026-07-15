import os
from pathlib import Path

import pytest

CSV = "source,target,mapping type\nheart_rate,hr,exact match\n"


def _write_dictionary(cohorts_root: Path, cohort_id: str, mtime_ns: int) -> Path:
    folder = cohorts_root / cohort_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{cohort_id}_datadictionary.csv"
    path.write_text("VARIABLENAME,VARIABLELABEL\nhr,Heart rate\n", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _write_mapping(output_dir: Path, filename: str, mtime_ns: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(CSV, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_artifact_listing_filters_cohorts_and_reports_only_fresh_files(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF", "OTHER"):
        _write_dictionary(cohorts_root, cohort_id, base)
    _write_mapping(output_dir, "time-chf_gissi-hf_full.csv", base + 10)
    _write_mapping(output_dir, "time-chf_other_full.csv", base + 20)
    store = MappingArtifactStore(output_dir, cohorts_root=cohorts_root)

    artifacts = store.list_for_cohorts({"TIME-CHF", "GISSI-HF"})

    assert [artifact.filename for artifact in artifacts] == [
        "time-chf_gissi-hf_full.csv"
    ]
    assert artifacts[0].cohorts == ("time-chf", "gissi-hf")
    assert artifacts[0].stats == {"total_mappings": 1}
    assert store.list_for_cohorts({"TIME-CHF"}) == []


def test_outdated_mapping_is_not_reused(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    source_dictionary = _write_dictionary(cohorts_root, "TIME-CHF", base)
    _write_dictionary(cohorts_root, "GISSI-HF", base)
    _write_mapping(output_dir, "time-chf_gissi-hf_full.csv", base + 10)
    store = MappingArtifactStore(output_dir, cohorts_root=cohorts_root)
    artifact = store.list_for_cohorts({"TIME-CHF", "GISSI-HF"})[0]

    assert store.cache_status(artifact).fresh is True

    os.utime(source_dictionary, ns=(base + 20, base + 20))
    status = store.cache_status(artifact)

    assert status.filename == artifact.filename
    assert status.fresh is False
    assert status.source_mtime_ns == base + 20
    assert status.target_mtime_ns == base + 10
    assert store.list_for_cohorts({"TIME-CHF", "GISSI-HF"}) == []


@pytest.mark.parametrize(
    "filename",
    [
        "../secret.json",
        "..\\secret.json",
        "/tmp/secret.json",
        "time-chf_gissi-hf_fixture.json.meta.json",
        "unrelated.txt",
    ],
)
def test_safe_path_rejects_traversal_sidecars_and_unrelated_files(
    tmp_path: Path,
    filename: str,
) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    store = MappingArtifactStore(tmp_path / "mappings", cohorts_root=tmp_path / "cohorts")

    with pytest.raises(ValueError):
        store.safe_path(filename)


def test_artifact_api_shape_preserves_existing_mapping_file_contract(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    _write_dictionary(cohorts_root, "TIME-CHF", base)
    _write_dictionary(cohorts_root, "GISSI-HF", base)
    _write_mapping(output_dir, "time-chf_gissi-hf_full.csv", base + 10)
    store = MappingArtifactStore(output_dir, cohorts_root=cohorts_root)

    payload = store.list_for_cohorts({"TIME-CHF", "GISSI-HF"})[0].to_api_dict()

    assert payload["cohorts"] == ["time-chf", "gissi-hf"]
    assert payload["filename"] == "time-chf_gissi-hf_full.csv"
    assert payload["display_name"] == "time-chf → gissi-hf"
    assert payload["stats"] == {"total_mappings": 1}
    assert payload["filepath"] == str(output_dir / payload["filename"])


def test_result_lookup_preserves_live_provider_member_study_expansion(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF", "MEMBER"):
        _write_dictionary(cohorts_root, cohort_id, base)
    result_path = output_dir / "time-chf_gissi-hf_member_biolord+no_llm_OEH.json"
    output_dir.mkdir(parents=True)
    result_path.write_text("{}\n", encoding="utf-8")
    os.utime(result_path, ns=(base + 10, base + 10))
    store = MappingArtifactStore(output_dir, cohorts_root=cohorts_root)

    result = store.result_for_request("TIME-CHF", ["GISSI-HF"])

    assert result is not None
    assert result.filename == result_path.name
