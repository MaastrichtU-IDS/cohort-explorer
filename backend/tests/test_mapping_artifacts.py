import hashlib
import json
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


def _write_result_sidecar(
    result_path: Path,
    *,
    source: str,
    targets: list[str],
    provider: str = "cohortvarlinker",
    synthetic: bool = False,
    digest: str | None = None,
) -> Path:
    sidecar = result_path.with_name(f"{result_path.name}.meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "provider": provider,
                "synthetic": synthetic,
                "parameters": {
                    "source_study": source,
                    "target_studies": targets,
                },
                "output_sha256": {
                    result_path.name: digest
                    or hashlib.sha256(result_path.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    return sidecar


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
    sidecar_path = _write_result_sidecar(
        result_path,
        source="time-chf",
        targets=["gissi-hf", "member"],
    )
    os.utime(result_path, ns=(base + 10, base + 10))
    os.utime(sidecar_path, ns=(base + 10, base + 10))
    store = MappingArtifactStore(output_dir, cohorts_root=cohorts_root)

    result = store.result_for_request("TIME-CHF", ["GISSI-HF"])

    assert result is not None
    assert result.filename == result_path.name


def test_result_lookup_rejects_json_without_a_completion_sidecar(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF"):
        _write_dictionary(cohorts_root, cohort_id, base)
    result_path = output_dir / "time-chf_gissi-hf_biolord+no_llm_OEH.json"
    output_dir.mkdir(parents=True)
    result_path.write_text("{", encoding="utf-8")
    os.utime(result_path, ns=(base + 10, base + 10))

    result = MappingArtifactStore(
        output_dir, cohorts_root=cohorts_root
    ).result_for_request("TIME-CHF", ["GISSI-HF"])

    assert result is None


def test_result_lookup_rejects_invalid_json_even_when_its_hash_is_attested(
    tmp_path: Path,
) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF"):
        _write_dictionary(cohorts_root, cohort_id, base)
    result_path = output_dir / "time-chf_gissi-hf_biolord+no_llm_OEH.json"
    output_dir.mkdir(parents=True)
    result_path.write_text("{", encoding="utf-8")
    sidecar = _write_result_sidecar(
        result_path,
        source="time-chf",
        targets=["gissi-hf"],
    )
    for path in (result_path, sidecar):
        os.utime(path, ns=(base + 10, base + 10))

    result = MappingArtifactStore(
        output_dir, cohorts_root=cohorts_root
    ).result_for_request("TIME-CHF", ["GISSI-HF"])

    assert result is None


def test_result_lookup_rejects_a_mismatched_completion_hash(tmp_path: Path) -> None:
    from src.mapping_artifacts import MappingArtifactStore

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF"):
        _write_dictionary(cohorts_root, cohort_id, base)
    result_path = output_dir / "time-chf_gissi-hf_biolord+no_llm_OEH.json"
    output_dir.mkdir(parents=True)
    result_path.write_text("{}\n", encoding="utf-8")
    sidecar = _write_result_sidecar(
        result_path,
        source="time-chf",
        targets=["gissi-hf"],
        digest="0" * 64,
    )
    for path in (result_path, sidecar):
        os.utime(path, ns=(base + 10, base + 10))

    result = MappingArtifactStore(
        output_dir, cohorts_root=cohorts_root
    ).result_for_request("TIME-CHF", ["GISSI-HF"])

    assert result is None


def test_failed_sidecar_publication_never_exposes_the_replaced_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import mapping_artifacts

    cohorts_root = tmp_path / "cohorts"
    output_dir = tmp_path / "mappings"
    base = 1_700_000_000_000_000_000
    for cohort_id in ("TIME-CHF", "GISSI-HF"):
        _write_dictionary(cohorts_root, cohort_id, base)
    result_path = output_dir / "time-chf_gissi-hf_biolord+no_llm_OEH.json"
    provenance = {
        "provider": "cohortvarlinker",
        "synthetic": False,
        "parameters": {
            "source_study": "time-chf",
            "target_studies": ["gissi-hf"],
        },
    }
    mapping_artifacts.publish_mapping_json(result_path, {"age": {}}, provenance)
    real_atomic_write = mapping_artifacts.atomic_write_bytes

    def fail_sidecar(path: Path, content: bytes) -> None:
        if path.name.endswith(".meta.json"):
            raise OSError("simulated sidecar publication failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(mapping_artifacts, "atomic_write_bytes", fail_sidecar)

    with pytest.raises(OSError, match="sidecar publication failure"):
        mapping_artifacts.publish_mapping_json(result_path, {"age": {"changed": True}}, provenance)

    store = mapping_artifacts.MappingArtifactStore(output_dir, cohorts_root=cohorts_root)
    assert store.result_for_request("TIME-CHF", ["GISSI-HF"]) is None
