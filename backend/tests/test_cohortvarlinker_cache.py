import hashlib
import json
import os
from pathlib import Path

import pytest

try:
    from CohortVarLinker import cache as cohortvarlinker_cache
except ModuleNotFoundError:
    cohortvarlinker_cache = None

require_cached_mapping_csvs = getattr(cohortvarlinker_cache, "require_cached_mapping_csvs", None)
select_cached_mapping_csv = getattr(cohortvarlinker_cache, "select_cached_mapping_csv", None)


def _dictionary(root: Path, cohort_id: str, *, mtime_ns: int) -> Path:
    path = root / cohort_id / f"{cohort_id}_datadictionary.csv"
    path.parent.mkdir(parents=True)
    path.write_text("VARIABLENAME,VARIABLELABEL\nage,Age\n", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _mapping(output_dir: Path, filename: str, *, mtime_ns: int) -> Path:
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("source,target\nage,age\n", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _selector():
    if select_cached_mapping_csv is None:
        pytest.fail("CohortVarLinker.cache.select_cached_mapping_csv is not implemented")
    return select_cached_mapping_csv


def test_normal_cache_rejects_a_newer_fixture_artifact(tmp_path: Path) -> None:
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    _dictionary(cohorts, "TIME-CHF", mtime_ns=200)
    _dictionary(cohorts, "GISSI-HF", mtime_ns=200)
    _mapping(output, "time-chf_gissi-hf_full.csv", mtime_ns=300)

    selected = _selector()(
        "TIME-CHF",
        "GISSI-HF",
        output,
        cohorts_root=cohorts,
    )

    assert selected is None


def test_normal_cache_rejects_then_accepts_the_current_artifact_by_dictionary_freshness(
    tmp_path: Path,
) -> None:
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    _dictionary(cohorts, "TIME-CHF", mtime_ns=200)
    _dictionary(cohorts, "GISSI-HF", mtime_ns=200)
    mapping = _mapping(
        output,
        "time-chf_gissi-hf_biolord+no_llm_OEH_full.csv",
        mtime_ns=100,
    )

    selector = _selector()
    assert selector(
        "time-chf",
        "gissi-hf",
        output,
        cohorts_root=cohorts,
    ) is None

    os.utime(mapping, ns=(300, 300))

    assert selector(
        "time-chf",
        "gissi-hf",
        output,
        cohorts_root=cohorts,
    ) == mapping


def test_normal_cache_keeps_safe_glob_compatibility_across_model_tags(tmp_path: Path) -> None:
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    _dictionary(cohorts, "TIME-CHF", mtime_ns=200)
    _dictionary(cohorts, "GISSI-HF", mtime_ns=200)
    _mapping(output, "time-chf_gissi-hf_full.csv", mtime_ns=400)
    alternate = _mapping(
        output,
        "time-chf_gissi-hf_alternative+local_OO_full.csv",
        mtime_ns=300,
    )

    assert _selector()(
        "time-chf",
        "gissi-hf",
        output,
        cohorts_root=cohorts,
    ) == alternate


def test_normal_cache_rejects_hash_attested_synthetic_model_tagged_output(tmp_path: Path) -> None:
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    _dictionary(cohorts, "TIME-CHF", mtime_ns=200)
    _dictionary(cohorts, "GISSI-HF", mtime_ns=200)
    candidate = _mapping(
        output,
        "time-chf_gissi-hf_biolord+no_llm_OEH_full.csv",
        mtime_ns=300,
    )
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    (output / "fixture.json.meta.json").write_text(
        json.dumps(
            {
                "provider": "fixture",
                "synthetic": True,
                "output_sha256": {candidate.name: digest},
            }
        ),
        encoding="utf-8",
    )

    assert _selector()(
        "time-chf",
        "gissi-hf",
        output,
        cohorts_root=cohorts,
        expected_suffix="biolord+no_llm_OEH_full.csv",
    ) is None


def test_synthetic_attestation_overrides_a_matching_normal_attestation(tmp_path: Path) -> None:
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    _dictionary(cohorts, "TIME-CHF", mtime_ns=200)
    _dictionary(cohorts, "GISSI-HF", mtime_ns=200)
    candidate = _mapping(
        output,
        "time-chf_gissi-hf_biolord+no_llm_OEH_full.csv",
        mtime_ns=300,
    )
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    for name, provider, synthetic in (
        ("a-normal.json.meta.json", "cohortvarlinker", False),
        ("z-fixture.json.meta.json", "fixture", True),
    ):
        (output / name).write_text(
            json.dumps(
                {
                    "provider": provider,
                    "synthetic": synthetic,
                    "output_sha256": {candidate.name: digest},
                }
            ),
            encoding="utf-8",
        )

    assert _selector()(
        "time-chf",
        "gissi-hf",
        output,
        cohorts_root=cohorts,
    ) is None


def test_normal_cache_batch_preflight_fails_before_partial_combination(tmp_path: Path) -> None:
    if require_cached_mapping_csvs is None:
        pytest.fail("CohortVarLinker.cache.require_cached_mapping_csvs is not implemented")
    cohorts = tmp_path / "cohorts"
    output = tmp_path / "mapping-output"
    for cohort_id in ("TIME-CHF", "GISSI-HF", "AACHEN-HF"):
        _dictionary(cohorts, cohort_id, mtime_ns=200)
    _mapping(
        output,
        "time-chf_gissi-hf_biolord+no_llm_OEH_full.csv",
        mtime_ns=300,
    )

    with pytest.raises(FileNotFoundError, match="aachen-hf"):
        require_cached_mapping_csvs(
            "time-chf",
            ["gissi-hf", "aachen-hf"],
            output,
            cohorts_root=cohorts,
        )
