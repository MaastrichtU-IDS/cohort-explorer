import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

VALID_DICTIONARY = b"""variable name,variable label,vartype,units,categorical,missing,count,na,min,max,formula,categorical value concept code,categorical value concept name,categorical value omop id,variable concept code,variable concept name,variable omop id,additional context concept name,additional context concept code,additional context omop id,unit concept name,unit concept code,unit omop id,domain,visits,visit omop id,visit concept name,visit concept code
heart_rate,Heart rate,INT,,,NA,10,0,1,100,,,,,snomedct:364075005,Heart rate,3027018,,,,,,,measurement,,,,
"""


class RecordingProvider:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[Path, Path, bytes]] = []

    def validate(self, dictionary_path: Path, report_path: Path) -> bool:
        self.calls.append((dictionary_path, report_path, dictionary_path.read_bytes()))
        report_path.write_text("overall_status\nPASS\n", encoding="utf-8")
        return self.result


def test_validate_dictionary_upload_is_non_mutating_and_normalizes_headers(tmp_path: Path) -> None:
    from src.dictionary_validation import validate_dictionary_upload

    canonical = tmp_path / "TIME-CHF_datadictionary.csv"
    canonical.write_bytes(b"old canonical bytes")
    provider = RecordingProvider()

    result = validate_dictionary_upload("TIME-CHF", VALID_DICTIONARY, provider, tmp_path / "work")

    assert result.cohort_id == "TIME-CHF"
    assert result.concepts_valid is True
    assert result.syntax_issues == ()
    assert result.normalized_csv.startswith(b"VARIABLENAME,VARIABLELABEL,VARTYPE")
    assert canonical.read_bytes() == b"old canonical bytes"
    assert provider.calls[0][2] == result.normalized_csv
    assert provider.calls[0][0] != canonical


def test_validate_dictionary_upload_rejects_syntax_before_calling_provider(tmp_path: Path) -> None:
    from src.dictionary_validation import InvalidDictionary, validate_dictionary_upload

    provider = RecordingProvider()
    with pytest.raises(InvalidDictionary, match="Missing required columns"):
        validate_dictionary_upload(
            "TIME-CHF",
            b"variable name,variable label\nhr,Heart rate\n",
            provider,
            tmp_path,
        )
    assert provider.calls == []


def test_validate_dictionary_upload_fails_closed_when_concept_provider_fails(tmp_path: Path) -> None:
    from src.dictionary_validation import InvalidDictionary, validate_dictionary_upload

    with pytest.raises(InvalidDictionary, match="Concept validation failed"):
        validate_dictionary_upload("TIME-CHF", VALID_DICTIONARY, RecordingProvider(False), tmp_path)


def test_canonical_dictionary_path_is_stable_and_rejects_path_traversal(tmp_path: Path) -> None:
    from src.metadata_paths import canonical_dictionary_path

    assert canonical_dictionary_path(tmp_path, "TIME-CHF") == (tmp_path / "TIME-CHF" / "TIME-CHF_datadictionary.csv")
    with pytest.raises(ValueError, match="Invalid cohort ID"):
        canonical_dictionary_path(tmp_path, "../escape")


def test_transactional_replacement_commits_file_graph_and_cache_together(tmp_path: Path) -> None:
    from src.triplestore import replace_metadata_transactionally

    canonical = tmp_path / "TIME-CHF_datadictionary.csv"
    canonical.write_bytes(b"old-file")
    state = {"graph": b"old-graph", "cache": {"value": "old-cache"}}

    replace_metadata_transactionally(
        canonical_path=canonical,
        content=b"new-file",
        staged_graph=b"new-graph",
        snapshot_graph=lambda: state["graph"],
        replace_graph=lambda graph: state.__setitem__("graph", graph),
        snapshot_cache=lambda: deepcopy(state["cache"]),
        replace_cache=lambda path: state.__setitem__("cache", {"value": path.read_text()}),
        restore_cache=lambda snapshot: state.__setitem__("cache", snapshot),
    )

    assert canonical.read_bytes() == b"new-file"
    assert state == {"graph": b"new-graph", "cache": {"value": "new-file"}}


def test_transactional_replacement_rolls_back_file_graph_and_cache(tmp_path: Path) -> None:
    from src.triplestore import replace_metadata_transactionally

    canonical = tmp_path / "TIME-CHF_datadictionary.csv"
    canonical.write_bytes(b"old-file")
    state = {"graph": b"old-graph", "cache": {"value": "old-cache"}}

    def fail_after_mutating_cache(path: Path) -> None:
        state["cache"] = {"value": path.read_text()}
        raise RuntimeError("cache replacement failed")

    with pytest.raises(RuntimeError, match="cache replacement failed"):
        replace_metadata_transactionally(
            canonical_path=canonical,
            content=b"new-file",
            staged_graph=b"new-graph",
            snapshot_graph=lambda: state["graph"],
            replace_graph=lambda graph: state.__setitem__("graph", graph),
            snapshot_cache=lambda: deepcopy(state["cache"]),
            replace_cache=fail_after_mutating_cache,
            restore_cache=lambda snapshot: state.__setitem__("cache", snapshot),
        )

    assert canonical.read_bytes() == b"old-file"
    assert state == {"graph": b"old-graph", "cache": {"value": "old-cache"}}


def test_canonical_studies_graph_uri_is_used_for_metadata() -> None:
    from src.metadata_paths import STUDIES_METADATA_GRAPH_URI

    assert STUDIES_METADATA_GRAPH_URI == "https://w3id.org/CMEO/graph/studies_metadata"


def test_validate_route_does_not_mutate_canonical_dictionary(
    tmp_path: Path,
    local_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ModuleType("src.mapping_generation.retriever")
    retriever.map_csv_to_standard_codes = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "src.mapping_generation.retriever", retriever)
    sys.modules.pop("src.upload", None)
    upload = importlib.import_module("src.upload")

    test_settings = local_settings
    test_settings.concept_validation_backend = "fixture"
    monkeypatch.setattr(upload, "settings", test_settings)
    from src import cohort_cache

    cohort_folder = tmp_path / "cohorts" / "TIME-CHF"
    cohort_folder.mkdir(parents=True)
    canonical = cohort_folder / "TIME-CHF_datadictionary.csv"
    canonical.write_bytes(b"old canonical bytes")
    cohort = SimpleNamespace(can_edit=True, folder_path=str(cohort_folder), variables={})
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {"TIME-CHF": cohort})

    app = FastAPI()
    app.include_router(upload.router)
    app.dependency_overrides[upload.get_current_user] = lambda: {"email": test_settings.local_auth_email}
    with TestClient(app) as client:
        response = client.post(
            "/validate-cohort-dictionary",
            data={"cohort_id": "TIME-CHF"},
            files={"cohort_dictionary": ("dictionary.csv", VALID_DICTIONARY, "text/csv")},
        )

    assert response.status_code == 200
    assert response.json()["concepts_valid"] is True
    assert canonical.read_bytes() == b"old canonical bytes"
    sys.modules.pop("src.upload", None)


def test_cohort_spreadsheet_returns_exact_canonical_dictionary(
    tmp_path: Path,
    local_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import explore

    monkeypatch.setattr(explore, "settings", local_settings)
    cohort_folder = tmp_path / "cohorts" / "TIME-CHF"
    cohort_folder.mkdir(parents=True)
    canonical = cohort_folder / "TIME-CHF_datadictionary.csv"
    canonical.write_bytes(b"canonical dictionary")
    (cohort_folder / "newer_decoy.csv").write_bytes(b"not canonical")

    app = FastAPI()
    app.include_router(explore.router)
    app.dependency_overrides[explore.get_current_user] = lambda: {"email": local_settings.local_auth_email}
    with TestClient(app) as client:
        response = client.get("/cohort-spreadsheet/TIME-CHF")

    assert response.status_code == 200
    assert response.content == b"canonical dictionary"
    assert "TIME-CHF_datadictionary.csv" in response.headers["content-disposition"]


def test_syntax_report_uses_canonical_dictionary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import metadata_reports

    settings = SimpleNamespace(
        data_folder=str(tmp_path),
        cohort_folder=str(tmp_path / "cohorts"),
        admins_list=["admin@example.test"],
        local_auth_email="admin@example.test",
    )
    cohort_folder = tmp_path / "cohorts" / "TIME-CHF"
    cohort_folder.mkdir(parents=True)
    (cohort_folder / "TIME-CHF_datadictionary.csv").write_bytes(VALID_DICTIONARY)
    (cohort_folder / "newer_invalid_datadictionary.csv").write_text(
        "variable name,variable label\nhr,Heart rate\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(metadata_reports, "settings", settings)
    monkeypatch.setattr(
        metadata_reports,
        "get_cohorts_from_cache",
        lambda _email: {"TIME-CHF": object()},
    )

    report = metadata_reports.build_syntax_report()

    assert report.total_cohorts == 1
    assert report.cohorts_with_errors == 0
    assert "Status: OK" in report.path.read_text(encoding="utf-8")
