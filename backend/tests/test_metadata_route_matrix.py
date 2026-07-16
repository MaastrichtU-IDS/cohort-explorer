import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import get_current_user
from src.data_analysis import compare_cache_objects
from src.models import Cohort, CohortVariable, VariableCategory

ADMIN_EMAIL = "nikolas.molyndris@decentriq.ch"


class RecordingValidationProvider:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[Path, Path]] = []

    def validate(self, dictionary_path: Path, report_path: Path) -> bool:
        self.calls.append((dictionary_path, report_path))
        report_path.write_text("variable name,overall_status\nage,PASS\n", encoding="utf-8")
        return self.result


@pytest.fixture
def route_context(tmp_path: Path, local_settings, monkeypatch: pytest.MonkeyPatch):
    cohort_cache = importlib.import_module("src.cohort_cache")
    config = importlib.import_module("src.config")
    data_analysis = importlib.import_module("src.data_analysis")
    explore = importlib.import_module("src.explore")
    metadata_reports = importlib.import_module("src.metadata_reports")
    upload = importlib.import_module("src.upload")

    original_cache = cohort_cache._cohorts_cache
    original_cache_initialized = cohort_cache._cache_initialized
    original_cache_disk_mtime_ns = cohort_cache._cache_disk_mtime_ns
    monkeypatch.setattr(config, "settings", local_settings)
    monkeypatch.setattr(cohort_cache, "settings", local_settings)
    monkeypatch.setattr(data_analysis, "settings", local_settings, raising=False)
    monkeypatch.setattr(explore, "settings", local_settings)
    monkeypatch.setattr(metadata_reports, "settings", local_settings)
    monkeypatch.setattr(upload, "settings", local_settings)
    Path(local_settings.cohort_folder).mkdir(parents=True, exist_ok=True)

    original_named_temporary_file = tempfile.NamedTemporaryFile

    def isolated_named_temporary_file(**kwargs):
        return original_named_temporary_file(dir=tmp_path, **kwargs)

    monkeypatch.setattr(data_analysis.tempfile, "NamedTemporaryFile", isolated_named_temporary_file)

    app = FastAPI()
    app.include_router(explore.router)
    app.include_router(upload.router)
    app.include_router(data_analysis.router, prefix="/api")
    context = SimpleNamespace(
        app=app,
        cohort_cache=cohort_cache,
        data_analysis=data_analysis,
        explore=explore,
        metadata_reports=metadata_reports,
        settings=local_settings,
        upload=upload,
    )
    yield context

    cohort_cache._cohorts_cache = original_cache
    cohort_cache._cache_initialized = original_cache_initialized
    cohort_cache._cache_disk_mtime_ns = original_cache_disk_mtime_ns


def authenticated_client(app: FastAPI, email: str = ADMIN_EMAIL) -> TestClient:
    app.dependency_overrides[get_current_user] = lambda: {"email": email}
    return TestClient(app)


def sample_cohort() -> Cohort:
    return Cohort(
        cohort_id="TIME-CHF",
        institution="Synthetic Maastricht Heart Institute",
        variables={
            "sex": CohortVariable(
                var_name="sex",
                var_label="Biological sex",
                var_type="STR",
                count=24,
                categories=[
                    VariableCategory(value="F", label="Female", concept_id="8532"),
                    VariableCategory(value="M", label="Male", concept_id="8507"),
                ],
            )
        },
    )


def test_compare_cache_objects_accepts_serialized_category_lists() -> None:
    cache_a = {
        "cohorts": {
            "TIME-CHF": {
                "variables": {
                    "sex": {
                        "var_label": "Biological sex",
                        "categories": [
                            {"value": "F", "label": "Female", "concept_id": "8532"},
                            {"value": "M", "label": "Male", "concept_id": "8507"},
                        ],
                    }
                }
            }
        }
    }
    cache_b = {
        "cohorts": {
            "TIME-CHF": {
                "variables": {
                    "sex": {
                        "var_label": "Biological sex",
                        "categories": [
                            {"value": "M", "label": "Male", "concept_id": "8507"},
                            {"value": "F", "label": "Female participant", "concept_id": "8532"},
                        ],
                    }
                }
            }
        }
    }

    report = compare_cache_objects(cache_a, cache_b)

    category_differences = report["cohort_comparisons"]["TIME-CHF"]["variable_differences"]["sex"][
        "category_differences"
    ]
    assert category_differences == {
        "categories_only_in_a": [],
        "categories_only_in_b": [],
        "category_value_differences": {
            "F": {
                "value_in_a": {"value": "F", "label": "Female", "concept_id": "8532"},
                "value_in_b": {"value": "F", "label": "Female participant", "concept_id": "8532"},
            }
        },
    }


def test_compare_cache_objects_preserves_legacy_dictionary_categories() -> None:
    cache_a = {
        "cohorts": {
            "TIME-CHF": {
                "variables": {
                    "sex": {
                        "categories": {"F": {"value": "F", "label": "Female"}},
                    }
                }
            }
        }
    }
    cache_b = {
        "cohorts": {
            "TIME-CHF": {
                "variables": {
                    "sex": {
                        "categories": {"F": {"value": "F", "label": "Female participant"}},
                    }
                }
            }
        }
    }

    report = compare_cache_objects(cache_a, cache_b)

    category_differences = report["cohort_comparisons"]["TIME-CHF"]["variable_differences"]["sex"][
        "category_differences"
    ]
    assert category_differences["categories_only_in_a"] == []
    assert category_differences["categories_only_in_b"] == []
    assert category_differences["category_value_differences"] == {
        "F": {
            "value_in_a": {"value": "F", "label": "Female"},
            "value_in_b": {"value": "F", "label": "Female participant"},
        }
    }


def test_sparql_metadata_route_delegates_and_preserves_summary_shape(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = sample_cohort()
    calls: list[tuple[str, bool]] = []

    def retrieve(user_email: str, include_sparql_metadata: bool = False):
        calls.append((user_email, include_sparql_metadata))
        return {
            "cohorts": {cohort.cohort_id: cohort},
            "sparql_metadata": {
                "row_count": 2,
                "query_duration_ms": 3,
                "processing_duration_ms": 4,
                "total_duration_ms": 7,
            },
        }

    monkeypatch.setattr(route_context.explore, "retrieve_cohorts_metadata", retrieve)

    with authenticated_client(route_context.app) as client:
        response = client.get("/cohorts-metadata-sparql", params={"summary": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["userEmail"] == ADMIN_EMAIL
    assert payload["sparql_metadata"] == {
        "row_count": 2,
        "query_duration_ms": 3,
        "processing_duration_ms": 4,
        "total_duration_ms": 7,
    }
    assert payload["cohorts"]["TIME-CHF"]["institution"] == "Synthetic Maastricht Heart Institute"
    assert "variables" not in payload["cohorts"]["TIME-CHF"]
    assert calls == [(ADMIN_EMAIL, True)]


def test_sparql_metadata_route_preserves_full_cohort_and_variable_shape(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = sample_cohort()
    calls: list[tuple[str, bool]] = []

    def retrieve(user_email: str, include_sparql_metadata: bool = False):
        calls.append((user_email, include_sparql_metadata))
        return {
            "cohorts": {cohort.cohort_id: cohort},
            "sparql_metadata": {
                "row_count": 2,
                "query_duration_ms": 3,
                "processing_duration_ms": 4,
                "total_duration_ms": 7,
            },
        }

    monkeypatch.setattr(route_context.explore, "retrieve_cohorts_metadata", retrieve)

    with authenticated_client(route_context.app) as client:
        response = client.get("/cohorts-metadata-sparql")

    assert response.status_code == 200
    payload = response.json()
    assert payload["userEmail"] == ADMIN_EMAIL
    assert payload["sparql_metadata"]["row_count"] == 2
    cohort_payload = payload["cohorts"]["TIME-CHF"]
    assert cohort_payload["cohort_id"] == "TIME-CHF"
    assert cohort_payload["institution"] == "Synthetic Maastricht Heart Institute"
    variable_payload = cohort_payload["variables"]["sex"]
    assert variable_payload["var_name"] == "sex"
    assert variable_payload["count"] == 24
    assert variable_payload["categories"] == [
        {
            "value": "F",
            "label": "Female",
            "concept_id": "8532",
            "mapped_id": None,
            "mapped_label": None,
        },
        {
            "value": "M",
            "label": "Male",
            "concept_id": "8507",
            "mapped_id": None,
            "mapped_label": None,
        },
    ]
    assert calls == [(ADMIN_EMAIL, True)]


def test_metadata_spreadsheet_route_returns_exact_file_and_last_modified_header(
    route_context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spreadsheet = tmp_path / "synthetic-cohorts.xlsx"
    spreadsheet.write_bytes(b"synthetic workbook bytes")
    monkeypatch.setattr(route_context.upload, "COHORTS_METADATA_FILEPATH", str(spreadsheet))

    with authenticated_client(route_context.app) as client:
        response = client.get("/download-cohorts-metadata-spreadsheet")

    assert response.status_code == 200
    assert response.content == b"synthetic workbook bytes"
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert "synthetic-cohorts.xlsx" in response.headers["content-disposition"]
    assert response.headers["x-file-last-modified"]


def test_syntax_report_route_delegates_to_canonical_report_service(
    route_context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "metadata-syntax-report.txt"
    report_path.write_text("deterministic syntax report\n", encoding="utf-8")
    calls: list[str] = []

    def build_report():
        calls.append("build")
        return SimpleNamespace(
            path=report_path,
            total_cohorts=2,
            cohorts_with_errors=1,
            cohorts_without_dictionary=0,
            total_errors=3,
        )

    monkeypatch.setattr(route_context.metadata_reports, "build_syntax_report", build_report)

    with authenticated_client(route_context.app) as client:
        response = client.post("/metadata-syntax-issues-report")

    assert response.status_code == 200
    assert response.content == b"deterministic syntax report\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["x-total-cohorts"] == "2"
    assert response.headers["x-cohorts-with-errors"] == "1"
    assert response.headers["x-cohorts-without-dict"] == "0"
    assert response.headers["x-total-errors"] == "3"
    assert calls == ["build"]


def test_normalize_headers_route_updates_canonical_dictionary_and_returns_report(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dictionary = Path(route_context.settings.cohort_folder) / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("variable name,variable label,vartype\nage,Age,INT\n", encoding="utf-8")
    monkeypatch.setattr(
        route_context.cohort_cache,
        "get_cohorts_from_cache",
        lambda _email: {"TIME-CHF": sample_cohort()},
    )

    with authenticated_client(route_context.app) as client:
        response = client.post("/normalize-all-dictionary-headers")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "header_normalization_report_" in response.headers["content-disposition"]
    assert "Performed by: nikolas.molyndris@decentriq.ch" in response.text
    assert "Total cohorts processed: 1" in response.text
    assert "Cohorts with normalized headers: 1" in response.text
    assert dictionary.read_text(encoding="utf-8").splitlines()[0] == "VARIABLENAME,VARIABLELABEL,VARTYPE"


def test_get_logs_route_returns_deterministic_line_array(route_context) -> None:
    logs_path = Path(route_context.settings.logs_filepath)
    logs_path.write_text("first synthetic event\nsecond synthetic event\n", encoding="utf-8")

    with authenticated_client(route_context.app) as client:
        response = client.post("/get-logs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == ["first synthetic event", "second synthetic event", ""]


def test_delete_cohort_route_removes_graphs_folder_and_cache_entry(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_folder = Path(route_context.settings.cohort_folder) / "TIME-CHF"
    cohort_folder.mkdir(parents=True)
    (cohort_folder / "TIME-CHF_datadictionary.csv").write_text("synthetic", encoding="utf-8")
    graph_calls: list[tuple[object, ...]] = []
    cache_calls: list[str] = []
    monkeypatch.setattr(route_context.upload, "get_cohort_mapping_uri", lambda cohort_id: f"mapping:{cohort_id}")
    monkeypatch.setattr(route_context.upload, "get_cohort_uri", lambda cohort_id: f"cohort:{cohort_id}")
    monkeypatch.setattr(route_context.upload, "get_cohort_graph_uri", lambda cohort_id: f"graph:{cohort_id}")
    monkeypatch.setattr(
        route_context.upload,
        "delete_existing_triples",
        lambda *args: graph_calls.append(args),
    )
    monkeypatch.setattr(
        route_context.cohort_cache,
        "remove_cohort_from_cache",
        lambda cohort_id: cache_calls.append(cohort_id),
    )

    with authenticated_client(route_context.app) as client:
        response = client.post("/delete-cohort", data={"cohort_id": "TIME-CHF"})

    assert response.status_code == 200
    assert response.json() == {"message": "Cohort TIME-CHF has been successfully deleted."}
    assert graph_calls == [
        ("mapping:TIME-CHF", "<cohort:TIME-CHF>", "icare:previewEnabled"),
        ("graph:TIME-CHF",),
    ]
    assert cache_calls == ["TIME-CHF"]
    assert not cohort_folder.exists()


def test_delete_cohort_route_rejects_traversal_before_any_mutation(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    victim = Path(route_context.settings.data_folder) / "victim"
    victim.mkdir()
    marker = victim / "must-survive.txt"
    marker.write_text("synthetic evidence", encoding="utf-8")
    graph_calls: list[tuple[object, ...]] = []
    cache_calls: list[str] = []
    monkeypatch.setattr(
        route_context.upload,
        "delete_existing_triples",
        lambda *args: graph_calls.append(args),
    )
    monkeypatch.setattr(
        route_context.cohort_cache,
        "remove_cohort_from_cache",
        lambda cohort_id: cache_calls.append(cohort_id),
    )

    with authenticated_client(route_context.app) as client:
        response = client.post("/delete-cohort", data={"cohort_id": "../victim"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid cohort ID: '../victim'"}
    assert marker.read_text(encoding="utf-8") == "synthetic evidence"
    assert graph_calls == []
    assert cache_calls == []


def test_single_cohort_validation_resolves_canonical_id_and_delegates_to_fixture_provider(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import cohort_cache

    dictionary = Path(route_context.settings.cohort_folder) / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("VARIABLENAME,VARIABLELABEL\nage,Age\n", encoding="utf-8")
    provider = RecordingValidationProvider()
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {"TIME-CHF": sample_cohort()})
    monkeypatch.setattr(route_context.upload, "get_concept_validation_provider", lambda _settings: provider)

    with authenticated_client(route_context.app) as client:
        response = client.post("/validate-athena-codes/time-chf")

    assert response.status_code == 200
    assert response.content == b"variable name,overall_status\nage,PASS\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["x-cohort-id"] == "TIME-CHF"
    assert response.headers["x-all-pass"] == "True"
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == dictionary
    assert provider.calls[0][1].parent == Path(route_context.settings.data_folder) / "ATHENA_VALIDATION_REPORTS"


def test_all_cohort_validation_summarizes_fixture_provider_results_and_missing_dictionary(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import cohort_cache

    dictionary = Path(route_context.settings.cohort_folder) / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("VARIABLENAME,VARIABLELABEL\nage,Age\n", encoding="utf-8")
    provider = RecordingValidationProvider()
    monkeypatch.setattr(
        cohort_cache,
        "get_cohorts_from_cache",
        lambda _email: {"GISSI-HF": sample_cohort(), "TIME-CHF": sample_cohort()},
    )
    monkeypatch.setattr(route_context.upload, "get_concept_validation_provider", lambda _settings: provider)

    with authenticated_client(route_context.app) as client:
        response = client.post("/validate-athena-codes-all-summary")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["x-total-cohorts"] == "2"
    assert response.headers["x-cohorts-passed"] == "1"
    assert response.headers["x-cohorts-failed"] == "0"
    assert response.headers["x-cohorts-without-dict"] == "1"
    assert response.headers["x-cohorts-errored"] == "0"
    report = response.text
    assert "COHORT: TIME-CHF" in report
    assert "PASS: 1  |  FAIL: 0  |  N/A: 0" in report
    assert "Cohorts without dictionary: 1" in report
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == dictionary


def test_cache_download_round_trips_through_compare_route_with_list_categories(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = sample_cohort()
    monkeypatch.setattr(route_context.data_analysis, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(
        route_context.data_analysis,
        "get_cohorts_from_cache",
        lambda email: {cohort.cohort_id: cohort} if email == ADMIN_EMAIL else {},
    )

    with authenticated_client(route_context.app) as client:
        download = client.get("/api/download-cohorts-cache")
        assert download.status_code == 200
        downloaded = download.json()
        modified = json.loads(json.dumps(downloaded))
        categories = modified["cohorts"]["TIME-CHF"]["variables"]["sex"]["categories"]
        categories.reverse()
        next(category for category in categories if category["value"] == "F")["label"] = "Female participant"

        comparison = client.post(
            "/api/compare-cohorts-cache",
            files={
                "cache_file_a": ("download-a.json", download.content, "application/json"),
                "cache_file_b": ("download-b.json", json.dumps(modified).encode(), "application/json"),
            },
        )

    assert download.headers["content-type"] == "application/json"
    assert "cohorts_cache_" in download.headers["content-disposition"]
    assert downloaded["userEmail"] == ADMIN_EMAIL
    assert downloaded["cache_status"] == "from_cache"
    assert downloaded["cohort_count"] == 1
    assert isinstance(downloaded["cohorts"]["TIME-CHF"]["variables"]["sex"]["categories"], list)

    assert comparison.status_code == 200
    assert comparison.headers["content-type"] == "application/json"
    comparison_payload = comparison.json()
    assert comparison_payload["summary"]["cohorts_with_differences"] == 1
    assert comparison_payload["comparison_metadata"]["file_a_name"] == "download-a.json"
    assert comparison_payload["comparison_metadata"]["file_b_name"] == "download-b.json"
    assert comparison_payload["comparison_metadata"]["performed_by"] == ADMIN_EMAIL
    category_diff = comparison_payload["cohort_comparisons"]["TIME-CHF"]["variable_differences"]["sex"][
        "category_differences"
    ]
    assert category_diff["categories_only_in_a"] == []
    assert category_diff["categories_only_in_b"] == []
    assert list(category_diff["category_value_differences"]) == ["F"]


def test_cache_compare_route_rejects_invalid_json_as_client_error(route_context) -> None:
    with authenticated_client(route_context.app) as client:
        response = client.post(
            "/api/compare-cohorts-cache",
            files={
                "cache_file_a": ("broken.json", b"{not-json", "application/json"),
                "cache_file_b": ("valid.json", b'{"cohorts": {}}', "application/json"),
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Invalid JSON in cache file:")


def test_refresh_and_clear_cache_routes_delegate_and_remove_local_state(
    route_context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import cohort_cache

    calls: list[tuple[str, str | None]] = []
    timestamp_file = tmp_path / "cache.timestamp"
    timestamp_file.write_text("stale", encoding="utf-8")
    init_lock = tmp_path / ".cache_init.lock"
    write_lock = tmp_path / ".cache_write.lock"
    init_lock.write_text("stale", encoding="utf-8")
    write_lock.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        cohort_cache,
        "initialize_cache_from_source_files",
        lambda email: calls.append(("refresh", email)),
    )
    monkeypatch.setattr(cohort_cache, "clear_cache", lambda: calls.append(("clear", None)))
    monkeypatch.setattr(cohort_cache, "get_cache_timestamp_file", lambda: timestamp_file)

    with authenticated_client(route_context.app) as client:
        refresh_response = client.post("/refresh-cache")
        clear_response = client.post("/clear-cache")

    assert refresh_response.status_code == 200
    assert refresh_response.json() == {
        "message": "Cache has been successfully refreshed from the source files (Excel + CSV dictionaries).",
        "refreshed_by": ADMIN_EMAIL,
    }
    assert clear_response.status_code == 200
    assert clear_response.json() == {
        "message": "Cache has been successfully cleared. It will be re-initialized on the next API request.",
        "cleared_by": ADMIN_EMAIL,
    }
    assert calls == [("refresh", ADMIN_EMAIL), ("clear", None)]
    assert not timestamp_file.exists()
    assert not init_lock.exists()
    assert not write_lock.exists()


def test_refresh_cache_reports_missing_source_workbook(route_context) -> None:
    spreadsheet = Path(route_context.settings.data_folder) / "iCARE4CVD_Cohorts.xlsx"
    assert not spreadsheet.exists()

    with authenticated_client(route_context.app) as client:
        response = client.post("/refresh-cache")

    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to refresh cache: Excel metadata file not found at {spreadsheet}"}


def test_clear_then_refresh_reconstructs_identical_synthetic_inventory_from_source_files(
    route_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    metadata_mappings = importlib.import_module("src.metadata_mappings")
    monkeypatch.setattr(metadata_mappings, "hydrate_manual_mappings", lambda _cohort_id, _variables: None)

    spreadsheet = Path(route_context.settings.data_folder) / "iCARE4CVD_Cohorts.xlsx"
    pd.DataFrame(
        [
            {
                "Study Name": "TIME-CHF",
                "Institute": "Synthetic Maastricht Heart Institute",
                "Number of participants": "24",
                "Mixed Sex": "50% male; 50% female",
                "Age Distribution": "40-64: 50%; 65+: 50%",
                "References": "Synthetic protocol",
                "Administrator Email Address": ADMIN_EMAIL,
                "Study Contact Person Email Address": "owner@example.test",
            }
        ]
    ).to_excel(spreadsheet, sheet_name="Descriptions", index=False)
    dictionary = Path(route_context.settings.cohort_folder) / "TIME-CHF" / "TIME-CHF_datadictionary.csv"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text(
        "variable name,variable label,vartype,count,na,categorical,categorical value concept code,"
        "categorical value concept name,categorical value omop id\n"
        'sex,Biological sex,STR,24,0,"F=Female | M=Male",8532|8507,Female|Male,8532|8507\n',
        encoding="utf-8",
    )

    route_context.cohort_cache.clear_cache()
    route_context.cohort_cache.initialize_cache_from_source_files(ADMIN_EMAIL)

    def inventory() -> dict[str, object]:
        cohorts = route_context.cohort_cache.get_cohorts_from_cache(ADMIN_EMAIL)
        return {cohort_id: route_context.cohort_cache.cohort_to_dict(cohort) for cohort_id, cohort in cohorts.items()}

    inventory_before = inventory()
    cache_file = route_context.cohort_cache.get_cache_file_path()
    timestamp_file = route_context.cohort_cache.get_cache_timestamp_file()
    timestamp_file.write_text("synthetic-session", encoding="utf-8")
    assert cache_file.is_file()

    with authenticated_client(route_context.app) as client:
        clear_response = client.post("/clear-cache")
        assert clear_response.status_code == 200
        assert route_context.cohort_cache._cohorts_cache == {}
        assert not cache_file.exists()
        assert not timestamp_file.exists()

        refresh_response = client.post("/refresh-cache")

    assert refresh_response.status_code == 200
    assert refresh_response.json() == {
        "message": "Cache has been successfully refreshed from the source files (Excel + CSV dictionaries).",
        "refreshed_by": ADMIN_EMAIL,
    }
    inventory_after = inventory()
    assert inventory_after == inventory_before
    assert list(inventory_after) == ["TIME-CHF"]
    variable = inventory_after["TIME-CHF"]["variables"]["sex"]
    assert variable["var_label"] == "Biological sex"
    assert variable["count"] == 24
    assert variable["categories"] == [
        {
            "value": "F",
            "label": "Female",
            "concept_id": "8532",
            "mapped_id": "8532",
            "mapped_label": "Female",
        },
        {
            "value": "M",
            "label": "Male",
            "concept_id": "8507",
            "mapped_id": "8507",
            "mapped_label": "Male",
        },
    ]
    assert cache_file.is_file()


@pytest.mark.parametrize(
    ("path", "files"),
    [
        ("/cohorts-metadata-sparql", None),
        ("/metadata-syntax-issues-report", None),
        ("/validate-athena-codes/TIME-CHF", None),
        ("/validate-athena-codes-all-summary", None),
        ("/download-cohorts-metadata-spreadsheet", None),
        ("/api/download-cohorts-cache", None),
        (
            "/api/compare-cohorts-cache",
            {
                "cache_file_a": ("a.json", b'{"cohorts": {}}', "application/json"),
                "cache_file_b": ("b.json", b'{"cohorts": {}}', "application/json"),
            },
        ),
        ("/refresh-cache", None),
        ("/clear-cache", None),
        ("/normalize-all-dictionary-headers", None),
        ("/get-logs", None),
        ("/delete-cohort", None),
    ],
)
def test_metadata_matrix_routes_require_authentication(route_context, path: str, files) -> None:
    method = (
        "GET"
        if path
        in {
            "/cohorts-metadata-sparql",
            "/download-cohorts-metadata-spreadsheet",
            "/api/download-cohorts-cache",
        }
        else "POST"
    )

    with TestClient(route_context.app) as client:
        response = client.request(method, path, files=files)

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


@pytest.mark.parametrize(
    "path",
    [
        "/metadata-syntax-issues-report",
        "/validate-athena-codes/TIME-CHF",
        "/validate-athena-codes-all-summary",
        "/refresh-cache",
        "/clear-cache",
        "/normalize-all-dictionary-headers",
        "/get-logs",
        "/delete-cohort",
    ],
)
def test_metadata_admin_routes_reject_authenticated_non_admin(route_context, path: str) -> None:
    with authenticated_client(route_context.app, "viewer@example.test") as client:
        response = client.post(path, data={"cohort_id": "TIME-CHF"} if path == "/delete-cohort" else None)

    assert response.status_code == 403
    assert response.json() == {"detail": "You need to be admin to perform this action."}
