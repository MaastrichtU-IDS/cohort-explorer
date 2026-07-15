import csv
import json
from pathlib import Path

import pandas as pd
import pytest

DICTIONARY_COLUMNS = [
    "variable name",
    "variable label",
    "vartype",
    "units",
    "categorical",
    "missing",
    "count",
    "na",
    "min",
    "max",
    "formula",
    "categorical value concept code",
    "categorical value concept name",
    "categorical value omop id",
    "variable concept code",
    "variable concept name",
    "variable omop id",
    "additional context concept name",
    "additional context concept code",
    "additional context omop id",
    "unit concept name",
    "unit concept code",
    "unit omop id",
    "domain",
    "visits",
    "visit omop id",
    "visit concept name",
    "visit concept code",
]


@pytest.fixture
def fixture_catalog(tmp_path: Path) -> tuple[Path, Path]:
    catalog_path = tmp_path / "concepts.json"
    catalog_path.write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "id": "snomedct:364075005",
                        "uri": "http://snomed.info/id/364075005",
                        "label": "Heart rate",
                        "domain": "Measurement",
                        "vocabulary": "SNOMED",
                        "omop_id": "3027018",
                    },
                    {
                        "id": "snomedct:248153007",
                        "uri": "http://snomed.info/id/248153007",
                        "label": "Male",
                        "domain": "Observation",
                        "vocabulary": "SNOMED",
                        "omop_id": "8507",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    relationships_path = tmp_path / "concept_relationship_enriched.csv"
    relationships_path.write_text(
        "concept_code,omop_id,concept_name,domain,vocabulary\n"
        "snomedct:364075005,3027018,Heart rate,Measurement,SNOMED\n"
        "snomedct:248153007,8507,Male,Observation,SNOMED\n",
        encoding="utf-8",
    )
    return catalog_path, relationships_path


def _dictionary(path: Path, **overrides: str) -> Path:
    row = dict.fromkeys(DICTIONARY_COLUMNS, "")
    row.update(
        {
            "variable name": "heart_rate",
            "variable label": "Heart rate",
            "vartype": "INT",
            "count": "10",
            "na": "0",
            "variable concept code": "snomedct:364075005",
            "variable concept name": "Heart rate",
            "variable omop id": "3027018",
            "domain": "measurement",
        }
    )
    row.update(overrides)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DICTIONARY_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_fixture_validator_accepts_a_catalogued_code_omop_pair(
    tmp_path: Path,
    fixture_catalog: tuple[Path, Path],
) -> None:
    from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

    catalog_path, relationships_path = fixture_catalog
    provider = FixtureConceptValidationProvider(catalog_path, relationships_path)
    report_path = tmp_path / "report.csv"

    assert provider.validate(_dictionary(tmp_path / "dictionary.csv"), report_path) is True
    report = pd.read_csv(report_path)
    assert report.loc[0, "variable_status"] == "PASS"
    assert report.loc[0, "overall_status"] == "PASS"


def test_fixture_validator_fails_closed_for_a_mismatched_pair(
    tmp_path: Path,
    fixture_catalog: tuple[Path, Path],
) -> None:
    from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

    catalog_path, relationships_path = fixture_catalog
    provider = FixtureConceptValidationProvider(catalog_path, relationships_path)
    report_path = tmp_path / "report.csv"

    valid = provider.validate(
        _dictionary(tmp_path / "dictionary.csv", **{"variable omop id": "8507"}),
        report_path,
    )

    assert valid is False
    report = pd.read_csv(report_path)
    assert report.loc[0, "variable_status"] == "FAIL"
    assert "not present in the offline fixture" in report.loc[0, "variable_reason"]
    assert report.loc[0, "overall_status"] == "FAIL"


@pytest.mark.parametrize(
    ("name_column", "code_column", "omop_column", "values_column", "status_column"),
    [
        (
            "categorical value concept name",
            "categorical value concept code",
            "categorical value omop id",
            "categorical",
            "categorical_value_status",
        ),
        (
            "additional context concept name",
            "additional context concept code",
            "additional context omop id",
            None,
            "additional_context_status",
        ),
        (
            "unit concept name",
            "unit concept code",
            "unit omop id",
            "units",
            "unit_status",
        ),
        (
            "visit concept name",
            "visit concept code",
            "visit omop id",
            "visits",
            "visit_status",
        ),
    ],
)
def test_fixture_validator_checks_each_supported_concept_context(
    tmp_path: Path,
    fixture_catalog: tuple[Path, Path],
    name_column: str,
    code_column: str,
    omop_column: str,
    values_column: str | None,
    status_column: str,
) -> None:
    from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

    overrides = {
        name_column: "Male",
        code_column: "snomedct:248153007",
        omop_column: "8507",
    }
    if values_column == "categorical":
        overrides[values_column] = "1=Male"
    elif values_column:
        overrides[values_column] = "baseline"
    catalog_path, relationships_path = fixture_catalog
    provider = FixtureConceptValidationProvider(catalog_path, relationships_path)
    report_path = tmp_path / "report.csv"

    assert provider.validate(_dictionary(tmp_path / "dictionary.csv", **overrides), report_path)
    status = pd.read_csv(report_path).loc[0, status_column]
    if status_column == "additional_context_status":
        assert "status='PASS'" in status
    else:
        assert status == "PASS"


def test_fixture_validator_runs_local_structure_validation_first(
    tmp_path: Path,
    fixture_catalog: tuple[Path, Path],
) -> None:
    from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

    catalog_path, relationships_path = fixture_catalog
    provider = FixtureConceptValidationProvider(catalog_path, relationships_path)
    dictionary_path = tmp_path / "dictionary.csv"
    dictionary_path.write_text("variable name,variable label\nhr,Heart rate\n", encoding="utf-8")
    report_path = tmp_path / "report.csv"

    assert provider.validate(dictionary_path, report_path) is False
    report = pd.read_csv(report_path)
    assert report.loc[0, "overall_status"] == "FAIL"
    assert "Missing required columns" in report.loc[0, "variable_reason"]
