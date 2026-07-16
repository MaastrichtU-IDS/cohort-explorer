from dataclasses import asdict

import pytest

from src.concept_usage import get_concept_usage
from src.metadata_providers.contracts import ConceptResult
from src.metadata_providers.factory import get_concept_search_provider
from src.metadata_providers.fixture_catalog import FixtureConceptSearchProvider
from src.models import Cohort, CohortVariable, VariableCategory

HEART_RATE_URI = "http://snomed.info/id/364075005"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fixture_search_provider(monkeypatch):
    from src import concept_usage

    def fake_run_query(_query: str):
        return {
            "results": {
                "bindings": [
                    {
                        "cohortId": {"value": "TIME-CHF"},
                        "varName": {"value": "heart_rate"},
                        "varLabel": {"value": "Heart rate"},
                        "omopDomain": {"value": "Measurement"},
                        "mappedId": {"value": HEART_RATE_URI},
                    }
                ]
            }
        }

    monkeypatch.setattr(concept_usage, "run_query", fake_run_query)
    return FixtureConceptSearchProvider(usage_lookup=get_concept_usage)


@pytest.mark.anyio
async def test_fixture_search_is_ranked_filtered_and_enriched(fixture_search_provider):
    results = await fixture_search_provider.search("heart rate", ["Measurement"])

    assert [item.id for item in results][:1] == ["snomedct:364075005"]
    assert results[0].domain == "Measurement"
    assert results[0].used_by == [
        {
            "cohort_id": "TIME-CHF",
            "var_name": "heart_rate",
            "var_label": "Heart rate",
            "omop_domain": "Measurement",
        }
    ]


@pytest.mark.anyio
async def test_fixture_search_filters_domains_case_insensitively(fixture_search_provider):
    results = await fixture_search_provider.search("heart", ["condition"])

    assert [item.id for item in results] == [
        "snomed:416683003",
        "snomedct:84114007",
    ]


@pytest.mark.anyio
async def test_fixture_search_returns_empty_for_unknown_or_blank_query(fixture_search_provider):
    assert await fixture_search_provider.search("not-in-the-catalog", []) == []
    assert await fixture_search_provider.search("  ", []) == []


@pytest.mark.anyio
async def test_fixture_search_uses_metadata_cache_without_sparql(monkeypatch):
    from src import cohort_cache, concept_usage

    def fail_if_sparql_is_queried(_query: str):
        raise AssertionError("fixture concept search must not query SPARQL")

    age = CohortVariable(
        var_name="age",
        var_label="Age at enrollment",
        var_type="FLOAT",
        count=2_500,
        mapped_id="loinc:30525-0",
        omop_domain="Person",
    )
    cohort = Cohort(cohort_id="TIME-CHF", variables={"age": age})
    monkeypatch.setattr(concept_usage, "run_query", fail_if_sparql_is_queried)
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {"TIME-CHF": cohort})

    results = await FixtureConceptSearchProvider().search("age", ["Person"])

    assert [item.id for item in results][:1] == ["loinc:30525-0"]
    assert results[0].used_by == [
        {
            "cohort_id": "TIME-CHF",
            "var_name": "age",
            "var_label": "Age at enrollment",
            "omop_domain": "Person",
        }
    ]


@pytest.mark.anyio
async def test_fixture_cache_usage_normalizes_aliases_categories_and_full_uris(monkeypatch):
    from src import cohort_cache

    heart_failure_admission_uri = "http://snomed.info/id/416683003"
    admission = CohortVariable(
        var_name="hf_hosp",
        var_label="Emergency hospital admission for heart failure",
        var_type="BOOL",
        count=2_500,
        mapped_id=heart_failure_admission_uri,
        omop_domain="Condition",
        categories=[
            VariableCategory(
                value="1",
                label="Yes",
                mapped_id="snomedct:416683003",
            )
        ],
    )
    cohort = Cohort(cohort_id="TIME-CHF", variables={"hf_hosp": admission})
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {"TIME-CHF": cohort})

    results = await FixtureConceptSearchProvider().search("emergency hospital admission", ["Condition"])

    concept = next(item for item in results if item.id == "snomed:416683003")
    assert concept.used_by == [
        {
            "cohort_id": "TIME-CHF",
            "var_name": "hf_hosp",
            "var_label": "Emergency hospital admission for heart failure",
            "omop_domain": "Condition",
        }
    ]


def test_concept_usage_queries_cmeo_study_and_data_element_graphs(monkeypatch):
    from src import concept_usage

    captured_query = ""

    def fake_run_query(query: str):
        nonlocal captured_query
        captured_query = query
        return {
            "results": {
                "bindings": [
                    {
                        "cohortId": {"value": "TIME-CHF"},
                        "varName": {"value": "heart_rate"},
                        "varLabel": {"value": "Heart rate"},
                        "omopDomain": {"value": "Measurement"},
                        "mappedId": {"value": HEART_RATE_URI},
                    }
                ]
            }
        }

    monkeypatch.setattr(concept_usage, "run_query", fake_run_query)

    usage = get_concept_usage([HEART_RATE_URI])

    assert "cmeo:study" in captured_query
    assert "cmeo:data_element" in captured_query
    assert "icare:Cohort" not in captured_query
    assert usage[HEART_RATE_URI][0]["var_name"] == "heart_rate"


@pytest.mark.anyio
async def test_search_route_serializes_provider_results(monkeypatch):
    from src import mapping

    result = ConceptResult(
        id="snomedct:364075005",
        uri=HEART_RATE_URI,
        label="Heart rate",
        domain="Measurement",
        vocabulary="SNOMED",
        used_by=[],
    )

    class StubProvider:
        async def search(self, query: str, domains: list[str]):
            assert query == "heart rate"
            assert domains == ["Measurement"]
            return [result]

    monkeypatch.setattr(mapping, "get_concept_search_provider", lambda _settings: StubProvider())

    response = await mapping.search_concepts(
        query="heart rate",
        domain=["Measurement"],
        user={"email": "admin@example.test"},
    )

    assert response == [asdict(result)]
