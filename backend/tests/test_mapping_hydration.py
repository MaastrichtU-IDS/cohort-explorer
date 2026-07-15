from pathlib import Path

import pytest

DICTIONARY = """VARIABLENAME,VARIABLELABEL,VARTYPE,CATEGORICAL,COUNT,NA
heart_rate,Heart rate,INT,,10,0
sex,Sex,INT,0=Female|1=Male,10,0
"""


def _binding(value: str) -> dict[str, str]:
    return {"type": "uri", "value": value}


def test_variable_and_category_mappings_survive_cache_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import cohort_cache, metadata_mappings
    from src.models import Cohort

    dictionary = tmp_path / "TIME-CHF_datadictionary.csv"
    dictionary.write_text(DICTIONARY, encoding="utf-8")
    monkeypatch.setattr(
        cohort_cache,
        "_cohorts_cache",
        {"TIME-CHF": Cohort(cohort_id="TIME-CHF")},
    )
    queries: list[str] = []

    def fake_query(query: str) -> dict[str, object]:
        queries.append(query)
        return {
            "results": {
                "bindings": [
                    {
                        "subject": _binding(
                            "https://w3id.org/icare4cvd/cohort/TIME-CHF/heart_rate"
                        ),
                        "mappedId": _binding("http://snomed.info/id/364075005"),
                        "mappedLabel": {
                            "type": "literal",
                            "value": "Heart rate (observable entity)",
                        },
                    },
                    {
                        "subject": _binding(
                            "https://w3id.org/icare4cvd/cohort/TIME-CHF/sex/"
                            "categorical_value_specification/female"
                        ),
                        "mappedId": _binding("http://snomed.info/id/248153007"),
                        "mappedLabel": {"type": "literal", "value": "Female"},
                    },
                ]
            }
        }

    monkeypatch.setattr(metadata_mappings, "run_query", fake_query)

    assert cohort_cache.build_cohort_variables_from_csv("TIME-CHF", str(dictionary)) is True
    rebuilt = cohort_cache._cohorts_cache["TIME-CHF"]

    assert rebuilt.variables["heart_rate"].mapped_id == "snomedct:364075005"
    assert rebuilt.variables["heart_rate"].mapped_label == "Heart rate (observable entity)"
    assert rebuilt.variables["sex"].categories[0].mapped_id == "snomedct:248153007"
    assert rebuilt.variables["sex"].categories[0].mapped_label == "Female"
    assert len(queries) == 1
    assert (
        "GRAPH <https://w3id.org/icare4cvd/cohort/TIME-CHF/mappings>" in queries[0]
    )
    assert "categorical_value_specification/female" in queries[0]
    assert "/category/0" not in queries[0]


def test_category_subject_uses_the_dictionary_import_uri() -> None:
    from src.metadata_mappings import category_subject_uri

    assert str(category_subject_uri("TIME-CHF", "sex", "Female")) == (
        "https://w3id.org/icare4cvd/cohort/TIME-CHF/sex/"
        "categorical_value_specification/female"
    )


def test_insert_triples_resolves_category_index_to_its_imported_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import cohort_cache, upload
    from src.models import Cohort, CohortVariable, VariableCategory

    cohort = Cohort(
        cohort_id="TIME-CHF",
        can_edit=True,
        variables={
            "sex": CohortVariable(
                var_name="sex",
                var_label="Sex",
                var_type="INT",
                count=10,
                categories=[VariableCategory(value="0", label="Female")],
            )
        },
    )
    monkeypatch.setattr(
        cohort_cache,
        "get_cohorts_from_cache",
        lambda _email: {"TIME-CHF": cohort},
    )
    monkeypatch.setattr(cohort_cache, "add_cohort_to_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(upload, "delete_existing_triples", lambda *_args, **_kwargs: True)

    sent_queries: list[str] = []

    class FakeEndpoint:
        def __init__(self, _url: str) -> None:
            self.query_text = ""

        def setMethod(self, _method: str) -> None:  # noqa: N802 - mirrors SPARQLWrapper
            pass

        def setRequestMethod(self, _method: str) -> None:  # noqa: N802 - mirrors SPARQLWrapper
            pass

        def setTimeout(self, _timeout: int) -> None:  # noqa: N802 - mirrors SPARQLWrapper
            pass

        def setQuery(self, query: str) -> None:  # noqa: N802 - mirrors SPARQLWrapper
            self.query_text = query

        def query(self) -> None:
            sent_queries.append(self.query_text)

    monkeypatch.setattr(upload, "SPARQLWrapper", FakeEndpoint)

    upload.insert_triples(
        cohort_id="TIME-CHF",
        var_id="sex",
        predicate="icare:mappedId",
        value="snomedct:248153007",
        label="Female",
        category_id="0",
        user={"email": "admin@example.test"},
    )

    assert len(sent_queries) == 1
    assert "categorical_value_specification/female" in sent_queries[0]
    assert "/category/0" not in sent_queries[0]


@pytest.mark.parametrize("category_id", ["9", "-1", "not-an-index"])
def test_insert_triples_rejects_an_unknown_category_index(
    monkeypatch: pytest.MonkeyPatch,
    category_id: str,
) -> None:
    from fastapi import HTTPException

    from src import cohort_cache, upload
    from src.models import Cohort, CohortVariable, VariableCategory

    cohort = Cohort(
        cohort_id="TIME-CHF",
        can_edit=True,
        variables={
            "sex": CohortVariable(
                var_name="sex",
                var_label="Sex",
                var_type="INT",
                count=10,
                categories=[VariableCategory(value="0", label="Female")],
            )
        },
    )
    monkeypatch.setattr(
        cohort_cache,
        "get_cohorts_from_cache",
        lambda _email: {"TIME-CHF": cohort},
    )

    with pytest.raises(HTTPException, match="Unknown category index") as error:
        upload.insert_triples(
            cohort_id="TIME-CHF",
            var_id="sex",
            predicate="icare:mappedId",
            value="snomedct:248153007",
            label="Female",
            category_id=category_id,
            user={"email": "admin@example.test"},
        )

    assert error.value.status_code == 422
