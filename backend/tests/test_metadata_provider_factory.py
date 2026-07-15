import sys

import pytest

from src.metadata_providers.factory import (
    get_concept_search_provider,
    get_concept_validation_provider,
    get_mapping_generation_provider,
)


class ImportBomb:
    def __getattr__(self, name: str):
        raise AssertionError(f"live provider import accessed {name}")


@pytest.fixture
def fixture_settings(settings_factory):
    return settings_factory(concept_search_backend="fixture")


def test_live_search_provider_is_not_imported_in_fixture_mode(monkeypatch, fixture_settings):
    monkeypatch.setitem(sys.modules, "src.metadata_providers.athena_search", ImportBomb())

    provider = get_concept_search_provider(fixture_settings)

    assert provider.__class__.__name__ == "FixtureConceptSearchProvider"


def test_default_live_adapters_defer_cohortvarlinker_engine_imports(monkeypatch, settings_factory):
    monkeypatch.setitem(sys.modules, "CohortVarLinker.validate_cde", ImportBomb())
    monkeypatch.setitem(sys.modules, "CohortVarLinker.main", ImportBomb())
    live_settings = settings_factory()

    search_provider = get_concept_search_provider(live_settings)
    validation_provider = get_concept_validation_provider(live_settings)
    mapping_provider = get_mapping_generation_provider(live_settings)

    assert search_provider.__class__.__name__ == "AthenaConceptSearchProvider"
    assert validation_provider.__class__.__name__ == "CohortVarLinkerConceptValidationProvider"
    assert mapping_provider.__class__.__name__ == "CohortVarLinkerMappingGenerationProvider"


@pytest.mark.parametrize(
    ("setting_name", "factory", "provider_name"),
    [
        ("concept_search_backend", get_concept_search_provider, "concept search"),
        ("concept_validation_backend", get_concept_validation_provider, "concept validation"),
        ("mapping_generation_backend", get_mapping_generation_provider, "mapping generation"),
    ],
)
def test_unknown_provider_selector_fails_closed(settings_factory, setting_name, factory, provider_name):
    provider_settings = settings_factory(**{setting_name: "unknown"})

    with pytest.raises(ValueError, match=f"Unsupported {provider_name} backend: unknown"):
        factory(provider_settings)
