from typing import Any

from src.metadata_providers.contracts import (
    ConceptSearchProvider,
    ConceptValidationProvider,
    MappingGenerationProvider,
)


def get_concept_search_provider(settings: Any) -> ConceptSearchProvider:
    backend = str(settings.concept_search_backend).strip().lower()
    if backend == "fixture":
        from src.metadata_providers.fixture_catalog import FixtureConceptSearchProvider

        return FixtureConceptSearchProvider()
    if backend == "athena":
        from src.metadata_providers.athena_search import AthenaConceptSearchProvider

        return AthenaConceptSearchProvider()
    raise ValueError(f"Unsupported concept search backend: {backend}")


def get_concept_validation_provider(settings: Any) -> ConceptValidationProvider:
    backend = str(settings.concept_validation_backend).strip().lower()
    if backend == "fixture":
        from src.metadata_providers.fixture_validation import FixtureConceptValidationProvider

        return FixtureConceptValidationProvider()
    if backend == "cohortvarlinker":
        from src.metadata_providers.cohortvarlinker_validation import CohortVarLinkerConceptValidationProvider

        return CohortVarLinkerConceptValidationProvider()
    raise ValueError(f"Unsupported concept validation backend: {backend}")


def get_mapping_generation_provider(settings: Any) -> MappingGenerationProvider:
    backend = str(settings.mapping_generation_backend).strip().lower()
    if backend == "fixture":
        from src.metadata_providers.fixture_mapping import FixtureMappingGenerationProvider

        return FixtureMappingGenerationProvider()
    if backend == "cohortvarlinker":
        from src.metadata_providers.cohortvarlinker_mapping import CohortVarLinkerMappingGenerationProvider

        return CohortVarLinkerMappingGenerationProvider()
    raise ValueError(f"Unsupported mapping generation backend: {backend}")
