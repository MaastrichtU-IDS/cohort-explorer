from collections.abc import Sequence

from src.metadata_providers.contracts import MappingGenerationResult


class CohortVarLinkerMappingGenerationProvider:
    def generate(
        self,
        source_study: str,
        target_studies: Sequence[tuple[str, bool]],
    ) -> MappingGenerationResult:
        from CohortVarLinker.main import generate_mapping_csv

        cache_info = generate_mapping_csv(source_study, list(target_studies))
        return MappingGenerationResult(cache_info=cache_info)
