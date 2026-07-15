from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ConceptResult:
    id: str
    uri: str
    label: str
    domain: str
    vocabulary: str
    used_by: list[dict[str, str]]


@dataclass(frozen=True)
class MappingGenerationResult:
    cache_info: object


class ConceptSearchProvider(Protocol):
    async def search(self, query: str, domains: Sequence[str]) -> list[ConceptResult]: ...


class ConceptValidationProvider(Protocol):
    def validate(self, dictionary_path: Path, report_path: Path) -> bool: ...


class MappingGenerationProvider(Protocol):
    def generate(
        self,
        source_study: str,
        target_studies: Sequence[tuple[str, bool]],
    ) -> MappingGenerationResult: ...
