import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from src.metadata_providers.contracts import ConceptResult

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "demo" / "metadata-fixtures" / "concepts.json"
UsageLookup = Callable[[Sequence[str]], dict[str, list[dict[str, str]]]]


@dataclass(frozen=True)
class CatalogConcept:
    id: str
    uri: str
    label: str
    domain: str
    vocabulary: str


@cache
def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> tuple[CatalogConcept, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["concepts"] if isinstance(payload, dict) else payload
    return tuple(
        CatalogConcept(
            id=str(record["id"]),
            uri=str(record["uri"]),
            label=str(record["label"]),
            domain=str(record["domain"]),
            vocabulary=str(record["vocabulary"]),
        )
        for record in records
    )


class FixtureConceptSearchProvider:
    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        usage_lookup: UsageLookup | None = None,
    ) -> None:
        self._catalog = load_catalog(catalog_path)
        self._usage_lookup = usage_lookup or self._usage_from_cache

    def _usage_from_cache(self, concept_uris: Sequence[str]) -> dict[str, list[dict[str, str]]]:
        from src.cohort_cache import get_cohorts_from_cache

        requested_uris = sorted(set(concept_uris))
        usage_by_uri: dict[str, list[dict[str, str]]] = {uri: [] for uri in requested_uris}
        if not requested_uris:
            return usage_by_uri

        uri_by_identifier: dict[str, str] = {}
        for concept in self._catalog:
            identifiers = {concept.uri, concept.id}
            if concept.uri.startswith("http://snomed.info/id/"):
                code = concept.uri.removeprefix("http://snomed.info/id/")
                identifiers.update({f"snomed:{code}", f"snomedct:{code}"})
            for identifier in identifiers:
                uri_by_identifier[identifier] = concept.uri
                uri_by_identifier[identifier.casefold()] = concept.uri

        for cohort_id, cohort in get_cohorts_from_cache("").items():
            for variable in cohort.variables.values():
                mapped_ids = [variable.mapped_id, *(category.mapped_id for category in variable.categories)]
                for mapped_id in mapped_ids:
                    identifier = str(mapped_id or "").strip()
                    mapped_uri = uri_by_identifier.get(identifier) or uri_by_identifier.get(identifier.casefold())
                    if mapped_uri not in usage_by_uri:
                        continue
                    entry = {
                        "cohort_id": cohort_id,
                        "var_name": variable.var_name,
                        "var_label": variable.var_label,
                        "omop_domain": variable.omop_domain or "",
                    }
                    if entry not in usage_by_uri[mapped_uri]:
                        usage_by_uri[mapped_uri].append(entry)

        for entries in usage_by_uri.values():
            entries.sort(key=lambda entry: (entry["cohort_id"], entry["var_name"]))
        return usage_by_uri

    async def search(self, query: str, domains: Sequence[str]) -> list[ConceptResult]:
        normalized_query = " ".join(query.casefold().split())
        if not normalized_query:
            return []

        tokens = normalized_query.split()
        domain_filter = {domain.strip().casefold() for domain in domains if domain.strip()}
        matches: list[tuple[int, CatalogConcept]] = []
        for concept in self._catalog:
            if domain_filter and concept.domain.casefold() not in domain_filter:
                continue
            label = " ".join(concept.label.casefold().split())
            label_tokens = set(label.split())
            if f" {normalized_query} " in f" {label} ":
                rank = 0
            elif all(token in label_tokens for token in tokens):
                rank = 1
            elif any(token in label_tokens for token in tokens):
                rank = 2
            else:
                continue
            matches.append((rank, concept))

        matches.sort(key=lambda match: (match[0], match[1].label.casefold(), match[1].id))
        usage_by_uri = self._usage_lookup([concept.uri for _, concept in matches])
        return [
            ConceptResult(
                id=concept.id,
                uri=concept.uri,
                label=concept.label,
                domain=concept.domain,
                vocabulary=concept.vocabulary,
                used_by=usage_by_uri.get(concept.uri, []),
            )
            for _, concept in matches
        ]
