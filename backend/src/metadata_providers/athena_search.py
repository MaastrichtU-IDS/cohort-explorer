from collections.abc import Sequence

import requests
from fastapi import HTTPException

from src.concept_usage import get_concept_usage
from src.metadata_providers.contracts import ConceptResult
from src.utils import curie_converter


class AthenaConceptSearchProvider:
    async def search(self, query: str, domains: Sequence[str]) -> list[ConceptResult]:
        response = None
        try:
            response = requests.get(
                "https://athena.ohdsi.org/api/v1/concepts",
                params={
                    "query": query,
                    "domain": list(domains),
                    "vocabulary": ["LOINC", "ATC", "SNOMED"],
                    "standardConcept": ["Standard", "Classification"],
                    "pageSize": 15,
                    "page": 1,
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.3"
                    )
                },
                timeout=60,
            )
            response.raise_for_status()
            search_results = response.json().get("content", [])
        except Exception as exc:
            status_code = response.status_code if response is not None else 502
            raise HTTPException(status_code=status_code, detail="Error fetching data from OHDSI Athena API") from exc

        concepts: list[ConceptResult] = []
        for result in search_results:
            vocabulary = str(result.get("vocabulary", ""))
            prefix = "snomedct" if vocabulary.casefold() == "snomed" else vocabulary.casefold()
            concept_id = f"{prefix}:{result.get('id')}"
            concepts.append(
                ConceptResult(
                    id=concept_id,
                    uri=curie_converter.expand(concept_id) or concept_id,
                    label=str(result.get("name", "")),
                    domain=str(result.get("domain", "")),
                    vocabulary=vocabulary,
                    used_by=[],
                )
            )

        usage_by_uri = get_concept_usage([concept.uri for concept in concepts])
        enriched = [
            ConceptResult(
                id=concept.id,
                uri=concept.uri,
                label=concept.label,
                domain=concept.domain,
                vocabulary=concept.vocabulary,
                used_by=usage_by_uri.get(concept.uri, []),
            )
            for concept in concepts
        ]
        enriched.sort(key=lambda concept: len(concept.used_by), reverse=True)
        return enriched
