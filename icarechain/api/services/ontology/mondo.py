import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

@dataclass
class DiseaseResult:
    \
    id: str
    name: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    xrefs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "synonyms": self.synonyms,
            "parents": self.parents,
            "children": self.children,
            "xrefs": self.xrefs,
        }

class MondoClient:
    \
\
\
\
\
\
\
\
\
\
    BASE_URL = "https://api.monarchinitiative.org/v3"

    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 1000

    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 3,
        cache_enabled: bool = True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled

        self._cache: dict[str, tuple[any, datetime]] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        \
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self.timeout,
                headers={"Accept": "application/json"}
            )
        return self._client

    async def close(self):
        \
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _cache_key(self, method: str, *args) -> str:
        \
        return f"mondo:{method}:{':'.join(str(a) for a in args)}"

    def _get_cached(self, key: str) -> Optional[any]:
        \
        if not self.cache_enabled:
            return None

        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, value: any):
        \
        if not self.cache_enabled:
            return

        if len(self._cache) >= self.MAX_CACHE_SIZE:

            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            for k in sorted_keys[:self.MAX_CACHE_SIZE // 10]:
                del self._cache[k]

        expiry = datetime.utcnow() + timedelta(seconds=self.CACHE_TTL_SECONDS)
        self._cache[key] = (value, expiry)

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> list[DiseaseResult]:
        \
\
\
\
\
\
\
\
\
\
\
        cache_key = self._cache_key("search", query, limit, offset)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()

        try:
            response = await client.get(
                "/search",
                params={
                    "q": query,
                    "category": "biolink:Disease",
                    "limit": limit,
                    "offset": offset,
                }
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):

                entity_id = item.get("id", "")
                if not entity_id.startswith("MONDO:"):

                    continue

                result = DiseaseResult(
                    id=entity_id,
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    synonyms=item.get("synonyms", []),
                )
                results.append(result)

            self._set_cached(cache_key, results)
            return results

        except httpx.HTTPError as e:
            logger.error(f"MONDO API error during search: {e}")
            return []

    async def get_disease(self, mondo_id: str) -> Optional[DiseaseResult]:
\
\
\
\
\
\
\
\
\

        if not mondo_id.startswith("MONDO:"):
            mondo_id = f"MONDO:{mondo_id}"

        cache_key = self._cache_key("get", mondo_id)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()

        try:
            response = await client.get(f"/entity/{mondo_id}")
            response.raise_for_status()
            data = response.json()

            result = DiseaseResult(
                id=mondo_id,
                name=data.get("name", ""),
                description=data.get("description", ""),
                synonyms=data.get("synonyms", []),
                xrefs=data.get("xrefs", []),
            )

            hierarchy = await self._get_hierarchy(mondo_id)
            result.parents = hierarchy.get("parents", [])
            result.children = hierarchy.get("children", [])

            self._set_cached(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"MONDO API error: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"MONDO API error: {e}")
            return None

    async def _get_hierarchy(self, mondo_id: str) -> dict:
        \
        client = await self._get_client()

        try:

            response = await client.get(
                f"/entity/{mondo_id}/hierarchicalDescendants",
                params={"limit": 20}
            )

            children = []
            if response.status_code == 200:
                data = response.json()
                children = [
                    item.get("id")
                    for item in data.get("items", [])
                    if item.get("id", "").startswith("MONDO:")
                ]

            response = await client.get(
                f"/entity/{mondo_id}/hierarchicalAncestors",
                params={"limit": 10}
            )

            parents = []
            if response.status_code == 200:
                data = response.json()
                parents = [
                    item.get("id")
                    for item in data.get("items", [])
                    if item.get("id", "").startswith("MONDO:")
                ]

            return {"parents": parents[:5], "children": children[:20]}

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get hierarchy for {mondo_id}: {e}")
            return {"parents": [], "children": []}

    async def validate_id(self, mondo_id: str) -> bool:
        \
        disease = await self.get_disease(mondo_id)
        return disease is not None

    async def is_subtype_of(self, disease_id: str, parent_id: str) -> bool:
        \
\
\
\
\
\
\
        if disease_id == parent_id:
            return True

        disease = await self.get_disease(disease_id)
        if not disease:
            return False

        if parent_id in disease.parents:
            return True

        for parent in disease.parents[:3]:
            if await self.is_subtype_of(parent, parent_id):
                return True

        return False

    def clear_cache(self):
        \
        self._cache.clear()

_mondo_client: Optional[MondoClient] = None

def get_mondo_client() -> MondoClient:
    \
    global _mondo_client
    if _mondo_client is None:
        _mondo_client = MondoClient()
    return _mondo_client

async def close_mondo_client():
    \
    global _mondo_client
    if _mondo_client:
        await _mondo_client.close()
        _mondo_client = None
