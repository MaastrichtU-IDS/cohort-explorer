import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

@dataclass
class InstitutionResult:
    id: str
    ror_id: str
    name: str
    country_code: str
    country_name: str
    types: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    city: str = ""
    region: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ror_id": self.ror_id,
            "name": self.name,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "types": self.types,
            "aliases": self.aliases,
            "links": self.links,
            "city": self.city,
            "region": self.region,
        }

class RORClient:
    BASE_URL = "https://api.ror.org"

    TYPES = {
        "education": "Education",
        "healthcare": "Healthcare",
        "company": "Company",
        "nonprofit": "Nonprofit",
        "government": "Government",
        "facility": "Facility",
        "funder": "Funder",
        "archive": "Archive",
        "other": "Other",
    }

    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 500

    def __init__(
        self,
        timeout: float = 10.0,
        cache_enabled: bool = True
    ):
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self._cache: dict[str, tuple[any, datetime]] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self.timeout,
                headers={"Accept": "application/json"}
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _cache_key(self, method: str, *args) -> str:
        return f"ror:{method}:{':'.join(str(a) for a in args)}"

    def _get_cached(self, key: str) -> Optional[any]:
        if not self.cache_enabled:
            return None
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: any):
        if not self.cache_enabled:
            return
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])
            for k in sorted_keys[:self.MAX_CACHE_SIZE // 10]:
                del self._cache[k]
        expiry = datetime.utcnow() + timedelta(seconds=self.CACHE_TTL_SECONDS)
        self._cache[key] = (value, expiry)

    def _parse_institution(self, data: dict) -> InstitutionResult:
        ror_id = data.get("id", "").split("/")[-1]

        locations = data.get("locations", [])
        city = ""
        region = ""
        country_code = ""
        country_name = ""

        if locations:
            loc = locations[0].get("geonames_details", {})
            city = loc.get("name", "")
            country_code = loc.get("country_code", "")
            country_name = loc.get("country_name", "")

            admin = locations[0].get("geonames_details", {})
            region = admin.get("admin1_name", "")

        return InstitutionResult(
            id=data.get("id", ""),
            ror_id=ror_id,
            name=data.get("name", ""),
            country_code=country_code,
            country_name=country_name,
            types=data.get("types", []),
            aliases=data.get("aliases", []),
            links=data.get("links", []),
            city=city,
            region=region,
        )

    async def search(
        self,
        query: str,
        country: Optional[str] = None,
        types: Optional[list[str]] = None,
        limit: int = 10
    ) -> list[InstitutionResult]:
        cache_key = self._cache_key("search", query, country, str(types), limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()

        try:

            params = {"query": query}

            filters = []
            if country:
                filters.append(f"locations.geonames_details.country_code:{country}")
            if types:
                type_filter = " OR ".join(f"types:{t}" for t in types)
                filters.append(f"({type_filter})")

            if filters:
                params["filter"] = ",".join(filters)

            response = await client.get("/organizations", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", [])[:limit]:
                results.append(self._parse_institution(item))

            self._set_cached(cache_key, results)
            return results

        except httpx.HTTPError as e:
            logger.error(f"ROR API error during search: {e}")
            return []

    async def get_institution(self, ror_id: str) -> Optional[InstitutionResult]:

        if ror_id.startswith("https://ror.org/"):
            ror_id = ror_id.split("/")[-1]

        cache_key = self._cache_key("get", ror_id)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()

        try:
            response = await client.get(f"/organizations/{ror_id}")
            response.raise_for_status()
            data = response.json()

            result = self._parse_institution(data)
            self._set_cached(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"ROR API error: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"ROR API error: {e}")
            return None

    async def validate_id(self, ror_id: str) -> bool:
        institution = await self.get_institution(ror_id)
        return institution is not None

    async def get_country(self, ror_id: str) -> Optional[str]:
        institution = await self.get_institution(ror_id)
        return institution.country_code if institution else None

    async def is_type(self, ror_id: str, inst_type: str) -> bool:
        institution = await self.get_institution(ror_id)
        if not institution:
            return False
        return inst_type in institution.types

    async def search_by_email_domain(self, email: str) -> list[InstitutionResult]:
        domain = email.split("@")[-1].lower()

        parts = domain.split(".")
        if len(parts) >= 2:

            search_term = parts[-2]

            if search_term.startswith("uni-"):
                search_term = search_term[4:]
        else:
            search_term = parts[0]

        return await self.search(search_term, limit=5)

    def clear_cache(self):
        self._cache.clear()

_ror_client: Optional[RORClient] = None

def get_ror_client() -> RORClient:
    global _ror_client
    if _ror_client is None:
        _ror_client = RORClient()
    return _ror_client

async def close_ror_client():
    global _ror_client
    if _ror_client:
        await _ror_client.close()
        _ror_client = None
