import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

class CacheBackend(Protocol):
    async def get_consent(self, cohort_hash: str) -> Optional[dict]: ...
    async def set_consent(self, cohort_hash: str, data: dict) -> None: ...
    async def delete_consent(self, cohort_hash: str) -> None: ...
    async def update_consent(self, cohort_hash: str, updates: dict) -> None: ...
    async def revoke_consent(self, cohort_hash: str) -> None: ...
    async def get_all_consents(self) -> list[dict]: ...

    async def get_access(self, cohort_hash: str, requester: str) -> Optional[dict]: ...
    async def set_access(self, cohort_hash: str, requester: str, data: dict) -> None: ...
    async def revoke_access(self, cohort_hash: str, requester: str) -> None: ...
    async def revoke_all_access(self, cohort_hash: str) -> int: ...
    async def has_access(self, cohort_hash: str, requester: str) -> bool: ...
    async def get_cohort_access_grants(self, cohort_hash: str) -> list[dict]: ...

    async def get_attestation(self, subject: str, att_type: str, scope: str) -> Optional[dict]: ...
    async def set_attestation(self, subject: str, att_type: str, scope: str, data: dict) -> None: ...
    async def has_valid_attestation(self, subject: str, att_type: str, scope: str) -> bool: ...
    async def get_subject_attestations(self, subject: str) -> list[dict]: ...

    async def get_collaboration(self, cohort_hash: str, requester: str) -> Optional[dict]: ...
    async def set_collaboration(self, cohort_hash: str, requester: str, data: dict) -> None: ...
    async def has_collaboration(self, cohort_hash: str, requester: str) -> bool: ...

    async def mark_synced(self) -> None: ...
    async def clear(self) -> None: ...
    def get_stats(self) -> dict: ...

    @property
    def is_synced(self) -> bool: ...

class InMemoryCache:
    def __init__(self):
        self.consents: dict[str, dict] = {}
        self.access_grants: dict[str, dict[str, dict]] = {}
        self.attestations: dict[str, dict[str, dict[str, dict]]] = {}
        self.collaborations: dict[str, dict[str, dict]] = {}
        self._lock = asyncio.Lock()
        self._synced = False
        self._last_sync: Optional[datetime] = None

    @property
    def is_synced(self) -> bool:
        return self._synced

    @property
    def consent_count(self) -> int:
        return len(self.consents)

    async def get_consent(self, cohort_hash: str) -> Optional[dict]:
        return self.consents.get(cohort_hash)

    async def set_consent(self, cohort_hash: str, data: dict) -> None:
        async with self._lock:
            self.consents[cohort_hash] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }

    async def delete_consent(self, cohort_hash: str) -> None:
        async with self._lock:
            if cohort_hash in self.consents:
                del self.consents[cohort_hash]
            if cohort_hash in self.access_grants:
                del self.access_grants[cohort_hash]

    async def update_consent(self, cohort_hash: str, updates: dict) -> None:
        async with self._lock:
            if cohort_hash in self.consents:
                self.consents[cohort_hash].update(updates)
                self.consents[cohort_hash]["cached_at"] = datetime.utcnow().isoformat()

    async def revoke_consent(self, cohort_hash: str) -> None:
        async with self._lock:
            if cohort_hash in self.consents:
                self.consents[cohort_hash]["active"] = False
                self.consents[cohort_hash]["cached_at"] = datetime.utcnow().isoformat()

    async def get_all_consents(self) -> list[dict]:
        return list(self.consents.values())

    async def get_access(self, cohort_hash: str, requester: str) -> Optional[dict]:
        cohort_grants = self.access_grants.get(cohort_hash, {})
        return cohort_grants.get(requester)

    async def set_access(self, cohort_hash: str, requester: str, data: dict) -> None:
        async with self._lock:
            if cohort_hash not in self.access_grants:
                self.access_grants[cohort_hash] = {}
            self.access_grants[cohort_hash][requester] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }

    async def revoke_access(self, cohort_hash: str, requester: str) -> None:
        async with self._lock:
            if cohort_hash in self.access_grants:
                if requester in self.access_grants[cohort_hash]:
                    self.access_grants[cohort_hash][requester]["approved"] = False
                    self.access_grants[cohort_hash][requester]["revoked_at"] = datetime.utcnow().isoformat()

    async def revoke_all_access(self, cohort_hash: str) -> int:
        async with self._lock:
            if cohort_hash not in self.access_grants:
                return 0

            count = 0
            for requester in self.access_grants[cohort_hash]:
                if self.access_grants[cohort_hash][requester].get("approved"):
                    self.access_grants[cohort_hash][requester]["approved"] = False
                    self.access_grants[cohort_hash][requester]["revoked_at"] = datetime.utcnow().isoformat()
                    count += 1

            return count

    async def has_access(self, cohort_hash: str, requester: str) -> bool:
        grant = await self.get_access(cohort_hash, requester)
        if not grant:
            return False

        consent = await self.get_consent(cohort_hash)
        if not consent or not consent.get("active"):
            return False

        return grant.get("approved", False)

    async def get_cohort_access_grants(self, cohort_hash: str) -> list[dict]:
        grants = self.access_grants.get(cohort_hash, {})
        return [
            {"requester": req, **data}
            for req, data in grants.items()
        ]

    async def get_attestation(self, subject: str, att_type: str, scope: str) -> Optional[dict]:
        subject_atts = self.attestations.get(subject, {})
        type_atts = subject_atts.get(att_type, {})
        return type_atts.get(scope)

    async def set_attestation(self, subject: str, att_type: str, scope: str, data: dict) -> None:
        async with self._lock:
            if subject not in self.attestations:
                self.attestations[subject] = {}
            if att_type not in self.attestations[subject]:
                self.attestations[subject][att_type] = {}

            self.attestations[subject][att_type][scope] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }

    async def has_valid_attestation(self, subject: str, att_type: str, scope: str) -> bool:
        att = await self.get_attestation(subject, att_type, scope)
        if not att:
            return False

        if att.get("revoked"):
            return False

        valid_until = att.get("valid_until")
        if valid_until and valid_until < datetime.utcnow().timestamp():
            return False

        return True

    async def get_subject_attestations(self, subject: str) -> list[dict]:
        out = []
        for att_type, scopes in self.attestations.get(subject, {}).items():
            for scope, data in scopes.items():
                out.append({"type": att_type, "scope": scope, **data})
        return out

    async def get_collaboration(self, cohort_hash: str, requester: str) -> Optional[dict]:
        cohort_colls = self.collaborations.get(cohort_hash, {})
        return cohort_colls.get(requester)

    async def set_collaboration(self, cohort_hash: str, requester: str, data: dict) -> None:
        async with self._lock:
            if cohort_hash not in self.collaborations:
                self.collaborations[cohort_hash] = {}
            self.collaborations[cohort_hash][requester] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }

    async def has_collaboration(self, cohort_hash: str, requester: str) -> bool:
        coll = await self.get_collaboration(cohort_hash, requester)
        if not coll:
            return False

        valid_until = coll.get("valid_until")
        if valid_until and valid_until < datetime.utcnow().timestamp():
            return False

        return coll.get("active", True)

    def __init_transactions(self):
        if not hasattr(self, 'transactions'):
            self.transactions: dict[str, dict] = {}
            self.transaction_index: list[str] = []

    async def store_transaction(self, tx_hash: str, data: dict) -> None:
        self.__init_transactions()
        async with self._lock:
            self.transactions[tx_hash] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }
            if tx_hash not in self.transaction_index:
                self.transaction_index.insert(0, tx_hash)

    async def get_transaction(self, tx_hash: str) -> Optional[dict]:
        self.__init_transactions()
        return self.transactions.get(tx_hash)

    async def get_transactions(
        self,
        tx_type: Optional[str] = None,
        cohort_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[dict]:
        self.__init_transactions()
        result = []

        for tx_hash in self.transaction_index:
            tx = self.transactions.get(tx_hash)
            if not tx:
                continue

            if tx_type and tx.get("type") != tx_type:
                continue
            if cohort_id and tx.get("cohort_id") != cohort_id:
                continue

            result.append({"tx_hash": tx_hash, **tx})

        return result[offset:offset + limit]

    async def get_transaction_count(self, tx_type: Optional[str] = None) -> int:
        self.__init_transactions()
        if not tx_type:
            return len(self.transactions)

        return sum(1 for tx in self.transactions.values() if tx.get("type") == tx_type)

    def __init_audit(self):
        if not hasattr(self, 'audit_entries'):
            self.audit_entries: dict[str, dict] = {}
            self.audit_index: list[str] = []

    async def add_audit_entry(self, entry_id: str, data: dict) -> None:
        self.__init_audit()
        async with self._lock:
            self.audit_entries[entry_id] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }
            if entry_id not in self.audit_index:
                self.audit_index.insert(0, entry_id)

    async def get_audit_entries(
        self,
        action: Optional[str] = None,
        admin_email: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[dict]:
        self.__init_audit()
        result = []

        for entry_id in self.audit_index:
            entry = self.audit_entries.get(entry_id)
            if not entry:
                continue

            if action and entry.get("action") != action:
                continue
            if admin_email and entry.get("admin_email") != admin_email:
                continue

            result.append({"id": entry_id, **entry})

        return result[offset:offset + limit]

    async def get_audit_count(self) -> int:
        self.__init_audit()
        return len(self.audit_entries)

    def __init_tokens(self):
        if not hasattr(self, 'auth_tokens'):
            self.auth_tokens: dict[str, dict] = {}

    async def set_authorization_token(self, token: str, data: dict, ttl: int = 3600) -> None:
        self.__init_tokens()
        async with self._lock:
            self.auth_tokens[token] = {
                **data,
                "cached_at": datetime.utcnow().isoformat()
            }

    async def get_authorization_token(self, token: str) -> Optional[dict]:
        self.__init_tokens()
        return self.auth_tokens.get(token)

    async def invalidate_authorization_token(self, token: str) -> None:
        self.__init_tokens()
        async with self._lock:
            if token in self.auth_tokens:
                del self.auth_tokens[token]

    async def mark_synced(self) -> None:
        self._synced = True
        self._last_sync = datetime.utcnow()

    async def clear(self) -> None:
        async with self._lock:
            self.consents.clear()
            self.access_grants.clear()
            self.attestations.clear()
            self.collaborations.clear()
            self._synced = False
            self._last_sync = None

    def get_stats(self) -> dict:
        total_access = sum(len(grants) for grants in self.access_grants.values())
        total_attestations = sum(
            sum(len(scopes) for scopes in types.values())
            for types in self.attestations.values()
        )

        return {
            "backend": "in_memory",
            "synced": self._synced,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "consents": len(self.consents),
            "access_grants": total_access,
            "attestations": total_attestations,
            "collaborations": sum(len(c) for c in self.collaborations.values())
        }

_cache: Optional[CacheBackend] = None
_cache_type: str = "auto"

def configure_cache(cache_type: str = "auto") -> None:
    global _cache_type
    _cache_type = cache_type

async def _create_cache() -> CacheBackend:
    global _cache_type

    if _cache_type == "auto":

        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            _cache_type = "redis"
        else:
            _cache_type = "memory"

    if _cache_type == "redis":
        from api.services.redis_cache import RedisCache
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        cache = RedisCache(redis_url=redis_url)
        await cache.connect()
        logger.info(f"Using Redis cache at {redis_url}")
        return cache
    else:
        logger.info("Using in-memory cache")
        return InMemoryCache()

def get_cache() -> CacheBackend:
    global _cache

    if _cache is None:

        _cache = InMemoryCache()
        logger.info("Using in-memory cache (sync access)")

    return _cache

async def get_cache_async() -> CacheBackend:
    global _cache

    if _cache is None:
        _cache = await _create_cache()

    return _cache

async def close_cache() -> None:
    global _cache

    if _cache is not None:
        if hasattr(_cache, 'disconnect'):
            await _cache.disconnect()
        _cache = None
