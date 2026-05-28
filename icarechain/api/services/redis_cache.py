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
\
\
\
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from api.config import get_settings

logger = logging.getLogger(__name__)

class RedisCache:
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
\

    CONSENT_PREFIX = "consent"
    ACCESS_PREFIX = "access"
    ATTESTATION_PREFIX = "attestation"
    COLLABORATION_PREFIX = "collaboration"
    TRANSACTION_PREFIX = "transaction"
    AUDIT_PREFIX = "audit"
    AUTH_TOKEN_PREFIX = "auth_token"

    CONSENT_CHANNEL = "consent:events"
    ACCESS_CHANNEL = "access:events"
    ATTESTATION_CHANNEL = "attestation:events"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        max_connections: int = 50,
        decode_responses: bool = True,
        retry_on_timeout: bool = True,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ):
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
\
        self.redis_url = redis_url
        self.db = db
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._connected = False
        self._synced = False
        self._last_sync: Optional[datetime] = None

        self._pool_config = {
            "max_connections": max_connections,
            "decode_responses": decode_responses,
            "retry_on_timeout": retry_on_timeout,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
        }

        self._event_handlers: dict[str, list[Callable[[dict], Awaitable[None]]]] = {
            self.CONSENT_CHANNEL: [],
            self.ACCESS_CHANNEL: [],
            self.ATTESTATION_CHANNEL: [],
        }

    async def connect(self) -> None:
        \
        if self._connected:
            return

        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                **self._pool_config
            )
            self.client = redis.Redis(connection_pool=self.pool)

            await self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        \
        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis")

    async def health_check(self) -> dict[str, Any]:
        \
        if not self._connected or not self.client:
            return {"healthy": False, "error": "Not connected"}

        try:
            info = await self.client.info("server", "memory", "stats")
            return {
                "healthy": True,
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    @asynccontextmanager
    async def pipeline(self, transaction: bool = True):
        \
\
\
\
\
\
\
\
\
        if not self.client:
            raise RuntimeError("Redis not connected")
        pipe = self.client.pipeline(transaction=transaction)
        try:
            yield pipe
            await pipe.execute()
        finally:
            await pipe.reset()

    def _serialize(self, data: dict) -> str:
        \
        def default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        return json.dumps(data, default=default)

    def _deserialize(self, data: str) -> dict:
        \
        if not data:
            return {}
        return json.loads(data)

    def _hash_to_dict(self, hash_data: dict[str, str]) -> dict[str, Any]:
        \
        result = {}
        for key, value in hash_data.items():

            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value
        return result

    def _dict_to_hash(self, data: dict[str, Any]) -> dict[str, str]:
        \
        result = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, bool):
                result[key] = json.dumps(value)
            elif value is None:
                result[key] = ""
            else:
                result[key] = str(value)
        return result

    async def get_consent(self, cohort_hash: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.CONSENT_PREFIX}:{cohort_hash}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        result = self._hash_to_dict(data)

        owners_key = f"{self.CONSENT_PREFIX}:{cohort_hash}:owners"
        owners = await self.client.smembers(owners_key)
        result["owners"] = list(owners) if owners else []

        return result

    async def set_consent(self, cohort_hash: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.CONSENT_PREFIX}:{cohort_hash}"
        owners_key = f"{self.CONSENT_PREFIX}:{cohort_hash}:owners"
        index_key = f"{self.CONSENT_PREFIX}:index"

        owners = data.pop("owners", [])

        data["cohort_hash"] = cohort_hash
        data["cached_at"] = datetime.utcnow().isoformat()

        async with self.pipeline() as pipe:

            pipe.hset(key, mapping=self._dict_to_hash(data))

            if owners:
                pipe.delete(owners_key)
                pipe.sadd(owners_key, *owners)

            pipe.sadd(index_key, cohort_hash)

        await self._publish_event(self.CONSENT_CHANNEL, {
            "type": "consent_updated",
            "cohort_hash": cohort_hash,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def delete_consent(self, cohort_hash: str) -> None:
        \
        if not self.client:
            return

        consent_key = f"{self.CONSENT_PREFIX}:{cohort_hash}"
        owners_key = f"{self.CONSENT_PREFIX}:{cohort_hash}:owners"
        index_key = f"{self.CONSENT_PREFIX}:index"

        requesters_key = f"{self.ACCESS_PREFIX}:{cohort_hash}:requesters"
        requesters = await self.client.smembers(requesters_key)

        async with self.pipeline() as pipe:
            pipe.delete(consent_key)
            pipe.delete(owners_key)
            pipe.srem(index_key, cohort_hash)

            for requester in (requesters or []):
                access_key = f"{self.ACCESS_PREFIX}:{cohort_hash}:{requester}"
                requester_index = f"{self.ACCESS_PREFIX}:requester:{requester}"
                pipe.delete(access_key)
                pipe.srem(requester_index, cohort_hash)
            pipe.delete(requesters_key)

    async def update_consent(self, cohort_hash: str, updates: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.CONSENT_PREFIX}:{cohort_hash}"

        if "owners" in updates:
            owners = updates.pop("owners")
            owners_key = f"{self.CONSENT_PREFIX}:{cohort_hash}:owners"
            await self.client.delete(owners_key)
            if owners:
                await self.client.sadd(owners_key, *owners)

        if updates:
            updates["cached_at"] = datetime.utcnow().isoformat()
            await self.client.hset(key, mapping=self._dict_to_hash(updates))

        await self._publish_event(self.CONSENT_CHANNEL, {
            "type": "consent_updated",
            "cohort_hash": cohort_hash,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def revoke_consent(self, cohort_hash: str) -> None:
        \
        await self.update_consent(cohort_hash, {
            "active": False,
            "revoked_at": datetime.utcnow().isoformat()
        })

        await self._publish_event(self.CONSENT_CHANNEL, {
            "type": "consent_revoked",
            "cohort_hash": cohort_hash,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def get_all_consents(self) -> list[dict]:
        \
        if not self.client:
            return []

        index_key = f"{self.CONSENT_PREFIX}:index"
        cohort_hashes = await self.client.smembers(index_key)

        consents = []
        for cohort_hash in (cohort_hashes or []):
            consent = await self.get_consent(cohort_hash)
            if consent:
                consents.append(consent)

        return consents

    async def get_access(self, cohort_hash: str, requester: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.ACCESS_PREFIX}:{cohort_hash}:{requester}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        return self._hash_to_dict(data)

    async def set_access(self, cohort_hash: str, requester: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.ACCESS_PREFIX}:{cohort_hash}:{requester}"
        requesters_key = f"{self.ACCESS_PREFIX}:{cohort_hash}:requesters"
        requester_index = f"{self.ACCESS_PREFIX}:requester:{requester}"

        data["cached_at"] = datetime.utcnow().isoformat()

        async with self.pipeline() as pipe:
            pipe.hset(key, mapping=self._dict_to_hash(data))
            pipe.sadd(requesters_key, requester)
            pipe.sadd(requester_index, cohort_hash)

        await self._publish_event(self.ACCESS_CHANNEL, {
            "type": "access_granted",
            "cohort_hash": cohort_hash,
            "requester": requester,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def revoke_access(self, cohort_hash: str, requester: str) -> None:
        \
        if not self.client:
            return

        key = f"{self.ACCESS_PREFIX}:{cohort_hash}:{requester}"

        await self.client.hset(key, mapping={
            "approved": "false",
            "revoked_at": datetime.utcnow().isoformat()
        })

        await self._publish_event(self.ACCESS_CHANNEL, {
            "type": "access_revoked",
            "cohort_hash": cohort_hash,
            "requester": requester,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def revoke_all_access(self, cohort_hash: str) -> int:
        \
        if not self.client:
            return 0

        requesters_key = f"{self.ACCESS_PREFIX}:{cohort_hash}:requesters"
        requesters = await self.client.smembers(requesters_key)

        if not requesters:
            return 0

        count = 0
        for requester in requesters:
            access = await self.get_access(cohort_hash, requester)
            if access and access.get("approved"):
                await self.revoke_access(cohort_hash, requester)
                count += 1

        return count

    async def has_access(self, cohort_hash: str, requester: str) -> bool:
        \
        access = await self.get_access(cohort_hash, requester)
        if not access:
            return False

        consent = await self.get_consent(cohort_hash)
        if not consent or not consent.get("active"):
            return False

        return access.get("approved", False)

    async def get_cohort_access_grants(self, cohort_hash: str) -> list[dict]:
        \
        if not self.client:
            return []

        requesters_key = f"{self.ACCESS_PREFIX}:{cohort_hash}:requesters"
        requesters = await self.client.smembers(requesters_key)

        grants = []
        for requester in (requesters or []):
            access = await self.get_access(cohort_hash, requester)
            if access:
                grants.append({"requester": requester, **access})

        return grants

    async def get_requester_access_grants(self, requester: str) -> list[dict]:
        \
        if not self.client:
            return []

        requester_index = f"{self.ACCESS_PREFIX}:requester:{requester}"
        cohort_hashes = await self.client.smembers(requester_index)

        grants = []
        for cohort_hash in (cohort_hashes or []):
            access = await self.get_access(cohort_hash, requester)
            if access and access.get("approved"):
                consent = await self.get_consent(cohort_hash)
                grants.append({
                    "cohort_hash": cohort_hash,
                    "cohort_id": consent.get("cohort_id") if consent else None,
                    **access
                })

        return grants

    async def get_attestation(self, subject: str, att_type: str, scope: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.ATTESTATION_PREFIX}:{subject}:{att_type}:{scope}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        return self._hash_to_dict(data)

    async def set_attestation(self, subject: str, att_type: str, scope: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.ATTESTATION_PREFIX}:{subject}:{att_type}:{scope}"
        index_key = f"{self.ATTESTATION_PREFIX}:{subject}:all"

        data["cached_at"] = datetime.utcnow().isoformat()

        async with self.pipeline() as pipe:
            pipe.hset(key, mapping=self._dict_to_hash(data))
            pipe.sadd(index_key, f"{att_type}:{scope}")

        await self._publish_event(self.ATTESTATION_CHANNEL, {
            "type": "attestation_submitted",
            "subject": subject,
            "attestation_type": att_type,
            "scope": scope,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def has_valid_attestation(self, subject: str, att_type: str, scope: str) -> bool:
        \
        att = await self.get_attestation(subject, att_type, scope)
        if not att:
            return False

        if att.get("revoked"):
            return False

        valid_until = att.get("valid_until")
        if valid_until:
            try:
                if isinstance(valid_until, str):

                    from datetime import datetime as dt
                    valid_until = dt.fromisoformat(valid_until).timestamp()
                if valid_until < datetime.utcnow().timestamp():
                    return False
            except (ValueError, TypeError):
                pass

        return True

    async def get_subject_attestations(self, subject: str) -> list[dict]:
        \
        if not self.client:
            return []

        index_key = f"{self.ATTESTATION_PREFIX}:{subject}:all"
        att_refs = await self.client.smembers(index_key)

        attestations = []
        for ref in (att_refs or []):
            parts = ref.split(":", 1)
            if len(parts) == 2:
                att_type, scope = parts
                att = await self.get_attestation(subject, att_type, scope)
                if att:
                    attestations.append({
                        "type": att_type,
                        "scope": scope,
                        **att
                    })

        return attestations

    async def get_collaboration(self, cohort_hash: str, requester: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.COLLABORATION_PREFIX}:{cohort_hash}:{requester}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        return self._hash_to_dict(data)

    async def set_collaboration(self, cohort_hash: str, requester: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.COLLABORATION_PREFIX}:{cohort_hash}:{requester}"
        index_key = f"{self.COLLABORATION_PREFIX}:{cohort_hash}:all"

        data["cached_at"] = datetime.utcnow().isoformat()

        async with self.pipeline() as pipe:
            pipe.hset(key, mapping=self._dict_to_hash(data))
            pipe.sadd(index_key, requester)

    async def has_collaboration(self, cohort_hash: str, requester: str) -> bool:
        \
        coll = await self.get_collaboration(cohort_hash, requester)
        if not coll:
            return False

        valid_until = coll.get("valid_until")
        if valid_until:
            try:
                if isinstance(valid_until, (int, float)):
                    if valid_until < datetime.utcnow().timestamp():
                        return False
            except (ValueError, TypeError):
                pass

        return coll.get("active", True)

    async def get_cohort_collaborations(self, cohort_hash: str) -> list[dict]:
        \
        if not self.client:
            return []

        index_key = f"{self.COLLABORATION_PREFIX}:{cohort_hash}:all"
        requesters = await self.client.smembers(index_key)

        collaborations = []
        for requester in (requesters or []):
            coll = await self.get_collaboration(cohort_hash, requester)
            if coll:
                collaborations.append({"requester": requester, **coll})

        return collaborations

    async def store_transaction(self, tx_hash: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.TRANSACTION_PREFIX}:{tx_hash}"
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())

        await self.client.hset(key, mapping=self._dict_to_hash(data))

        tx_type = data.get("type", "unknown")
        cohort_id = data.get("cohort_id")

        score = datetime.fromisoformat(timestamp).timestamp() if isinstance(timestamp, str) else timestamp
        await self.client.zadd(f"{self.TRANSACTION_PREFIX}:index:all", {tx_hash: score})
        await self.client.zadd(f"{self.TRANSACTION_PREFIX}:index:by_type:{tx_type}", {tx_hash: score})

        if cohort_id:
            await self.client.sadd(f"{self.TRANSACTION_PREFIX}:index:by_cohort:{cohort_id}", tx_hash)

    async def get_transaction(self, tx_hash: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.TRANSACTION_PREFIX}:{tx_hash}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        return self._hash_to_dict(data)

    async def get_transactions(
        self,
        tx_type: Optional[str] = None,
        cohort_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[dict]:
        \
        if not self.client:
            return []

        if cohort_id:

            tx_hashes = await self.client.smembers(
                f"{self.TRANSACTION_PREFIX}:index:by_cohort:{cohort_id}"
            )
            tx_hashes = list(tx_hashes)[offset:offset + limit]
        elif tx_type:

            tx_hashes = await self.client.zrevrange(
                f"{self.TRANSACTION_PREFIX}:index:by_type:{tx_type}",
                offset,
                offset + limit - 1
            )
        else:

            tx_hashes = await self.client.zrevrange(
                f"{self.TRANSACTION_PREFIX}:index:all",
                offset,
                offset + limit - 1
            )

        transactions = []
        for tx_hash in tx_hashes:
            tx = await self.get_transaction(tx_hash)
            if tx:
                tx["tx_hash"] = tx_hash
                transactions.append(tx)

        return transactions

    async def get_transaction_count(self, tx_type: Optional[str] = None) -> int:
        \
        if not self.client:
            return 0

        if tx_type:
            return await self.client.zcard(f"{self.TRANSACTION_PREFIX}:index:by_type:{tx_type}")
        return await self.client.zcard(f"{self.TRANSACTION_PREFIX}:index:all")

    async def add_audit_entry(self, entry_id: str, data: dict) -> None:
        \
        if not self.client:
            return

        key = f"{self.AUDIT_PREFIX}:{entry_id}"
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())

        await self.client.hset(key, mapping=self._dict_to_hash(data))

        score = datetime.fromisoformat(timestamp).timestamp() if isinstance(timestamp, str) else timestamp
        await self.client.zadd(f"{self.AUDIT_PREFIX}:index:all", {entry_id: score})

        action = data.get("action")
        admin_email = data.get("admin_email")

        if action:
            await self.client.zadd(f"{self.AUDIT_PREFIX}:index:by_action:{action}", {entry_id: score})
        if admin_email:
            await self.client.zadd(f"{self.AUDIT_PREFIX}:index:by_admin:{admin_email}", {entry_id: score})

    async def get_audit_entries(
        self,
        action: Optional[str] = None,
        admin_email: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[dict]:
        \
        if not self.client:
            return []

        if action:
            entry_ids = await self.client.zrevrange(
                f"{self.AUDIT_PREFIX}:index:by_action:{action}",
                offset,
                offset + limit - 1
            )
        elif admin_email:
            entry_ids = await self.client.zrevrange(
                f"{self.AUDIT_PREFIX}:index:by_admin:{admin_email}",
                offset,
                offset + limit - 1
            )
        else:
            entry_ids = await self.client.zrevrange(
                f"{self.AUDIT_PREFIX}:index:all",
                offset,
                offset + limit - 1
            )

        entries = []
        for entry_id in entry_ids:
            key = f"{self.AUDIT_PREFIX}:{entry_id}"
            data = await self.client.hgetall(key)
            if data:
                entry = self._hash_to_dict(data)
                entry["id"] = entry_id
                entries.append(entry)

        return entries

    async def get_audit_count(self) -> int:
        \
        if not self.client:
            return 0
        return await self.client.zcard(f"{self.AUDIT_PREFIX}:index:all")

    async def set_authorization_token(self, token: str, data: dict, ttl: int = 3600) -> None:
        \
        if not self.client:
            return

        key = f"{self.AUTH_TOKEN_PREFIX}:{token}"
        await self.client.hset(key, mapping=self._dict_to_hash(data))
        await self.client.expire(key, ttl)

    async def get_authorization_token(self, token: str) -> Optional[dict]:
        \
        if not self.client:
            return None

        key = f"{self.AUTH_TOKEN_PREFIX}:{token}"
        data = await self.client.hgetall(key)

        if not data:
            return None

        return self._hash_to_dict(data)

    async def invalidate_authorization_token(self, token: str) -> None:
        \
        if not self.client:
            return

        key = f"{self.AUTH_TOKEN_PREFIX}:{token}"
        await self.client.delete(key)

    async def _publish_event(self, channel: str, event: dict) -> None:
        \
        if not self.client:
            return
        try:
            await self.client.publish(channel, self._serialize(event))
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")

    def on_event(self, channel: str, handler: Callable[[dict], Awaitable[None]]) -> None:
        \
        if channel in self._event_handlers:
            self._event_handlers[channel].append(handler)

    async def start_event_listener(self) -> None:
        \
        if not self.client:
            return

        self.pubsub = self.client.pubsub()
        await self.pubsub.subscribe(
            self.CONSENT_CHANNEL,
            self.ACCESS_CHANNEL,
            self.ATTESTATION_CHANNEL
        )

        logger.info("Started Redis event listener")

        async for message in self.pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"]
                try:
                    data = self._deserialize(message["data"])
                    handlers = self._event_handlers.get(channel, [])
                    for handler in handlers:
                        await handler(data)
                except Exception as e:
                    logger.error(f"Error handling event: {e}")

    async def mark_synced(self) -> None:
        \
        self._synced = True
        self._last_sync = datetime.utcnow()

        if self.client:
            await self.client.set("_meta:last_sync", self._last_sync.isoformat())
            await self.client.set("_meta:synced", "true")

    async def clear(self) -> None:
        \
        if not self.client:
            return

        async for key in self.client.scan_iter(f"{self.CONSENT_PREFIX}:*"):
            await self.client.delete(key)
        async for key in self.client.scan_iter(f"{self.ACCESS_PREFIX}:*"):
            await self.client.delete(key)
        async for key in self.client.scan_iter(f"{self.ATTESTATION_PREFIX}:*"):
            await self.client.delete(key)
        async for key in self.client.scan_iter(f"{self.COLLABORATION_PREFIX}:*"):
            await self.client.delete(key)

        self._synced = False
        self._last_sync = None

    def get_stats(self) -> dict:
        \
        return {
            "backend": "redis",
            "connected": self._connected,
            "synced": self._synced,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "redis_url": self.redis_url,
        }

    @property
    def is_synced(self) -> bool:
        return self._synced

    @property
    def consent_count(self) -> int:
        \
        return 0

    async def get_consent_count(self) -> int:
        \
        if not self.client:
            return 0
        return await self.client.scard(f"{self.CONSENT_PREFIX}:index")

_redis_cache: Optional[RedisCache] = None

async def get_redis_cache() -> RedisCache:
    \
    global _redis_cache

    if _redis_cache is None:
        settings = get_settings()
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')

        _redis_cache = RedisCache(redis_url=redis_url)
        await _redis_cache.connect()

    return _redis_cache

async def close_redis_cache() -> None:
    \
    global _redis_cache

    if _redis_cache is not None:
        await _redis_cache.disconnect()
        _redis_cache = None
