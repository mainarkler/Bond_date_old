from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    async def get_json(self, key: str) -> dict[str, Any] | None:
        ...

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        ...


@dataclass(slots=True)
class InMemoryRedisCompatibleCache:
    _store: dict[str, tuple[datetime, str]] = field(default_factory=dict)

    async def get_json(self, key: str) -> dict[str, Any] | None:
        record = self._store.get(key)
        if record is None:
            return None

        expires_at, raw = record
        if expires_at <= datetime.now(timezone.utc):
            self._store.pop(key, None)
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._store[key] = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            json.dumps(value, ensure_ascii=False),
        )


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Any | None = None

    async def _client_or_none(self) -> Any | None:
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as redis  # type: ignore

            self._client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            return self._client
        except Exception as exc:
            logger.warning("redis_unavailable", extra={"error": str(exc)})
            return None

    async def get_json(self, key: str) -> dict[str, Any] | None:
        client = await self._client_or_none()
        if client is None:
            return None
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        client = await self._client_or_none()
        if client is None:
            return
        await client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))


class HybridCache(CacheBackend):
    def __init__(self, redis_url: str | None = None) -> None:
        self.redis = RedisCache(redis_url) if redis_url else None
        self.memory = InMemoryRedisCompatibleCache()

    async def get_json(self, key: str) -> dict[str, Any] | None:
        if self.redis:
            cached = await self.redis.get_json(key)
            if cached is not None:
                return cached
        return await self.memory.get_json(key)

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if self.redis:
            await self.redis.set_json(key, value, ttl_seconds)
        await self.memory.set_json(key, value, ttl_seconds)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest()


def make_signal_cache_key(query: str) -> str:
    return f"signal:{query_hash(query)}"


def make_analysis_cache_key(query: str) -> str:
    return f"analysis:{query_hash(query)}"
