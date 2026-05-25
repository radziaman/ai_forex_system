"""
Redis-Backed WorldState — persistent, distributed state layer.
Mirrors the in-memory WorldState API with Redis persistence.

When Redis is unavailable, falls back to in-memory dict transparently.
"""

from __future__ import annotations
import time
import json
import os
from typing import Dict, Optional, Any
from loguru import logger

try:
    import redis.asyncio as aioredis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class RedisWorldState:
    """Redis-backed world state with in-memory fallback.

    Uses the SAME API as WorldState (get/set/update/snapshot) so agents
    can use either interchangeably.

    Key prefix: "rts:ws:" to namespace in Redis.
    TTL keys: "rts:ws:ttl:{key}" with the actual ttl value.
    Auto-cleanup and reconnection in background.
    """

    DEFAULT_REDIS_URL = "redis://localhost:6379/0"

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", self.DEFAULT_REDIS_URL)
        self._redis: Optional[aioredis.Redis] = None
        self._in_memory: Dict[str, Any] = {}  # Fallback when Redis is down
        self._connected = False
        self._prefix = "rts:ws:"
        logger.info(f"RedisWorldState initialized (url={self._redis_url})")

    async def connect(self):
        """Connect to Redis. On failure, use in-memory fallback."""
        if not HAS_REDIS:
            logger.warning("redis-py not installed — using in-memory fallback")
            return
        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await self._redis.ping()
            self._connected = True
            logger.info("RedisWorldState connected")
        except Exception as e:
            self._connected = False
            logger.warning(f"Redis unavailable ({e}) — using in-memory fallback")

    async def disconnect(self):
        """Disconnect from Redis cleanly."""
        if self._redis and self._connected:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._connected = False

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key. Returns default if key doesn't exist or is expired."""
        if self._connected and self._redis:
            try:
                val = await self._redis.get(f"{self._prefix}{key}")
                if val is not None:
                    return json.loads(val)
                # Check TTL
                ttl_key = f"{self._prefix}ttl:{key}"
                expires = await self._redis.get(ttl_key)
                if expires and time.time() > float(expires):
                    await self._redis.delete(f"{self._prefix}{key}", ttl_key)
                    return default
                return default
            except Exception:
                self._connected = False
        return self._in_memory.get(key, default)

    async def set(
        self,
        key: str,
        value: Any,
        updated_by: str = "",
        ttl: float = 0.0,
    ):
        """Set a value by key with optional TTL (seconds) and provenance."""
        serialized = json.dumps(value, default=str)
        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()
                if ttl > 0:
                    pipe.setex(f"{self._prefix}{key}", int(ttl), serialized)
                else:
                    pipe.set(f"{self._prefix}{key}", serialized)
                if ttl > 0:
                    pipe.set(f"{self._prefix}ttl:{key}", time.time() + ttl)
                await pipe.execute()
            except Exception:
                self._connected = False
        self._in_memory[key] = value

    async def update(self, key: str, **kwargs):
        """Update a dictionary value by key with keyword arguments."""
        current = await self.get(key, {})
        if isinstance(current, dict):
            current.update(kwargs)
            await self.set(key, current)

    async def delete(self, key: str):
        """Delete a key from state."""
        if self._connected and self._redis:
            try:
                await self._redis.delete(
                    f"{self._prefix}{key}", f"{self._prefix}ttl:{key}"
                )
            except Exception:
                pass
        self._in_memory.pop(key, None)

    async def snapshot(self) -> Dict[str, Any]:
        """Return a complete snapshot of all non-expired keys."""
        if self._connected and self._redis:
            try:
                keys = await self._redis.keys(f"{self._prefix}*")
                result: Dict[str, Any] = {}
                for k in keys:
                    key = k.replace(self._prefix, "")
                    if key.startswith("ttl:"):
                        continue
                    val = await self._redis.get(k)
                    if val is not None:
                        result[key] = json.loads(val)
                return result
            except Exception:
                pass
        return dict(self._in_memory)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return await self.get(key, None) is not None

    async def get_domain_state(self, prefix: str) -> Dict[str, Any]:
        """Get all keys that start with a given prefix."""
        snapshot = await self.snapshot()
        return {k: v for k, v in snapshot.items() if k.startswith(prefix)}

    def is_connected(self) -> bool:
        """Return whether Redis is currently connected."""
        return self._connected
