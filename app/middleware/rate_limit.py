"""
Per-API-key rate limiting via a token bucket, kept in-process (a plain dict
guarded by an asyncio lock). This is intentionally not Redis-backed: for a
single-instance deployment (which is what the docker-compose.yml in this
project ships) an in-memory bucket is simpler and has one less moving part.
If you scale this to multiple app instances behind a load balancer, swap
`_BUCKETS` for a Redis-backed implementation (the RateLimiter interface
below stays the same) -- don't add that complexity before you need it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded, retry after {retry_after_seconds:.1f}s")


class RateLimiter:
    """Token-bucket limiter keyed by API key. Capacity == requests-per-minute,
    refilled continuously (capacity / 60 tokens per second) rather than in a
    fixed per-minute window, so it doesn't allow a burst of 2x the limit at
    a minute boundary."""

    def __init__(self, requests_per_minute: int):
        self._capacity = float(requests_per_minute)
        self._refill_rate = self._capacity / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, api_key: str, cost: float = 1.0) -> None:
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(api_key)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, last_refill=now)
                self._buckets[api_key] = bucket

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            bucket.last_refill = now

            if bucket.tokens < cost:
                deficit = cost - bucket.tokens
                retry_after = deficit / self._refill_rate
                raise RateLimitExceeded(retry_after)

            bucket.tokens -= cost


_limiter_singleton: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter_singleton
    if _limiter_singleton is None:
        _limiter_singleton = RateLimiter(get_settings().default_rate_limit_per_minute)
    return _limiter_singleton
