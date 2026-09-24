"""Tests the token-bucket rate limiter in isolation. Time is faked via a
monkeypatched time.monotonic() rather than real sleeps, so these tests run
in milliseconds and are deterministic regardless of machine speed."""
import pytest

from app.middleware.rate_limit import RateLimitExceeded, RateLimiter


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr("app.middleware.rate_limit.time.monotonic", clock)
    return clock


@pytest.mark.asyncio
async def test_allows_requests_up_to_capacity(fake_clock):
    limiter = RateLimiter(requests_per_minute=5)
    for _ in range(5):
        await limiter.check("key-a")  # should not raise


@pytest.mark.asyncio
async def test_rejects_once_capacity_is_exhausted(fake_clock):
    limiter = RateLimiter(requests_per_minute=3)
    for _ in range(3):
        await limiter.check("key-a")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("key-a")


@pytest.mark.asyncio
async def test_refills_over_time(fake_clock):
    limiter = RateLimiter(requests_per_minute=60)  # 1 token/sec refill rate
    for _ in range(60):
        await limiter.check("key-a")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("key-a")

    fake_clock.advance(1.0)  # should refill ~1 token
    await limiter.check("key-a")  # should not raise now


@pytest.mark.asyncio
async def test_keys_have_independent_buckets(fake_clock):
    limiter = RateLimiter(requests_per_minute=1)
    await limiter.check("key-a")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("key-a")
    await limiter.check("key-b")  # different key, untouched bucket


@pytest.mark.asyncio
async def test_retry_after_is_positive_when_exceeded(fake_clock):
    limiter = RateLimiter(requests_per_minute=1)
    await limiter.check("key-a")
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check("key-a")
    assert exc_info.value.retry_after_seconds > 0
