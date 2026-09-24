"""Tests the fallback/retry orchestration in app/providers/router_client.py
against fake providers -- no real network calls, no real API keys. Fakes
implement the same BaseProvider interface so the orchestrator can't tell
the difference."""
import pytest

from app.models import ChatMessage
from app.providers import router_client
from app.providers.base import BaseProvider, CompletionResult, ProviderError
from app.providers.router_client import complete_with_fallback
from app.router.tiers import ModelSpec, Tier


class _FakeProvider(BaseProvider):
    """Scripted provider: raises the queued exceptions in order, then
    returns `result` on the call after the queue is empty."""

    def __init__(self, name: str, configured: bool = True, raises: list[Exception] | None = None, result: CompletionResult | None = None):
        self.name = name
        self._configured = configured
        self._raises = list(raises or [])
        self._result = result or CompletionResult(content=f"response from {name}", prompt_tokens=10, completion_tokens=5)
        self.call_count = 0

    def is_configured(self) -> bool:
        return self._configured

    async def complete(self, model, messages, temperature=0.7, max_tokens=None) -> CompletionResult:
        self.call_count += 1
        if self._raises:
            raise self._raises.pop(0)
        return self._result


@pytest.fixture
def messages():
    return [ChatMessage(role="user", content="test prompt")]


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # The real backoff sleeps between retries -- irrelevant to what these
    # tests check and would slow the suite down for no reason.
    monkeypatch.setattr(router_client, "_RETRY_BACKOFF_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _pinned_registry(monkeypatch):
    # These tests care about escalation *mechanics* (retry, skip-if-
    # unconfigured, escalate-on-exhaustion), not about which real provider
    # tiers.py currently points MID/STRONG at -- that mapping changes
    # depending on which API keys are available (see tiers.py's own note
    # about testing against a single OpenAI key). Pinning CHEAP->openai,
    # MID->anthropic, STRONG->anthropic here keeps these tests correct
    # regardless of how tiers.py is configured for real traffic.
    test_registry = {
        Tier.CHEAP: ModelSpec(tier=Tier.CHEAP, provider="openai", model="test-cheap", cost_per_1k_input=0.0001, cost_per_1k_output=0.0001),
        Tier.MID: ModelSpec(tier=Tier.MID, provider="anthropic", model="test-mid", cost_per_1k_input=0.001, cost_per_1k_output=0.001),
        Tier.STRONG: ModelSpec(tier=Tier.STRONG, provider="anthropic", model="test-strong", cost_per_1k_input=0.01, cost_per_1k_output=0.01),
    }
    monkeypatch.setattr(router_client, "MODEL_REGISTRY", test_registry)


@pytest.mark.asyncio
async def test_succeeds_on_first_try(monkeypatch, messages):
    fake_openai = _FakeProvider("openai")
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)

    result = await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)

    assert result.tier_served == Tier.CHEAP
    assert result.provider == "openai"
    assert result.escalated_from_failure is False
    assert fake_openai.call_count == 1


@pytest.mark.asyncio
async def test_retries_retryable_error_then_succeeds(monkeypatch, messages):
    fake_openai = _FakeProvider("openai", raises=[ProviderError("openai", "timeout", retryable=True)])
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)

    result = await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)

    assert result.tier_served == Tier.CHEAP
    assert result.escalated_from_failure is False
    assert fake_openai.call_count == 2  # one failure + one success


@pytest.mark.asyncio
async def test_escalates_to_next_tier_after_exhausting_retries(monkeypatch, messages):
    fake_openai = _FakeProvider(
        "openai",
        raises=[
            ProviderError("openai", "timeout", retryable=True),
            ProviderError("openai", "timeout", retryable=True),
            ProviderError("openai", "timeout", retryable=True),
        ],
    )
    fake_anthropic = _FakeProvider("anthropic")
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)
    monkeypatch.setitem(router_client._PROVIDERS, "anthropic", fake_anthropic)

    result = await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)

    assert result.tier_served == Tier.MID  # MODEL_REGISTRY[MID].provider == "anthropic"
    assert result.escalated_from_failure is True


@pytest.mark.asyncio
async def test_non_retryable_error_escalates_immediately_without_retrying(monkeypatch, messages):
    fake_openai = _FakeProvider("openai", raises=[ProviderError("openai", "bad api key", retryable=False)])
    fake_anthropic = _FakeProvider("anthropic")
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)
    monkeypatch.setitem(router_client._PROVIDERS, "anthropic", fake_anthropic)

    result = await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)

    assert result.tier_served == Tier.MID
    assert fake_openai.call_count == 1  # no retries burned on a non-retryable error


@pytest.mark.asyncio
async def test_unconfigured_provider_is_skipped(monkeypatch, messages):
    fake_openai = _FakeProvider("openai", configured=False)
    fake_anthropic = _FakeProvider("anthropic")
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)
    monkeypatch.setitem(router_client._PROVIDERS, "anthropic", fake_anthropic)

    result = await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)

    assert result.tier_served == Tier.MID
    assert fake_openai.call_count == 0


@pytest.mark.asyncio
async def test_all_tiers_exhausted_raises_runtime_error(monkeypatch, messages):
    # MODEL_REGISTRY routes both MID and STRONG through the "anthropic"
    # provider name, so the fake needs one non-retryable failure queued per
    # tier it'll actually be invoked for (MID, then STRONG) -- not just one.
    fake_openai = _FakeProvider("openai", raises=[ProviderError("openai", "down", retryable=False)])
    fake_anthropic = _FakeProvider(
        "anthropic",
        raises=[
            ProviderError("anthropic", "down", retryable=False),
            ProviderError("anthropic", "down", retryable=False),
        ],
    )
    monkeypatch.setitem(router_client._PROVIDERS, "openai", fake_openai)
    monkeypatch.setitem(router_client._PROVIDERS, "anthropic", fake_anthropic)

    with pytest.raises(RuntimeError):
        await complete_with_fallback(starting_tier=Tier.CHEAP, messages=messages)
