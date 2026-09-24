"""
Orchestrates a completion request across the tier ladder: try the assigned
tier's provider, and on a retryable failure (timeout, 5xx, rate limit),
retry a couple of times, then escalate to the next tier up rather than
failing the request outright. A non-retryable failure (e.g. missing API
key) skips straight to escalation without wasting retries on something
that can't succeed.

This is where "fallback and retries across providers" actually lives --
policy.py decides the *starting* tier from difficulty; this module decides
what happens when that tier doesn't work out.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.models import ChatMessage
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import BaseProvider, CompletionResult, ProviderError
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.router.tiers import MODEL_REGISTRY, Tier, next_tier_up

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, BaseProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
}

_MAX_RETRIES_PER_TIER = 2
_RETRY_BACKOFF_SECONDS = 0.75


@dataclass
class RoutedCompletion:
    result: CompletionResult
    tier_served: Tier
    provider: str
    model: str
    attempts: int
    escalated_from_failure: bool


async def complete_with_fallback(
    starting_tier: Tier,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> RoutedCompletion:
    tier: Tier | None = starting_tier
    attempts = 0
    escalated_from_failure = False
    last_error: Exception | None = None

    while tier is not None:
        spec = MODEL_REGISTRY[tier]
        provider = _PROVIDERS.get(spec.provider)
        if provider is None:
            logger.error("No provider implementation registered for %s", spec.provider)
            tier = next_tier_up(tier)
            escalated_from_failure = True
            continue

        if not provider.is_configured():
            logger.info("Skipping tier=%s: provider %s is not configured", tier.value, spec.provider)
            tier = next_tier_up(tier)
            escalated_from_failure = True
            continue

        for retry_num in range(_MAX_RETRIES_PER_TIER + 1):
            attempts += 1
            try:
                result = await provider.complete(
                    model=spec.model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                return RoutedCompletion(
                    result=result,
                    tier_served=tier,
                    provider=spec.provider,
                    model=spec.model,
                    attempts=attempts,
                    escalated_from_failure=escalated_from_failure,
                )
            except ProviderError as exc:
                last_error = exc
                logger.warning(
                    "Provider call failed (tier=%s provider=%s attempt=%d retryable=%s): %s",
                    tier.value,
                    spec.provider,
                    retry_num + 1,
                    exc.retryable,
                    exc,
                )
                if not exc.retryable:
                    break
                if retry_num < _MAX_RETRIES_PER_TIER:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (retry_num + 1))

        # Exhausted retries (or hit a non-retryable error) at this tier -- escalate.
        escalated_from_failure = True
        tier = next_tier_up(tier)

    raise RuntimeError(
        f"All tiers exhausted without a successful completion. Last error: {last_error}"
    )
