"""
Provider abstraction. Every backend (OpenAI, Anthropic, a local Ollama
model) implements this same tiny interface, so the fallback/retry
orchestration in router_client.py doesn't need to know which provider it's
talking to -- it just calls `.complete(...)` and gets the same shape back
or a `ProviderError` it can catch and fall back from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models import ChatMessage


@dataclass
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class ProviderError(Exception):
    """Raised for any provider-side failure (timeout, rate limit, auth,
    malformed response) that should trigger a fallback to the next
    provider/tier rather than propagate as a 500 to the caller."""

    def __init__(self, provider: str, message: str, retryable: bool = True):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Raises ProviderError on any failure -- never returns a partial
        or malformed result silently."""
        raise NotImplementedError

    @abstractmethod
    def is_configured(self) -> bool:
        """False if this provider is missing required config (e.g. no API
        key) -- router_client.py skips unconfigured providers rather than
        attempting them and failing every time."""
        raise NotImplementedError
