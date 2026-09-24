"""Anthropic provider. Uses the official `anthropic` async client."""
from __future__ import annotations

from app.config import get_settings
from app.models import ChatMessage
from app.providers.base import BaseProvider, CompletionResult, ProviderError


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self):
        self._settings = get_settings()
        self._client = None

    def is_configured(self) -> bool:
        return bool(self._settings.anthropic_api_key)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        if not self.is_configured():
            raise ProviderError(self.name, "ANTHROPIC_API_KEY not set", retryable=False)

        client = self._get_client()

        # Anthropic's API separates the system prompt from the message list.
        system_parts = [m.content for m in messages if m.role == "system"]
        turn_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        try:
            response = await client.messages.create(
                model=model,
                system="\n".join(system_parts) if system_parts else None,
                messages=turn_messages,
                temperature=temperature,
                max_tokens=max_tokens or 1024,  # Anthropic requires max_tokens; OpenAI does not
            )
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return CompletionResult(
            content=text,
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason or "stop",
        )
