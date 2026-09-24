"""OpenAI provider. Uses the official `openai` async client."""
from __future__ import annotations

from app.config import get_settings
from app.models import ChatMessage
from app.providers.base import BaseProvider, CompletionResult, ProviderError


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        self._settings = get_settings()
        self._client = None  # constructed lazily so importing this module never requires an API key

    def is_configured(self) -> bool:
        return bool(self._settings.openai_api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # local import: keep module import side-effect-free

            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        if not self.is_configured():
            raise ProviderError(self.name, "OPENAI_API_KEY not set", retryable=False)

        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # the SDK raises its own exception hierarchy; normalize at the boundary
            raise ProviderError(self.name, str(exc)) from exc

        choice = response.choices[0]
        usage = response.usage
        return CompletionResult(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )
