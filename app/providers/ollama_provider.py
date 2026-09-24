"""
Local-model provider via Ollama's HTTP API. No API key required -- "is
configured" here means "the Ollama server is reachable", checked lazily on
first use rather than at import time (so the app can start up without a
local Ollama instance running at all; this tier just won't be available).

Token counts: Ollama's /api/chat response includes prompt_eval_count /
eval_count, which are real token counts from the local model's own
tokenizer -- not estimated.
"""
from __future__ import annotations

import httpx

from app.config import get_settings
from app.models import ChatMessage
from app.providers.base import BaseProvider, CompletionResult, ProviderError


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self):
        self._settings = get_settings()
        self._reachable: bool | None = None  # cached after first real check, not assumed

    def is_configured(self) -> bool:
        # Static config check only (a base URL is set) -- this does not
        # make a network call. router_client.py should treat a subsequent
        # ProviderError from an unreachable server as a normal fallback
        # trigger, same as any other provider failure.
        return bool(self._settings.ollama_base_url)

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})},
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ProviderError(
                self.name, f"could not reach Ollama at {self._settings.ollama_base_url} -- is it running?", retryable=False
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(self.name, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        return CompletionResult(
            content=data.get("message", {}).get("content", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            finish_reason="stop" if data.get("done") else "length",
        )
