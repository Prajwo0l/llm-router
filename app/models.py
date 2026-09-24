"""
OpenAI-compatible request/response schemas. Kept deliberately close to the
real /v1/chat/completions shape (only the fields this router actually reads
or sets are modelled -- not a full spec reimplementation) so any existing
OpenAI-client-based app can point its `base_url` here with no code changes.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="router:auto",
        description="Accepted for OpenAI-client compatibility. 'router:auto' (the default) lets the "
        "router choose a tier; any other value is treated as a tier override (e.g. 'router:strong').",
    )
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    # Router-specific, ignored by strict OpenAI clients that don't set them.
    router_min_tier: str | None = Field(
        default=None, description="Force a minimum tier (e.g. caller knows this request is high-stakes)."
    )
    router_force_tier: str | None = Field(
        default=None,
        description="Bypass difficulty-based routing entirely and use exactly this tier. Unlike "
        "router_min_tier (a floor the router's own decision can still exceed), this is an exact override -- "
        "used by the benchmark harness to run 'always-cheapest' / 'always-strongest' baseline arms that "
        "don't get to benefit from difficulty routing at all.",
    )
    router_bypass_cache: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class RouterMetadata(BaseModel):
    """Not part of the OpenAI spec -- extra fields a router-aware caller
    (like the benchmark harness) can read, that a plain OpenAI client will
    just ignore since it doesn't know to look for them."""

    # model_used starts with "model_", which pydantic v2 reserves for its
    # own internals by default and warns about on every import -- the name
    # matches this project's OpenAI-compatible vocabulary (model, model_used
    # read naturally together) and is worth keeping, so silence the warning
    # instead of renaming the field.
    model_config = ConfigDict(protected_namespaces=())

    tier_used: str
    provider: str
    model_used: str
    difficulty_label: str
    difficulty_confidence: float
    escalated: bool
    cache_hit: bool
    closest_cache_similarity: float | None = Field(
        default=None,
        description="Cosine similarity to the nearest cached prompt, regardless of whether it cleared "
        "semantic_cache_similarity_threshold. Populated even on a miss -- a miss that scored 0.91 against a "
        "0.94 threshold is a different situation than one that scored 0.2, and this is what lets you tune the "
        "threshold by watching real traffic instead of guessing. None only when the cache is disabled, bypassed, "
        "or was empty at lookup time.",
    )
    latency_ms: float
    estimated_cost_usd: float


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    router: RouterMetadata
