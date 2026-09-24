"""
The FastAPI app: exposes an OpenAI-compatible POST /v1/chat/completions,
wiring together (in order) rate limiting -> semantic cache lookup ->
difficulty scoring -> tier decision -> provider call with fallback ->
cost tracking -> cache store -> tracing. Every one of those stages is
independently optional/degradable (see each module's own docstring) so
this endpoint works end-to-end even with zero API keys, no trained
classifier, no faiss installed, and tracing disabled -- which is exactly
the state this repo is in until you add keys and run the Colab training.

Streaming (`stream: true`) is explicitly not implemented -- see the 501
below. Wiring streaming through the fallback/retry/cache logic is a real
design problem (you can't "retry" after you've already streamed half a
response to the caller) and doing it properly is out of scope for getting
the whole system running end-to-end today.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app.cache.semantic_cache import get_cache, get_embedder
from app.config import get_settings
from app.dashboard.routes import router as dashboard_router
from app.middleware.cost_tracking import CostEvent, get_cost_ledger
from app.middleware.rate_limit import RateLimitExceeded, get_rate_limiter
from app.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    RouterMetadata,
    UsageInfo,
)
from app.providers.router_client import complete_with_fallback
from app.router.classifier import get_classifier
from app.router.policy import decide_tier
from app.router.tiers import MODEL_REGISTRY, Tier
from app.telemetry.tracing import trace_completion

logging.basicConfig(level=get_settings().log_level.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Router",
    description="Eval-driven router that picks the cheapest model tier likely to answer a prompt well, "
    "with a semantic cache in front of it.",
    version="0.1.0",
)
app.include_router(dashboard_router)


def _resolve_api_key(authorization: str | None) -> str:
    """No real auth/issuance system here (out of scope) -- the bearer token
    is used purely as a rate-limit/cost-ledger partition key, same as
    OpenAI's own key model. Missing header falls back to a shared
    "anonymous" bucket rather than rejecting the request, since local/dev
    use without a key should still work."""
    if not authorization:
        return "anonymous"
    return authorization.removeprefix("Bearer ").strip() or "anonymous"


# Same rung ordering as policy.py's internal _tier_rank -- kept in sync
# manually since that one is private to policy.py. LOCAL is ranked just
# above CHEAP (an availability choice, not a quality claim), matching how
# policy.py already treats it.
_TIER_RANK_ORDER = [Tier.LOCAL, Tier.CHEAP, Tier.MID, Tier.STRONG]


def _resolve_starting_tier(request: ChatCompletionRequest, difficulty_label: str, difficulty_confidence: float, source: str):
    from app.router.heuristic import DifficultyResult

    min_tier = None
    if request.model and request.model.startswith("router:") and request.model != "router:auto":
        requested = request.model.removeprefix("router:")
        try:
            min_tier = Tier(requested)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown router tier in model='{request.model}'")
    if request.router_min_tier:
        try:
            explicit_min = Tier(request.router_min_tier)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown router_min_tier='{request.router_min_tier}'")
        min_tier = explicit_min if min_tier is None else max(min_tier, explicit_min, key=_TIER_RANK_ORDER.index)

    difficulty = DifficultyResult(label=difficulty_label, confidence=difficulty_confidence, source=source)
    return decide_tier(difficulty, min_tier=min_tier)


def _estimate_cost(tier: Tier, prompt_tokens: int, completion_tokens: int) -> float:
    spec = MODEL_REGISTRY[tier]
    return (prompt_tokens / 1000) * spec.cost_per_1k_input + (completion_tokens / 1000) * spec.cost_per_1k_output


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": f"router:{tier.value}", "object": "model", "owned_by": "llm-router"} for tier in Tier]
        + [{"id": "router:auto", "object": "model", "owned_by": "llm-router"}],
    }


@app.get("/v1/usage")
async def usage(authorization: str | None = Header(default=None)):
    """Not part of the OpenAI spec -- lets a caller (or the dashboard)
    check their own spend. Scoped to the caller's own key only; there's no
    admin/all-keys view here since there's no auth system to gate it."""
    api_key = _resolve_api_key(authorization)
    ledger = get_cost_ledger()
    return await ledger.summary(api_key=api_key)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)):
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented in this router yet.")

    api_key = _resolve_api_key(authorization)
    started_at = time.monotonic()

    rate_limiter = get_rate_limiter()
    try:
        await rate_limiter.check(api_key)
    except RateLimitExceeded as exc:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": str(exc), "type": "rate_limit_exceeded"}},
            headers={"Retry-After": str(int(exc.retry_after_seconds) + 1)},
        )

    async with trace_completion("chat_completion", {"api_key": api_key}) as span:
        user_prompt = "\n".join(m.content for m in request.messages if m.role == "user") or request.messages[-1].content

        # --- 1. semantic cache lookup ---
        cache_hit_entry = None
        closest_cache_similarity = None
        cache = get_cache() if not request.router_bypass_cache else None
        query_embedding = None
        if cache is not None:
            try:
                embedder = get_embedder()
                query_embedding = await embedder.embed(user_prompt)
                closest = cache.lookup(query_embedding)
                if closest is not None:
                    closest_cache_similarity = closest.similarity
                    if closest.is_hit(cache.similarity_threshold):
                        cache_hit_entry = closest.entry
                        closest.entry.hit_count += 1
                        span.set("cache_hit", True)
                        span.set("cache_similarity", closest.similarity)
            except Exception:
                logger.exception("Semantic cache lookup failed -- continuing without cache")

        if cache_hit_entry is not None:
            latency_ms = (time.monotonic() - started_at) * 1000
            await get_cost_ledger().record(
                CostEvent(
                    api_key=api_key,
                    tier=cache_hit_entry.tier_served,
                    provider="cache",
                    model="cache",
                    prompt_tokens=0,
                    completion_tokens=0,
                    estimated_cost_usd=0.0,
                    cache_hit=True,
                    latency_ms=latency_ms,
                )
            )
            return ChatCompletionResponse(
                model=request.model,
                choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=cache_hit_entry.response))],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                router=RouterMetadata(
                    tier_used=cache_hit_entry.tier_served,
                    provider="cache",
                    model_used="cache",
                    difficulty_label="n/a",
                    difficulty_confidence=1.0,
                    escalated=False,
                    cache_hit=True,
                    closest_cache_similarity=closest_cache_similarity,
                    latency_ms=latency_ms,
                    estimated_cost_usd=0.0,
                ),
            )

        span.set("cache_hit", False)

        # --- 2. difficulty scoring + tier decision ---
        # Difficulty is always scored, even under router_force_tier, purely
        # for observability (the benchmark's baseline arms still want to
        # know what the router *would* have picked, to report alongside
        # "what tier was actually forced").
        classifier = get_classifier()
        difficulty = classifier.score(user_prompt)

        if request.router_force_tier:
            try:
                starting_tier = Tier(request.router_force_tier)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown router_force_tier='{request.router_force_tier}'")
            escalated_by_policy = False
        else:
            decision = _resolve_starting_tier(request, difficulty.label, difficulty.confidence, difficulty.source)
            starting_tier = decision.tier
            escalated_by_policy = decision.escalated

        span.set("difficulty_label", difficulty.label)
        span.set("difficulty_confidence", difficulty.confidence)
        span.set("tier_starting", starting_tier.value)

        # --- 3. provider call with fallback across the tier ladder ---
        try:
            routed = await complete_with_fallback(
                starting_tier=starting_tier,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        latency_ms = (time.monotonic() - started_at) * 1000
        estimated_cost = _estimate_cost(routed.tier_served, routed.result.prompt_tokens, routed.result.completion_tokens)

        span.set("tier_served", routed.tier_served.value)
        span.set("provider", routed.provider)
        span.set("estimated_cost_usd", estimated_cost)
        span.set("escalated", routed.escalated_from_failure or escalated_by_policy)

        # --- 4. cost tracking ---
        await get_cost_ledger().record(
            CostEvent(
                api_key=api_key,
                tier=routed.tier_served.value,
                provider=routed.provider,
                model=routed.model,
                prompt_tokens=routed.result.prompt_tokens,
                completion_tokens=routed.result.completion_tokens,
                estimated_cost_usd=estimated_cost,
                cache_hit=False,
                latency_ms=latency_ms,
            )
        )

        # --- 5. cache store (best-effort; a failure here shouldn't fail the request) ---
        if cache is not None and query_embedding is not None:
            try:
                cache.store(query_embedding, user_prompt, routed.result.content, routed.tier_served.value)
            except Exception:
                logger.exception("Semantic cache store failed -- response still returned to caller")

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=routed.result.content),
                    finish_reason=routed.result.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=routed.result.prompt_tokens,
                completion_tokens=routed.result.completion_tokens,
                total_tokens=routed.result.prompt_tokens + routed.result.completion_tokens,
            ),
            router=RouterMetadata(
                tier_used=routed.tier_served.value,
                provider=routed.provider,
                model_used=routed.model,
                difficulty_label=difficulty.label,
                difficulty_confidence=difficulty.confidence,
                escalated=routed.escalated_from_failure or escalated_by_policy,
                cache_hit=False,
                closest_cache_similarity=closest_cache_similarity,
                latency_ms=latency_ms,
                estimated_cost_usd=estimated_cost,
            ),
        )
