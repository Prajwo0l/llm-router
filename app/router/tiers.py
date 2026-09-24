"""
Model tier definitions. A "tier" is a named rung on the cost/quality ladder
-- the router picks a tier, not a specific model; the tier maps to a real
(provider, model) pair here, in one place, so swapping which actual model
backs "strong" doesn't touch any routing logic.

Real model IDs change over time and across accounts -- the ones below are
realistic placeholders (accurate as of when this was written), not a claim
that these are the only or best choices. Treat this file as the one place
you'd edit to point at whatever models you actually have access to.
"""
from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    CHEAP = "cheap"
    MID = "mid"
    STRONG = "strong"
    LOCAL = "local"  # Ollama/vLLM -- zero marginal API cost, used as an extra-cheap or offline-fallback rung


@dataclass(frozen=True)
class ModelSpec:
    tier: Tier
    provider: str  # "openai" | "anthropic" | "ollama"
    model: str
    # Cost per 1K tokens, input/output -- used by cost_tracking.py and the
    # benchmark. Local models are free at the API-cost level (compute cost
    # is real but out of scope for this ledger).
    cost_per_1k_input: float
    cost_per_1k_output: float


# Ordered cheapest -> strongest within the *paid* tiers; LOCAL sits outside
# this ordering since it's an availability/cost tradeoff, not a quality rung.
TIER_LADDER: list[Tier] = [Tier.CHEAP, Tier.MID, Tier.STRONG]

MODEL_REGISTRY: dict[Tier, ModelSpec] = {
    Tier.CHEAP: ModelSpec(
        tier=Tier.CHEAP,
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    # MID and STRONG temporarily point at gpt-4o-mini too (same as CHEAP)
    # instead of Anthropic -- testing against a single OpenAI key with a
    # very small balance ($1.98 as of when this was set), and MID/STRONG
    # routing to an unconfigured provider would 502 on every "medium"/
    # "hard" prompt. This tests the routing/escalation *logic*, not real
    # quality tiers, since all three cost the same right now. Once an
    # Anthropic key (or more OpenAI budget) is available, swap MID back to
    # a mid-cost model and STRONG to something like "gpt-4o" -- that's the
    # whole point of this file being the one place tier->model is decided.
    Tier.MID: ModelSpec(
        tier=Tier.MID,
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    Tier.STRONG: ModelSpec(
        tier=Tier.STRONG,
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    Tier.LOCAL: ModelSpec(
        tier=Tier.LOCAL,
        provider="ollama",
        model="llama3.1:8b",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
}


def next_tier_up(tier: Tier) -> Tier | None:
    """The tier the router escalates to when confidence is low or a
    provider call fails. Returns None if already at the top of the ladder."""
    if tier not in TIER_LADDER:
        return None
    idx = TIER_LADDER.index(tier)
    if idx + 1 >= len(TIER_LADDER):
        return None
    return TIER_LADDER[idx + 1]
