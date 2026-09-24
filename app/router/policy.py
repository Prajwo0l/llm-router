"""
Turns a DifficultyResult into a Tier decision. This is the one place that
encodes "what do we actually do with a difficulty label" -- kept separate
from both the scorer (heuristic.py / classifier.py) and the tier registry
(tiers.py) so each can be reasoned about and tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.router.heuristic import DifficultyResult
from app.router.tiers import Tier, next_tier_up

_LABEL_TO_TIER = {
    "easy": Tier.CHEAP,
    "medium": Tier.MID,
    "hard": Tier.STRONG,
}


@dataclass
class RoutingDecision:
    tier: Tier
    difficulty: DifficultyResult
    escalated: bool
    escalation_reason: str | None


def decide_tier(difficulty: DifficultyResult, min_tier: Tier | None = None) -> RoutingDecision:
    settings = get_settings()
    base_tier = _LABEL_TO_TIER.get(difficulty.label, Tier.MID)

    escalated = False
    reason = None

    # Low-confidence predictions escalate one rung rather than being
    # trusted outright -- the whole point of a cost-saving router is that a
    # confidently-easy prompt is safe to send cheap; an uncertain one isn't,
    # and the cost of over-escalating a few prompts is much smaller than
    # the cost of a wrong answer on one that mattered.
    if difficulty.confidence < settings.classifier_confidence_floor:
        escalated_tier = next_tier_up(base_tier)
        if escalated_tier is not None:
            base_tier = escalated_tier
            escalated = True
            reason = (
                f"confidence {difficulty.confidence:.2f} below floor "
                f"{settings.classifier_confidence_floor:.2f} ({difficulty.source})"
            )

    # A caller-specified floor (router_min_tier on the request) always wins
    # over the router's own judgement -- e.g. a caller who knows a request
    # is high-stakes shouldn't have to fight the classifier to get it.
    if min_tier is not None and _tier_rank(min_tier) > _tier_rank(base_tier):
        base_tier = min_tier
        escalated = True
        reason = f"caller-specified router_min_tier={min_tier.value}"

    return RoutingDecision(tier=base_tier, difficulty=difficulty, escalated=escalated, escalation_reason=reason)


def _tier_rank(tier: Tier) -> int:
    order = [Tier.LOCAL, Tier.CHEAP, Tier.MID, Tier.STRONG]
    return order.index(tier) if tier in order else 1
