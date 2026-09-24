"""Tests app/router/policy.py's decide_tier() in isolation -- constructs
DifficultyResult objects directly rather than going through the heuristic
or classifier, and overrides settings via monkeypatch rather than
depending on .env / environment state."""
import pytest

from app.router.heuristic import DifficultyResult
from app.router.policy import decide_tier
from app.router.tiers import Tier


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    # get_settings() is @lru_cache'd -- clear it so each test's monkeypatched
    # env vars actually take effect instead of reusing a cached Settings
    # instance from a previous test.
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_confident_easy_stays_cheap():
    difficulty = DifficultyResult(label="easy", confidence=0.9, source="heuristic")
    decision = decide_tier(difficulty)
    assert decision.tier == Tier.CHEAP
    assert decision.escalated is False


def test_confident_medium_maps_to_mid():
    difficulty = DifficultyResult(label="medium", confidence=0.9, source="heuristic")
    decision = decide_tier(difficulty)
    assert decision.tier == Tier.MID


def test_confident_hard_maps_to_strong():
    difficulty = DifficultyResult(label="hard", confidence=0.9, source="heuristic")
    decision = decide_tier(difficulty)
    assert decision.tier == Tier.STRONG


def test_low_confidence_escalates_one_rung(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_CONFIDENCE_FLOOR", "0.6")
    difficulty = DifficultyResult(label="easy", confidence=0.5, source="heuristic")
    decision = decide_tier(difficulty)
    assert decision.tier == Tier.MID
    assert decision.escalated is True
    assert decision.escalation_reason is not None


def test_low_confidence_at_top_of_ladder_does_not_escalate_past_strong():
    difficulty = DifficultyResult(label="hard", confidence=0.1, source="heuristic")
    decision = decide_tier(difficulty)
    assert decision.tier == Tier.STRONG
    # next_tier_up(STRONG) is None, so escalation is a no-op here even
    # though confidence is below the floor.
    assert decision.escalated is False


def test_caller_min_tier_overrides_a_lower_base_tier():
    difficulty = DifficultyResult(label="easy", confidence=0.9, source="heuristic")
    decision = decide_tier(difficulty, min_tier=Tier.STRONG)
    assert decision.tier == Tier.STRONG
    assert decision.escalated is True


def test_caller_min_tier_does_not_downgrade_a_higher_base_tier():
    difficulty = DifficultyResult(label="hard", confidence=0.9, source="heuristic")
    decision = decide_tier(difficulty, min_tier=Tier.CHEAP)
    assert decision.tier == Tier.STRONG
