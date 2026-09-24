from app.router.tiers import MODEL_REGISTRY, TIER_LADDER, Tier, next_tier_up


def test_tier_ladder_covers_every_paid_tier_exactly_once():
    assert set(TIER_LADDER) == {Tier.CHEAP, Tier.MID, Tier.STRONG}
    assert len(TIER_LADDER) == len(set(TIER_LADDER))


def test_next_tier_up_climbs_the_ladder():
    assert next_tier_up(Tier.CHEAP) == Tier.MID
    assert next_tier_up(Tier.MID) == Tier.STRONG


def test_next_tier_up_returns_none_at_the_top():
    assert next_tier_up(Tier.STRONG) is None


def test_next_tier_up_returns_none_for_local():
    # LOCAL sits outside the cost/quality ladder (availability tradeoff,
    # not a quality rung) -- escalating past it isn't a defined operation.
    assert next_tier_up(Tier.LOCAL) is None


def test_every_tier_has_a_registry_entry():
    for tier in Tier:
        assert tier in MODEL_REGISTRY
        spec = MODEL_REGISTRY[tier]
        assert spec.tier == tier
        assert spec.provider in {"openai", "anthropic", "ollama"}


def test_local_tier_has_zero_api_cost():
    spec = MODEL_REGISTRY[Tier.LOCAL]
    assert spec.cost_per_1k_input == 0.0
    assert spec.cost_per_1k_output == 0.0


def test_paid_tiers_never_get_cheaper_up_the_ladder():
    # Non-decreasing, not strictly increasing: MID/STRONG are temporarily
    # tied with CHEAP (see tiers.py's note on testing against a single,
    # budget-constrained OpenAI key) -- ties are a valid registry state,
    # a tier getting *cheaper* than the one below it would not be.
    costs = [MODEL_REGISTRY[t].cost_per_1k_output for t in TIER_LADDER]
    assert costs == sorted(costs)
