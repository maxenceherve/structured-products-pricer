"""Product finder and equivalent constructions."""

import numpy as np
import pytest

from spricer.finder import (
    CARDS,
    Objective,
    Profile,
    Protection,
    View,
    VolView,
    find_card,
    recommend,
)
from spricer.packaging import equivalences
from spricer.products.capital_protection import NOTE_CATALOG
from spricer.products.coupon_notes import COUPON_NOTE_CATALOG
from spricer.products.participation import (
    PARTICIPATION_CATALOG,
    SHARK_WARRANTS,
    YIELD_CERTIFICATES,
)
from spricer.strategies import CATALOG


def pick(profile: Profile, name: str):
    return next(r for r in recommend(profile) if r.card.name == name)


def top_names(profile: Profile, n: int = 3) -> list[str]:
    return [r.card.name for r in recommend(profile, n)]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (Profile(View.MILD_BULL, 0.0, Objective.GROWTH), "Shark note (up)"),
        (Profile(View.BULLISH, 0.0, Objective.GROWTH), "Participation note"),
        (Profile(View.BEARISH, 0.0, Objective.GROWTH), "Bearish participation note"),
        (
            Profile(View.RANGE, 0.0, Objective.INCOME, accept_barrier=True),
            "Phoenix autocall",
        ),
        (Profile(View.BIG_MOVE, 100.0, Objective.GROWTH, VolView.RISE, 1.0), "Straddle"),
        (Profile(View.RANGE, 100.0, Objective.INCOME, VolView.FALL, 0.5), "Iron condor"),
        (Profile(View.BULLISH, 100.0, Objective.GROWTH, horizon=1.0), "Call"),
    ],
)
def test_the_finder_picks_the_textbook_product(profile, expected):
    assert top_names(profile)[0] == expected


def test_full_protection_excludes_capital_at_risk():
    ranked = recommend(Profile(View.RANGE, 0.0, Objective.INCOME))
    for r in ranked[:3]:
        assert r.card.protection is Protection.FULL
    risky = next(r for r in ranked if r.card.name == "Reverse convertible")
    assert any("Can lose up to 100 %" in w for w in risky.warnings)
    assert risky.score < ranked[0].score - 3.0
    brc = next(r for r in ranked if r.card.name == "Barrier reverse convertible")
    assert any("only while the barrier holds" in w for w in brc.warnings)


def test_the_largest_loss_accepted_selects_partial_protections():
    buffer = pick(Profile(View.MILD_BULL, 90.0, Objective.GROWTH), "Buffer certificate")
    assert any("Loses at most 90 %" in line for line in buffer.reasons)
    bonus = pick(Profile(View.MILD_BULL, 20.0, Objective.GROWTH), "Bonus note")
    assert any("within the 20 % you accept" in line for line in bonus.reasons)
    too_risky = pick(Profile(View.MILD_BULL, 80.0, Objective.GROWTH), "Buffer certificate")
    assert any("more than the 80 % you accept" in w for w in too_risky.warnings)
    accepted = pick(
        Profile(View.RANGE, 0.0, Objective.INCOME, accept_barrier=True), "Phoenix autocall"
    )
    assert any("unless the barrier is breached" in line for line in accepted.reasons)
    texts = {c.name: c.protection_text for c in CARDS}
    assert texts["Bonus note"] == "Loss limited to 10 %"
    assert texts["Call"] == "Capital at risk"
    with pytest.raises(ValueError, match="largest loss"):
        recommend(Profile(View.RANGE, 120.0, Objective.INCOME))


def test_every_point_is_explained():
    profile = Profile(View.RANGE, 100.0, Objective.INCOME, VolView.RISE, 12.0)
    for r in recommend(profile):
        assert r.warnings[-1].startswith("Risk:")
        assert all(line.endswith(".") for line in (*r.reasons, *r.warnings))
    phoenix = pick(profile, "Phoenix autocall")
    assert any("against your view" in w for w in phoenix.warnings)
    assert any("Capital at risk, as you accept" in line for line in phoenix.reasons)
    assert any("Usually built" in w for w in phoenix.warnings)
    protected = pick(profile, "Digital coupon note")
    assert any("more protection than you need" in line for line in protected.reasons)
    big = Profile(View.BIG_MOVE, 100.0, Objective.GROWTH, VolView.RISE, 1.0)
    straddle = pick(big, "Straddle")
    assert any("Long volatility: it gains" in line for line in straddle.reasons)
    calm = Profile(View.RANGE, 100.0, Objective.INCOME, VolView.FALL, 0.5)
    condor = pick(calm, "Iron condor")
    assert any("Short volatility" in line for line in condor.reasons)
    bull = Profile(View.BULLISH, 100.0, Objective.GROWTH, VolView.FALL, 1.0)
    call = pick(bull, "Call")
    assert any("Long volatility, against" in w for w in call.warnings)


def test_every_card_is_a_product_of_the_app():
    names = {s.name for s in CATALOG} | {s.name for s in NOTE_CATALOG}
    names |= {s.name for s in COUPON_NOTE_CATALOG}
    names |= {s.name for s in PARTICIPATION_CATALOG + YIELD_CERTIFICATES + SHARK_WARRANTS}
    for card in CARDS:
        assert card.name in names
        assert find_card(card.name) is card
    with pytest.raises(KeyError):
        find_card("Unknown")
    with pytest.raises(ValueError, match="horizon"):
        recommend(Profile(View.RANGE, 100.0, Objective.INCOME, horizon=0.0))


@pytest.mark.parametrize("view", list(View))
def test_equivalent_constructions_share_payoff_value_and_greeks(market, view):
    pairs = equivalences(view, market.spot, 2.0)
    assert pairs
    grid = np.linspace(0.0, 3.0 * market.spot, 601)
    for pair in pairs:
        a, b = pair.first.strategy, pair.second.strategy
        np.testing.assert_allclose(a.payoff(grid), b.payoff(grid), atol=1e-9)
        assert pair.cash_gap(market) == pytest.approx(0.0, abs=1e-9)
        ga, gb = a.greeks(market), b.greeks(market)
        for name in ("delta", "gamma", "vega"):
            assert float(getattr(ga, name)) == pytest.approx(float(getattr(gb, name)), abs=1e-9)
        assert pair.lesson


def test_the_seller_receives_a_premium_and_posts_collateral(market):
    bullish = equivalences(View.MILD_BULL, market.spot, 2.0)[0]
    buyer, seller = bullish.first, bullish.second
    assert buyer.option_premium(market) > 0.0
    assert seller.option_premium(market) < 0.0
    assert buyer.collateral(market) == 0.0
    assert seller.collateral(market) == pytest.approx(20.0)  # width of the put spread
    assert seller.bond() == pytest.approx(market.spot + 20.0)
    cash_secured = equivalences(View.RANGE, market.spot, 1.0)[0].second
    assert cash_secured.collateral(market) == pytest.approx(100.0)
    assert cash_secured.bond() == pytest.approx(market.spot)
