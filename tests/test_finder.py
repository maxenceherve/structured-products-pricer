"""Product finder and equivalent constructions."""

from pathlib import Path

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
from spricer.strategies import CATALOG

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def pick(profile: Profile, name: str):
    return next(r for r in recommend(profile) if r.card.name == name)


def top_names(profile: Profile, n: int = 3) -> list[str]:
    return [r.card.name for r in recommend(profile, n)]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (Profile(View.MILD_BULL, Protection.FULL, Objective.GROWTH), "Shark note (up)"),
        (Profile(View.BULLISH, Protection.FULL, Objective.GROWTH), "Participation note"),
        (Profile(View.BEARISH, Protection.FULL, Objective.GROWTH), "Bearish participation note"),
        (Profile(View.RANGE, Protection.CONDITIONAL, Objective.INCOME), "Phoenix autocall"),
        (Profile(View.BIG_MOVE, Protection.NONE, Objective.GROWTH, VolView.RISE, 1.0), "Straddle"),
        (Profile(View.RANGE, Protection.NONE, Objective.INCOME, VolView.FALL, 0.5), "Iron condor"),
        (Profile(View.BULLISH, Protection.NONE, Objective.GROWTH, horizon=1.0), "Call"),
    ],
)
def test_the_finder_picks_the_textbook_product(profile, expected):
    assert top_names(profile)[0] == expected


def test_full_protection_excludes_capital_at_risk():
    ranked = recommend(Profile(View.RANGE, Protection.FULL, Objective.INCOME))
    for r in ranked[:3]:
        assert r.card.protection is Protection.FULL
    risky = next(r for r in ranked if r.card.name == "Reverse convertible")
    assert any("Protects less" in w for w in risky.warnings)
    assert risky.score < ranked[0].score - 3.0


def test_every_point_is_explained():
    profile = Profile(View.RANGE, Protection.NONE, Objective.INCOME, VolView.RISE, 12.0)
    for r in recommend(profile):
        assert r.warnings[-1].startswith("Risk:")
        assert all(line.endswith(".") for line in (*r.reasons, *r.warnings))
    phoenix = pick(profile, "Phoenix autocall")
    assert any("against your view" in w for w in phoenix.warnings)
    assert any("more than you need" in line for line in phoenix.reasons)
    assert any("Usually built" in w for w in phoenix.warnings)
    big = Profile(View.BIG_MOVE, Protection.NONE, Objective.GROWTH, VolView.RISE, 1.0)
    straddle = pick(big, "Straddle")
    assert any("Long volatility: it gains" in line for line in straddle.reasons)
    calm = Profile(View.RANGE, Protection.NONE, Objective.INCOME, VolView.FALL, 0.5)
    condor = pick(calm, "Iron condor")
    assert any("Short volatility" in line for line in condor.reasons)
    bull = Profile(View.BULLISH, Protection.NONE, Objective.GROWTH, VolView.FALL, 1.0)
    call = pick(bull, "Call")
    assert any("Long volatility, against" in w for w in call.warnings)


def test_every_card_is_a_product_of_the_app():
    names = {s.name for s in CATALOG} | {s.name for s in NOTE_CATALOG}
    names |= {s.name for s in COUPON_NOTE_CATALOG}
    for card in CARDS:
        assert card.name in names
        assert find_card(card.name) is card
    with pytest.raises(KeyError):
        find_card("Unknown")
    with pytest.raises(ValueError, match="horizon"):
        recommend(Profile(View.RANGE, Protection.NONE, Objective.INCOME, horizon=0.0))


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


# --- App ---------------------------------------------------------------------------------------


@pytest.fixture
def app():
    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_the_home_page_is_the_finder(app):
    titles = [m.value for m in app.markdown if m.value.startswith("### ")]
    assert titles[0] == "### 1. Digital coupon note"
    assert any("packaging" in s.value for s in app.subheader)


@pytest.mark.parametrize(
    ("view", "protection", "page_key", "expected"),
    [
        ("Strongly bullish", "Capital at risk", "vanilla_product_SINGLE_OPTIONS", "Call"),
        ("Range-bound", "Protected unless a barrier is breached", "cn_autocallables_product", None),
        ("Range-bound", "Capital at risk", "cn_yield_product", None),
        ("Moderately bullish", "Full capital protection", "cp_product", "Digital coupon note"),
    ],
)
def test_price_it_opens_the_product_on_its_page(app, view, protection, page_key, expected):
    app.pills(key="widget_finder_view").set_value(view).run()
    app.pills(key="widget_finder_protection").set_value(protection).run()
    if view == "Strongly bullish":
        app.radio(key="widget_finder_objective").set_value("Participation in the move").run()
        app.slider(key="widget_finder_horizon").set_value(1.0).run()
    if protection == "Capital at risk" and view == "Range-bound":
        # pick the yield enhancement product among the top three
        names = [m.value for m in app.markdown if m.value.startswith("### ")]
        rank = next(i for i, n in enumerate(names, 1) if "convertible" in n)
    elif page_key == "cn_autocallables_product":
        rank = 1
    else:
        rank = 1
    app.button(key=f"finder_open_{rank}").click().run()
    assert not app.exception
    assert page_key in app.session_state
    if expected is not None:
        assert app.session_state[page_key] == expected
