"""Greeks lab: second-order Greeks against Black-Scholes, moneyness, profiles, explanations."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from spricer import strategies as strat
from spricer.greeks_lab import (
    FIRST_ORDER,
    SECOND_ORDER,
    Moneyness,
    greeks_at,
    moneyness,
    peak,
    second_order_at,
    sign_changes,
    spot_profile,
    time_profile,
)
from spricer.market import FlatVol, Market
from web import explain as gx
from web.options import display_greeks

S, R, Q, V = 100.0, 0.03, 0.01, 0.25
FLAT = Market(S, R, Q, FlatVol(V))


def bs_second_order(k: float, t: float) -> dict[str, float]:
    """Black-Scholes call vanna, volga, charm, speed, converted to the lab's units."""
    sq = math.sqrt(t)
    d1 = (math.log(S / k) + (R - Q + 0.5 * V * V) * t) / (V * sq)
    d2 = d1 - V * sq
    pdf, disc = norm.pdf(d1), math.exp(-Q * t)
    gamma = disc * pdf / (S * V * sq)
    vega = S * disc * pdf * sq
    vanna = -disc * pdf * d2 / V  # d(delta)/d(sigma)
    volga = vega * d1 * d2 / V
    charm = Q * disc * norm.cdf(d1) - disc * pdf * (2 * (R - Q) * t - d2 * V * sq) / (
        2 * t * V * sq
    )
    speed = -gamma / S * (d1 / (V * sq) + 1.0)
    return {
        "Vanna": vanna * 0.01 * 100.0,  # Delta (%) points per vol point, notional = spot
        "Volga": volga * 1e-4,  # Vega (%) per vol point
        "Charm": charm / 365.0 * 100.0,  # Delta (%) points per day
        # Gamma (%) = 0.01 gamma S^2 (notional = spot): its change for +1 % spot
        "Speed": 0.01 * (speed * S**2 + 2.0 * gamma * S) * 0.01 * S,
    }


@pytest.mark.parametrize(("k", "t"), [(100.0, 1.0), (85.0, 0.5), (120.0, 2.0)])
def test_second_order_greeks_match_black_scholes(k, t):
    lab = second_order_at(lambda m: strat.call(k, m), FLAT, t)
    expected = bs_second_order(k, t)
    # the lab uses the desk bumps (1 vol point, 1 day, 1 % spot): central differences that
    # differ from the analytic derivatives by a fraction of a percent
    for name in SECOND_ORDER:
        assert float(lab[name]) == pytest.approx(expected[name], rel=1e-2, abs=1e-6), name


def test_first_order_greeks_are_the_app_display_units(market):
    strategy = strat.bull_call_spread(95.0, 115.0, 1.5)
    lab = greeks_at(strategy, market)
    shown = display_greeks(strategy, market)
    for name in ("Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)"):
        assert float(lab[name]) == pytest.approx(shown[name], rel=1e-10)
    assert float(lab["Price (%)"]) == pytest.approx(float(strategy.value(market)))


def test_moneyness_is_measured_in_standard_deviations():
    assert moneyness("call", 101.0, FLAT, 1.0) is Moneyness.ATM
    assert moneyness("call", 80.0, FLAT, 1.0) is Moneyness.ITM
    assert moneyness("put", 80.0, FLAT, 1.0) is Moneyness.OTM
    assert moneyness("put", 130.0, FLAT, 1.0) is Moneyness.ITM
    # the same strike is at the money for a long maturity, not for a short one
    assert moneyness("call", 106.0, FLAT, 3.0) is Moneyness.ATM
    assert moneyness("call", 106.0, FLAT, 0.05) is Moneyness.OTM


def test_profiles_have_every_greek_and_gamma_explodes_at_the_money_near_expiry():
    spots = np.linspace(60.0, 140.0, 81)
    profile = spot_profile(lambda t: strat.call(100.0, t), FLAT, 1.0, spots)
    assert set(profile) == set(FIRST_ORDER) | set(SECOND_ORDER)
    assert profile["Gamma (%)"].shape == spots.shape
    assert peak(spots, profile["Gamma (%)"]) == pytest.approx(100.0, abs=5.0)
    times = np.array([1.0, 0.25, 0.04])
    atm = time_profile(lambda t: strat.call(100.0, t), FLAT, times, 100.0)["Gamma (%)"]
    otm = time_profile(lambda t: strat.call(100.0, t), FLAT, times, 75.0)["Gamma (%)"]
    assert atm[0] < atm[1] < atm[2]
    assert otm[2] < otm[0]


def test_sign_changes_ignore_noise():
    x = np.linspace(-1.0, 1.0, 201)
    assert sign_changes(x, x) == pytest.approx([0.0], abs=1e-9)
    noise = 1e-12 * np.sin(50 * x)
    assert sign_changes(x, noise) == []
    assert sign_changes(x, x**2 + 1.0) == []


# --- Explanations -----------------------------------------------------------------------------


def explain(strategy_of_t, name, family, short=False, money=None, knock_out=False, t=1.0):
    spots_pct = np.linspace(50.0, 150.0, 101)
    profile = spot_profile(strategy_of_t, FLAT, t, spots_pct)
    now = {k: float(v) for k, v in greeks_at(strategy_of_t(t), FLAT).items()}
    quarter = float(greeks_at(strategy_of_t(t / 4), FLAT)["Gamma (%)"])
    situation = gx.Situation(name, family, short, money, knock_out)
    return _text(*gx.first_order(situation, now, spots_pct, profile, quarter))


def _text(cards, notes) -> str:
    parts = [f"{c.headline} {' '.join(c.details)}" for c in cards]
    return " ".join([*parts, *notes])


def test_a_long_atm_call_is_explained():
    text = explain(lambda t: strat.call(100.0, t), "Call", "vanilla", money=Moneyness.ATM)
    assert "If the spot rises by 1 %, the position gains" in text
    assert "close to 50 %" in text
    assert "1/√T" in text
    assert "Theta pays for gamma" in text


def test_a_short_put_and_a_digital_are_explained():
    short_put = explain(
        lambda t: strat.put(100.0, t).short(), "Put", "vanilla", short=True, money=Moneyness.ATM
    )
    assert "Short position" in short_put
    assert "the position gains" in short_put
    digital = explain(lambda t: strat.digital_call(100.0, t, 10.0), "Digital call", "digital")
    assert "changes sign" in digital
    assert "decides whether it pays" in digital


def test_explanations_of_a_worthless_position_and_of_a_far_strike():
    _, dead = gx.first_order(
        gx.Situation("Barrier option", "barrier", False),
        dict.fromkeys(FIRST_ORDER, 0.0),
        np.linspace(50, 150, 11),
        dict.fromkeys(FIRST_ORDER, np.zeros(11)),
        None,
    )
    assert "worth nothing" in dead[0]
    deep_put = explain(lambda t: strat.put(160.0, t), "Put", "vanilla", t=2.0)
    assert "carry" in deep_put
    flat_value = _text(
        *gx.first_order(
            gx.Situation("Zero", "vanilla", False),
            {
                "Price (%)": 90.0,
                "Delta (%)": 0.0,
                "Gamma (%)": 0.0,
                "Vega (%)": 0.0,
                "Theta (%)": 0.01,
                "Rho (%)": -0.01,
            },
            np.linspace(50, 150, 11),
            dict.fromkeys(FIRST_ORDER, np.zeros(11)),
            None,
        )
    )
    assert "only the carry" in flat_value


def test_a_knock_out_has_a_negative_vega_explained():
    text = _text(
        *gx.first_order(
            gx.Situation("Barrier option", "barrier", False, None, knock_out=True),
            {
                "Price (%)": 2.0,
                "Delta (%)": -10.0,
                "Gamma (%)": -1.0,
                "Vega (%)": -0.2,
                "Theta (%)": 0.01,
                "Rho (%)": 0.0,
            },
            np.linspace(50, 150, 11),
            {k: np.linspace(-1, 1, 11) for k in FIRST_ORDER},
            None,
        )
    )
    assert "losing the option" in text
    assert "the position loses" in text


def test_second_order_and_note_readings():
    cards = gx.second_order({"Vanna": 0.4, "Volga": 0.001, "Charm": -0.02, "Speed": 0.03})
    assert [c.title for c in cards] == ["Vanna", "Volga", "Charm", "Speed"]
    assert "+0.40 %" in cards[0].headline
    assert "gain more and more" in cards[1].details[0]
    negative = gx.second_order({"Vanna": -0.4, "Volga": -0.001, "Charm": 0.02, "Speed": -0.03})
    assert "-0.40 %" in negative[0].headline
    assert "shrinks" in negative[1].details[0]
    reading = gx.coupon_greeks(["A", "B"], [30.0, -5.0], [-0.3, 0.1])
    assert "gains 0.30 %" in reading[0]
    assert "Volatility hurts" in reading[0]
    assert "loses" in reading[1]
    assert "Volatility helps" in reading[1]
    assert "most exposed to **A**" in reading[2]
