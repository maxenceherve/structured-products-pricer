from dataclasses import replace

import numpy as np
import pytest

from spricer.bricks import DigitalOption, EuropeanOption, Underlying, ZeroCoupon
from spricer.market import FlatVol, Market

T = 1.0
BRICKS = [
    ZeroCoupon(T),
    Underlying(T),
    *(EuropeanOption(kind, k, T) for kind in ("call", "put") for k in (60.0, 95.0, 100.0, 130.0)),
    *(DigitalOption(kind, k, T) for kind in ("call", "put") for k in (60.0, 95.0, 100.0, 130.0)),
]


@pytest.mark.parametrize("brick", BRICKS, ids=lambda b: b.label)
def test_value_is_the_discounted_expected_payoff_under_the_smile_density(
    market, discounted_expectation, brick
):
    breakpoints = () if brick.strike is None else (brick.strike,)
    expected = discounted_expectation(market, brick.payoff, brick.maturity, breakpoints)
    assert float(brick.value(market)) == pytest.approx(expected, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_flat_digital_misses_the_skew_term(market, kind):
    skewed = DigitalOption(kind, 100.0, T)
    flat = DigitalOption(kind, 100.0, T, skew_adjusted=False)
    gap = float(skewed.value(market) - flat.value(market))
    assert gap == pytest.approx(float(skewed.skew_adjustment(market)), abs=1e-15)
    assert float(flat.skew_adjustment(market)) == 0.0
    # equity skew: a digital call is worth more than its flat-volatility formula
    assert (gap > 0.02) if kind == "call" else (gap < -0.02)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_digital_is_the_limit_of_a_tight_call_spread(market, strike):
    h = 1e-3
    spread = (
        EuropeanOption("call", strike - h, T).value(market)
        - EuropeanOption("call", strike + h, T).value(market)
    ) / (2 * h)
    assert float(DigitalOption("call", strike, T).value(market)) == pytest.approx(
        float(spread), abs=1e-7
    )


GREEK_BRICKS = [b for b in BRICKS if not isinstance(b, ZeroCoupon)] + [ZeroCoupon(2.0)]


@pytest.mark.parametrize("brick", GREEK_BRICKS, ids=lambda b: b.label)
def test_greeks_match_finite_differences_on_the_surface(market, brick):
    g = brick.greeks(market)
    hs, hv, hr = 1e-3, 1e-5, 1e-6
    s = market.spot

    def v(m=market, spot=None):
        return float(brick.value(m, spot))

    assert float(g.delta) == pytest.approx(
        (v(spot=s + hs) - v(spot=s - hs)) / (2 * hs), rel=1e-5, abs=1e-8
    )
    assert float(g.gamma) == pytest.approx(
        (v(spot=s + hs) - 2 * v() + v(spot=s - hs)) / hs**2, rel=1e-3, abs=1e-6
    )
    assert float(g.vega) == pytest.approx(
        (v(market.with_vol_shift(hv)) - v(market.with_vol_shift(-hv))) / (2 * hv),
        rel=1e-5,
        abs=1e-8,
    )
    assert float(g.rho) == pytest.approx(
        (v(market.with_rate(market.rate + hr)) - v(market.with_rate(market.rate - hr))) / (2 * hr),
        rel=1e-5,
        abs=1e-8,
    )


@pytest.mark.parametrize("brick", GREEK_BRICKS, ids=lambda b: b.label)
def test_theta_holds_implied_volatility_constant(market, brick):
    sigma = brick.implied_vol(market)
    flat = replace(market, vol=FlatVol(0.2 if sigma is None else sigma))
    if isinstance(brick, DigitalOption):  # keep the smile slope too
        flat = replace(flat, vol=_FrozenSmile(sigma, brick.vol_slope(market)))
    dt = 1e-6
    fd = -(
        float(replace(brick, maturity=brick.maturity + dt).value(flat))
        - float(replace(brick, maturity=brick.maturity - dt).value(flat))
    ) / (2 * dt)
    assert float(brick.greeks(market).theta) == pytest.approx(fd, rel=1e-5, abs=1e-8)


class _FrozenSmile:
    def __init__(self, sigma, slope):
        self.sigma, self.slope = sigma, slope

    def vol(self, strike, maturity):
        return np.full(np.broadcast_shapes(np.shape(strike), np.shape(maturity)), self.sigma)

    def vol_slope(self, strike, maturity):
        return np.full(np.broadcast_shapes(np.shape(strike), np.shape(maturity)), self.slope)


def test_payoffs():
    s = np.array([50.0, 100.0, 150.0])
    np.testing.assert_array_equal(EuropeanOption("call", 100.0, T).payoff(s), [0.0, 0.0, 50.0])
    np.testing.assert_array_equal(EuropeanOption("put", 100.0, T).payoff(s), [50.0, 0.0, 0.0])
    np.testing.assert_array_equal(DigitalOption("call", 100.0, T).payoff(s), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(DigitalOption("put", 100.0, T).payoff(s), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(Underlying(T).payoff(s), s)
    np.testing.assert_array_equal(ZeroCoupon(T).payoff(s), [1.0, 1.0, 1.0])


def test_values_are_vectorised_in_spot(market):
    spots = np.linspace(50.0, 150.0, 11)
    option = EuropeanOption("put", 95.0, T)
    np.testing.assert_allclose(
        option.value(market, spots),
        [float(option.value(market.with_spot(x))) for x in spots],
        rtol=1e-14,
    )


def test_sticky_strike_the_smile_does_not_move_with_spot(market):
    option = EuropeanOption("put", 90.0, T)
    assert option.implied_vol(market.with_spot(80.0)) == option.implied_vol(market)


@pytest.mark.parametrize(
    "build",
    [
        lambda: EuropeanOption("call", -1.0, T),
        lambda: EuropeanOption("call", 100.0, -1.0),
        lambda: EuropeanOption("forward", 100.0, T),  # type: ignore[arg-type]
        lambda: DigitalOption("put", np.nan, T),
        lambda: ZeroCoupon(np.inf),
    ],
)
def test_invalid_bricks_raise(build):
    with pytest.raises(ValueError, match=r"strike|maturity|kind"):
        build()


def test_market_validation():
    with pytest.raises(ValueError, match="spot"):
        Market(spot=0.0, rate=0.0, dividend_yield=0.0, vol=FlatVol(0.2))
    with pytest.raises(ValueError, match="rate"):
        Market(spot=100.0, rate=np.nan, dividend_yield=0.0, vol=FlatVol(0.2))


def test_digital_kind_is_validated():
    with pytest.raises(ValueError, match="kind"):
        DigitalOption("forward", 100.0, T)  # type: ignore[arg-type]
