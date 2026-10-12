from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from spricer.bricks import EuropeanOption
from spricer.conventions import Greeks, percent_of_notional, to_desk, year_fraction


def test_year_fraction_is_act_365_fixed():
    assert year_fraction(date(2026, 1, 1), date(2027, 1, 1)) == 1.0
    assert year_fraction(date(2028, 1, 1), date(2029, 1, 1)) == pytest.approx(366 / 365)
    assert year_fraction(date(2026, 3, 1), date(2026, 2, 1)) == pytest.approx(-28 / 365)


def test_greeks_arithmetic():
    a = Greeks(*(np.asarray(x) for x in (1.0, 2.0, 3.0, 4.0, 5.0)))
    b = a * 2.0 + (-a)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert float(getattr(b, name)) == float(getattr(a, name))
    assert float((Greeks.zero() + a).rho) == 5.0


def test_percent_of_notional():
    assert float(percent_of_notional(2.5, 50.0)) == 5.0


def test_desk_greeks_give_the_taylor_expansion(market):
    """Each desk Greek predicts the P&L of its own unit move, in % of notional."""
    option = EuropeanOption("call", 105.0, 1.0)
    notional = market.spot
    desk = to_desk(option.greeks(market), spot=market.spot, notional=notional)
    v0 = float(option.value(market))

    def pnl_pct(value):
        return float(percent_of_notional(value - v0, notional))

    m = 0.5  # spot move in %
    exact = pnl_pct(option.value(market, market.spot * (1 + m / 100)))
    assert exact == pytest.approx(float(desk.delta) * m + 0.5 * float(desk.gamma) * m**2, abs=2e-5)

    assert pnl_pct(option.value(market.with_vol_shift(0.001))) == pytest.approx(
        float(desk.vega) * 0.1, rel=1e-3
    )
    assert pnl_pct(option.value(market.with_rate(market.rate + 1e-4))) == pytest.approx(
        float(desk.rho), rel=1e-3
    )

    # theta holds the implied volatility constant: shorten the maturity at a frozen volatility
    frozen = replace(market, vol=_Frozen(option.implied_vol(market)))
    aged = replace(option, maturity=option.maturity - 1 / 365)
    assert pnl_pct(aged.value(frozen)) == pytest.approx(float(desk.theta), rel=1e-3)


class _Frozen:
    def __init__(self, sigma):
        self.sigma = sigma

    def vol(self, strike, maturity):
        return np.full(np.broadcast_shapes(np.shape(strike), np.shape(maturity)), self.sigma)

    def vol_slope(self, strike, maturity):
        return np.zeros(np.broadcast_shapes(np.shape(strike), np.shape(maturity)))
