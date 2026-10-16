"""Greeks as functions of the spot, the time and the volatility, for the Greeks lab.

Everything is in the desk display units of the app, for a notional equal to today's spot:

* ``Price (%)``: value in % of the notional;
* ``Delta (%)``: exposure, the change in value for +1 % spot times 100 (a call with a 0.52
  Black-Scholes delta shows 52 %);
* ``Gamma (%)``: change of Delta (%) for +1 % spot;
* ``Vega (%)``: change in value for +1 volatility point;
* ``Theta (%)``: change in value after one calendar day;
* ``Rho (%)``: change in value for +1 basis point.

Second-order Greeks (expert view), by central differences of the first-order ones:

* ``Vanna``: change of Delta (%) for +1 volatility point;
* ``Volga``: change of Vega (%) for +1 volatility point;
* ``Charm``: change of Delta (%) after one calendar day;
* ``Speed``: change of Gamma (%) for +1 % spot.

Instruments are given as a function of the maturity, so that time can be moved too. Implied
volatilities are held fixed per strike (sticky strike), as everywhere in the package.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.conventions import DAYS_PER_YEAR, to_desk
from spricer.market.core import Market
from spricer.strategies import Strategy

FIRST_ORDER = ("Price (%)", "Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)")
SECOND_ORDER = ("Vanna", "Volga", "Charm", "Speed")
Instrument = Callable[[float], Strategy]  # maturity in years -> strategy

_VOL_STEP = 0.01
_SPOT_STEP = 0.01
_DAY = 1.0 / DAYS_PER_YEAR
ATM_BAND = 0.25  # |ln(F/K)| / (sigma sqrt T) below which an option counts as at the money


class Moneyness(StrEnum):
    """Where the spot sits against an option's strike."""

    ITM = "In the money"
    ATM = "At the money"
    OTM = "Out of the money"


def moneyness(kind: str, strike: float, market: Market, maturity: float) -> Moneyness:
    """ATM, ITM or OTM, measured in standard deviations of the forward.

    The option is at the money when ``|ln(F/K)| < 0.25 sigma sqrt(T)`` (a Black-Scholes delta
    between about 0.40 and 0.60); otherwise a call is in the money above the strike and a put
    below it.
    """
    vol = float(market.vol.vol(strike, maturity))
    spread = vol * np.sqrt(max(maturity, 1e-8))
    z = float(np.log(market.forward(maturity) / strike)) / spread
    if abs(z) < ATM_BAND:
        return Moneyness.ATM
    in_the_money = z > 0 if kind == "call" else z < 0
    return Moneyness.ITM if in_the_money else Moneyness.OTM


def greeks_at(
    strategy: Strategy, market: Market, spot: ArrayLike | None = None
) -> dict[str, FloatArray]:
    """Price and first-order Greeks at one or many spots, notional = today's spot."""
    s = np.asarray(market.spot if spot is None else spot, dtype=np.float64)
    notional = market.spot
    desk = to_desk(strategy.greeks(market, s), spot=s, notional=notional)
    return {
        "Price (%)": 100.0 * np.asarray(strategy.value(market, s)) / notional,
        "Delta (%)": 100.0 * desk.delta,
        "Gamma (%)": 100.0 * desk.gamma,
        "Vega (%)": np.asarray(desk.vega),
        "Theta (%)": np.asarray(desk.theta),
        "Rho (%)": np.asarray(desk.rho),
    }


def second_order_at(
    instrument: Instrument, market: Market, maturity: float, spot: ArrayLike | None = None
) -> dict[str, FloatArray]:
    """Vanna, volga, charm and speed at one or many spots (see the module docstring)."""
    s = np.asarray(market.spot if spot is None else spot, dtype=np.float64)
    strategy = instrument(maturity)
    up, down = market.with_vol_shift(_VOL_STEP), market.with_vol_shift(-_VOL_STEP)
    g_up, g_down = greeks_at(strategy, up, s), greeks_at(strategy, down, s)
    day = min(_DAY, 0.5 * maturity)
    later = greeks_at(instrument(maturity - day), market, s)
    now = greeks_at(strategy, market, s)
    s_up = greeks_at(strategy, market, s * (1.0 + _SPOT_STEP))
    s_down = greeks_at(strategy, market, s * (1.0 - _SPOT_STEP))
    return {
        "Vanna": (g_up["Delta (%)"] - g_down["Delta (%)"]) / 2.0,
        "Volga": (g_up["Vega (%)"] - g_down["Vega (%)"]) / 2.0,
        "Charm": (later["Delta (%)"] - now["Delta (%)"]) * _DAY / day,
        "Speed": (s_up["Gamma (%)"] - s_down["Gamma (%)"]) / 2.0,
    }


def spot_profile(
    instrument: Instrument, market: Market, maturity: float, spots: ArrayLike
) -> dict[str, FloatArray]:
    """First- and second-order Greeks along a range of spots (implied vols per strike fixed)."""
    s = np.asarray(spots, dtype=np.float64)
    return greeks_at(instrument(maturity), market, s) | second_order_at(
        instrument, market, maturity, s
    )


def time_profile(
    instrument: Instrument, market: Market, maturities: ArrayLike, spot: float
) -> dict[str, FloatArray]:
    """First-order Greeks at ``spot`` for each remaining maturity."""
    rows = [greeks_at(instrument(float(t)), market, spot) for t in np.asarray(maturities)]
    return {name: np.array([float(r[name]) for r in rows]) for name in FIRST_ORDER}


def peak(x: ArrayLike, y: ArrayLike) -> float:
    """Abscissa where ``|y|`` is largest."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(xs[int(np.argmax(np.abs(ys)))])


def sign_changes(x: ArrayLike, y: ArrayLike, tol: float = 1e-6) -> list[float]:
    """Abscissas where ``y`` changes sign (linear interpolation).

    Values below ``tol`` times the largest ``|y|`` (and below 1e-8 in absolute terms) count as
    zero, so numerical noise around zero is not taken for a change of sign.
    """
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.abs(ys) > max(tol * float(np.max(np.abs(ys))), 1e-8)
    xs, ys = xs[keep], ys[keep]
    out = []
    for i in range(len(ys) - 1):
        if np.sign(ys[i]) != np.sign(ys[i + 1]):
            out.append(float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])))
    return out
