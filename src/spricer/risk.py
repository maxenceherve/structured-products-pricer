"""P&L explain: Taylor attribution at the starting point, full revaluation, residual.

A position is any function ``value(market, maturity)`` returning its value in % of a notional:
a vanilla strategy, a structured note. A market move is a :class:`Shock` (spot, volatility,
time, rate). Three views of the same P&L:

* **Taylor** at the starting point, term by term, in the desk units of
  :mod:`spricer.conventions`::

      dV ~ Delta m + 1/2 Gamma m^2 + Vega dv + Theta days + Rho dr
           + Vanna m dv + 1/2 Volga dv^2

  with ``m`` the spot move in %, ``dv`` in volatility points, ``dr`` in basis points. The
  Greeks are central differences of ``value`` at today's market (1 % spot, 1 volatility point,
  1 day, 1 bp), never at the arrival point;
* **full revaluation** after the move, and the **unexplained residual**, full minus Taylor:
  the only check that the approximation holds;
* **step by step**: revalue moving time, then the spot, then the volatility, then the rate.
  Each step is exact, they add up to the full revaluation, and the order is shown.

Large moves (stress tests) are always revalued (:func:`scenario_matrix`), never expanded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.conventions import DAYS_PER_YEAR
from spricer.market.core import Market

Valuation = Callable[[Market, float], float]  # (market, maturity in years) -> % of notional

_H_SPOT = 0.01  # 1 % of the spot
_H_VOL = 0.01  # 1 volatility point
_H_RATE = 1e-4  # 1 basis point
TERMS = ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga")
STEPS = ("Time", "Spot", "Volatility", "Rate")


@dataclass(frozen=True)
class Shock:
    """A market move.

    Attributes
    ----------
    spot_pct
        Spot move in % (relative).
    vol_pts
        Parallel move of the implied volatility surface, in points.
    days
        Calendar days elapsed.
    rate_bp
        Rate move in basis points.
    """

    spot_pct: float = 0.0
    vol_pts: float = 0.0
    days: float = 0.0
    rate_bp: float = 0.0

    def apply(self, market: Market) -> Market:
        """Market after the move (time is applied to the position's maturity)."""
        return (
            market.with_spot(market.spot * (1.0 + self.spot_pct / 100.0))
            .with_vol_shift(self.vol_pts / 100.0)
            .with_rate(market.rate + self.rate_bp * 1e-4)
        )


@dataclass(frozen=True)
class Attribution:
    """P&L of a position for a shock, explained three ways (all in % of notional).

    Attributes
    ----------
    start, end
        Values before and after the move.
    terms
        Taylor contributions of :data:`TERMS`.
    greeks
        Sensitivities used, per unit of each driver (per 1 %, per point, per day, per bp).
    steps
        Step-by-step contributions of :data:`STEPS`.
    """

    start: float
    end: float
    terms: dict[str, float]
    greeks: dict[str, float]
    steps: dict[str, float]

    @property
    def full(self) -> float:
        """Full revaluation P&L."""
        return self.end - self.start

    @property
    def first_and_second_order(self) -> float:
        """Taylor P&L without the cross and volga terms."""
        return sum(self.terms[k] for k in ("Delta", "Gamma", "Vega", "Theta", "Rho"))

    @property
    def taylor(self) -> float:
        """Taylor P&L with every term."""
        return sum(self.terms.values())

    @property
    def residual(self) -> float:
        """Unexplained: full revaluation minus Taylor."""
        return self.full - self.taylor


def greeks(value: Valuation, market: Market, maturity: float) -> dict[str, float]:
    """Sensitivities of ``value`` at today's market, per unit of each driver."""
    v0 = value(market, maturity)

    def at(spot: float = 0.0, vol: float = 0.0, rate: float = 0.0) -> float:
        m = market.with_spot(market.spot * (1.0 + spot)).with_vol_shift(vol)
        return value(m.with_rate(market.rate + rate), maturity)

    up, down = at(spot=_H_SPOT), at(spot=-_H_SPOT)
    v_up, v_down = at(vol=_H_VOL), at(vol=-_H_VOL)
    day = min(1.0 / DAYS_PER_YEAR, 0.5 * maturity)
    return {
        "Delta": (up - down) / 2.0,  # per 1 % (the bump is 1 %)
        "Gamma": up - 2.0 * v0 + down,  # per (1 %)^2
        "Vega": (v_up - v_down) / 2.0,  # per volatility point
        "Theta": (value(market, maturity - day) - v0) * (1.0 / DAYS_PER_YEAR) / day,
        "Rho": (at(rate=_H_RATE) - at(rate=-_H_RATE)) / 2.0,  # per bp
        "Vanna": (
            at(_H_SPOT, _H_VOL)
            - at(_H_SPOT, -_H_VOL)
            - at(-_H_SPOT, _H_VOL)
            + at(-_H_SPOT, -_H_VOL)
        )
        / 4.0,
        "Volga": v_up - 2.0 * v0 + v_down,
    }


def explain(value: Valuation, market: Market, maturity: float, shock: Shock) -> Attribution:
    """Explain the P&L of ``value`` for ``shock``: Taylor, full revaluation, steps."""
    years = shock.days / DAYS_PER_YEAR
    if years >= maturity:
        raise ValueError("the shock must leave some time before maturity")
    g = greeks(value, market, maturity)
    m, dv, dr = shock.spot_pct, shock.vol_pts, shock.rate_bp
    terms = {
        "Delta": g["Delta"] * m,
        "Gamma": 0.5 * g["Gamma"] * m**2,
        "Vega": g["Vega"] * dv,
        "Theta": g["Theta"] * shock.days,
        "Rho": g["Rho"] * dr,
        "Vanna": g["Vanna"] * m * dv,
        "Volga": 0.5 * g["Volga"] * dv**2,
    }
    t1 = maturity - years
    v0 = value(market, maturity)
    after_time = value(market, t1)
    spot_moved = market.with_spot(market.spot * (1.0 + m / 100.0))
    after_spot = value(spot_moved, t1)
    vol_moved = spot_moved.with_vol_shift(dv / 100.0)
    after_vol = value(vol_moved, t1)
    after_rate = value(vol_moved.with_rate(market.rate + dr * 1e-4), t1)
    steps = {
        "Time": after_time - v0,
        "Spot": after_spot - after_time,
        "Volatility": after_vol - after_spot,
        "Rate": after_rate - after_vol,
    }
    return Attribution(v0, after_rate, terms, g, steps)


def scenario_matrix(
    value: Valuation,
    market: Market,
    maturity: float,
    spot_moves_pct: Sequence[float],
    vol_moves_pts: Sequence[float],
    days: float = 0.0,
) -> FloatArray:
    """Full-revaluation P&L for each (volatility move, spot move): rows vol, columns spot."""
    t1 = maturity - days / DAYS_PER_YEAR
    if t1 <= 0.0:
        raise ValueError("the horizon must be before maturity")
    v0 = value(market, maturity)
    out = np.empty((len(vol_moves_pts), len(spot_moves_pct)))
    for i, dv in enumerate(vol_moves_pts):
        for j, m in enumerate(spot_moves_pct):
            out[i, j] = value(Shock(m, dv).apply(market), t1) - v0
    return out


def residual_order(
    value: Valuation, market: Market, maturity: float, spot_moves_pct: ArrayLike
) -> FloatArray:
    """Residual of the Taylor attribution for pure spot moves (to check its order)."""
    return np.array(
        [
            explain(value, market, maturity, Shock(spot_pct=float(m))).residual
            for m in np.asarray(spot_moves_pct)
        ]
    )
