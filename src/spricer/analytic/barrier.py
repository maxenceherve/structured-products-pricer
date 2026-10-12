"""Closed forms for barrier options monitored continuously, and their discrete correction.

Standard barrier options (Merton 1973, Reiner and Rubinstein 1991), in the notation of Haug,
*The Complete Guide to Option Pricing Formulas*, ch. 4 (the same results as Wilmott, *Paul
Wilmott Introduces Quantitative Finance*, ch. 26). With ``b = r - q``, ``v = sigma sqrt(T)``,
``mu = (b - sigma^2/2) / sigma^2``, ``phi = +1`` for a call and ``-1`` for a put, ``eta = +1``
for a down barrier and ``-1`` for an up barrier:

    A = phi S e^{(b-r)T} N(phi x1) - phi K e^{-rT} N(phi x1 - phi v)
    B = phi S e^{(b-r)T} N(phi x2) - phi K e^{-rT} N(phi x2 - phi v)
    C = phi S e^{(b-r)T} (H/S)^{2(mu+1)} N(eta y1) - phi K e^{-rT} (H/S)^{2 mu} N(eta y1 - eta v)
    D = phi S e^{(b-r)T} (H/S)^{2(mu+1)} N(eta y2) - phi K e^{-rT} (H/S)^{2 mu} N(eta y2 - eta v)

with ``x1 = ln(S/K)/v + (1+mu) v``, ``x2 = ln(S/H)/v + (1+mu) v``,
``y1 = ln(H^2/(S K))/v + (1+mu) v``, ``y2 = ln(H/S)/v + (1+mu) v``. Each of the eight options is
a combination of these terms, depending on whether the strike is above or below the barrier.
Rebates are not part of these options: a rebate paid at maturity when the barrier was touched
is a one-touch, priced by :func:`one_touch_price`.

If the spot is already beyond the barrier, the barrier is touched: an out option is worth zero
and an in option is the vanilla. The history before today is not tracked.

Discrete monitoring at ``n`` equally spaced dates is priced with the continuous formula and a
barrier shifted away from the spot by ``exp(beta sigma sqrt(T/n))``, ``beta = 0.5826``
(Broadie, Glasserman and Kou 1997). The correction is accurate when the barrier is not too
close to the spot and ``n`` is not too small.

These formulas use one volatility for the whole path: they are Black-Scholes prices, not
smile-consistent ones (that needs a local or stochastic volatility model).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.analytic import black_scholes as bs
from spricer.analytic.black_scholes import OptionKind, norm_cdf

BarrierType = Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
Direction = Literal["up", "down"]

BGK_BETA = 0.5826  # -zeta(1/2) / sqrt(2 pi), Broadie, Glasserman and Kou (1997)
# Below about weekly observations the correction drifts away from a discretely monitored
# Monte Carlo (tests: ~5 % of the price with 12 dates, within noise with 52 and 252).
BGK_MIN_OBSERVATIONS = 50


def bgk_barrier(barrier: float, direction: Direction, vol: float, maturity: float, n: int) -> float:
    """Continuous-equivalent barrier for ``n`` equally spaced observations (BGK shift).

    An up barrier moves up and a down barrier moves down: observing less often makes the
    barrier harder to touch.
    """
    if n < 1:
        raise ValueError("the number of observations must be at least 1")
    shift = BGK_BETA * vol * math.sqrt(maturity / n)
    return barrier * math.exp(shift if direction == "up" else -shift)


def bgk_warning(observations: int) -> str | None:
    """Message when the BGK correction is not reliable for this number of dates, else None."""
    if observations >= BGK_MIN_OBSERVATIONS:
        return None
    return (
        f"With {observations} observation dates the BGK correction is approximate (a few % of "
        "the price); a Monte Carlo on the actual dates is more accurate."
    )


def barrier_price(
    kind: OptionKind,
    barrier_type: BarrierType,
    *,
    spot: ArrayLike,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> FloatArray:
    """Price of a continuously monitored barrier option, without rebate (vectorised in spot)."""
    s = np.asarray(spot, dtype=np.float64)
    if np.any(s <= 0.0) or strike <= 0.0 or barrier <= 0.0:
        raise ValueError("spot, strike and barrier must be positive")
    vanilla = bs.vanilla_price(
        kind,
        spot=s,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        vol=vol,
    )
    up = barrier_type.startswith("up")
    knock_in = barrier_type.endswith("in")
    touched = (s >= barrier) if up else (s <= barrier)
    if maturity <= 0.0 or vol <= 0.0:
        # no time or no randomness left: the barrier is touched only if it already is
        alive = touched if knock_in else ~touched
        return np.where(alive, vanilla, 0.0)

    s_safe = np.where(touched, barrier * (0.5 if up else 2.0), s)  # any inside point
    phi = 1.0 if kind == "call" else -1.0
    eta = -1.0 if up else 1.0
    b = rate - dividend_yield
    v = vol * math.sqrt(maturity)
    mu = (b - 0.5 * vol * vol) / (vol * vol)
    fwd = s_safe * math.exp((b - rate) * maturity)
    pv_k = strike * math.exp(-rate * maturity)
    ratio = barrier / s_safe
    x1 = np.log(s_safe / strike) / v + (1.0 + mu) * v
    x2 = np.log(s_safe / barrier) / v + (1.0 + mu) * v
    y1 = np.log(barrier * barrier / (s_safe * strike)) / v + (1.0 + mu) * v
    y2 = np.log(barrier / s_safe) / v + (1.0 + mu) * v
    a = phi * fwd * norm_cdf(phi * x1) - phi * pv_k * norm_cdf(phi * x1 - phi * v)
    b_ = phi * fwd * norm_cdf(phi * x2) - phi * pv_k * norm_cdf(phi * x2 - phi * v)
    c = phi * fwd * ratio ** (2 * (mu + 1)) * norm_cdf(eta * y1) - phi * pv_k * ratio ** (
        2 * mu
    ) * norm_cdf(eta * y1 - eta * v)
    d = phi * fwd * ratio ** (2 * (mu + 1)) * norm_cdf(eta * y2) - phi * pv_k * ratio ** (
        2 * mu
    ) * norm_cdf(eta * y2 - eta * v)

    above = strike > barrier
    if kind == "call" and not up:  # down-and-in call
        price_in = np.where(above, c, a - b_ + d)
    elif kind == "call":  # up-and-in call
        price_in = np.where(above, a, b_ - c + d)
    elif not up:  # down-and-in put
        price_in = np.where(above, b_ - c + d, a)
    else:  # up-and-in put
        price_in = np.where(above, a - b_ + d, c)

    if knock_in:
        return np.where(touched, vanilla, price_in)
    return np.where(touched, 0.0, vanilla - price_in)


def one_touch_price(
    direction: Direction,
    *,
    spot: ArrayLike,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> FloatArray:
    """Value of 1 paid at maturity if the barrier is touched before (continuous monitoring).

    This is the discount factor minus Haug's rebate term ``E`` (a rebate paid at maturity when
    the barrier was never touched): ``e^{-rT} [1 - N(eta x2 - eta v) + (H/S)^{2 mu}
    N(eta y2 - eta v)]``.
    """
    s = np.asarray(spot, dtype=np.float64)
    if np.any(s <= 0.0) or barrier <= 0.0:
        raise ValueError("spot and barrier must be positive")
    df = math.exp(-rate * maturity)
    up = direction == "up"
    touched = (s >= barrier) if up else (s <= barrier)
    if maturity <= 0.0 or vol <= 0.0:
        return np.where(touched, df, 0.0)
    s_safe = np.where(touched, barrier * (0.5 if up else 2.0), s)
    eta = -1.0 if up else 1.0
    v = vol * math.sqrt(maturity)
    mu = (rate - dividend_yield - 0.5 * vol * vol) / (vol * vol)
    x2 = np.log(s_safe / barrier) / v + (1.0 + mu) * v
    y2 = np.log(barrier / s_safe) / v + (1.0 + mu) * v
    no_touch = norm_cdf(eta * x2 - eta * v) - (barrier / s_safe) ** (2 * mu) * norm_cdf(
        eta * y2 - eta * v
    )
    return np.where(touched, df, df * (1.0 - no_touch))
