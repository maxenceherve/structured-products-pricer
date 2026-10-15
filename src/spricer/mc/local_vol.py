"""Dupire local volatility implied by a market's volatility surface.

The local volatility is the unique diffusion coefficient ``sigma(t, S)`` under which the
model reprices every European option of the implied surface (Dupire 1994). In total implied
variance ``w(k, T) = sigma_imp^2 T`` and forward log-moneyness ``k = ln(K / F(T))`` it reads
(Gatheral, *The Volatility Surface*, 2006, eq. 1.10):

    sigma_loc^2(K, T) = (dw/dT) / g(k),
    g(k) = 1 - (k/w) w' + (1/4)(-1/4 - 1/w + k^2/w^2) w'^2 + (1/2) w'',

with ``w'`` and ``w''`` the derivatives in ``k`` at fixed ``T`` and ``dw/dT`` taken at fixed
``k``. The denominator is the Durrleman condition of the surface (positive when there is no
butterfly arbitrage) and the numerator is positive when there is no calendar arbitrage: a
surface with arbitrage has no local volatility and is refused.

The derivatives are central finite differences of the surface, so any
:class:`~spricer.market.VolSurface` works, including a parallel shift for a vega bump. A flat
surface gives back its constant volatility. Point overrides of an
:class:`~spricer.market.OverriddenVol` are invisible here: the diffusion follows the smile.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.market.core import Market
from spricer.market.vol_surface import FlatVol, ShiftedVol, VolSurface

_H_K = 1e-3  # log-moneyness step of the finite differences
_H_T = 1e-4  # maturity step (years), reduced near zero
_ARBITRAGE_TOL = 1e-10
MAX_LOCAL_VOL = 3.0  # cap (300 %) on the far wings, where the formula is ill-conditioned


def flat_sigma(surface: VolSurface) -> float | None:
    """Constant volatility of a flat surface (possibly shifted), None for a smile."""
    if isinstance(surface, FlatVol):
        return surface.sigma
    if isinstance(surface, ShiftedVol):
        base = flat_sigma(surface.base)
        return None if base is None else base + surface.shift
    return None


def _total_variance(market: Market, k: FloatArray, maturity: FloatArray) -> FloatArray:
    strike = market.forward(maturity) * np.exp(k)
    return np.asarray(market.vol.vol(strike, maturity) ** 2 * maturity, dtype=np.float64)


def local_variance_at_moneyness(market: Market, k: ArrayLike, time: ArrayLike) -> FloatArray:
    """Dupire local variance at forward log-moneyness ``k`` and ``time`` (years, positive).

    ``k`` and ``time`` broadcast together, so a whole table (times in a column, moneyness in
    rows) is computed in one call.

    Raises
    ------
    ValueError
        If the surface has calendar arbitrage (total variance decreasing in maturity) or
        butterfly arbitrage (non-positive Durrleman denominator) at these points.
    """
    kk, t = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(time, dtype=np.float64))
    if not np.all(t > 0.0):
        raise ValueError("the local volatility needs a positive time")
    h_t = np.minimum(_H_T, 0.5 * t)
    w = _total_variance(market, kk, t)
    w_up, w_down = _total_variance(market, kk + _H_K, t), _total_variance(market, kk - _H_K, t)
    dw = (w_up - w_down) / (2.0 * _H_K)
    d2w = (w_up - 2.0 * w + w_down) / _H_K**2
    dw_dt = (_total_variance(market, kk, t + h_t) - _total_variance(market, kk, t - h_t)) / (
        2.0 * h_t
    )
    g = 1.0 - kk / w * dw + 0.25 * (-0.25 - 1.0 / w + kk**2 / w**2) * dw**2 + 0.5 * d2w
    calendar = dw_dt < -_ARBITRAGE_TOL
    if np.any(calendar):
        raise ValueError(
            "the volatility surface has calendar arbitrage (total variance decreases with "
            f"maturity around t = {float(t[calendar].min()):.3g}y): no local volatility exists"
        )
    butterfly = g <= _ARBITRAGE_TOL
    if np.any(butterfly):
        raise ValueError(
            "the volatility surface has butterfly arbitrage (negative implied density around "
            f"t = {float(t[butterfly].min()):.3g}y): no local volatility exists"
        )
    return np.asarray(np.minimum(np.maximum(dw_dt, 0.0) / g, MAX_LOCAL_VOL**2))


def local_vol(market: Market, strike: ArrayLike, time: ArrayLike) -> FloatArray:
    """Dupire local volatility at absolute ``strike`` (spot level) and ``time`` (years)."""
    k = np.log(np.asarray(strike, dtype=np.float64) / market.forward(time))
    return np.sqrt(local_variance_at_moneyness(market, k, time))
