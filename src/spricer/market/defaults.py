"""Synthetic default market, shared by the app, the tests and the notebooks.

Nothing here is market data. The volatility grid is sampled from an SSVI surface (Gatheral and
Jacquier 2014, power-law parameterisation with ``eta (1 + |rho|) <= 2`` and ``gamma = 1/2``, a
choice free of static arbitrage) with an upward-sloping at-the-money term structure, then
rounded to 0.1 volatility point, the way a user would type it. A test checks that the
interpolated grid is still free of butterfly and calendar arbitrage after rounding.
"""

from __future__ import annotations

import math

from spricer.market.core import Market
from spricer.market.vol_surface import InterpolatedVolSurface

DEFAULT_SPOT = 100.0
DEFAULT_RATE = 0.03
DEFAULT_DIVIDEND_YIELD = 0.02
DEFAULT_MATURITIES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_STRIKES_PCT: tuple[float, ...] = (
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    95.0,
    100.0,
    105.0,
    110.0,
    120.0,
    130.0,
    150.0,
)

_SSVI_RHO = -0.6
_SSVI_ETA = 0.8
_SSVI_GAMMA = 0.5


def _atm_vol(maturity: float) -> float:
    return 0.20 - 0.03 * math.exp(-maturity / 1.5)


def ssvi_vol(strike_pct: float, maturity: float) -> float:
    """Implied volatility of the SSVI surface used to fill the default grid (decimal).

    Parameters
    ----------
    strike_pct
        Strike in percent of the default spot.
    maturity
        Maturity in years.
    """
    theta = _atm_vol(maturity) ** 2 * maturity
    phi = _SSVI_ETA / (theta**_SSVI_GAMMA * (1.0 + theta) ** (1.0 - _SSVI_GAMMA))
    forward_pct = 100.0 * math.exp((DEFAULT_RATE - DEFAULT_DIVIDEND_YIELD) * maturity)
    k = math.log(strike_pct / forward_pct)
    rho = _SSVI_RHO
    w = 0.5 * theta * (1.0 + rho * phi * k + math.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))
    return math.sqrt(w / maturity)


DEFAULT_VOLS: tuple[tuple[float, ...], ...] = tuple(
    tuple(round(ssvi_vol(m, t), 3) for m in DEFAULT_STRIKES_PCT) for t in DEFAULT_MATURITIES
)


def default_vol_surface() -> InterpolatedVolSurface:
    """Interpolated surface built from the default grid."""
    return InterpolatedVolSurface(
        spot_ref=DEFAULT_SPOT,
        rate=DEFAULT_RATE,
        dividend_yield=DEFAULT_DIVIDEND_YIELD,
        maturities=DEFAULT_MATURITIES,
        strikes_pct=DEFAULT_STRIKES_PCT,
        vols=DEFAULT_VOLS,
    )


def default_market() -> Market:
    """Default market: spot 100, rate 3 %, dividend yield 2 %, default smile."""
    return Market(
        spot=DEFAULT_SPOT,
        rate=DEFAULT_RATE,
        dividend_yield=DEFAULT_DIVIDEND_YIELD,
        vol=default_vol_surface(),
    )
