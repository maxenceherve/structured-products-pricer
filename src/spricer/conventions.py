"""Units and conventions shared by every pricer in the package.

Sensitivities live in two layers.

* :class:`Greeks` holds the raw mathematical derivatives of a value ``V`` expressed in
  currency: ``delta = dV/dS``, ``gamma = d2V/dS2``, ``vega = dV/dsigma``,
  ``theta = -dV/dT`` (per year of remaining maturity) and ``rho = dV/dr``.
* :class:`DeskGreeks` expresses the same sensitivities the way a structuring desk reads them,
  all in percent of a notional ``N``, so that a small market move gives the P&L directly::

      dV / N [%] ~ delta * m + 0.5 * gamma * m**2 + vega * dvol + theta * days + rho * dr

  with ``m`` the spot move in percent, ``dvol`` the volatility move in points, ``days`` the
  number of calendar days elapsed and ``dr`` the rate move in basis points. This is the
  second-order Taylor expansion used for P&L attribution.

Time is measured in years on an ACT/365 Fixed basis everywhere: no business-day calendar and
no dependence on the system clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray

DAYS_PER_YEAR: Final = 365.0


def year_fraction(start: date, end: date) -> float:
    """Return the ACT/365 Fixed year fraction between two dates.

    Parameters
    ----------
    start, end
        Start and end dates. ``end`` may precede ``start``, giving a negative fraction.

    Returns
    -------
    float
        Number of calendar days divided by 365.
    """
    return (end - start).days / DAYS_PER_YEAR


def _as_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Greeks:
    """Raw sensitivities of a value expressed in currency.

    Attributes
    ----------
    delta
        ``dV/dS``.
    gamma
        ``d2V/dS2``.
    vega
        ``dV/dsigma``, for a volatility expressed as a decimal (0.20 for 20 %).
    theta
        ``-dV/dT``: change in value per year as time passes and the remaining maturity shrinks.
    rho
        ``dV/dr``, for a rate expressed as a decimal.
    """

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray

    @classmethod
    def zero(cls) -> Greeks:
        """Return Greeks that are all zero (neutral element of addition)."""
        z = np.zeros((), dtype=np.float64)
        return cls(z, z, z, z, z)

    def __add__(self, other: Greeks) -> Greeks:
        return Greeks(
            self.delta + other.delta,
            self.gamma + other.gamma,
            self.vega + other.vega,
            self.theta + other.theta,
            self.rho + other.rho,
        )

    def __mul__(self, factor: ArrayLike) -> Greeks:
        f = _as_array(factor)
        return Greeks(self.delta * f, self.gamma * f, self.vega * f, self.theta * f, self.rho * f)

    __rmul__ = __mul__

    def __neg__(self) -> Greeks:
        return self * -1.0


@dataclass(frozen=True)
class DeskGreeks:
    """Sensitivities in percent of a notional, ready for a P&L explain.

    Attributes
    ----------
    delta
        Change in value, in % of notional, for a +1 % spot move.
    gamma
        Second-order coefficient, in % of notional per (1 %)², of the Taylor expansion in
        the spot move expressed in percent.
    vega
        Change in value, in % of notional, for a +1 volatility point move.
    theta
        Change in value, in % of notional, for one calendar day elapsed.
    rho
        Change in value, in % of notional, for a +1 basis point rate move.
    """

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray


def to_desk(greeks: Greeks, *, spot: ArrayLike, notional: ArrayLike) -> DeskGreeks:
    """Convert raw Greeks into desk units (see :class:`DeskGreeks`).

    Parameters
    ----------
    greeks
        Raw sensitivities of a value in currency.
    spot
        Spot level at which the Greeks were computed.
    notional
        Notional in currency used to express everything in percent.

    Returns
    -------
    DeskGreeks
        Sensitivities in percent of ``notional``.
    """
    s = _as_array(spot)
    pct = 100.0 / _as_array(notional)
    one_pct_move = 0.01 * s
    return DeskGreeks(
        delta=greeks.delta * one_pct_move * pct,
        gamma=greeks.gamma * one_pct_move**2 * pct,
        vega=greeks.vega * 0.01 * pct,
        theta=greeks.theta / DAYS_PER_YEAR * pct,
        rho=greeks.rho * 1e-4 * pct,
    )


def percent_of_notional(value: ArrayLike, notional: ArrayLike) -> FloatArray:
    """Express a value in currency as a percentage of a notional."""
    return 100.0 * _as_array(value) / _as_array(notional)
