"""Black-Scholes-Merton closed forms: European calls and puts, cash-or-nothing digitals.

Model: lognormal spot with a continuous dividend yield ``q`` and a flat, continuously
compounded rate ``r`` (Black and Scholes 1973, Merton 1973). Formulas follow Hull, *Options,
Futures, and Other Derivatives*, ch. 15 and 19, and Haug, *The Complete Guide to Option Pricing
Formulas*, ch. 1 and 2.

Every function is vectorised: its arguments are broadcast against each other with NumPy rules.
Degenerate inputs are exact limits rather than errors: zero spot, zero strike, zero volatility
and zero time to maturity are all valid. Negative spot, strike, maturity or volatility raise
``ValueError``.

Digitals pay one unit of currency. The call pays when ``S_T >= K`` and the put when
``S_T < K``, so that call + put = discount factor in every case. With a volatility smile, a
digital is not the Black-Scholes formula at the volatility of its strike: it is minus the
derivative of the call price with respect to the strike *along the smile*, which adds the term
``-vega * dsigma/dK``. Pass that slope as ``vol_slope`` (zero for a flat volatility).

Greeks are raw derivatives, see :class:`spricer.conventions.Greeks`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from spricer._typing import FloatArray
from spricer.conventions import Greeks

OptionKind = Literal["call", "put"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: ArrayLike) -> FloatArray:
    """Standard normal cumulative distribution function."""
    return np.asarray(ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def norm_pdf(x: ArrayLike) -> FloatArray:
    """Standard normal probability density function."""
    a = np.asarray(x, dtype=np.float64)
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * a * a), dtype=np.float64)


def _sign(kind: OptionKind) -> float:
    if kind == "call":
        return 1.0
    if kind == "put":
        return -1.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


@dataclass(frozen=True)
class _Inputs:
    """Broadcast inputs and the quantities shared by every formula.

    ``regular`` flags the points with a strictly positive spot, strike and total variance,
    where the lognormal formulas apply. Elsewhere the price is the deterministic forward
    intrinsic value. The ``*_safe`` arrays replace non-regular entries by harmless values so
    that no division by zero happens; results at those entries are discarded.
    """

    spot: FloatArray
    strike: FloatArray
    maturity: FloatArray
    rate: FloatArray
    dividend_yield: FloatArray
    vol: FloatArray
    regular: NDArray[np.bool_]
    spot_safe: FloatArray
    maturity_safe: FloatArray
    vol_safe: FloatArray
    sqrt_t: FloatArray
    std_safe: FloatArray
    d1: FloatArray
    d2: FloatArray
    discount: FloatArray
    dividend_discount: FloatArray


def _prepare(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
) -> _Inputs:
    raw = {
        "spot": np.asarray(spot, dtype=np.float64),
        "strike": np.asarray(strike, dtype=np.float64),
        "maturity": np.asarray(maturity, dtype=np.float64),
        "vol": np.asarray(vol, dtype=np.float64),
    }
    for name, arr in raw.items():
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
    s, k, t, r, q, v = np.broadcast_arrays(
        raw["spot"],
        raw["strike"],
        raw["maturity"],
        np.asarray(rate, dtype=np.float64),
        np.asarray(dividend_yield, dtype=np.float64),
        raw["vol"],
    )
    sqrt_t = np.sqrt(t)
    std = v * sqrt_t
    regular = (std > 0.0) & (s > 0.0) & (k > 0.0)
    s_safe = np.where(regular, s, 1.0)
    k_safe = np.where(regular, k, 1.0)
    t_safe = np.where(regular, t, 1.0)
    v_safe = np.where(regular, v, 1.0)
    std_safe = np.where(regular, std, 1.0)
    d1 = (np.log(s_safe / k_safe) + (r - q) * t) / std_safe + 0.5 * std_safe
    return _Inputs(
        spot=s,
        strike=k,
        maturity=t,
        rate=r,
        dividend_yield=q,
        vol=v,
        regular=regular,
        spot_safe=s_safe,
        maturity_safe=t_safe,
        vol_safe=v_safe,
        sqrt_t=sqrt_t,
        std_safe=std_safe,
        d1=d1,
        d2=d1 - std_safe,
        discount=np.exp(-r * t),
        dividend_discount=np.exp(-q * t),
    )


# --- European calls and puts ------------------------------------------------------------


def vanilla_price(
    kind: OptionKind,
    *,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
) -> FloatArray:
    """Price a European call or put (Black-Scholes-Merton).

    Parameters
    ----------
    kind
        ``"call"`` or ``"put"``.
    spot, strike
        Spot and strike, in the same currency unit.
    maturity
        Time to maturity in years.
    rate
        Continuously compounded risk-free rate (decimal).
    dividend_yield
        Continuous dividend yield (decimal).
    vol
        Implied volatility (decimal).

    Returns
    -------
    FloatArray
        Option value, broadcast over the inputs.
    """
    sign = _sign(kind)
    x = _prepare(spot, strike, maturity, rate, dividend_yield, vol)
    pv_spot = x.spot * x.dividend_discount
    pv_strike = x.strike * x.discount
    lognormal = sign * (pv_spot * norm_cdf(sign * x.d1) - pv_strike * norm_cdf(sign * x.d2))
    intrinsic = np.maximum(sign * (pv_spot - pv_strike), 0.0)
    return np.where(x.regular, lognormal, intrinsic)


def vanilla_greeks(
    kind: OptionKind,
    *,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
) -> Greeks:
    """Raw Greeks of a European call or put (parameters as in :func:`vanilla_price`).

    Returns
    -------
    Greeks
        ``delta``, ``gamma``, ``vega``, ``theta = -dV/dT`` and ``rho``. Where the total
        variance is zero, these are the derivatives of the deterministic forward intrinsic
        value (zero gamma and vega).
    """
    sign = _sign(kind)
    x = _prepare(spot, strike, maturity, rate, dividend_yield, vol)
    s, k, t, r, q = x.spot, x.strike, x.maturity, x.rate, x.dividend_yield
    dq, df = x.dividend_discount, x.discount
    n1 = norm_pdf(x.d1)
    cdf1 = norm_cdf(sign * x.d1)
    cdf2 = norm_cdf(sign * x.d2)

    delta = sign * dq * cdf1
    gamma = dq * n1 / (x.spot_safe * x.std_safe)
    vega = s * dq * n1 * x.sqrt_t
    theta = -s * dq * n1 * x.vol_safe / (2.0 * np.sqrt(x.maturity_safe)) + sign * (
        q * s * dq * cdf1 - r * k * df * cdf2
    )
    rho = sign * k * t * df * cdf2

    in_the_money = sign * (s * dq - k * df) > 0.0
    zero = np.zeros_like(s)
    return Greeks(
        delta=np.where(x.regular, delta, np.where(in_the_money, sign * dq, 0.0)),
        gamma=np.where(x.regular, gamma, zero),
        vega=np.where(x.regular, vega, zero),
        theta=np.where(
            x.regular, theta, np.where(in_the_money, sign * (q * s * dq - r * k * df), 0.0)
        ),
        rho=np.where(x.regular, rho, np.where(in_the_money, sign * k * t * df, 0.0)),
    )


# --- Cash-or-nothing digitals -------------------------------------------------------------


def digital_price(
    kind: OptionKind,
    *,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
    vol_slope: ArrayLike = 0.0,
) -> FloatArray:
    """Price a cash-or-nothing digital paying one unit of currency.

    Parameters
    ----------
    kind
        ``"call"`` (pays if ``S_T >= K``) or ``"put"`` (pays if ``S_T < K``).
    spot, strike, maturity, rate, dividend_yield, vol
        As in :func:`vanilla_price`; ``vol`` is the implied volatility at the strike.
    vol_slope
        Slope of the implied volatility smile at the strike, ``dsigma/dK`` (per unit of
        strike). Zero gives the flat-volatility Black-Scholes digital.

    Returns
    -------
    FloatArray
        Digital value: ``e^{-rT} N(d2) - vega * dsigma/dK`` for the call, discount factor
        minus the call for the put.
    """
    sign = _sign(kind)
    x = _prepare(spot, strike, maturity, rate, dividend_yield, vol)
    slope = np.asarray(vol_slope, dtype=np.float64)
    vega = x.spot * x.dividend_discount * norm_pdf(x.d1) * x.sqrt_t
    call_lognormal = x.discount * norm_cdf(x.d2) - slope * vega
    call_deterministic = x.discount * (x.spot * x.dividend_discount >= x.strike * x.discount)
    call = np.where(x.regular, call_lognormal, call_deterministic)
    return call if sign > 0 else x.discount - call


def digital_greeks(
    kind: OptionKind,
    *,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
    vol_slope: ArrayLike = 0.0,
) -> Greeks:
    """Raw Greeks of a cash-or-nothing digital (parameters as in :func:`digital_price`).

    The smile is sticky-strike: when the spot moves, the volatility and its slope at the
    strike are unchanged. ``vega`` is the sensitivity to a parallel shift of the smile and
    ``theta`` holds the implied volatility constant.

    Returns
    -------
    Greeks
        Raw sensitivities, including the derivatives of the skew term ``-vega * dsigma/dK``.
    """
    sign = _sign(kind)
    x = _prepare(spot, strike, maturity, rate, dividend_yield, vol)
    s, t, r, q = x.spot, x.maturity, x.rate, x.dividend_yield
    v, t_safe, sqrt_t = x.vol_safe, x.maturity_safe, x.sqrt_t
    d1, d2, df, dq = x.d1, x.d2, x.discount, x.dividend_discount
    slope = np.asarray(vol_slope, dtype=np.float64)
    n1, n2 = norm_pdf(d1), norm_pdf(d2)

    # Flat-volatility digital call: V0 = e^{-rT} N(d2) (Haug, ch. 2).
    v0 = df * norm_cdf(d2)
    delta = df * n2 / (x.spot_safe * x.std_safe)
    gamma = -df * n2 * d1 / (x.spot_safe**2 * x.std_safe**2)
    vega = -df * n2 * d1 / v
    theta = r * v0 + df * n2 * (d1 / (2.0 * t_safe) - (r - q) / (v * np.sqrt(t_safe)))
    rho = -t * v0 + df * n2 * sqrt_t / v

    # Skew term -slope * C_vega, where C_vega = S e^{-qT} n(d1) sqrt(T) is the call vega.
    call_vega = s * dq * n1 * sqrt_t
    vanna = -dq * n1 * d2 / v
    dvanna_dspot = dq * n1 * (d1 * d2 - 1.0) / (x.spot_safe * v * x.std_safe)
    volga = call_vega * d1 * d2 / v
    dd1_dt = (r - q) / (v * np.sqrt(t_safe)) - d2 / (2.0 * t_safe)
    dvega_dt = call_vega * (1.0 / (2.0 * t_safe) - q - d1 * dd1_dt)
    dvega_dr = -call_vega * d1 * sqrt_t / v

    deterministic = df * (s * dq >= x.strike * df)
    zero = np.zeros_like(s)
    call = Greeks(
        delta=np.where(x.regular, delta - slope * vanna, zero),
        gamma=np.where(x.regular, gamma - slope * dvanna_dspot, zero),
        vega=np.where(x.regular, vega - slope * volga, zero),
        theta=np.where(x.regular, theta + slope * dvega_dt, r * deterministic),
        rho=np.where(x.regular, rho - slope * dvega_dr, -t * deterministic),
    )
    if sign > 0:
        return call
    zero_coupon = Greeks(delta=zero, gamma=zero, vega=zero, theta=r * df, rho=-t * df)
    return zero_coupon + (-call)
