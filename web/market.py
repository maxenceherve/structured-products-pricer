"""Market data shared by every page: kept in the browser session as plain JSON."""

from __future__ import annotations

from typing import Any

import numpy as np

from spricer.market import InterpolatedVolSurface, Market
from spricer.market.defaults import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_MATURITIES,
    DEFAULT_RATE,
    DEFAULT_SPOT,
    DEFAULT_STRIKES_PCT,
    DEFAULT_VOLS,
)

MATURITY = "Maturity (y)"
MarketData = dict[str, Any]


def strike_field(strike_pct: float) -> str:
    """Column field of a grid strike."""
    return f"{strike_pct:g}%"


def default_data() -> MarketData:
    """The synthetic default market, as stored in the session."""
    return {
        "spot": DEFAULT_SPOT,
        "rate_pct": 100.0 * DEFAULT_RATE,
        "dividend_pct": 100.0 * DEFAULT_DIVIDEND_YIELD,
        "funding_bp": 0.0,
        "strikes_pct": list(DEFAULT_STRIKES_PCT),
        "rows": grid_rows(DEFAULT_MATURITIES, DEFAULT_STRIKES_PCT, DEFAULT_VOLS),
    }


def grid_rows(maturities: Any, strikes_pct: Any, vols: Any) -> list[dict[str, float]]:
    """Rows of the volatility grid in volatility points, one per maturity."""
    rows = []
    for t, row in zip(maturities, vols, strict=True):
        record = {MATURITY: float(t)}
        record |= {
            strike_field(k): round(100.0 * float(v), 2)
            for k, v in zip(strikes_pct, row, strict=True)
        }
        rows.append(record)
    return rows


def _number(value: object, what: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number") from None
    if not np.isfinite(out):
        raise ValueError(f"{what} must be a number")
    return out


def market_from_data(data: MarketData) -> Market:
    """Market of the session data.

    Raises
    ------
    ValueError
        With a message for the user when an input is missing or the grid is inconsistent.
    """
    spot = _number(data.get("spot"), "The spot")
    if spot <= 0.0:
        raise ValueError("The spot must be positive")
    rate = _number(data.get("rate_pct"), "The rate") / 100.0
    dividend = _number(data.get("dividend_pct"), "The dividend yield") / 100.0
    strikes = [float(k) for k in data.get("strikes_pct", [])]
    rows = [r for r in data.get("rows", []) if r.get(MATURITY) not in (None, "")]
    if not rows:
        raise ValueError("The volatility grid needs at least one maturity")
    rows = sorted(rows, key=lambda r: _number(r[MATURITY], "Every maturity"))
    maturities = [_number(r[MATURITY], "Every maturity") for r in rows]
    vols = [
        [_number(r.get(strike_field(k)), f"The volatility at {k:g}%") / 100.0 for k in strikes]
        for r in rows
    ]
    surface = InterpolatedVolSurface(
        spot_ref=spot,
        rate=rate,
        dividend_yield=dividend,
        maturities=maturities,
        strikes_pct=strikes,
        vols=vols,
    )
    return Market(spot=spot, rate=rate, dividend_yield=dividend, vol=surface)


def funding_bp(data: MarketData) -> float:
    """Issuer's funding spread over the risk-free rate (bp): it discounts the notes' flows."""
    try:
        value = float(data.get("funding_bp") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0
