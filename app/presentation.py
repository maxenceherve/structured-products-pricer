"""Presentation helpers shared by the pricing pages (no Streamlit calls, unit-tested).

Every amount is expressed in percent of the notional. Greeks follow the desk display:

* Delta (%): exposure to the underlying, in % of notional (a +1 % spot move changes the value
  by Delta / 100 % of notional);
* Gamma (%): change of Delta (%) for a +1 % spot move;
* Vega (%): change in value for +1 volatility point;
* Theta (%): change in value after one calendar day;
* Rho (%): change in value for a +1 basis point rate move.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from spricer._typing import FloatArray
from spricer.bricks import DigitalOption, EuropeanOption, Underlying, ZeroCoupon
from spricer.conventions import to_desk
from spricer.market import Market, OverriddenVol
from spricer.strategies import Family, Leg, Strategy

GREEK_HELP = {
    "Delta (%)": "Exposure to the underlying in % of notional: +1 % spot moves the value by "
    "Delta / 100 % of notional.",
    "Gamma (%)": "Change of Delta (%) for a +1 % spot move.",
    "Vega (%)": "Change in value, in % of notional, for +1 volatility point.",
    "Theta (%)": "Change in value, in % of notional, after one calendar day.",
    "Rho (%)": "Change in value, in % of notional, for a +1 basis point rate move.",
}

INSTRUMENTS = ("Call", "Put", "Digital call", "Digital put", "Underlying", "Zero-coupon")
CASH_INSTRUMENTS = ("Digital call", "Digital put", "Zero-coupon")
CUSTOM_COLUMNS = ("Position", "Instrument", "Strike (% of spot)", "Quantity")

DEFAULT_CUSTOM_LEGS = pd.DataFrame(
    {
        "Position": ["Long", "Short", "Short"],
        "Instrument": ["Call", "Call", "Put"],
        "Strike (% of spot)": [100.0, 120.0, 85.0],
        "Quantity": [1.0, 1.0, 1.0],
    }
)


# --- Names --------------------------------------------------------------------------------


def _noun(leg: Leg) -> str:
    brick = leg.brick
    if isinstance(brick, EuropeanOption):
        return brick.kind
    if isinstance(brick, DigitalOption):
        return f"digital {brick.kind}"
    if isinstance(brick, Underlying):
        return "underlying"
    if isinstance(brick, ZeroCoupon):
        return "zero-coupon"
    return brick.label


def leg_name(leg: Leg) -> str:
    """'Long call', 'Short 2 calls', 'Long digital call'... (digital quantities are cash)."""
    side = "Long" if leg.quantity >= 0 else "Short"
    noun = _noun(leg)
    size = abs(leg.quantity)
    if isinstance(leg.brick, DigitalOption | ZeroCoupon) or math.isclose(size, 1.0):
        return f"{side} {noun}"
    plural = noun if noun == "underlying" else f"{noun}s"
    return f"{side} {size:g} {plural}"


def _legs_at(strategy: Strategy, strike: float) -> list[Leg]:
    return [
        leg
        for leg in strategy.legs
        if leg.brick.strike == strike and not isinstance(leg.brick, Underlying | ZeroCoupon)
    ]


def _roles(legs: list[Leg]) -> str:
    names: list[str] = []
    for leg in legs:
        name = leg_name(leg)
        if name not in names:
            names.append(name)
    return " + ".join(names)


def strike_levels(strategy: Strategy, spot: float) -> list[tuple[float, str]]:
    """Distinct strikes with the legs struck there, e.g. 'Short call strike 110%'."""
    return [
        (strike, f"{_roles(_legs_at(strategy, strike))} strike {100.0 * strike / spot:g}%")
        for strike in strategy.strikes
    ]


def first_leg_index(strategy: Strategy, strike: float) -> int:
    """Index of the first leg struck at ``strike`` (to colour the strike line like its leg)."""
    return next(i for i, leg in enumerate(strategy.legs) if leg.brick.strike == strike)


# --- Market ----------------------------------------------------------------------------------


def smile_vols(strategy: Strategy, market: Market) -> pd.DataFrame:
    """One row per strike of the strategy, with the volatility read on the smile (editable)."""
    strikes = list(strategy.strikes)
    return pd.DataFrame(
        {
            "Leg": [_roles(_legs_at(strategy, k)) for k in strikes],
            "Strike (% of spot)": [100.0 * k / market.spot for k in strikes],
            "Volatility (%)": [
                100.0 * float(market.vol.vol(k, strategy.maturity)) for k in strikes
            ],
        }
    )


def changed_vols(edited: pd.DataFrame, smile: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``edited`` whose volatility differs from the smile's, with the smile's value."""
    typed = edited["Volatility (%)"].to_numpy(dtype=float)
    base = smile["Volatility (%)"].to_numpy(dtype=float)
    mask = ~np.isclose(typed, base, rtol=0.0, atol=1e-9)
    out = pd.DataFrame(edited[["Leg", "Strike (% of spot)", "Volatility (%)"]][mask])
    out["Smile (%)"] = base[mask]
    return out


def with_vols(market: Market, edited: pd.DataFrame, smile: pd.DataFrame) -> Market:
    """Market where the strikes whose volatility was edited use the typed value."""
    typed = edited["Volatility (%)"].to_numpy(dtype=float)
    if np.isnan(typed).any() or (typed <= 0).any():
        raise ValueError("Every volatility must be a positive number.")
    changed = changed_vols(edited, smile)
    strikes = changed["Strike (% of spot)"].to_numpy(dtype=float) / 100.0 * market.spot
    overrides = tuple(
        (float(k), float(v) / 100.0)
        for k, v in zip(strikes, changed["Volatility (%)"].to_numpy(dtype=float), strict=True)
    )
    return Market(
        spot=market.spot,
        rate=market.rate,
        dividend_yield=market.dividend_yield,
        vol=OverriddenVol(market.vol, overrides),
    )


# --- Custom strategy -------------------------------------------------------------------------


def custom_strategy(legs: pd.DataFrame, maturity: float, spot: float) -> Strategy:
    """Strategy from the custom legs table.

    Quantities are numbers of units for options and the underlying, and cash amounts in % of
    spot for digitals and zero-coupons.
    """
    rows = legs.dropna(how="all")
    if rows.empty:
        raise ValueError("Add at least one leg.")
    built = []
    for position, instrument, strike_pct, quantity in rows[list(CUSTOM_COLUMNS)].itertuples(
        index=False
    ):
        if position not in ("Long", "Short") or instrument not in INSTRUMENTS:
            raise ValueError("Every leg needs a position and an instrument.")
        if pd.isna(quantity) or quantity <= 0:
            raise ValueError("Quantities must be positive (use Short to sell).")
        needs_strike = instrument not in ("Underlying", "Zero-coupon")
        if needs_strike and (pd.isna(strike_pct) or strike_pct <= 0):
            raise ValueError(f"The {str(instrument).lower()} leg needs a positive strike.")
        strike = float(strike_pct) / 100.0 * spot if needs_strike else 0.0
        size = float(quantity) * (spot / 100.0 if instrument in CASH_INSTRUMENTS else 1.0)
        sign = 1.0 if position == "Long" else -1.0
        built.append(Leg(_brick(str(instrument), strike, maturity), sign * size))
    return Strategy("Custom", Family.CUSTOM, tuple(built))


def _brick(
    instrument: str, strike: float, maturity: float
) -> EuropeanOption | DigitalOption | Underlying | ZeroCoupon:
    if instrument in ("Call", "Put"):
        return EuropeanOption("call" if instrument == "Call" else "put", strike, maturity)
    if instrument in ("Digital call", "Digital put"):
        return DigitalOption("call" if instrument == "Digital call" else "put", strike, maturity)
    if instrument == "Underlying":
        return Underlying(maturity)
    return ZeroCoupon(maturity)


# --- Results ---------------------------------------------------------------------------------


def display_greeks(strategy: Strategy | Leg, market: Market) -> dict[str, float]:
    """Greeks of a strategy or a single leg in the display units of this module."""
    if isinstance(strategy, Leg):
        raw = strategy.brick.greeks(market) * strategy.quantity
    else:
        raw = strategy.greeks(market)
    desk = to_desk(raw, spot=market.spot, notional=market.spot)
    return {
        "Delta (%)": 100.0 * float(desk.delta),
        "Gamma (%)": 100.0 * float(desk.gamma),
        "Vega (%)": float(desk.vega),
        "Theta (%)": float(desk.theta),
        "Rho (%)": float(desk.rho),
    }


def leg_description(leg: Leg, market: Market) -> str:
    """Leg name with its terms, e.g. 'Long call · K 100% · σ 18.70%'."""
    parts = [leg_name(leg)]
    brick = leg.brick
    if brick.strike is not None:
        parts.append(f"K {100.0 * brick.strike / market.spot:g}%")
    if isinstance(brick, DigitalOption | ZeroCoupon):
        parts.append(f"pays {100.0 * abs(leg.quantity) / market.spot:g}%")
    vol = brick.implied_vol(market)
    if vol is not None:
        parts.append(f"σ {100.0 * vol:.2f}%")
    return " · ".join(parts)


def decomposition_table(strategy: Strategy, market: Market) -> pd.DataFrame:
    """One row per leg and a total row, in % of notional (one unit of underlying)."""
    rows = []
    for leg, valuation in zip(strategy.legs, strategy.decompose(market), strict=True):
        rows.append(
            {
                "Leg": leg_description(leg, market),
                "Price (%)": 100.0 * valuation.value / market.spot,
                "of which skew (%)": 100.0 * valuation.skew_adjustment / market.spot,
                **display_greeks(leg, market),
            }
        )
    table = pd.DataFrame(rows)
    if len(rows) > 1:
        total: dict[str, object] = {"Leg": "Total"}
        for column in ("Price (%)", "of which skew (%)", *GREEK_HELP):
            total[column] = float(table[column].sum())
        table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    if not np.any(table["of which skew (%)"].to_numpy(dtype=float)):
        table = table.drop(columns="of which skew (%)")
    return table


def breakevens(x: FloatArray, y: FloatArray) -> list[float]:
    """Points where a piecewise-linear curve crosses zero (linear interpolation)."""
    out = []
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if a == 0.0 and (i == 0 or y[i - 1] != 0.0):
            out.append(float(x[i]))
        elif a * b < 0.0:
            out.append(float(x[i] - a * (x[i + 1] - x[i]) / (b - a)))
    return out


def signature(*parts: object) -> str:
    """Stable fingerprint of the inputs, used to tell whether the last pricing is stale."""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, pd.DataFrame):
            chunks.append(part.to_json())
        elif isinstance(part, Sequence) and not isinstance(part, str):
            chunks.append(repr(tuple(part)))
        else:
            chunks.append(repr(part))
    return "|".join(chunks)


# --- Styling -----------------------------------------------------------------------------------

POSITIVE_COLOR = "#1a9850"
NEGATIVE_COLOR = "#d73027"
TOTAL_STYLE = (
    "font-weight: bold; font-size: 1.08em; background-color: rgba(31, 78, 121, 0.16); "
    "border-top: 2px solid #1f4e79"
)

FAMILY_BLURB = {
    "Single option": "1 leg: a call or a put on its own.",
    "Spreads": "2 legs (box spread: 4): same type, two strikes. Cheaper than the single "
    "option, with gain and loss both capped.",
    "Volatility plays": "2 legs: a call and a put bought together. You bet on how far the "
    "spot moves, not on its direction.",
    "Butterflies & condors": "3 to 4 legs: you bet that the spot ends in a zone. The price "
    "reads the density the market implies there.",
    "With the underlying": "2 to 3 legs: the share held together with options, for hedging "
    "or yield enhancement.",
    "Digitals": "1 or 2 legs: pays a fixed amount if the spot ends above or below a level.",
    "Custom": "Build your own: add as many legs as you like.",
}


PRICE_COLUMNS = ("Price (%)", "Closed form (%)", "Monte Carlo (%)")


def _sign_color(value: object) -> str:
    if not isinstance(value, int | float) or not math.isfinite(value) or abs(value) < 5e-5:
        return ""
    color = POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR
    return f"color: {color}; font-weight: 600"


def _side_color(value: object) -> str:
    text = str(value)
    if text.startswith("Long"):
        return f"color: {POSITIVE_COLOR}; font-weight: 600"
    if text.startswith("Short"):
        return f"color: {NEGATIVE_COLOR}; font-weight: 600"
    return ""


def _total_row_style(row: pd.Series) -> list[str]:
    return [TOTAL_STYLE if row["Leg"] == "Total" else ""] * len(row)


def style_decomposition(table: pd.DataFrame, formats: dict[str, str]) -> Styler:
    """Table styler: positives green, negatives red, long/short coloured, total highlighted."""
    numeric = pd.Index(
        [c for c in table.columns if c in (*PRICE_COLUMNS, "of which skew (%)", *GREEK_HELP)]
    )
    styler = table.style.format(formats, na_rep="").hide(axis="index")  # type: ignore[arg-type]
    return (
        styler.map(_sign_color, subset=numeric)
        .map(_side_color, subset=["Leg"])
        .apply(_total_row_style, axis=1)
    )
