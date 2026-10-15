"""Presentation helpers for structured notes (no Streamlit calls, unit-tested).

Amounts are in % of the notional, levels in % of the underlying's initial level. Greeks use the
same desk display as the vanilla page (see :mod:`presentation`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from presentation import GREEK_HELP
from spricer.bricks import BarrierOption, DigitalOption, OneTouch, ZeroCoupon
from spricer.conventions import Greeks, to_desk
from spricer.market import Market, OverriddenVol
from spricer.mc import NoteMonteCarlo
from spricer.monitoring import Monitoring
from spricer.products.notes import NOTIONAL, LevelKind, Note, NoteLeg

STRIKE_COLUMN = "Strike (% of initial)"
VOL_COLUMN = "Volatility (%)"


def performance_pct(note: Note, market: Market) -> float:
    """Current level of the underlying in % of its initial level."""
    return 100.0 * market.spot / note.initial


def _level_name(note: Note, strike: float) -> str:
    pct = 100.0 * strike / note.initial
    names = [
        level.name
        for level in note.levels
        if level.kind is LevelKind.PERFORMANCE and math.isclose(level.value_pct, pct)
    ]
    if names:
        return " / ".join(names)
    roles = [leg.role for leg in note.legs if leg.brick.strike == strike]
    return roles[0].split(":")[0] if roles else f"Strike {pct:g}%"


def smile_vols(note: Note, market: Market) -> pd.DataFrame:
    """One row per strike of the note, with the volatility read on the smile (editable)."""
    strikes = list(note.strikes)
    return pd.DataFrame(
        {
            "Level": [_level_name(note, k) for k in strikes],
            STRIKE_COLUMN: [100.0 * k / note.initial for k in strikes],
            "% of spot": [100.0 * k / market.spot for k in strikes],
            VOL_COLUMN: [100.0 * float(market.vol.vol(k, note.maturity)) for k in strikes],
        }
    )


def with_vols(market: Market, note: Note, edited: pd.DataFrame, smile: pd.DataFrame) -> Market:
    """Market where the strikes whose volatility was edited use the typed value."""
    typed = edited[VOL_COLUMN].to_numpy(dtype=float)
    if np.isnan(typed).any() or (typed <= 0).any():
        raise ValueError("Every volatility must be a positive number.")
    base = smile[VOL_COLUMN].to_numpy(dtype=float)
    strikes = edited[STRIKE_COLUMN].to_numpy(dtype=float) / 100.0 * note.initial
    overrides = tuple(
        (float(k), float(v) / 100.0)
        for k, v, b in zip(strikes, typed, base, strict=True)
        if not math.isclose(v, b, rel_tol=0.0, abs_tol=1e-9)
    )
    return Market(
        market.spot, market.rate, market.dividend_yield, OverriddenVol(market.vol, overrides)
    )


def display_greeks(
    raw_delta: float,
    raw_gamma: float,
    raw_vega: float,
    raw_theta: float,
    raw_rho: float,
    spot: float,
) -> dict[str, float]:
    """Raw Greeks of a value in % of notional, in the desk display units."""
    g = Greeks(*(np.asarray(x) for x in (raw_delta, raw_gamma, raw_vega, raw_theta, raw_rho)))
    desk = to_desk(g, spot=spot, notional=NOTIONAL)
    return {
        "Delta (%)": 100.0 * float(desk.delta),
        "Gamma (%)": 100.0 * float(desk.gamma),
        "Vega (%)": float(desk.vega),
        "Theta (%)": float(desk.theta),
        "Rho (%)": float(desk.rho),
    }


def leg_description(note: Note, leg: NoteLeg, market: Market) -> str:
    """Role with the brick's terms, e.g. 'Short put down-and-in · K 90% · H 90% · σ 21.40%'.

    Levels are in % of the initial level; amounts paid in % of notional; a participation other
    than 100 % is shown next to the role.
    """
    brick = leg.brick
    parts = [leg.role]
    if isinstance(brick, ZeroCoupon | DigitalOption | OneTouch):
        if brick.strike is not None:
            parts.append(f"K {100.0 * brick.strike / note.initial:g}%")
        if isinstance(brick, OneTouch):
            parts.append(f"H {100.0 * brick.barrier / note.initial:g}%")
        parts.append(f"pays {abs(leg.quantity):g}%")
    else:
        participation = 100.0 * abs(leg.quantity) * note.initial / NOTIONAL
        if not math.isclose(participation, 100.0):
            parts.append(f"× {participation:g}%")
        if brick.strike is not None:
            parts.append(f"K {100.0 * brick.strike / note.initial:g}%")
    if isinstance(brick, BarrierOption):
        parts.append(f"H {100.0 * brick.barrier / note.initial:g}%")
        if brick.monitoring is Monitoring.EUROPEAN:
            parts.append(
                f"σ {100.0 * brick.implied_vol(market):.2f}% / "
                f"{100.0 * brick.barrier_vol(market):.2f}%"
            )
        else:
            parts.append(f"σ {100.0 * brick.implied_vol(market):.2f}% (one vol, closed form)")
    else:
        vol = brick.implied_vol(market)
        if vol is not None:
            parts.append(f"σ {100.0 * vol:.2f}%")
    return " · ".join(parts)


def decomposition_table(note: Note, market: Market, credit_spread: float) -> pd.DataFrame:
    """Bond part, then every brick, then the total; everything in % of notional."""
    df = note.discount_spread(credit_spread)
    values = note.leg_values(market, credit_spread)
    rows = []
    for leg, value, g in zip(
        note.legs, values, note.leg_greeks(market, credit_spread), strict=True
    ):
        brick = leg.brick
        skew = (
            leg.quantity * float(brick.skew_adjustment(market)) * df
            if isinstance(brick, DigitalOption | BarrierOption | OneTouch)
            else 0.0
        )
        rows.append(
            {
                "Leg": leg_description(note, leg, market),
                "Price (%)": value,
                "of which skew (%)": skew,
                **display_greeks(
                    float(g.delta),
                    float(g.gamma),
                    float(g.vega),
                    float(g.theta),
                    float(g.rho),
                    market.spot,
                ),
            }
        )
    table = pd.DataFrame(rows)
    total: dict[str, object] = {"Leg": "Total"}
    for column in ("Price (%)", "of which skew (%)", *GREEK_HELP):
        total[column] = float(table[column].sum())
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    if not np.any(table["of which skew (%)"].to_numpy(dtype=float)):
        table = table.drop(columns="of which skew (%)")
    return table


def bond_value(note: Note, market: Market, credit_spread: float) -> float:
    """Value of the bond part (the zero-coupon legs), in % of notional."""
    return sum(
        value
        for leg, value in zip(note.legs, note.leg_values(market, credit_spread), strict=True)
        if isinstance(leg.brick, ZeroCoupon)
    )


MC_GAP_COLUMN = "Gap (std errors)"


def monte_carlo_table(
    note: Note, market: Market, credit_spread: float, mc: NoteMonteCarlo
) -> pd.DataFrame:
    """Closed form against Monte Carlo, leg by leg then in total, in % of notional."""
    rows = []
    pairs = [
        (leg.role, closed, est)
        for leg, closed, est in zip(
            note.legs, note.leg_values(market, credit_spread), mc.legs, strict=True
        )
    ]
    pairs.append(("Total", float(note.value(market, credit_spread)), mc.total))
    for name, closed, est in pairs:
        # a leg without randomness (the zero-coupon) has no error to measure the gap in
        gap = (est.value - closed) / est.std_error if est.std_error > 1e-9 else math.nan
        rows.append(
            {
                "Leg": name,
                "Closed form (%)": closed,
                "Monte Carlo (%)": est.value,
                "± 1 std error (%)": est.std_error,
                MC_GAP_COLUMN: gap,
            }
        )
    return pd.DataFrame(rows)
