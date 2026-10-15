"""Market inputs shared by every page, kept in Streamlit's session state.

Streamlit forgets the state of a widget as soon as a run does not draw it, which happens when
the user switches page. Every input is therefore stored under its own key and copied back
into its widget when the page is drawn again (:func:`persistent`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import streamlit as st

from spricer.market import InterpolatedVolSurface, Market
from spricer.market.defaults import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_MATURITIES,
    DEFAULT_RATE,
    DEFAULT_SPOT,
    DEFAULT_STRIKES_PCT,
    DEFAULT_VOLS,
)

SPOT = "market_spot"
RATE_PCT = "market_rate_pct"
DIVIDEND_PCT = "market_dividend_pct"
GRID = "market_vol_grid"
MATURITY_COLUMN = "Maturity (y)"

T = TypeVar("T")
V = TypeVar("V")


def strike_column(strike_pct: float) -> str:
    """Column label of a grid strike."""
    return f"{strike_pct:g}%"


def grid_strikes(grid: pd.DataFrame) -> list[float]:
    """Strikes (% of spot) of a volatility grid, read from its column labels."""
    return [float(str(c).rstrip("%")) for c in grid.columns if c != MATURITY_COLUMN]


def default_grid() -> pd.DataFrame:
    """Default volatility grid, in volatility points: one row per maturity."""
    data: dict[str, list[float]] = {MATURITY_COLUMN: list(DEFAULT_MATURITIES)}
    for j, strike in enumerate(DEFAULT_STRIKES_PCT):
        data[strike_column(strike)] = [round(100.0 * row[j], 2) for row in DEFAULT_VOLS]
    return pd.DataFrame(data)


def init() -> None:
    """Fill the session state with the synthetic default market (first run only)."""
    st.session_state.setdefault(SPOT, DEFAULT_SPOT)
    st.session_state.setdefault(RATE_PCT, 100.0 * DEFAULT_RATE)
    st.session_state.setdefault(DIVIDEND_PCT, 100.0 * DEFAULT_DIVIDEND_YIELD)
    st.session_state.setdefault(GRID, default_grid())


def reset_market() -> None:
    """Restore the synthetic default market."""
    for key in (SPOT, RATE_PCT, DIVIDEND_PCT, GRID):
        st.session_state.pop(key, None)
        st.session_state.pop(_widget_key(key), None)
    init()


def build_market(spot: float, rate_pct: float, dividend_pct: float, grid: pd.DataFrame) -> Market:
    """Market from the user's inputs.

    Raises
    ------
    ValueError
        With a message meant for the user when the inputs are incomplete or inconsistent.
    """
    rows = grid.dropna(how="all")
    if rows.empty:
        raise ValueError("The volatility grid is empty.")
    if rows[MATURITY_COLUMN].isna().any():
        raise ValueError("Every row of the volatility grid needs a maturity.")
    rows = rows.sort_values(MATURITY_COLUMN)
    vols = rows.drop(columns=MATURITY_COLUMN).to_numpy(dtype=float)
    if np.isnan(vols).any():
        raise ValueError("The volatility grid has empty cells.")
    surface = InterpolatedVolSurface(
        spot_ref=spot,
        rate=rate_pct / 100.0,
        dividend_yield=dividend_pct / 100.0,
        maturities=rows[MATURITY_COLUMN].to_numpy(dtype=float),
        strikes_pct=grid_strikes(rows),
        vols=vols / 100.0,
    )
    return Market(
        spot=spot, rate=rate_pct / 100.0, dividend_yield=dividend_pct / 100.0, vol=surface
    )


def current_market() -> Market:
    """Market built from the session state (raises ``ValueError`` on invalid inputs)."""
    return build_market(
        float(st.session_state[SPOT]),
        float(st.session_state[RATE_PCT]),
        float(st.session_state[DIVIDEND_PCT]),
        st.session_state[GRID],
    )


def regrid(grid: pd.DataFrame, strikes_pct: Sequence[float], market: Market | None) -> pd.DataFrame:
    """Same maturities on new strike columns, filled from the current surface when possible.

    Existing columns keep their values; new ones are read on the interpolated surface, or left
    empty for the user to fill if the current inputs do not form a valid market.
    """
    maturities = grid[MATURITY_COLUMN].to_numpy(dtype=float)
    data: dict[str, Any] = {MATURITY_COLUMN: maturities}
    for strike in sorted(set(strikes_pct)):
        column = strike_column(strike)
        if column in grid.columns:
            data[column] = grid[column].to_numpy(dtype=float)
        elif market is not None:
            vols = market.vol.vol(strike / 100.0 * market.spot, maturities)
            data[column] = np.round(100.0 * vols, 1)
        else:
            data[column] = np.full(maturities.shape, np.nan)
    return pd.DataFrame(data)


def _widget_key(key: str) -> str:
    return f"widget_{key}"


def persistent(
    widget: Callable[..., T],
    label: str,
    key: str,
    default: Any,
    options: Sequence[Any] | None = None,
    **kwargs: Any,
) -> T:
    """Draw ``widget`` with a value that survives page switches.

    The value lives under ``key``. The widget uses a derived key that is reset from that stored
    value on every draw, and a change by the user is copied back by an ``on_change`` callback
    (callbacks run before the script, so the user's input is never overwritten). For widgets
    with ``options``, a stored value that is no longer offered falls back to the first option.
    """
    st.session_state.setdefault(key, default)
    if options is not None:
        if st.session_state[key] not in options:
            st.session_state[key] = options[0]
        kwargs["options"] = options
    widget_key = _widget_key(key)
    st.session_state[widget_key] = st.session_state[key]

    def store() -> None:
        st.session_state[key] = st.session_state[widget_key]

    return widget(label, key=widget_key, on_change=store, **kwargs)


def choice(
    widget: Callable[..., Any],
    label: str,
    key: str,
    options: Sequence[V],
    default: V | None = None,
    **kwargs: Any,
) -> V:
    """:func:`persistent` for a selectbox or radio: always returns one of ``options``.

    The first option is selected until the user picks another, unless ``default`` is given.
    """
    first = options[0] if default is None else default
    value = persistent(widget, label, key, first, options=options, **kwargs)
    return options[list(options).index(value)]
