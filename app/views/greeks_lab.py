"""Greeks lab: move the strike, the time and the volatility, and watch the Greeks react."""

from __future__ import annotations

import numpy as np
import streamlit as st

import charts
import greeks_explain as gx
import state
from instruments import crossed_barrier_warning, draw_inputs
from spricer.greeks_lab import (
    ATM_BAND,
    FIRST_ORDER,
    SECOND_ORDER,
    Moneyness,
    greeks_at,
    moneyness,
    second_order_at,
    spot_profile,
    time_profile,
)
from spricer.market import Market

SPOTS_PCT = np.linspace(50.0, 150.0, 201)


def _atm_band(market: Market, strike: float, maturity: float) -> tuple[float, float]:
    """Spots (% of today) where an option struck at ``strike`` is at the money."""
    sd = float(market.vol.vol(strike, maturity)) * np.sqrt(maturity)
    carry = np.exp((market.rate - market.dividend_yield) * maturity)
    centre = strike / carry / market.spot * 100.0
    return centre * float(np.exp(-ATM_BAND * sd)), centre * float(np.exp(ATM_BAND * sd))


st.title("Greeks lab")
st.caption(
    "Pick an option, move its strike, its maturity and the volatility, and see how the price "
    "and the Greeks react, with what they mean. Notional = today's spot; Greeks in the desk "
    "units of the other pages."
)

with st.container(border=True):
    selection = draw_inputs("lab")
name, family, values, maturity = (
    selection.name,
    selection.family,
    selection.values,
    selection.maturity,
)

try:
    base = state.current_market()
except ValueError as exc:
    st.error(f"The volatility surface is invalid: {exc}")
    st.stop()
market = selection.market(base)
instrument = selection.instrument(market.spot)
try:
    strategy = instrument(maturity)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

crossed_barrier_warning(selection)
spots = SPOTS_PCT / 100.0 * market.spot
profile = spot_profile(instrument, market, maturity, spots)
now = {
    k: float(v)
    for k, v in (
        greeks_at(strategy, market) | second_order_at(instrument, market, maturity)
    ).items()
}
strikes_pct = selection.strikes_pct
single = len(strikes_pct) == 1 and family != "volatility"
kind = selection.kind
money = moneyness(kind, strikes_pct[0] / 100.0 * market.spot, market, maturity) if single else None
quarter = (
    float(greeks_at(instrument(maturity / 4.0), market)["Gamma (%)"]) if maturity > 0.08 else None
)

with st.container(border=True):
    head = st.columns([3, 2])
    head[0].subheader(f"{values['side']} {strategy.name}")
    if money is not None:
        color = {Moneyness.ITM: "green", Moneyness.ATM: "blue", Moneyness.OTM: "red"}[money]
        head[1].markdown(f"### :{color}-badge[{money.value}]")
    row1, row2 = st.columns(3), st.columns(3)
    for col, key in zip([*row1, *row2], FIRST_ORDER, strict=True):
        fmt = "{:+.4f}" if key in ("Theta (%)", "Rho (%)") else "{:+.2f}"
        col.metric(key, fmt.format(now[key]) if key != "Price (%)" else f"{now[key]:.2f}")

    situation = gx.Situation(
        name,
        family,
        values["side"] == "Short",
        money,
        knock_out=family == "barrier" and str(values["barrier_type"]).endswith("out"),
    )
    st.markdown("#### What the Greeks say")
    for line in gx.first_order(situation, now, SPOTS_PCT, profile, quarter):
        st.markdown(f"- {line}")

bands = tuple(_atm_band(market, k / 100.0 * market.spot, maturity) for k in strikes_pct)
itm_side = ("above" if kind == "call" else "below") if single else None
spot_tab, time_tab, expert_tab = st.tabs(
    ["Against the spot", "Against time", "Second order (expert)"]
)
with spot_tab:
    st.plotly_chart(
        charts.greeks_grid_figure(
            SPOTS_PCT,
            profile,
            FIRST_ORDER,
            strikes_pct=strikes_pct,
            atm_bands=bands,
            itm_side=itm_side,
            title="Price and Greeks against the spot (blue band: at the money)",
        )
    )
    st.caption(
        "Dotted line: today's spot. Dashed lines: strikes. Green: in the money, red: out of "
        "the money, blue band: at the money (within a quarter of a standard deviation)."
    )
with time_tab:
    greek = state.choice(
        st.selectbox, "Greek", "lab_time_greek", list(FIRST_ORDER[1:]), "Gamma (%)"
    )
    times = np.linspace(maturity, 0.02, 60)
    k = strikes_pct[0] / 100.0 * market.spot
    labels = (
        {"In the money": 1.15, "At the money": 1.0, "Out of the money": 0.85}
        if kind == "call"
        else {"In the money": 0.85, "At the money": 1.0, "Out of the money": 1.15}
    )
    curves = {
        f"{label} (spot {100 * f * k / market.spot:.0f} %)": time_profile(
            instrument, market, times, f * k
        )[greek]
        for label, f in labels.items()
    }
    st.plotly_chart(charts.greek_time_figure(times, curves, greek))
    st.caption(
        "The curves run toward expiry (right). At the money, gamma and theta grow without "
        "bound as expiry approaches; in and out of the money they fade to zero."
    )
with expert_tab:
    st.plotly_chart(
        charts.greeks_grid_figure(
            SPOTS_PCT,
            profile,
            SECOND_ORDER,
            strikes_pct=strikes_pct,
            atm_bands=bands,
            itm_side=itm_side,
            title="Second-order Greeks against the spot",
        )
    )
    for line in gx.second_order(now):
        st.markdown(f"- {line}")
