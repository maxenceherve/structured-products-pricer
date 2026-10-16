"""Greeks lab: move the strike, the time and the volatility, and watch the Greeks react."""

from __future__ import annotations

import numpy as np
import streamlit as st

import charts
import greeks_explain as gx
import state
from spricer import strategies as strat
from spricer.bricks import BARRIER_TYPES, BarrierOption
from spricer.greeks_lab import (
    ATM_BAND,
    FIRST_ORDER,
    SECOND_ORDER,
    Instrument,
    Moneyness,
    greeks_at,
    moneyness,
    second_order_at,
    spot_profile,
    time_profile,
)
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from spricer.strategies import Family, Leg, Strategy

INSTRUMENTS = {
    "Call": "vanilla",
    "Put": "vanilla",
    "Digital call": "digital",
    "Digital put": "digital",
    "Bull call spread": "spread",
    "Straddle": "volatility",
    "Call butterfly": "spread",
    "Barrier option": "barrier",
}
SPOTS_PCT = np.linspace(50.0, 150.0, 201)
DIGITAL_PAYOUT_PCT = 10.0  # % of the spot paid by a digital


def _instrument(name: str, spot: float, values: dict[str, float | str]) -> Instrument:
    """Strategy as a function of the maturity, for the inputs on screen."""
    k = float(values["strike"]) / 100.0 * spot
    w = float(values["width"]) / 100.0 * spot

    def build(t: float) -> Strategy:
        if name == "Call":
            s = strat.call(k, t)
        elif name == "Put":
            s = strat.put(k, t)
        elif name == "Digital call":
            s = strat.digital_call(k, t, DIGITAL_PAYOUT_PCT / 100.0 * spot)
        elif name == "Digital put":
            s = strat.digital_put(k, t, DIGITAL_PAYOUT_PCT / 100.0 * spot)
        elif name == "Bull call spread":
            s = strat.bull_call_spread(k, k + w, t)
        elif name == "Straddle":
            s = strat.straddle(k, t)
        elif name == "Call butterfly":
            s = strat.call_butterfly(max(k - w, 1e-6), k, k + w, t)
        else:
            option = BarrierOption(
                str(values["kind"]),  # type: ignore[arg-type]
                str(values["barrier_type"]),  # type: ignore[arg-type]
                k,
                float(values["barrier"]) / 100.0 * spot,
                t,
                Monitoring.CONTINUOUS
                if values["monitoring"] == "Continuously"
                else Monitoring.EUROPEAN,
            )
            s = Strategy(option.label, Family.CUSTOM, (Leg(option, 1.0),))
        return s.short() if values["side"] == "Short" else s

    return build


def _strikes_pct(name: str, values: dict[str, float | str]) -> tuple[float, ...]:
    k, w = float(values["strike"]), float(values["width"])
    if name == "Bull call spread":
        return (k, k + w)
    if name == "Call butterfly":
        return (k - w, k, k + w)
    return (k,)


def _kind(name: str, values: dict[str, float | str]) -> str:
    if name == "Barrier option":
        return str(values["kind"])
    return "put" if "Put" in name or name == "Digital put" else "call"


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
    name = state.choice(st.pills, "Instrument", "lab_instrument", list(INSTRUMENTS))
    family = INSTRUMENTS[name]
    values: dict[str, float | str] = {}
    top = st.columns(3)
    with top[0]:
        values["side"] = state.choice(
            st.radio, "Position", "lab_side", ["Long", "Short"], horizontal=True
        )
    with top[1]:
        vol_source = state.choice(
            st.radio,
            "Volatility",
            "lab_vol_source",
            ["Flat", "Smile of the Volatility surface page"],
            horizontal=True,
        )
    sliders = st.columns(3)
    with sliders[0]:
        values["strike"] = state.persistent(
            st.slider,
            "Strike (% of spot)",
            "lab_strike",
            100.0,
            min_value=50.0,
            max_value=150.0,
            step=1.0,
        )
    with sliders[1]:
        maturity = state.persistent(
            st.slider,
            "Time to maturity (years)",
            "lab_maturity",
            1.0,
            min_value=0.02,
            max_value=5.0,
            step=0.01,
        )
    with sliders[2]:
        flat_vol = state.persistent(
            st.slider,
            "Flat volatility (%)",
            "lab_vol",
            20.0,
            min_value=5.0,
            max_value=80.0,
            step=0.5,
            disabled=vol_source != "Flat",
        )
    values["width"] = 10.0
    if family == "spread":
        values["width"] = state.persistent(
            st.slider,
            "Distance between strikes (% of spot)",
            "lab_width",
            10.0,
            min_value=2.0,
            max_value=40.0,
            step=1.0,
        )
    if family == "barrier":
        b = st.columns(4)
        with b[0]:
            values["kind"] = state.choice(st.radio, "Option", "lab_kind", ["call", "put"])
        with b[1]:
            values["barrier_type"] = state.choice(
                st.selectbox, "Barrier", "lab_barrier_type", list(BARRIER_TYPES), "down-and-out"
            )
        with b[2]:
            values["barrier"] = state.persistent(
                st.slider,
                "Barrier (% of spot)",
                "lab_barrier",
                80.0,
                min_value=40.0,
                max_value=160.0,
                step=1.0,
            )
        with b[3]:
            values["monitoring"] = state.choice(
                st.radio, "Observed", "lab_monitoring", ["Continuously", "At maturity"]
            )

try:
    base = state.current_market()
except ValueError as exc:
    st.error(f"The volatility surface is invalid: {exc}")
    st.stop()
market = (
    Market(base.spot, base.rate, base.dividend_yield, FlatVol(flat_vol / 100.0))
    if vol_source == "Flat"
    else base
)
instrument = _instrument(name, market.spot, values)
try:
    strategy = instrument(maturity)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if family == "barrier":
    up = str(values["barrier_type"]).startswith("up")
    crossed = float(values["barrier"]) <= 100.0 if up else float(values["barrier"]) > 100.0
    if crossed:
        st.warning(
            "The barrier is already crossed at today's spot: a knock-out option is dead, a "
            "knock-in option is a vanilla. Move the barrier to the other side of 100 %."
        )
spots = SPOTS_PCT / 100.0 * market.spot
profile = spot_profile(instrument, market, maturity, spots)
now = {
    k: float(v)
    for k, v in (
        greeks_at(strategy, market) | second_order_at(instrument, market, maturity)
    ).items()
}
strikes_pct = _strikes_pct(name, values)
single = len(strikes_pct) == 1 and family != "volatility"
kind = _kind(name, values)
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
