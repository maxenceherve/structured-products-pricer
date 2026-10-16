"""Hedging: neutralise the Greeks of a position, simulate a delta hedge, explain the P&L."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
import hedging_explain as hx
import presentation as pres
import state
from instruments import crossed_barrier_warning, draw_inputs
from spricer.analytic.black_scholes import OptionKind
from spricer.bricks import EuropeanOption
from spricer.conventions import Greeks, to_desk
from spricer.hedging import derman_kamal_std, simulate_delta_hedge, static_hedge
from spricer.market import FlatVol, Market
from spricer.risk import STEPS, TERMS, Shock, explain, scenario_matrix

REBALANCING = {"Daily": 252, "Weekly": 52, "Monthly": 12}
SIM_PATHS = {"2,000": 2_000, "5,000": 5_000, "10,000": 10_000}
HEDGES = ["Delta", "Delta + gamma", "Delta + gamma + vega"]
SPOT_GRID = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0]
VOL_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0]


def _desk_row(name: str, g: Greeks, spot: float, notional: float) -> dict[str, object]:
    d = to_desk(g, spot=spot, notional=notional)
    return {
        "Leg": name,
        "Delta (%)": 100.0 * float(d.delta),
        "Gamma (%)": 100.0 * float(d.gamma),
        "Vega (%)": float(d.vega),
        "Theta (%)": float(d.theta),
    }


def _hedge_option(
    prefix: str, label: str, kind: str, strike: float, maturity: float
) -> tuple[OptionKind, float, float]:
    st.markdown(f"**{label}**")
    c = st.columns(3)
    with c[0]:
        k = state.choice(st.radio, "Type", f"{prefix}_kind", ["call", "put"], kind, horizontal=True)
    with c[1]:
        s = state.persistent(
            st.number_input,
            "Strike (% of spot)",
            f"{prefix}_strike",
            strike,
            min_value=1.0,
            step=5.0,
            format="%.1f",
        )
    with c[2]:
        t = state.persistent(
            st.number_input,
            "Maturity (years)",
            f"{prefix}_maturity",
            maturity,
            min_value=0.02,
            step=0.25,
            format="%.2f",
        )
    option_kind: OptionKind = "call" if k == "call" else "put"
    return option_kind, float(s), float(t)


st.title("Hedging")
st.caption(
    "Hedge an option position as a market maker or as an option buyer: neutralise its Greeks, "
    "simulate a delta hedge through time, and explain its P&L after a market move."
)

with st.container(border=True):
    st.subheader("Position")
    selection = draw_inputs("hedge")
    options = state.persistent(
        st.number_input,
        "Number of options",
        "hedge_options",
        10_000,
        min_value=1,
        step=1_000,
    )
try:
    base = state.current_market()
except ValueError as exc:
    st.error(f"The volatility surface is invalid: {exc}")
    st.stop()
crossed_barrier_warning(selection)
market = selection.market(base)
spot = market.spot
instrument = selection.instrument(spot)
maturity = selection.maturity
try:
    strategy = instrument(maturity)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
position = strategy.greeks(market)
notional = spot  # per option: values in % of today's spot

static_tab, sim_tab, pnl_tab = st.tabs(["Static hedge", "Delta-hedging simulator", "P&L explain"])

with static_tab:
    choice = state.choice(st.radio, "Neutralise", "hedge_target", HEDGES, horizontal=True)
    hedges = []
    if choice != "Delta":
        k1, s1, t1 = _hedge_option("hedge_h1", "First hedging option", "call", 100.0, 0.25)
        o1 = EuropeanOption(k1, s1 / 100.0 * spot, t1)
        hedges.append((f"{k1}s K={s1:g}% {t1:g}y", o1.greeks(market)))
    if choice == "Delta + gamma + vega":
        k2, s2, t2 = _hedge_option("hedge_h2", "Second hedging option", "put", 100.0, 2.0)
        o2 = EuropeanOption(k2, s2 / 100.0 * spot, t2)
        hedges.append((f"{k2}s K={s2:g}% {t2:g}y", o2.greeks(market)))
    try:
        hedge = static_hedge(position, tuple(hedges))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    rows = [
        _desk_row(f"Position: {selection.values['side']} {strategy.name}", position, spot, notional)
    ]
    for leg in hedge.legs:
        units = leg.quantity * options
        verb = "Buy" if units > 0 else "Sell"
        rows.append(
            _desk_row(
                f"{verb} {abs(units):,.0f} × {leg.name}", leg.greeks * leg.quantity, spot, notional
            )
        )
    rows.append(_desk_row("Total", hedge.hedged, spot, notional))
    table = pd.DataFrame(rows)
    st.table(pres.style_decomposition(table, {c: "{:+.4f}" for c in table.columns if c != "Leg"}))
    st.caption(
        "Greeks per option, in % of today's spot (the quantities are for the whole position). "
        "The neutralised Greeks show 0 in the total row."
    )
    for line in hx.static_reading(hedge, int(options), selection.short):
        st.markdown(f"- {line}")

with sim_tab:
    c = st.columns(4)
    implied = float(market.vol.vol(selection.strikes_pct[0] / 100.0 * spot, maturity))
    with c[0]:
        realised_pct = state.persistent(
            st.slider,
            "Realised volatility (%)",
            "hedge_realised",
            round(100 * implied, 1),
            min_value=5.0,
            max_value=80.0,
            step=0.5,
        )
    with c[1]:
        frequency = state.choice(st.selectbox, "Rebalancing", "hedge_frequency", list(REBALANCING))
    with c[2]:
        cost_bp = state.persistent(
            st.number_input,
            "Costs (bp per trade)",
            "hedge_cost",
            0.0,
            min_value=0.0,
            step=1.0,
            format="%.1f",
        )
    with c[3]:
        paths_label = state.choice(st.selectbox, "Paths", "hedge_paths", list(SIM_PATHS), "5,000")
    st.caption(
        f"The hedger values the option at its implied volatility ({100 * implied:.1f} % at the "
        "main strike) and rebalances the delta; the spot actually moves with the realised one."
    )
    if st.button("Simulate", type="primary", icon=":material/play_arrow:", key="hedge_simulate"):
        rebalancings = max(1, round(REBALANCING[frequency] * maturity))
        with st.spinner("Hedging along every path..."):
            sim = simulate_delta_hedge(
                instrument,
                market,
                maturity,
                realised_vol=realised_pct / 100.0,
                rebalancings=rebalancings,
                cost_bp=cost_bp,
                n_paths=SIM_PATHS[paths_label],
            )
        st.session_state["hedge_sim"] = (realised_pct, frequency, cost_bp, sim, rebalancings)
    stored = st.session_state.get("hedge_sim")
    if stored is not None:
        realised_used, freq_used, cost_used, sim, rebalancings = stored
        realised_market = Market(
            spot, market.rate, market.dividend_yield, FlatVol(realised_used / 100.0)
        )
        theory = (
            100.0
            * (
                float(strategy.value(realised_market))
                - float(
                    strategy.value(
                        Market(spot, market.rate, market.dividend_yield, FlatVol(implied))
                    )
                )
            )
            / spot
        )
        vega_pct = 100.0 * float(strategy.greeks(market).vega) / spot
        dk = derman_kamal_std(vega_pct, implied, rebalancings)
        m = st.columns(3)
        m[0].metric(
            "Mean P&L",
            f"{sim.mean.value:+.3f} %",
            f"± {sim.mean.std_error:.3f}",
            delta_color="off",
            delta_arrow="off",
        )
        m[1].metric(
            "Theory",
            f"{theory:+.3f} %",
            help="BS price at the realised vol minus at the implied vol.",
        )
        m[2].metric(
            "Standard deviation",
            f"{sim.std:.3f} %",
            f"Derman-Kamal {dk:.3f}",
            delta_color="off",
            delta_arrow="off",
        )
        st.plotly_chart(charts.hedge_histogram_figure(sim.pnl, theory))
        st.plotly_chart(
            charts.hedge_path_figure(sim.times, sim.spot, sim.delta, sim.gamma_pnl, sim.theta_pnl)
        )
        for line in hx.simulation_reading(
            sim.mean.value,
            theory,
            sim.std,
            dk,
            realised_used,
            100.0 * implied,
            float(position.gamma) > 0.0,
            cost_used,
        ):
            st.markdown(f"- {line}")
        st.caption(
            f"{freq_used} rebalancing ({rebalancings} dates); P&L discounted, in % of notional. "
            "The path chart shows the first simulated path, undiscounted."
        )

with pnl_tab:
    c = st.columns(4)
    with c[0]:
        move = state.persistent(
            st.number_input, "Spot move (%)", "hedge_move", -10.0, step=1.0, format="%.1f"
        )
    with c[1]:
        dvol = state.persistent(
            st.number_input, "Volatility move (pts)", "hedge_dvol", 2.0, step=0.5, format="%.1f"
        )
    with c[2]:
        days = state.persistent(
            st.number_input,
            "Days elapsed",
            "hedge_days",
            7.0,
            min_value=0.0,
            step=1.0,
            format="%.0f",
        )
    with c[3]:
        dr = state.persistent(
            st.number_input, "Rate move (bp)", "hedge_dr", 0.0, step=5.0, format="%.0f"
        )

    def value(m: Market, t: float) -> float:
        return 100.0 * float(instrument(t).value(m)) / spot

    try:
        attribution = explain(value, market, maturity, Shock(move, dvol, days, dr))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    rows = [
        {"Term": k, "P&L (%)": attribution.terms[k], "Greek": attribution.greeks[k]} for k in TERMS
    ]
    rows += [
        {"Term": "Taylor total", "P&L (%)": attribution.taylor, "Greek": np.nan},
        {"Term": "Full revaluation", "P&L (%)": attribution.full, "Greek": np.nan},
        {"Term": "Unexplained", "P&L (%)": attribution.residual, "Greek": np.nan},
    ]
    left, right = st.columns(2)
    with left:
        st.markdown("**Taylor at today's Greeks**")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "P&L (%)": st.column_config.NumberColumn(format="%+.4f"),
                "Greek": st.column_config.NumberColumn(
                    format="%+.5f",
                    help="Per 1 % spot, per (1 %)², per vol point, per day, per bp.",
                ),
            },
        )
    with right:
        st.markdown("**Step by step (exact)**")
        steps = pd.DataFrame(
            [{"Step": k, "P&L (%)": attribution.steps[k]} for k in STEPS]
            + [{"Step": "Total", "P&L (%)": sum(attribution.steps.values())}]
        )
        st.dataframe(
            steps,
            hide_index=True,
            column_config={"P&L (%)": st.column_config.NumberColumn(format="%+.4f")},
        )
        st.caption("Revalued moving time, then the spot, then the volatility, then the rate.")
    for line in hx.attribution_reading(attribution):
        st.markdown(f"- {line}")
    try:
        grid = scenario_matrix(value, market, maturity, SPOT_GRID, VOL_GRID, days)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.plotly_chart(charts.scenario_heatmap_figure(SPOT_GRID, VOL_GRID, grid, days))
        st.caption("Stress test: every cell is a full revaluation, never a Taylor expansion.")
