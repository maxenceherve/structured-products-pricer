"""Capital protected notes: product, market data, Price button, then results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import charts
import note_presentation as npres
import presentation as pres
import state
import termsheet
from spricer.analytic.barrier import bgk_warning
from spricer.market import OverriddenVol
from spricer.mc import price_note
from spricer.mc.paths import Z_95
from spricer.monitoring import Monitoring
from spricer.products.capital_protection import NOTE_CATALOG, NoteSpec
from spricer.products.notes import Note
from spricer.solve import solve_note

PRICED = "cp_priced_signature"
MC_RESULT = "cp_mc_result"
SOLUTION = "cp_solution"
SOLVE_BOUNDS = (0.0, 400.0)  # every term is a level or an amount in %
MC_PATHS = {"50,000": 50_000, "100,000": 100_000, "200,000": 200_000}
VOLS_EDITOR, VOLS_BASE, VOLS_SOURCE, VOLS = (
    "cp_vols_editor",
    "cp_vols_base",
    "cp_vols_src",
    "cp_vols",
)
NAMES = [spec.name for spec in NOTE_CATALOG]
MONITORING_CHOICES = {
    "At maturity": Monitoring.EUROPEAN,
    "Continuously": Monitoring.CONTINUOUS,
    "Daily (252 dates a year)": Monitoring.DISCRETE,
}


def _editor(data: pd.DataFrame, key: str, base_key: str, **kwargs: Any) -> pd.DataFrame:
    """Data editor whose edits survive page switches (the edited table becomes the base)."""
    if key not in st.session_state:
        st.session_state[base_key] = data
    edited: pd.DataFrame = st.data_editor(st.session_state[base_key], key=key, **kwargs)
    return edited


def _terms(spec: NoteSpec) -> dict[str, float]:
    """One input per term of the note, in % of initial level or of notional."""
    values: dict[str, float] = {}
    toggles = {p.key: p for p in spec.params if p.kind == "toggle"}
    shown = [p for p in spec.params if p.kind != "toggle"]
    cols = st.columns(4)
    for i, param in enumerate(shown):
        # a capped/limited level is only asked when its toggle is on
        toggle = next(
            (
                t
                for t in toggles.values()
                if param.key in ("cap", "floor")
                and t.key in ("capped" if param.key == "cap" else "limited",)
            ),
            None,
        )
        with cols[i % 4]:
            if toggle is not None:
                on = state.persistent(
                    st.checkbox, toggle.label, f"cp_{spec.name}_{toggle.key}", bool(toggle.default)
                )
                values[toggle.key] = 1.0 if on else 0.0
                if not on:
                    continue
            values[param.key] = state.persistent(
                st.number_input,
                f"{param.label} (%)",
                f"cp_{spec.name}_{param.key}",
                param.default,
                min_value=0.0,
                step=1.0,
                format="%.1f",
                help=param.help or None,
            )
    return values


def _product() -> tuple[Note | None, str, dict[str, Any]]:
    """Draw the product section; return the note (None if invalid) and its fingerprint."""
    name = state.choice(st.pills, "Product", "cp_product", NAMES, required=True)
    spec = next(s for s in NOTE_CATALOG if s.name == name)
    st.markdown(f"**{spec.name}** — {spec.description}")
    top = st.columns(4)
    with top[0]:
        maturity = state.persistent(
            st.number_input,
            "Maturity (years)",
            "cp_maturity",
            3.0,
            min_value=0.1,
            max_value=30.0,
            step=0.5,
            format="%.2f",
        )
    with top[1]:
        state.persistent(
            st.number_input,
            "Notional",
            "cp_notional",
            1_000_000.0,
            min_value=1.0,
            step=100_000.0,
            format="%.0f",
        )
    with top[2]:
        initial = state.persistent(
            st.number_input,
            "Initial level of the underlying",
            "cp_initial",
            100.0,
            min_value=0.01,
            step=1.0,
            format="%.2f",
            help="The 100 % of the term sheet: every level below is in % of it.",
        )
    st.markdown("**Terms** (levels in % of the initial level, amounts in % of the notional)")
    values = _terms(spec)
    monitoring, observations = Monitoring.EUROPEAN, None
    if spec.monitoring_choice:
        choice = state.choice(
            st.radio,
            "Barrier observation",
            "cp_monitoring",
            list(MONITORING_CHOICES),
            horizontal=True,
            help="At maturity: exact replication with the smile. Continuously or daily: "
            "closed form (Reiner-Rubinstein), daily with the Broadie-Glasserman-Kou "
            "correction.",
        )
        monitoring = MONITORING_CHOICES[choice]
        if monitoring is Monitoring.DISCRETE:
            observations = max(1, round(252 * maturity))
            warning = bgk_warning(observations)
            if warning:
                st.warning(warning)
        if monitoring is not Monitoring.EUROPEAN:
            st.caption(
                "The chart shows the payoff when the barrier was never touched before maturity; "
                "once touched, only the protection and the rebate are paid."
            )
    elif spec.has_barrier:
        st.caption("Barrier observed at maturity (European).")
    fingerprint = pres.signature(
        name, maturity, initial, sorted(values.items()), monitoring.value, observations
    )
    context: dict[str, Any] = {
        "spec": spec,
        "values": values,
        "maturity": maturity,
        "initial": initial,
        "monitoring": monitoring,
        "observations": observations,
    }
    try:
        return (
            spec.build(values, maturity, initial, monitoring, observations),
            fingerprint,
            context,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None, fingerprint, context


st.title("Capital protection")

with st.container(border=True):
    st.subheader("1 · Product")
    note, product_fingerprint, product_context = _product()

with st.container(border=True):
    st.subheader("2 · Market data")
    m = st.columns(4)
    with m[0]:
        spot = state.persistent(
            st.number_input, "Spot", state.SPOT, 100.0, min_value=0.01, step=1.0, format="%.2f"
        )
    with m[1]:
        state.persistent(
            st.number_input,
            "Interest rate (%)",
            state.RATE_PCT,
            3.0,
            step=0.25,
            format="%.2f",
            help="Continuously compounded risk-free rate: drift of the underlying.",
        )
    with m[2]:
        state.persistent(
            st.number_input,
            "Dividend yield (%)",
            state.DIVIDEND_PCT,
            2.0,
            step=0.25,
            format="%.2f",
            help="Continuous dividend yield.",
        )
    with m[3]:
        spread_bp = state.persistent(
            st.number_input,
            "Issuer credit spread (bp)",
            "cp_spread_bp",
            0.0,
            step=10.0,
            format="%.0f",
            help="Discounts every cash flow of the note (bond and options), not the drift.",
        )
    vols_box = st.container()

try:
    market = state.current_market()
    smile_market = market  # before any volatility typed on this page
except ValueError as exc:
    with vols_box:
        st.error(f"The volatility surface is invalid: {exc}")
        st.page_link("views/volatility_surface.py", label="Fix the volatility surface")
    st.stop()
if note is None:
    st.stop()

with vols_box:
    perf = npres.performance_pct(note, market)
    st.markdown(
        f"**Underlying**: initial {note.initial:,.2f} · spot {market.spot:,.2f} · "
        f"performance **{perf:.1f} %** of initial"
    )
    smile = npres.smile_vols(note, market)
    if len(smile):
        source = pres.signature(smile)
        if st.session_state.get(VOLS_SOURCE) != source:  # new strikes or new smile
            st.session_state[VOLS_SOURCE] = source
            st.session_state.pop(VOLS_EDITOR, None)
            st.session_state[VOLS] = smile
        vols = _editor(
            st.session_state[VOLS],
            VOLS_EDITOR,
            VOLS_BASE,
            hide_index=True,
            disabled=["Level", npres.STRIKE_COLUMN, "% of spot"],
            column_config={
                npres.STRIKE_COLUMN: st.column_config.NumberColumn(format="%.1f"),
                "% of spot": st.column_config.NumberColumn(
                    format="%.1f", help="Where the strike sits on today's smile."
                ),
                npres.VOL_COLUMN: st.column_config.NumberColumn(
                    "✏️ Volatility (%)",
                    min_value=0.01,
                    max_value=500.0,
                    format="%.2f",
                    help="Read on the smile at the strike's true moneyness. Double-click to edit.",
                ),
            },
        )
        st.session_state[VOLS] = vols
        st.caption(
            "Each strike is read on the smile at its true moneyness (% of today's spot). "
            "Double-click a ✏️ volatility to type your own."
        )
        try:
            market = npres.with_vols(market, note, vols, smile)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    else:
        vols = pd.DataFrame()

fingerprint = pres.signature(
    product_fingerprint,
    spot,
    market.rate,
    market.dividend_yield,
    spread_bp,
    vols,
    st.session_state[state.GRID],
)
if st.button("Price", type="primary", icon=":material/calculate:", key="cp_price_button"):
    st.session_state[PRICED] = fingerprint
if st.session_state.get(PRICED) != fingerprint:
    st.info("Set the product and the market data, then click **Price**.")
    st.stop()

spread = spread_bp / 1e4
notional = float(st.session_state["cp_notional"])
price = float(note.value(market, spread))
bond = npres.bond_value(note, market, spread)

with st.container(border=True):
    st.subheader(f"Result · {note.name}")
    cols = st.columns(4)
    cols[0].metric("Price", f"{price:.2f} %", help="In % of notional.")
    cols[1].metric("Value", f"{price / 100.0 * notional:,.0f}", help="Price × notional.")
    cols[2].metric("Bond part", f"{bond:.2f} %", help="Zero-coupon paying the protection.")
    cols[3].metric("Options", f"{price - bond:+.2f} %", help="Everything else: the bricks.")

    st.markdown("#### Payoff")
    st.plotly_chart(charts.note_payoff_figure(note, market))

    st.markdown("#### Brick decomposition")
    table = npres.decomposition_table(note, market, spread)
    formats = {
        column: ("{:+.4f}" if column in pres.GREEK_HELP else "{:.2f}")
        for column in table.columns
        if column != "Leg"
    }
    st.table(pres.style_decomposition(table, formats))
    st.caption(
        "Price and Greeks in % of notional. Green: gain, red: cost. Delta (%): exposure to the "
        "underlying · Gamma (%): change of Delta for +1 % spot · Vega: +1 vol point · "
        "Theta: one day · Rho: +1 bp."
    )
    if "of which skew (%)" in table.columns:
        skew = float(table["of which skew (%)"].iloc[-1])
        st.info(
            f"The smile slope changes the digitals by {skew:+.2f} % of notional: with flat "
            f"Black-Scholes digitals the note would be worth {price - skew:.2f} % instead of "
            f"{price:.2f} %."
        )
    sheet_spec: NoteSpec = product_context["spec"]
    labels = {p.key: f"{p.label} (%)" for p in sheet_spec.params if p.kind != "toggle"}
    shown = {k: v for k, v in product_context["values"].items() if k in labels}
    performances = np.arange(40.0, 170.0, 10.0)
    sheet = termsheet.product_sheet(
        note.name,
        sheet_spec.description,
        [
            (
                "Terms",
                termsheet.terms_table(
                    {**shown, "Maturity (years)": float(product_context["maturity"])}, labels
                ),
            ),
            (
                "Valuation",
                f"Fair value {price:.2f} % of notional, of which the bond part {bond:.2f} % and "
                f"the options {price - bond:+.2f} %. Barrier observation: "
                f"{(note.monitoring or Monitoring.EUROPEAN).value.lower()}.",
            ),
            ("Brick decomposition", table),
            (
                "Payoff at maturity",
                termsheet.maturity_scenarios(note.payoff(performances), performances),
            ),
        ],
    )
    st.download_button(
        "Download product sheet",
        sheet,
        file_name=termsheet.file_name(note.name),
        mime="text/html",
        icon=":material/description:",
        key="cp_sheet",
    )

with st.container(border=True):
    st.subheader("Monte Carlo check")
    st.caption(
        "Simulates the underlying under the local volatility of the whole smile (Dupire) and "
        "prices every brick on the same paths."
    )
    left, right = st.columns([1.4, 1], vertical_alignment="bottom")
    with left:
        paths_label = state.choice(
            st.radio, "Paths", "cp_mc_paths", list(MC_PATHS), "100,000", horizontal=True
        )
    mc_key = (fingerprint, paths_label)
    with right:
        run_mc = st.button("Check with Monte Carlo", icon=":material/casino:", key="cp_mc_button")
    if run_mc:
        with st.spinner("Simulating paths..."):
            try:
                result = price_note(note, smile_market, spread, n_paths=MC_PATHS[paths_label])
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
        st.session_state[MC_RESULT] = (mc_key, result)
    stored = st.session_state.get(MC_RESULT)
    if stored is not None and stored[0] == mc_key:
        mc = stored[1]
        closed = float(note.value(smile_market, spread))
        gap = (mc.total.value - closed) / mc.total.std_error
        c = st.columns(3)
        c[0].metric("Monte Carlo", f"{mc.total.value:.2f} %")
        c[1].metric(
            "95 % interval",
            f"± {Z_95 * mc.total.std_error:.2f} %",
            help=f"1.96 standard errors; one standard error is {mc.total.std_error:.3f} %.",
        )
        c[2].metric(
            "MC − closed form",
            f"{mc.total.value - closed:+.2f} %",
            f"{gap:+.1f} std errors",
            delta_color="off",
            delta_arrow="off",
        )
        mc_table = npres.monte_carlo_table(note, smile_market, spread, mc)
        mc_formats = {
            "Closed form (%)": "{:.3f}",
            "Monte Carlo (%)": "{:.3f}",
            "± 1 std error (%)": "{:.3f}",
            npres.MC_GAP_COLUMN: "{:+.1f}",
        }
        st.table(pres.style_decomposition(mc_table, mc_formats))
        if note.monitoring in (None, Monitoring.EUROPEAN):
            st.caption(
                "Both prices come from the same smile: the gap is Monte Carlo noise and stays "
                "within ±2 standard errors 95 % of the time."
            )
        else:
            st.info(
                "The closed form reads one volatility, the strike's, for the whole path; the "
                "Monte Carlo diffuses on the local volatility of the whole smile. With a "
                "barrier observed along the path the gap is the smile's effect on the barrier, "
                "not noise."
            )
        overridden = isinstance(market.vol, OverriddenVol) and market.vol.overrides
        if overridden:
            st.caption(
                "Volatilities typed above apply to the price at the top only: this check "
                "compares both methods on the smile of the volatility surface."
            )

with st.container(border=True):
    st.subheader("Solve for")
    st.caption(
        "Fix every term but one and find the value that prices the note at your target: for "
        "instance the participation that an issue price of 100 % minus a 2 % margin can pay."
    )
    spec: NoteSpec = product_context["spec"]
    terms = [p for p in spec.params if p.kind != "toggle" and p.key in product_context["values"]]
    s_cols = st.columns([2, 1, 1], vertical_alignment="bottom")
    with s_cols[0]:
        term_label = state.choice(
            st.selectbox, "Term to solve", "cp_solve_term", [p.label for p in terms]
        )
    term = next(p for p in terms if p.label == term_label)
    with s_cols[1]:
        target = state.persistent(
            st.number_input,
            "Target price (%)",
            "cp_solve_target",
            98.0,
            step=0.25,
            format="%.2f",
        )
    with s_cols[2]:
        solve_clicked = st.button("Solve", icon=":material/target:", key="cp_solve_button")
    solve_fingerprint = pres.signature(fingerprint, term.key, target)
    if solve_clicked:

        def build(x: float) -> Note:
            return spec.build(
                {**product_context["values"], term.key: x},
                product_context["maturity"],
                product_context["initial"],
                product_context["monitoring"],
                product_context["observations"],
            )

        try:
            solution = solve_note(build, market, target, SOLVE_BOUNDS, spread, term.label.lower())
            st.session_state[SOLUTION] = (solve_fingerprint, solution)
        except ValueError as exc:
            st.error(str(exc))
    solved = st.session_state.get(SOLUTION)
    if solved is not None and solved[0] == solve_fingerprint:
        st.success(
            f"**{term.label} = {solved[1].value:.2f} %** prices the note at "
            f"{solved[1].price:.2f} % (closed form, exact)."
        )
        if st.button("Use this value", key="cp_solve_apply"):
            st.session_state[f"cp_{spec.name}_{term.key}"] = round(solved[1].value, 2)
            st.rerun()

with st.expander("How is this priced?"):
    st.markdown(
        """
- The note is replicated exactly by a zero-coupon (the capital protection), vanilla options
  and digitals; its price is the sum of their closed-form prices.
- Every strike is read on the smile at its true moneyness: a strike at 108 % of the initial
  level, with the underlying at 122 %, sits at 88.5 % of today's spot.
- Digitals include the smile slope (`-vega × dσ/dK`).
- The issuer's credit spread discounts every cash flow of the note; the underlying keeps its
  risk-free drift.
"""
    )
