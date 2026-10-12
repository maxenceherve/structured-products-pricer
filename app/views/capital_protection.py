"""Capital protected notes: product, market data, Price button, then results."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import charts
import note_presentation as npres
import presentation as pres
import state
from spricer.products.capital_protection import NOTE_CATALOG, NoteSpec
from spricer.products.notes import Note

PRICED = "cp_priced_signature"
VOLS_EDITOR, VOLS_BASE, VOLS_SOURCE, VOLS = (
    "cp_vols_editor",
    "cp_vols_base",
    "cp_vols_src",
    "cp_vols",
)
NAMES = [spec.name for spec in NOTE_CATALOG]


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


def _product() -> tuple[Note | None, str]:
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
    if spec.has_barrier:
        st.caption(
            "Barrier observed at maturity (European). Continuous and daily observation come "
            "with the barrier module."
        )
    fingerprint = pres.signature(name, maturity, initial, sorted(values.items()))
    try:
        return spec.build(values, maturity, initial), fingerprint
    except ValueError as exc:
        st.error(str(exc))
        return None, fingerprint


st.title("Capital protection")

with st.container(border=True):
    st.subheader("1 · Product")
    note, product_fingerprint = _product()

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
