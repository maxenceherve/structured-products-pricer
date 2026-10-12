"""Vanilla options pricer: product, market data, Price button, then results."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import charts
import presentation as pres
import state
from spricer.strategies import CATALOG, Family, Strategy, StrategySpec, specs_by_family

CUSTOM = "Custom"
PRICED = "vanilla_priced_signature"
VOLS_EDITOR, VOLS_BASE, VOLS_SOURCE, VOLS = (
    "vanilla_vols_editor",
    "vanilla_vols_base",
    "vanilla_vols_src",
    "vanilla_vols",
)
LEGS_EDITOR, LEGS_BASE, LEGS = "vanilla_legs_editor", "vanilla_legs_base", "vanilla_legs"
PERCENT = "{:.2f}"


def _spec(name: str) -> StrategySpec:
    return next(spec for spec in CATALOG if spec.name == name)


def _editor(data: pd.DataFrame, key: str, base_key: str, **kwargs: Any) -> pd.DataFrame:
    """Data editor whose edits survive page switches (the edited table becomes the base)."""
    if key not in st.session_state:
        st.session_state[base_key] = data
    edited: pd.DataFrame = st.data_editor(st.session_state[base_key], key=key, **kwargs)
    return edited


def _pick_product() -> str | None:
    """Search box, then family, then product. Returns the product name (or ``CUSTOM``)."""
    query = state.persistent(
        st.text_input,
        "Find a product",
        "vanilla_search",
        "",
        placeholder="Search: butterfly, collar, straddle, digital, bull spread...",
    )
    words = query.lower().split()
    if words:
        found = [
            spec
            for spec in CATALOG
            if all(
                w in f"{spec.name} {spec.family.value} {spec.description}".lower() for w in words
            )
        ]
        # products whose own name matches come first, those that only mention it after
        found.sort(key=lambda spec: not all(w in spec.name.lower() for w in words))
        matches = [spec.name for spec in found]
        if any(w in "custom build your own legs" for w in words):
            matches.append(CUSTOM)
        if not matches:
            st.warning(f"No product matches “{query}”. Clear the search to browse by family.")
            return None
        return state.choice(st.pills, "Results", "vanilla_product_search", matches, required=True)

    families = [family.value for family in Family]
    family = Family(state.choice(st.pills, "Family", "vanilla_family", families, required=True))
    st.caption(pres.FAMILY_BLURB[family.value])
    if family is Family.CUSTOM:
        return CUSTOM
    names = [spec.name for spec in specs_by_family()[family]]
    return state.choice(st.pills, "Product", f"vanilla_product_{family.name}", names, required=True)


def _product_inputs(spot: float) -> tuple[Strategy | None, str]:
    """Draw the product section; return the strategy (None if invalid) and its fingerprint."""
    product = _pick_product()
    if product is None:
        return None, "no-product"

    cols = st.columns([1, 1, 1.4])
    with cols[1]:
        maturity = state.persistent(
            st.number_input,
            "Maturity (years)",
            "vanilla_maturity",
            1.0,
            min_value=0.01,
            max_value=30.0,
            step=0.25,
            format="%.2f",
        )
    with cols[2]:
        state.persistent(
            st.number_input,
            "Notional",
            "vanilla_notional",
            1_000_000.0,
            min_value=1.0,
            step=100_000.0,
            format="%.0f",
        )

    if product == CUSTOM:
        st.caption(
            "One row per leg. Quantity: number of units for options and the underlying; cash "
            "paid, in % of spot, for digitals and zero-coupons. The strike is ignored for the "
            "underlying and zero-coupons."
        )
        st.session_state.setdefault(LEGS, pres.DEFAULT_CUSTOM_LEGS)
        legs = _editor(
            st.session_state[LEGS],
            LEGS_EDITOR,
            LEGS_BASE,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Position": st.column_config.SelectboxColumn(
                    options=["Long", "Short"], required=True
                ),
                "Instrument": st.column_config.SelectboxColumn(
                    options=pres.INSTRUMENTS, required=True
                ),
                "Strike (% of spot)": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
                "Quantity": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            },
        )
        st.session_state[LEGS] = legs
        fingerprint = pres.signature(product, maturity, legs)
        try:
            return pres.custom_strategy(legs, maturity, spot), fingerprint
        except ValueError as exc:
            st.error(str(exc))
            return None, fingerprint

    spec = _spec(product)
    with cols[0]:
        side = state.choice(
            st.radio, "Position", "vanilla_side", ["Long", "Short"], horizontal=True
        )
    st.markdown(f"**{spec.name}** — {spec.description}")
    inputs = st.columns(max(len(spec.strike_labels) + int(spec.uses_payout), 3))
    strikes_pct = []
    for i, (label, default) in enumerate(
        zip(spec.strike_labels, spec.default_strikes_pct, strict=True)
    ):
        with inputs[i]:
            strikes_pct.append(
                state.persistent(
                    st.number_input,
                    f"{label} (% of spot)",
                    f"vanilla_strike_{spec.name}_{i}",
                    default,
                    min_value=0.5,
                    step=1.0,
                    format="%.1f",
                )
            )
    payout_pct = 10.0
    if spec.uses_payout:
        with inputs[len(spec.strike_labels)]:
            payout_pct = state.persistent(
                st.number_input,
                "Amount paid (% of notional)",
                "vanilla_payout",
                10.0,
                min_value=0.0,
                step=1.0,
                format="%.1f",
            )
    fingerprint = pres.signature(product, side, maturity, strikes_pct, payout_pct)
    try:
        strategy = spec.build(
            [spot * k / 100.0 for k in strikes_pct], maturity, payout_pct / 100.0 * spot
        )
    except ValueError as exc:
        st.error(str(exc))
        return None, fingerprint
    return (strategy.short() if side == "Short" else strategy), fingerprint


st.title("Vanilla options")

# --- 1. Product ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1 · Product")
    product_box = st.container()

# --- 2. Market data -----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("2 · Market data")
    m = st.columns(3)
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
            help="Continuously compounded risk-free rate.",
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
    vols_box = st.container()

with product_box:
    strategy, product_fingerprint = _product_inputs(spot)

try:
    market = state.current_market()
except ValueError as exc:
    with vols_box:
        st.error(f"The volatility surface is invalid: {exc}")
        st.page_link("views/volatility_surface.py", label="Fix the volatility surface")
    st.stop()

if strategy is None:
    st.stop()

with vols_box:
    if strategy.strikes:
        st.markdown("**Volatility of each leg**")
        smile = pres.smile_vols(strategy, market)
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
            disabled=["Leg", "Strike (% of spot)"],
            column_config={
                "Leg": st.column_config.TextColumn("Leg"),
                "Strike (% of spot)": st.column_config.NumberColumn(
                    "Strike (% of spot)", format="%.1f"
                ),
                "Volatility (%)": st.column_config.NumberColumn(
                    "✏️ Volatility (%)",
                    min_value=0.01,
                    max_value=500.0,
                    format="%.2f",
                    help="Read on the smile at this strike. Double-click to type your own.",
                ),
            },
        )
        st.session_state[VOLS] = vols
        left, right = st.columns([3, 1])
        left.caption(
            "Pre-filled from the volatility surface at each leg's strike. "
            "Double-click a ✏️ volatility to type your own."
        )
        with right:
            st.page_link(
                "views/volatility_surface.py", label="Edit the smile", icon=":material/tune:"
            )
        try:
            market = pres.with_vols(market, vols, smile)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        edited = pres.changed_vols(vols, smile)
        if len(edited):
            lines = ", ".join(
                f"**{row['Leg']}** {row['Volatility (%)']:.2f} % (smile {row['Smile (%)']:.2f} %)"
                for _, row in edited.iterrows()
            )
            st.warning(f"Using your own volatility for {lines}.", icon=":material/edit:")
    else:
        vols = pd.DataFrame()
        st.caption("No strike: this product uses no volatility.")

# --- 3. Price -----------------------------------------------------------------------------
fingerprint = pres.signature(
    product_fingerprint,
    spot,
    market.rate,
    market.dividend_yield,
    vols,
    st.session_state[state.GRID],
)
if st.button("Price", type="primary", icon=":material/calculate:", key="price_button"):
    st.session_state[PRICED] = fingerprint
if st.session_state.get(PRICED) != fingerprint:
    st.info("Set the product and the market data, then click **Price**.")
    st.stop()

# --- Results ------------------------------------------------------------------------------
notional = float(st.session_state["vanilla_notional"])
price_pct = 100.0 * float(strategy.value(market)) / market.spot

with st.container(border=True):
    st.subheader(f"Result · {strategy.name}")
    cols = st.columns(2)
    cols[0].metric("Price", f"{price_pct:.2f} %", help="In % of notional.")
    cols[1].metric("Premium", f"{price_pct / 100.0 * notional:,.0f}", help="Price × notional.")

    st.markdown("#### Payoff")
    net = st.checkbox(
        "Net of premium (P&L at maturity)",
        key="vanilla_net_of_premium",
        help="Payoff minus the premium compounded at the risk-free rate: shows the breakevens.",
    )
    st.plotly_chart(charts.payoff_figure(strategy, market, net_of_premium=net))

    st.markdown("#### Brick decomposition")
    table = pres.decomposition_table(strategy, market)
    formats = {
        column: ("{:+.4f}" if column in pres.GREEK_HELP else PERCENT)
        for column in table.columns
        if column != "Leg"
    }
    st.table(pres.style_decomposition(table, formats))
    st.caption(
        "Price and Greeks in % of notional. Green: gain, red: loss. Delta (%): exposure to the "
        "underlying · Gamma (%): change of Delta for +1 % spot · Vega: +1 vol point · "
        "Theta: one day · Rho: +1 bp."
    )
    if "of which skew (%)" in table.columns:
        skew = float(table["of which skew (%)"].iloc[-1])
        st.info(
            f"The smile slope adds {skew:+.2f} % to the digitals: priced with the flat "
            f"Black-Scholes formula at the volatility of their strike they would be worth "
            f"{price_pct - skew:.2f} % instead of {price_pct:.2f} %."
        )

with st.expander("How is this priced?"):
    st.markdown(
        """
- Each leg is a closed-form Black-Scholes-Merton price at the implied volatility of its own
  strike; the product is the sum of its legs.
- A digital is minus the derivative of the call price with respect to the strike along the
  smile: the flat formula plus `-vega × dσ/dK`.
- The underlying leg is a prepaid forward worth `S e^(-qT)`: it delivers the share at maturity
  without the dividends paid meanwhile.
- *Net of premium* subtracts the premium compounded at the risk-free rate to maturity.
"""
    )
