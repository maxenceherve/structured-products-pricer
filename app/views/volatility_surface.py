"""Volatility surface page: the implied volatility grid, its smile, density and checks."""

import streamlit as st

import charts
import state
from spricer.market import InterpolatedVolSurface

GRID_EDITOR = "market_grid_editor"
GRID_BASE = "market_grid_base"
STRIKES_TEXT = "market_strikes_text"


def _restart_editor() -> None:
    st.session_state.pop(GRID_EDITOR, None)
    st.session_state[GRID_BASE] = st.session_state[state.GRID]


def _reset() -> None:
    state.reset_market()
    st.session_state.pop(STRIKES_TEXT, None)
    st.session_state.pop(f"widget_{STRIKES_TEXT}", None)
    _restart_editor()


def _apply_strikes() -> None:
    # the widget holds what was just typed; the stored copy is only refreshed by the next run
    text = st.session_state.get(f"widget_{STRIKES_TEXT}", st.session_state.get(STRIKES_TEXT, ""))
    try:
        strikes = [float(s) for s in text.replace(";", ",").split(",") if s.strip()]
    except ValueError:
        st.session_state["market_strikes_error"] = f"Could not read strikes from {text!r}."
        return
    if len(strikes) < 2 or min(strikes) <= 0:
        st.session_state["market_strikes_error"] = "Enter at least two positive strikes."
        return
    st.session_state.pop("market_strikes_error", None)
    try:
        market = state.current_market()
    except ValueError:
        market = None
    st.session_state[state.GRID] = state.regrid(st.session_state[state.GRID], strikes, market)
    _restart_editor()


st.title("Volatility surface")
st.caption(
    "Typed by hand, starting from synthetic defaults (no market data is downloaded). "
    "Volatilities are quoted at strikes in % of the spot; products read them by absolute strike, "
    "so the smile stays put when the spot moves (sticky strike). Spot, rate and dividend yield "
    "are shared with the pricing pages."
)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    state.persistent(
        st.number_input, "Spot", state.SPOT, 100.0, min_value=0.01, step=1.0, format="%.2f"
    )
with c2:
    state.persistent(
        st.number_input, "Rate (%, continuous)", state.RATE_PCT, 3.0, step=0.25, format="%.2f"
    )
with c3:
    state.persistent(
        st.number_input,
        "Dividend yield (%, continuous)",
        state.DIVIDEND_PCT,
        2.0,
        step=0.25,
        format="%.2f",
    )
with c4:
    st.write("")
    st.button("Reset to defaults", on_click=_reset, icon=":material/restart_alt:")

st.subheader("Implied volatility grid (vol points)")
default_strikes = ", ".join(f"{k:g}" for k in state.grid_strikes(st.session_state[state.GRID]))
left, right = st.columns([3, 1])
with left:
    state.persistent(
        st.text_input, "Strikes (% of spot), comma separated", STRIKES_TEXT, default_strikes
    )
with right:
    st.write("")
    st.button(
        "Apply strikes",
        on_click=_apply_strikes,
        help="New strikes are filled from the current interpolated smile; edit them as needed.",
    )
if "market_strikes_error" in st.session_state:
    st.error(st.session_state["market_strikes_error"])

if GRID_EDITOR not in st.session_state:
    st.session_state[GRID_BASE] = st.session_state[state.GRID]
st.session_state[state.GRID] = st.data_editor(
    st.session_state[GRID_BASE],
    key=GRID_EDITOR,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        state.MATURITY_COLUMN: st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
        **{
            state.strike_column(k): st.column_config.NumberColumn(min_value=0.0, format="%.1f")
            for k in state.grid_strikes(st.session_state[GRID_BASE])
        },
    },
)
st.caption("Add or delete maturities with the table controls; each row needs every strike.")

try:
    market = state.current_market()
except ValueError as exc:
    st.error(f"These inputs do not form a valid market: {exc}")
    st.stop()

assert isinstance(market.vol, InterpolatedVolSurface)
surface = market.vol
report = surface.check_arbitrage()
if report.is_arbitrage_free:
    st.success("No arbitrage: the implied density is positive and total variance increases.")
else:
    lines = [
        f"- butterfly at {v.maturity:g}y, strikes {v.strike_pct_from:.0f}-{v.strike_pct_to:.0f} %"
        for v in report.butterfly
    ] + [
        f"- calendar between {v.maturity_from:g}y and {v.maturity_to:g}y, forward moneyness "
        f"{v.moneyness_pct_from:.0f}-{v.moneyness_pct_to:.0f} %"
        for v in report.calendar
    ]
    st.warning("Arbitrage detected (prices remain computable):\n" + "\n".join(lines))

smile_col, density_col = st.columns(2)
with smile_col:
    st.markdown("**Smile**: typed points and interpolation")
    st.plotly_chart(charts.smile_figure(surface))
with density_col:
    maturity = state.choice(
        st.selectbox,
        "Implied density at maturity (years)",
        "market_density_maturity",
        [float(t) for t in surface.maturities],
    )
    st.plotly_chart(charts.density_figure(surface, maturity))
