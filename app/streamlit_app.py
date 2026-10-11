"""Streamlit entry point: ``uv run streamlit run app/streamlit_app.py``."""

import streamlit as st

import spricer

st.set_page_config(page_title="Structured Products Pricer", page_icon="📈", layout="wide")

st.title("Structured Products Pricer")
st.caption(f"spricer {spricer.__version__} · work in progress")

st.markdown(
    """
Price vanilla option strategies and structured products, look at their payoff,
and break each one down into elementary bricks with their individual price and Greeks.

Every market input (spot, rates, dividends, volatility by strike and maturity, correlations)
is entered by hand, with editable synthetic defaults. No market data is downloaded.
"""
)

st.subheader("Coming next")
st.markdown(
    """
- **Vanilla strategies**, grouped by family: vertical spreads, volatility combinations
  (straddle, strangle, strip, strap), butterflies and condors, option + underlying
  (covered call, protective put, collar, risk reversal), digitals, and a free-form builder.
- **Volatility surface**: strike × maturity input grid, smooth interpolation,
  butterfly and calendar arbitrage checks, implied density.
- **Structured products**: reverse convertibles, autocallables (Phoenix, Athena),
  capital protected notes, worst-of baskets, priced by Monte Carlo and checked
  against a closed-form brick decomposition.
"""
)
